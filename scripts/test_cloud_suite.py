"""Real 27B + immutable image, with a live SwanLab experiment audit trail."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import test_real_sdk as live
from scripts.cloud_experiment import CloudJournal
from scripts.live_platforms import load_env, configuration, write_receipt, termination_handler
from scripts.docker_rehearsal import DockerRuntime

DEFAULT_IMAGE = "gas-rsi-a:wyx-sdk104-v4-20261122"
DEFAULT_IMAGE_ID = "sha256:388fb0d9721fa11d0cb9befcb6dee6dbadb23597dfdb59415913d276b4ec25fe"


def snapshot(out, baseline):
    files = list((ROOT / "src/gas_a").rglob("*.py"))
    files += [ROOT / name for name in ("harness.py", "contest_submission.py", "submission_config.json", "Dockerfile",
              ".dockerignore", ".python-version", "requirements.txt", "requirements-platform.txt", "run.py", "rebuild_models.py",
              "scripts/test_real_sdk.py", "scripts/cloud_experiment.py", "scripts/test_cloud_suite.py",
              "scripts/docker_rehearsal.py", "scripts/live_platforms.py", "scripts/simulate_competition.py",
              "scripts/simulation_audit.py", "scripts/validate_platform.py", "tests/__init__.py", "tests/test_submission.py",
              "vendor/contest_harness-1.0.4-py3-none-any.whl")]
    manifest = {}
    for path in files:
        name = path.relative_to(ROOT)
        target = out / "reproduction" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[name.as_posix()] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("checkpoint.npz", "baseline-info.json", "START.md", "TASK.md", "TRAINING_GUIDE.md",
                 "public/train.jsonl", "public/validation.jsonl"):
        target = out / "official-baseline" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline / name, target)
    write_receipt(out / "source-sha256.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--expected-image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    parser.add_argument("--normal-max-calls", type=int, default=16)
    parser.add_argument("--recovery-max-calls", type=int, default=6)
    args = parser.parse_args()
    load_env(args.env_file)
    config = configuration()
    if any(not os.environ.get(k, "").strip() for k in ("PYTRIO_API_KEY", "SWANLAB_API_KEY")):
        raise ValueError("PYTRIO_API_KEY and SWANLAB_API_KEY are required on the host")
    if config["model"] != live.MODEL or config["base_url"] != "https://pytrio.com":
        raise ValueError("This test requires the exact configured PyTRiO Qwen3.8-27B model")
    if not 1 <= args.normal_max_calls <= 24 or not 1 <= args.recovery_max_calls <= 8:
        raise ValueError("Model call budgets exceed this suite's bounded limits")
    out, baseline = args.output.resolve(), args.baseline_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import swanlab
    result = {"status": "preflight", "model": live.MODEL, "image": args.image,
              "image_id": args.expected_image_id, "normal_exit_code": None, "recovery_exit_code": None,
              "cloud_verified": False, "started_at": datetime.now(timezone.utc).isoformat(),
              "automatic_model_retries": False, "official_host_or_hidden_score_verified": False}
    write_receipt(out / "suite.json", result)
    runtime = DockerRuntime(args.image, executable=args.docker_executable,
                            context=args.docker_context, expected_id=args.expected_image_id)
    probe = runtime.preflight()
    source = snapshot(out, baseline)
    app_files = {name: digest for name, digest in source.items()
                 if name.startswith("src/") or name in ("harness.py", "contest_submission.py", "submission_config.json",
                                                       "requirements.txt", "vendor/contest_harness-1.0.4-py3-none-any.whl")}
    script = "import hashlib,json,pathlib,sys; print(json.dumps({p:'sha256:'+hashlib.sha256((pathlib.Path('/app')/p).read_bytes()).hexdigest() for p in json.loads(sys.argv[1])}))"
    actual = runtime._run(["run", "--rm", "--pull=never", "--network", "none", "--read-only", "--entrypoint", "python3",
                           runtime.image, "-c", script, json.dumps(list(app_files))])
    if json.loads(actual.stdout) != app_files:
        raise ValueError("Current runtime source differs from the immutable image")
    probe["application_source_files_matched"] = len(app_files)
    write_receipt(out / "image-preflight.json", probe)
    task_config = {"purpose": "full_real_model_training_experiment_audit", "dataset_key": "A",
                   "model": live.MODEL, "image_id": args.expected_image_id, "sdk_version": "1.0.4",
                   "controller_version": "gas-a-controller-zh-v4", "source_files": source,
                   "data_sha256": {n: "sha256:" + hashlib.sha256((baseline / "public" / n).read_bytes()).hexdigest()
                                   for n in ("train.jsonl", "validation.jsonl")},
                   "normal_max_calls": args.normal_max_calls, "recovery_max_calls": args.recovery_max_calls,
                   "public_validation_only": True, "hidden_labels_available": False}
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    journal = CloudJournal(swanlab, config, out, task_config, "gas-a-v4-real-cloud-" + timestamp)
    original_gateway, original_round = live.RealGateway, live.run_native_round

    class LoggedGateway(original_gateway):
        def invoke(self, frame, start):
            journal.before_call(frame, start)
            call_id, response, bound = super().invoke(frame, start)
            journal.after_call(call_id, response, bound)
            return call_id, response, bound

    def logged_round(host, directory, state, dataset_root):
        original_round(host, directory, state, dataset_root)
        journal.after_round(host)

    def run_phase(phase, budget, recovery=None):
        journal.phase = phase
        argv = ["test_real_sdk.py", "--baseline-dir", str(baseline), "--output", str(out / phase),
                "--env-file", str(args.env_file), "--image", args.image, "--expected-image-id", args.expected_image_id,
                "--docker-executable", args.docker_executable, "--docker-context", args.docker_context,
                "--rounds", "4", "--max-rounds", "8", "--max-model-calls", str(budget)]
        if recovery:
            argv += ["--recovery-from", str(recovery)]
        with patch.object(sys, "argv", argv), patch.object(live, "RealGateway", LoggedGateway), \
                patch.object(live, "run_native_round", logged_round):
            code = live.main()
        path = out / phase / "live-report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report.update(cloud_experiment_record_created=True, cloud_run_path=journal.receipt["run_path"],
                      cloud_recording_component="host_side_cloud_suite_journal")
        write_receipt(path, report)
        return code

    finish_state = "crashed"
    error = None
    with termination_handler():
        try:
            live.ensure_model(config, os.environ["PYTRIO_API_KEY"])
            journal.begin()
            result.update(status="running", cloud_url=journal.receipt["url"], cloud_run_path=journal.receipt["run_path"])
            write_receipt(out / "suite.json", result)
            print(json.dumps({"event": "cloud_run_started", "url": journal.receipt["url"]}), flush=True)
            result["normal_exit_code"] = run_phase("normal", args.normal_max_calls)
            write_receipt(out / "suite.json", result)
            if result["normal_exit_code"] == 0:
                result["recovery_exit_code"] = run_phase("recovery", args.recovery_max_calls, out / "normal")
            passed = result["normal_exit_code"] == 0 and result["recovery_exit_code"] == 0
            result["test_requirements_passed"] = passed
            for phase in ("normal", "recovery"):
                path = out / phase / "live-report.json"
                if path.is_file():
                    report = json.loads(path.read_text(encoding="utf-8"))
                    journal.emit(texts={f"{phase}/summary": report})
            journal.emit({"suite/requirements_passed": int(passed),
                          "suite/complete_history_cycles": sum(r["complete_history_experiment_decision"] for r in journal.rounds)})
            result["status"] = "experiment_completed" if passed else "requirements_unmet"
            write_receipt(out / "suite.json", result)
            journal.archive()
            finish_state = "success" if passed else "crashed"
        except BaseException as caught:
            error = caught
            result.update(status="failed", error_type=type(caught).__name__)
            finish_state = "aborted" if isinstance(caught, (KeyboardInterrupt, SystemExit)) else "crashed"
        finally:
            write_receipt(out / "suite.json", result)
            journal.finish(finish_state)
    if error is None:
        try:
            cloud = journal.verify_cloud()
            result.update(cloud_verified=True, cloud_url=cloud["url"], cloud_state=cloud["state"])
        except Exception as caught:
            result.update(cloud_verification_error_type=type(caught).__name__)
    result.update(status="completed" if result.get("test_requirements_passed") and result["cloud_verified"]
                  else "needs_attention", finished_at=datetime.now(timezone.utc).isoformat())
    write_receipt(out / "suite.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if error is not None and isinstance(error, (KeyboardInterrupt, SystemExit)):
        raise error
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
