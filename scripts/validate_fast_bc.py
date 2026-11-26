"""At most two new B/C candidates each; hard 30-second children and 300-second batch."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {"B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}


def sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_bounded(job_path, log_path, timeout):
    # The child performs no RPC/network, and spawns no helper process.
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                                    "--worker-job", str(job_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return process.returncode, True


def worker(path):
    sys.path.insert(0, str(ROOT))
    import contest_submission  # Set numerical thread limits before importing NumPy.
    from gas_a.regression.data import load_model, load_pair
    from gas_a.regression.evaluation import evaluate, promotion_gate
    from gas_a.regression.training import continue_candidate
    job = json.loads(path.read_text(encoding="utf-8"))
    started = time.monotonic()
    if any(sha(ROOT / relative) != expected for relative, expected in job["sources"].items()):
        raise ValueError("Numerical implementation changed after registration")
    train, validation = load_pair(Path(job["train"]), Path(job["validation"]), job["dataset"])
    if train["sha256"] != job["train_sha256"] or validation["sha256"] != job["validation_sha256"]:
        raise ValueError("Dataset changed after registration")
    before = evaluate(load_model(Path(job["parent"]), job["dataset"]), validation, job["dataset"])
    remaining = job["hard_seconds"] - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Dataset loading exhausted candidate budget")
    report = continue_candidate(Path(job["parent"]), train, validation, Path(job["candidate"]), job["spec"],
                                {**job["limits"], "max_seconds_per_candidate": remaining}, job["rules"],
                                job["dataset"], job["parent_sha256"], reference=before)
    gate = promotion_gate(before, report["validation_after"], job["rules"], reference=before)
    timing = report.get("prediction_timing") or {}
    gate["eligible"] &= timing.get("eligible") is True
    if timing.get("eligible") is not True:
        gate.setdefault("guard_failures", []).append("prediction_cpu_budget")
    write(job["result"], {"before": before, "after": report["validation_after"], "gate": gate,
                          "prediction_timing": timing, "report_sha256": sha(Path(job["candidate"]) / "training.json"),
                          "checkpoint_sha256": report["checkpoint_sha256"],
                          "elapsed_worker_seconds": time.monotonic() - started})


def screen(args):
    started = time.monotonic()
    deadline = started + args.batch_seconds
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / "src").rglob("*.py")}
    registration = {"scope": "same_parent_public_short_screen", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "batch_seconds": args.batch_seconds, "hard_seconds_per_child": args.candidate_seconds,
                    "maximum_candidates_per_dataset": 2, "configuration": config, "sources": sources,
                    "real_model_calls": 0, "official_platform_submission": False}
    write(args.output / "registration.json", registration)
    ledger_path = ROOT / "runtime/fast-bc-short-screen-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    if not isinstance(ledger, dict):
        raise ValueError("Invalid short-screen deduplication ledger")
    rows = []
    for dataset in "BC":
        data = args.baseline_root / BASELINES[dataset] / "public"
        report = json.loads((args.parent_run / dataset / "report.json").read_text(encoding="utf-8"))
        identifier = report["selected_artifact_id"]
        if identifier == "initial":
            parent = args.parent_run / dataset / "initial/checkpoint.npz"
        elif identifier.startswith("candidate-") and identifier[10:].isdigit():
            parent = args.parent_run / dataset / f"host-{identifier[10:]}/candidate/checkpoint.npz"
        else:
            raise ValueError("Cannot resolve actual selected parent from the supplied run")
        specs = [s for s in config["regression"]["dataset_experiments"][dataset] if s.get("component_update") == "fixed_rbf"]
        if len(specs) != 2:
            raise ValueError("Register exactly two fixed RBF candidates per dataset")
        shared = config["regression"]
        rules = {**shared["controller"], "guards": shared["guards"][dataset], "prediction_budget": shared["prediction_budget"]}
        binding = {"dataset": dataset, "parent_sha256": sha(parent), "train_sha256": sha(data / "train.jsonl"),
                   "validation_sha256": sha(data / "validation.jsonl"), "rules": rules,
                   "sources": sources, "candidate_seconds": args.candidate_seconds, "limits": shared["limits"],
                   "screen_implementation_sha256": sha(__file__),
                   "runtime": {"python": sys.version, "numpy": importlib.metadata.version("numpy"), "platform": sys.platform}}
        for spec in specs:
            identity = hashlib.sha256(json.dumps([binding, spec], sort_keys=True).encode()).hexdigest()
            row = {"dataset": dataset, "experiment": spec["name"], "identity": identity}
            if identity in ledger:
                rows.append({**row, "status": "reused_identical_completed_screen", "measured_this_run": False,
                             "origin": ledger[identity]})
                continue
            remaining = deadline - time.monotonic()
            if remaining < 1:
                rows.append({**row, "status": "batch_budget_exhausted", "measured_this_run": False})
                continue
            hard_seconds = min(args.candidate_seconds, remaining)
            prefix = args.output / f"{dataset}-{spec['name']}"
            job = {"dataset": dataset, "parent": str(parent), "parent_sha256": binding["parent_sha256"],
                   "train": str(data / "train.jsonl"), "validation": str(data / "validation.jsonl"),
                   "train_sha256": binding["train_sha256"], "validation_sha256": binding["validation_sha256"], "sources": sources,
                   "spec": spec, "rules": rules, "limits": shared["limits"], "hard_seconds": hard_seconds,
                   "candidate": str(prefix), "result": str(prefix.with_suffix(".result.json"))}
            job_path = prefix.with_suffix(".job.json")
            write(job_path, job)
            trial_started = time.monotonic()
            code, expired = run_bounded(job_path, prefix.with_suffix(".log"), hard_seconds)
            row.update(status="hard_timeout" if expired else "completed" if code == 0 else "failed",
                       child_exit_code=code, elapsed_seconds=time.monotonic() - trial_started, measured_this_run=True)
            if code == 0 and not expired:
                result = json.loads(prefix.with_suffix(".result.json").read_text(encoding="utf-8"))
                row.update(result=result, status="eligible" if result["gate"]["eligible"] else "rejected")
            ledger[identity] = {"status": row["status"], "source": str(prefix.with_suffix(".result.json")),
                                "elapsed_seconds": row["elapsed_seconds"], "result": row.get("result")}
            write(ledger_path, ledger)
            rows.append(row)
            write(args.output / "progress.json", rows)
            print(json.dumps({"dataset": dataset, "experiment": spec["name"], "status": row["status"],
                              "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
    write(args.output / "summary.json", {"scope": registration["scope"], "elapsed_seconds": time.monotonic() - started,
          "batch_seconds": args.batch_seconds, "real_model_calls": 0, "official_score_verified": False,
          "automatic_activation": False, "ten_round_validation_included": False, "trials": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-seconds", type=float, default=300.)
    parser.add_argument("--candidate-seconds", type=float, default=30.)
    args = parser.parse_args()
    if args.worker_job:
        worker(args.worker_job.resolve())
        return
    if (any(getattr(args, name) is None for name in ("baseline_root", "parent_run", "output"))
            or not 1 <= args.batch_seconds <= 300 or not 1 <= args.candidate_seconds <= 30):
        parser.error("Require all paths, batch <=300 seconds and candidate <=30 seconds")
    args.baseline_root, args.parent_run, args.output = (p.resolve() for p in (args.baseline_root, args.parent_run, args.output))
    screen(args)


if __name__ == "__main__":
    main()
