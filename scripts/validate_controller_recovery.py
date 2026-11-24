"""Inject completed invalid replies into the unmodified SDK in a fixed image.

No provider requests or hidden data are used. This validates failure handling,
not the unidentified cause of a platform run.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from tests.test_submission import OfflineHost
from tests.test_regression import RegressionHost
from scripts.docker_rehearsal import DockerRuntime, child_environment
from scripts.validate_platform import exchange
from scripts.validate_multitask import BASELINES
from gas_a.sensor_training.io import digest, write_json

CASES = ("malformed_json", "outside_menu", "wrong_parent_id", "invalid_selection", "oversized_text")


def run_case(runtime, baseline, out, dataset, case):
    out.mkdir(parents=True)
    shutil.copytree(baseline / "public", out / "data")
    (out / "initial").mkdir()
    parent = out / "initial/checkpoint.npz"
    shutil.copyfile(baseline / "checkpoint.npz", parent)
    info = json.loads((baseline / "baseline-info.json").read_text(encoding="utf-8"))
    if digest(parent) != info["checkpoint_sha256"]:
        raise ValueError("Official baseline digest mismatch")
    host = (OfflineHost(out, parent) if dataset == "A" else RegressionHost(out, parent, dataset))
    original = host.write

    def inject(text):
        result = original(text)
        frame = json.loads(text)
        if frame["type"] != "model_request":
            return result
        phase = json.loads(frame["messages"][0]["content"])["phase"]
        content = host.pending[-1]["response"]["choices"][0]["message"]
        value = json.loads(content["content"])
        if case == "malformed_json":
            content["content"] = "{invalid JSON"
            return result
        if phase == "plan":
            if case == "outside_menu":
                value["experiment"] = "not_in_current_menu"
            elif case == "wrong_parent_id":
                value["target_artifact_id"] = "unapproved_history_id"
            elif case == "oversized_text":
                value["hypothesis"] = "中" * 1334  # 4002 UTF-8 bytes, fewer than 4000 characters.
        elif case == "invalid_selection":
            value["selected_artifact_id"] = "unapproved_candidate_id"
        content["content"] = json.dumps(value, ensure_ascii=False)
        return result

    host.write = inject
    host.frame["task_contract"]["public_projection"].update(
        train_path="/data/train.jsonl", validation_path="/data/validation.jsonl")
    argv, name = runtime.train(parent.parent, out / "data", None, host.store / "container.jsonl")
    exchange(host, argv, child_environment(), host.store, name, docker_argv=runtime.cli)
    state = host.saved_state
    record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
    timing = json.loads((host.store / "sdk-timing.json").read_text(encoding="utf-8"))
    evidence = [json.loads(line) for line in (state / "model-responses.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_source = "selection_calls" if case == "invalid_selection" else "planning_calls"
    checks = {
        "selected_actual_parent": host.record["selected_artifact_id"] == host.frame["parent_artifact"]["artifact_id"],
        "guard_labeled_as_program": record[expected_source]["decision_source"] == "program_guard",
        "all_completed_receipts_cited": host.record["model_call_ids"] == host.calls,
        "all_completed_responses_persisted": [r["call_id"] for r in evidence] == host.calls,
        "no_candidate_from_invalid_plan": case == "invalid_selection" or host.candidate is None,
        "sdk_order_and_lengths_valid": timing["order_valid"],
        "round_ack_received": timing["acknowledged"],
    }
    result = {"dataset": dataset, "case": case, "checks": checks, "passed": all(checks.values()),
        "completed_stub_replies": len(host.calls), "round_seconds": timing["process_exit_seconds"],
        "declaration_sizes": timing["declaration_sizes"]}
    write_json(out / "verification.json", result)
    if not result["passed"]:
        raise ValueError("Controller recovery check failed: " + str(out))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-executable", default="docker")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = DockerRuntime(args.image, executable=args.docker_executable, expected_id=args.expected_image_id,
                            required_tasks=("classification", "regression"))
    write_json(out / "image-preflight.json", runtime.preflight())
    cases = []
    for dataset in "ABC":
        for case in CASES:
            cases.append(run_case(runtime, args.baseline_root / BASELINES[dataset], out / dataset / case, dataset, case))
            print(json.dumps({"dataset": dataset, "case": case, "passed": True}), flush=True)
    write_json(out / "verification.json", {"passed": True, "scope": "offline_completed_reply_fault_injection_in_docker",
        "new_real_model_calls": 0, "platform_root_cause_confirmed": False, "cases": cases})


if __name__ == "__main__":
    main()
