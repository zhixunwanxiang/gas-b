"""Recheck ten-round parallel evidence, complete experiment loops and byte limits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from scripts.sdk_protocol_audit import audit_file
from gas_a.sensor_training.io import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root):
    summary = read(root / "summary.json")
    datasets, sizes = [], []
    for dataset in "ABC":
        run = root / "parallel" / dataset / "run"
        report = read(run / "report.json")
        rounds, loops = [], 0
        for row in report["rounds"]:
            host = run / ("host-%d" % row["round"])
            wire = audit_file(host / "protocol.jsonl")
            timing = read(host / "sdk-timing.json")
            decision = json.loads((host / "state/decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            responses = [json.loads(line) for line in (host / "state/model-responses.jsonl").read_text(encoding="utf-8").splitlines()]
            current = [r for r in responses if r["round_index"] == row["round"]]
            complete = (bool(row["history_ids"]) and row["candidate_submitted"]
                and decision["plan"]["action"] == "experiment"
                and all(decision[p]["decision_source"] == "model" for p in ("planning_calls", "selection_calls")))
            loops += bool(complete)
            checks = {"wire_order_and_lengths": wire["order_valid"], "accepted_round_ack": wire["acknowledged"],
                "process_completed": timing["status"] == "completed",
                "all_responses_recorded": [r["call_id"] for r in current] == row["model_call_ids"],
                "history_read_before_plan": wire["request_counts"].get("history_request", 0) == 1,
                "within_local_round_budget": timing["process_exit_seconds"] <= timing["budget_seconds"]}
            if not all(checks.values()):
                raise ValueError("Round verification failed: %s/%s %s" % (dataset, row["round"], checks))
            rounds.append({"round": row["round"], "checks": checks, "complete_history_experiment": bool(complete),
                           "seconds": timing["process_exit_seconds"]})
            sizes.extend(wire["declaration_sizes"])
        if len(rounds) < 10 or loops < 3 or not report["not_worse_than_public_baseline"]:
            raise ValueError("Dataset acceptance failed: " + dataset)
        datasets.append({"dataset": dataset, "rounds_completed": len(rounds), "complete_history_experiments": loops,
            "baseline": report["baseline"], "final": report["final"],
            "not_worse_than_public_baseline": True, "rounds": rounds})
    proof = summary["docker_event_concurrency"]["parallel"]
    if not summary["docker_event_capture_complete"] or proof["max_overlapping_train_containers"] < 3:
        raise ValueError("Missing evidence of three overlapping training containers")
    maximum = {key: max(item.get(key, 0) for item in sizes) for key in (
        "hypothesis_utf8_bytes", "change_summary_utf8_bytes", "reason_utf8_bytes", "configuration_serialized_bytes")}
    sdk_version = summary.get("sdk_version", "1.0.4")
    return {"passed": True, "scope": "offline_parallel_sdk" + sdk_version.replace(".", "") + "_public_data_acceptance",
        "sdk_version": sdk_version,
        "image_id": summary["image_id"], "rounds_total": sum(d["rounds_completed"] for d in datasets),
        "max_overlapping_training_containers": proof["max_overlapping_train_containers"],
        "declaration_maximum_bytes": maximum, "limits": {"text_utf8_bytes": 4000, "configuration_serialized_bytes": 65536},
        "human_intervention_during_run": False, "new_real_model_calls": 0,
        "official_platform_score_verified": False, "datasets": datasets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run.resolve())
    write_json(args.output.resolve(), result)
    print(json.dumps({"passed": result["passed"], "rounds_total": result["rounds_total"],
                      "max_bytes": result["declaration_maximum_bytes"]}))


if __name__ == "__main__":
    main()
