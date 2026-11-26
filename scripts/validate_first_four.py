"""Exercise current A and restored V18 B/C with SDK 1.0.4 and an offline controller."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_multitask import run_dataset, BASELINES
from scripts.docker_rehearsal import DockerRuntime
from gas_a.sensor_training.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, choices=range(5, 11), default=10)
    parser.add_argument("--image")
    parser.add_argument("--expected-image-id")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                            expected_id=args.expected_image_id,
                            required_tasks=("classification", "regression")) if args.image else None
    if runtime:
        write_json(args.output / "image-preflight.json", runtime.preflight())
    sequences = {
        "A": ["frozen_router", "frozen_router", "conservative_head", "frozen_router", "conservative_head",
              "four_sensor", "conservative_head", "five_sensor", "conservative_head", "frozen_router"],
        "B": ["robust_huber", "standard_rate", "head_only", "low_rate", "robust_huber",
              "standard_rate", "head_only", "low_rate", "robust_huber", "standard_rate"],
        "C": ["robust_huber", "standard_rate", "head_only", "low_rate", "robust_huber",
              "standard_rate", "head_only", "low_rate", "robust_huber", "standard_rate"],
    }
    reports = [run_dataset(args.baseline_root / BASELINES[key], args.output / key, key, args.rounds, runtime,
                           compact_validation=True, aggregate_parent=True, experiment_sequence=sequences[key])
               for key in ("A", "B", "C")]
    checks = {"all_rounds_completed": all(len(r["rounds"]) == args.rounds for r in reports),
              "no_public_regression": all(r["not_worse_than_public_baseline"] for r in reports),
              "all_sdk_model_receipts_present": all(row["model_call_ids"] for r in reports for row in r["rounds"]),
              "bc_v18_menu_only": all(row["record"]["configuration"].get("experiment", {}).get("name") in
                  ("low_rate", "standard_rate", "head_only", "robust_huber")
                  for report in reports[1:] for row in report["rounds"] if row["record"]["configuration"].get("experiment")),
              "a_retained_bundle_executed": any(row["record"]["configuration"].get("checkpoint_selection", {}).get(
                  "predictor_scope") == "complete_bundle_with_retained_auxiliaries" for row in reports[0]["rounds"])}
    audit = runtime.audit() if runtime else None
    if runtime:
        checks["container_isolation_verified"] = audit["container_policy_verified"]
    summary = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
               "scope": "public_data_sdk_acceptance_with_offline_controller", "real_model_calls": 0,
               "official_platform_score_verified": False, "runtime": audit or "native",
               "datasets": [{"dataset": r["dataset"], "baseline": r["baseline"], "final": r["final"],
                             "completed_rounds": len(r["rounds"]), "selected_artifact_id": r["selected_artifact_id"]}
                            for r in reports]}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
