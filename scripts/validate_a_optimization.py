"""Replay task-A rounds with compact SDK measurements and no paid model calls."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_multitask as validation
from scripts.docker_rehearsal import DockerRuntime
from gas_a.sensor_training.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10, choices=range(4, 11))
    parser.add_argument("--image")
    parser.add_argument("--expected-image-id")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    parser.add_argument("--expect-promotion", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                            expected_id=args.expected_image_id) if args.image else None
    if runtime:
        write_json(out / "image-preflight.json", runtime.preflight())
    original_exchange = validation.exchange

    def compact_exchange(host, *positional, **keywords):
        # Remove only optional measurement metadata. Keep scores computed by the host predictor.
        host.frame["current_validation"] = {"metrics": host.frame["current_validation"]["metrics"]}
        original_write = host.write
        def receive(text):
            result = original_write(text)
            if json.loads(text)["type"] == "candidate_request":
                host.pending[-1]["summary"] = {"metrics": host.pending[-1]["summary"]["metrics"]}
            return result
        host.write = receive
        return original_exchange(host, *positional, **keywords)

    with patch.object(validation, "exchange", compact_exchange):
        report = validation.run_dataset(args.baseline_dir.resolve(), out / "A", "A", args.rounds, runtime)
    promotions = sum(r["selected_artifact_id"] == f"candidate-{r['round']}" for r in report["rounds"])
    reasons = [r["record"]["configuration"].get("host_candidate_gate", {}).get("reason")
               for r in report["rounds"] if r["candidate_submitted"]]
    result = {"status": "passed" if report["not_worse_than_public_baseline"] and
              (promotions > 0 or not args.expect_promotion) else "failed",
              "scope": "local_sdk_host_compact_measurement_replay_with_offline_controller",
              "real_model_calls": 0, "official_platform_verified": False,
              "baseline_macro_f1": report["baseline"]["macro_f1"],
              "final_macro_f1": report["final"]["macro_f1"], "promotions": promotions,
              "completed_rounds": len(report["rounds"]), "host_admission_reasons": reasons,
              "runtime": runtime.audit() if runtime else "native"}
    write_json(out / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
