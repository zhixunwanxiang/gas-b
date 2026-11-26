"""Same-parent numerical probes and SDK acceptance; no real model or hidden score claims."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission  # Fix BLAS threads before importing numerical modules.
from scripts.validate_multitask import BASELINES, run_dataset
from scripts.docker_rehearsal import DockerRuntime
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.evaluation import evaluate, promotion_gate
from gas_a.regression.submission import dataset_config
from gas_a.regression.training import continue_candidate
from gas_a.sensor_training.io import digest, write_json


def actual_record(out, row):
    path = out / f"host-{row['round']}/state/decisions.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def probe(args, config):
    rows = []
    for dataset, names in (("B", ("low_rate", "balanced_short", "balanced_mild")), ("C", ("fine_short", "tail_short", "condition_ridge"))):
        base = args.baseline_root / BASELINES[dataset]
        train, validation = load_pair(base / "public/train.jsonl", base / "public/validation.jsonl", dataset)
        cfg = dataset_config(config, dataset)
        parents = {"official_initial": base / "checkpoint.npz"}
        if args.reference_run:
            report = json.loads((args.reference_run / dataset / "report.json").read_text(encoding="utf-8"))
            selected = report["selected_artifact_id"]
            match = re.fullmatch(r"candidate-(\d+)", selected)
            parents["selected_reference"] = (args.reference_run / dataset /
                (f"host-{match.group(1)}/candidate/checkpoint.npz" if match else "initial/checkpoint.npz"))
        for label, parent in parents.items():
            before = evaluate(load_model(parent, dataset), validation, dataset)
            trials = []
            for name in names:
                spec = next(s for s in cfg["experiments"] if s["name"] == name)
                report = continue_candidate(parent, train, validation, args.output / f"{dataset}-{label}-{name}",
                    spec, cfg["limits"], cfg["controller"], dataset, digest(parent), reference=before)
                gate = promotion_gate(before, report["validation_after"], cfg["controller"], reference=before)
                trials.append({"experiment": name, "spec": spec, "metrics": report["validation_after"], "gate": gate,
                               "selected_epoch": report["selected_epoch"], "weighting": report["weighting_evidence"]})
            rows.append({"dataset": dataset, "parent_kind": label, "parent_sha256": digest(parent),
                "train_sha256": train["sha256"], "validation_sha256": validation["sha256"], "before": before,
                "trials": trials, "weighted_candidate_better_than_control": bool(trials[1]["gate"]["eligible"]
                    and trials[1]["metrics"]["rmse"] < trials[0]["metrics"]["rmse"] - 1e-9)})
    write_json(args.output / "probes.json", {"scope": "same-parent-public-numerical-probes", "real_model_calls": 0,
        "independent_generalization_verified": False, "rows": rows})
    for row in rows:
        print(json.dumps({"dataset": row["dataset"], "parent": row["parent_kind"],
            "before_rmse": row["before"]["rmse"], "trials": [{"name": t["experiment"], "rmse": t["metrics"]["rmse"],
            "eligible": t["gate"]["eligible"], "failures": t["gate"].get("guard_failures")} for t in row["trials"]]}))


def proxy_projection(base, dataset):
    rows = [json.loads(s) for s in (base / "public/validation.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    if dataset == "B":
        groups = sorted({r["meta"]["bearing"] for r in rows})
        parts = [rows, [r for r in rows if r["meta"]["bearing"] != groups[-1]]]
    else:
        groups = sorted({(r["meta"]["source_subset"], r["meta"]["unit"]) for r in rows})
        held = set(groups[::2])
        parts = [rows, [r for r in rows if (r["meta"]["source_subset"], r["meta"]["unit"]) in held]]
    previous = [0]
    def change(index, directory):
        part = 1 if 4 <= index <= 6 else 0
        if part == previous[0]:
            return False
        previous[0] = part
        (directory / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in parts[part]), encoding="utf-8")
        return True
    return change


def rounds(args):
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context="desktop-linux",
        expected_id=args.expected_image_id, required_tasks=("classification", "regression")) if args.image else None
    if runtime:
        write_json(args.output / "image-preflight.json", runtime.preflight())
    sequences = {
        "A": ["frozen_router", "frozen_router", "conservative_head", "frozen_router", "conservative_head",
              "four_sensor", "conservative_head", "five_sensor", "conservative_head", "frozen_router"],
        "B": ["balanced_short", "low_rate", "head_short", "standard_rate", "robust_huber",
              "balanced_short", "low_rate", "head_short", "balanced_short", "standard_rate"],
        "C": ["fine_short", "condition_ridge", "head_short", "condition_ridge", "fine_rate",
              "condition_ridge", "low_rate", "head_short", "condition_ridge", "robust_huber"]}
    summary = []
    for dataset in args.datasets:
        base = args.baseline_root / BASELINES[dataset]
        out = args.output / dataset
        report = run_dataset(base, out, dataset, 10, runtime, compact_validation=True, aggregate_parent=True,
            experiment_sequence=sequences[dataset],
            projection_callback=proxy_projection(base, dataset) if args.proxy_stages and dataset != "A" else None)
        records = [actual_record(out, row) for row in report["rounds"]]
        resources = [json.loads((out / f"host-{r['round']}/state/resources.json").read_text(encoding="utf-8"))
                     for r in report["rounds"]]
        key = "macro_f1" if dataset == "A" else "rmse"
        # On changing projections, only same-scope stage-relative gains are summarized below.
        metric_summary = {"mean_selected_metric": sum(r["metrics"][key] for r in report["rounds"]) / len(report["rounds"]),
                          "scope": "mixed_projection_descriptive_only" if args.proxy_stages else "fixed_public_validation"}
        row = {"dataset": dataset, "baseline": report["baseline"], "final": report["final"], **metric_summary,
               "rounds": len(report["rounds"]), "all_finish_acknowledged": all(r["finish_acknowledged"] for r in report["rounds"]),
               "adoptions": sum(r["configuration"].get("selection_action") == "adopt_candidate" for r in records),
               "local_rejections_not_submitted": sum(bool(r["configuration"].get("submission_skipped")) for r in records),
               "single_call_observe_rounds": sum(len(r["model_call_ids"]) == 1 for r in records),
               "cached_measurements": sum(not e["measured_this_round"] for r in records
                                          for e in r["configuration"].get("measurement_events", [])),
               "participant_cpu_seconds": sum(r["cpu_seconds"] for r in resources),
               "participant_wall_seconds_before_state_save": sum(r["wall_seconds"] for r in resources),
               "token_usage_complete": all(r["usage_complete"] for r in resources),
               "candidate_payload_bytes": sum(p.stat().st_size for p in out.glob("host-*/candidate/**/*") if p.is_file()),
               "state_payload_bytes": sum(p.stat().st_size for p in out.glob("host-*/state/**/*") if p.is_file())}
        if dataset != "A":
            ledger = json.loads((out / "host-10/state/regression_stage_evidence.json").read_text(encoding="utf-8"))
            stage_keys = [r["stage_key"] for r in ledger["rounds"]]
            row.update(mean_relative_gain=ledger.get("mean_relative_gain"), worst_relative_gain=ledger.get("worst_relative_gain"),
                       stage_events=1 + sum(a != b for a, b in zip(stage_keys, stage_keys[1:])),
                       recovery_observations=[{k: r[k] for k in ("round", "stage_key", "relative_gain", "recovered_round", "confirmed_round")}
                                              for r in ledger["rounds"]])
        summary.append(row)
    audit = runtime.audit() if runtime else None
    write_json(args.output / "summary.json", {"scope": "public_proxy_projection_sdk_test" if args.proxy_stages else "fixed_public_sdk_test",
        "real_model_calls": 0, "official_platform_score_verified": False, "runtime": audit or "native", "datasets": summary})
    print(json.dumps({"output": str(args.output), "datasets": [{k: r[k] for k in ("dataset", "rounds", "adoptions", "local_rejections_not_submitted", "single_call_observe_rounds")} for r in summary]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("probes", "rounds"), default="probes")
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=list("ABC"), default=list("BC"))
    parser.add_argument("--proxy-stages", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--expected-image-id")
    parser.add_argument("--docker-executable", default="docker")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
    write_json(args.output / "registration.json", {"configuration": config,
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / "src").rglob("*.py")},
        "real_model_calls": 0, "official_platform_submission": False})
    probe(args, config) if args.mode == "probes" else rounds(args)


if __name__ == "__main__":
    main()
