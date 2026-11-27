"""B/C 独立消融与原生预测核验；仅用公开数据，不调用外部模型或正式平台。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
import numpy as np
from gas_a.regression.data import load_pair, load_model
from gas_a.regression.diagnostics import groups
from gas_a.regression.evaluation import evaluate, promotion_gate
from gas_a.regression.monitoring import fit_reference, monitor, diagnose
from gas_a.regression.submission import experiment_menu, validate_config
from gas_a.regression.training import continue_candidate
from gas_a.sensor_training.io import digest, write_json
from scripts.validate_multitask import BASELINES, PREDICT_ARGV, measure


def distribution(train, validation, dataset):
    cuts = np.unique(np.quantile(train["y"], [.25, .5, .75]))
    result = {"training_target_cuts": cuts.tolist(), "splits": {}}
    for name, data in (("train", train), ("validation", validation)):
        axes = groups(data, dataset)
        result["splits"][name] = {"rows": len(data["y"]), "sha256": data["sha256"],
            "target_quantiles": np.quantile(data["y"], [0, .25, .5, .75, 1]).tolist(),
            "rul_segment_counts": np.bincount(np.searchsorted(cuts, data["y"], side="right"), minlength=len(cuts) + 1).tolist(),
            "groups": {axis: dict(zip(*[v.tolist() for v in np.unique(values, return_counts=True)])) for axis, values in axes.items()}}
    train_ids, val_ids = set(groups(train, dataset)["device"]), set(groups(validation, dataset)["device"])
    result["known_device_overlap"] = sorted((train_ids & val_ids) - {"unknown"})
    result["device_split_verified"] = "unknown" not in train_ids | val_ids and not result["known_device_overlap"]
    result["protocol"] = "public full-cycle/snapshot record-weighted metrics; not official truncated-test endpoints"
    return result


def run(base, out, dataset, config):
    out.mkdir()
    checkpoint = base / "checkpoint.npz"
    train_path, validation_path = base / "public/train.jsonl", base / "public/validation.jsonl"
    identities = {str(p): digest(p) for p in (checkpoint, train_path, validation_path)}
    info = json.loads((base / "baseline-info.json").read_text(encoding="utf-8"))
    if info["dataset_key"] != dataset or info["checkpoint_sha256"] != identities[str(checkpoint)]:
        raise ValueError("Official baseline identity mismatch")
    train, validation = load_pair(train_path, validation_path, dataset)
    parent = load_model(checkpoint, dataset)
    descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": PREDICT_ARGV}
    before = measure(None, base, descriptor, validation_path, validation, dataset, out / "initial-native")
    reference = fit_reference(train, dataset)
    thresholds = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["controller"]["drift_thresholds"]
    drift = monitor(reference, validation, train, dataset, thresholds)
    write_json(out / "distribution.json", distribution(train, validation, dataset))
    write_json(out / "initial-diagnosis.json", diagnose(parent, validation, reference, before, drift))
    rows = []
    for name, spec in experiment_menu(config, dataset, parent, train).items():
        started = time.perf_counter()
        attempt = out / name
        attempt.mkdir()
        row = {"experiment": name, "configuration": spec, "parent_checkpoint_sha256": digest(checkpoint)}
        try:
            report = continue_candidate(checkpoint, train, validation, attempt / "candidate", spec,
                config["limits"], config["controller"], dataset, identities[str(checkpoint)])
            after = measure(None, attempt / "candidate", descriptor, validation_path, validation, dataset, attempt / "native")
            prediction = load_model(attempt / "candidate/checkpoint.npz", dataset)
            native = [json.loads(line)["outputs"]["value"] for line in (attempt / "native/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            if not np.allclose(native, prediction.predict(validation["x"]), rtol=1e-8, atol=1e-10):
                raise ValueError("Native predictions differ from local complete predictor")
            if promotion_gate(before, after, config["controller"])["eligible"] != report["selected_epoch_eligible"]:
                raise ValueError("Native and local admission gates disagree")
            write_json(attempt / "diagnosis.json", diagnose(prediction, validation, reference, after, drift))
            row.update(status="completed", metrics=after, gate=promotion_gate(before, after, config["controller"]),
                       native_prediction_verified=True, continuation_checks=report["checks"])
        except (ValueError, TimeoutError, ArithmeticError, np.linalg.LinAlgError) as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
        row["wall_seconds"] = time.perf_counter() - started
        rows.append(row)
        write_json(attempt / "result.json", row)
        print(json.dumps({"dataset": dataset, "experiment": name, "status": row["status"],
            "rmse": row.get("metrics", {}).get("rmse"), "eligible": row.get("gate", {}).get("eligible")}), flush=True)
    for path, identity in identities.items():
        if digest(path) != identity:
            raise ValueError("Read-only baseline or data changed")
    eligible = [r for r in rows if r.get("gate", {}).get("eligible")]
    best = min(eligible, key=lambda r: r["metrics"]["rmse"]) if eligible else None
    result = {"dataset": dataset, "scope": "fixed_public_independent_ablation_native_predictions",
        "initial": before, "experiments": rows, "best_eligible_experiment": best["experiment"] if best else None,
        "selected_metrics": best["metrics"] if best else before,
        "all_experiments_completed": all(r["status"] == "completed" for r in rows),
        "original_inputs_unchanged": True, "real_model_calls": 0, "official_score_verified": False}
    write_json(out / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", choices=("B", "C"), nargs="+", default=["B", "C"])
    args = parser.parse_args()
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]
    validate_config(config, config["limits"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "submission_config.json", Path(__file__).resolve(),
               ROOT / "scripts/validate_multitask.py", ROOT / "contest_submission.py", ROOT / "harness.py"]
    def hashes():
        return {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    registered = hashes()
    write_json(out / "registration.json", {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__},
        "source_sha256": registered, "configuration": config, "scope": "local_public_ablation", "real_model_calls": 0,
        "method": "Each candidate starts from the same official initial checkpoint; validation fits no model/scaler/weights."})
    reports = [run(args.baseline_root / BASELINES[k], out / k, k, config) for k in args.datasets]
    if registered != hashes():
        raise ValueError("Code changed during registered validation")
    passed = all(r["all_experiments_completed"] for r in reports)
    write_json(out / "summary.json", {"status": "completed" if passed else "requirements_unmet", "reports": reports,
        "real_model_calls": 0, "official_score_verified": False})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
