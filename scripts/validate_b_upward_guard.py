"""Check the B residual guard on public validation and whole-bearing holdouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gas_a.regression import bearing_guard
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.diagnostics import error_metrics, groups
from gas_a.regression.model import CORRECTION_METHODS, RegressionModel, frozen_correction, transformed
from gas_a.regression.training import fit_epoch
from gas_a.sensor_training.io import digest, read_json, write_json

METHODS = ("parent", "bounded_residual", "upward_guard")


def read_parent_run(run, baseline, train_hash, validation_hash):
    report = read_json(run / "report.json")
    if report["dataset"] != "B" or report["final"]["validation_sha256"] != validation_hash:
        raise ValueError("Parent run must be a B report for the same public validation data")
    initial = run / "initial/checkpoint.npz"
    initial_hash = digest(initial)
    if initial_hash != digest(baseline / "checkpoint.npz"):
        raise ValueError("Parent run did not start from the supplied B baseline checkpoint")
    models = {"initial": initial_hash}
    stages = {}
    selected = {row["selected_artifact_id"] for row in report["rounds"]}
    for row in report["rounds"]:
        number = row["round"]
        artifact = f"candidate-{number}"
        if artifact not in selected:
            continue
        directory = run / f"host-{number}" / "candidate"
        path = directory / "checkpoint.npz"
        record = read_json(directory / "training.json")
        checkpoint_hash = digest(path)
        if (record["dataset"] != "B" or record["checkpoint_sha256"] != checkpoint_hash
                or record["train_sha256"] != train_hash or record["validation_sha256"] != validation_hash):
            raise ValueError(f"Invalid B training record for round {number}")
        models[artifact] = checkpoint_hash
        stages[checkpoint_hash] = (path, record)

    final_hash = models[report["selected_artifact_id"]]
    metrics = report["final"]
    if final_hash in stages:
        selected_training = stages[final_hash][1]
        if selected_training["configuration"].get("method") == bearing_guard.METHOD:
            # Replay the comparison against the selected guard's parent, without stacking a second guard.
            final_hash = selected_training["parent_checkpoint_sha256"]
            metrics = selected_training["validation_before"]
    lineage = []
    seen = set()
    current = final_hash
    while current != initial_hash:
        if current in seen or current not in stages:
            raise ValueError("Broken B checkpoint lineage")
        seen.add(current)
        path, record = stages[current]
        spec = record.get("resolved_training_configuration", record["configuration"])
        strength = None
        if spec.get("method") in CORRECTION_METHODS:
            strength = load_model(path, "B").meta["corrections"][-1]["strength"]
        lineage.append((spec, record["selected_epoch"] - record["initial_epochs"], strength))
        current = record["parent_checkpoint_sha256"]
    final_path = initial if final_hash == initial_hash else stages[final_hash][0]
    return initial, final_path, lineage[::-1], metrics


def rebuild_parent(data, initial, lineage):
    keys = ("hidden", "seed", "learning_rate", "batch_size")
    settings = {key: initial[key] for key in keys}
    model = RegressionModel("regression", data["paths"], [], data["x"], data["y"], **settings)
    model.meta.update(profile="B", target_unit="minutes")
    model.fit(data["x"], data["y"], initial["epochs"])
    for spec, epochs, strength in lineage:
        deadline = time.monotonic() + 30.
        if strength is not None:
            model = frozen_correction(model, data, spec["transform"], spec["ridge"], deadline,
                                      basis=spec.get("basis", "linear"))
            model.meta["corrections"][-1]["strength"] = strength
        else:
            for _ in range(epochs):
                fit_epoch(model, data, spec, deadline)
    return model


def subset(data, mask):
    ids = [identity for identity, keep in zip(data["ids"], mask) if keep]
    identity = hashlib.sha256(json.dumps([data["sha256"], ids]).encode()).hexdigest()
    return {**data, "ids": ids, "x": data["x"][mask], "y": data["y"][mask], "sha256": "sha256:" + identity,
            "metadata": {name: np.asarray(column)[mask].tolist() for name, column in data["metadata"].items()}}


def predict_comparison(fit, check, parent):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fit_prediction, original = [parent.predict(data["x"]) for data in (fit, check)]
    fit_features, check_features = [transformed(data["x"], "bearing_energy", {}) for data in (fit, check)]
    support_state = bearing_guard.fit_support(fit_features, groups(fit, "B")["device"])
    support = bearing_guard.support_weight(check_features, bearing_guard.load_support(support_state))
    residual_model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(64, 32), alpha=.04, batch_size=256, learning_rate_init=.0015,
        max_iter=90, early_stopping=False, random_state=42))
    # Match the fixed-iteration offline experiment. The model remains usable at the iteration limit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        residual_model.fit(np.column_stack([fit_features, fit_prediction]), (fit["y"] - fit_prediction) / bearing_guard.TARGET_SCALE)
    residual = residual_model.predict(np.column_stack([check_features, original])) * bearing_guard.TARGET_SCALE
    delta = bearing_guard.MAX_CORRECTION * np.tanh(residual / bearing_guard.MAX_CORRECTION)
    return {"parent": original, "bounded_residual": np.maximum(0., original + support * delta),
            "upward_guard": np.maximum(0., original + support * bearing_guard.guarded_delta(original, delta))}


def compare(fit, check, parent):
    fit_names, check_names = [groups(data, "B")["device"] for data in (fit, check)]
    if ("unknown" in set(fit_names) | set(check_names) or set(fit_names) & set(check_names)
            or set(fit["ids"]) & set(check["ids"])):
        raise ValueError("Validation requires complete, disjoint bearing metadata and sample IDs")
    forecasts = predict_comparison(fit, check, parent)
    y = check["y"]

    def score(mask):
        result = {name: error_metrics(prediction[mask] - y[mask], "B", detailed=False)
                  for name, prediction in forecasts.items()}
        return result

    def by_life(mask):
        low, high = bearing_guard.LOW_CUTOFF, bearing_guard.HIGH_CUTOFF
        bands = {"le_230": y <= low, "231_to_758": (y > low) & (y < high), "ge_759": y >= high}
        return {"metrics": score(mask), "rul_bands": {name: score(mask & band) for name, band in bands.items()
                                                       if (mask & band).any()}}

    return {"fit_bearings": sorted(set(fit_names)), "overall": by_life(np.ones(len(y), dtype=bool)),
            "bearings": {name: by_life(check_names == name) for name in sorted(set(check_names))}}


def both_lower(metrics):
    return all(metrics["upward_guard"][name] < metrics["parent"][name] for name in ("rmse", "mae"))


def pooled_individual(folds):
    overall = [result["overall"]["metrics"] for result in folds.values()]
    count = sum(result["parent"]["rows"] for result in overall)
    pooled = {}
    for method in METHODS:
        rmse = np.sqrt(sum(row[method]["rmse"] ** 2 * row[method]["rows"] for row in overall) / count)
        mae = sum(row[method]["mae"] * row[method]["rows"] for row in overall) / count
        pooled[method] = {"rmse": float(rmse), "mae": mae, "rows": count}
    return {"metrics": pooled, "bearings_both_improved": sum(both_lower(row) for row in overall)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True, help="B package containing public/ and checkpoint.npz")
    parser.add_argument("--parent-run", type=Path, required=True, help="B run directory containing report.json and host rounds")
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    try:
        import sklearn
        from threadpoolctl import threadpool_limits
    except ImportError:
        parser.error("This offline B validation requires scikit-learn and threadpoolctl")

    train, public = load_pair(args.baseline_root / "public/train.jsonl", args.baseline_root / "public/validation.jsonl", "B")
    initial, final, stages, parent_metrics = read_parent_run(args.parent_run, args.baseline_root,
                                                             train["sha256"], public["sha256"])
    initial_meta = load_model(initial, "B").meta
    bearings = groups(train, "B")["device"]
    condition = tuple(name for name in np.unique(bearings) if name.startswith("Bearing3_"))
    long = tuple(name for name in condition if train["y"][bearings == name].max() >= 1000.)
    short = tuple(name for name in condition if name not in long)
    if len(long) != 2 or len(short) != 1:
        raise ValueError("Expected two long bearings and one shorter bearing for B's 40 Hz / 10 kN condition")

    def holdout(names):
        held = np.isin(bearings, names)
        fit, check = subset(train, ~held), subset(train, held)
        result = compare(fit, check, rebuild_parent(fit, initial_meta, stages))
        print("Measured holdout " + ", ".join(names), flush=True)
        return result

    with threadpool_limits(2):
        public_result = compare(train, public, load_model(final, "B"))
        for metric in ("rmse", "mae"):
            if not np.isclose(public_result["overall"]["metrics"]["parent"][metric], parent_metrics[metric], rtol=1e-7):
                raise ValueError(f"Public parent {metric} does not match the recorded run")
        individual = {name: holdout((name,)) for name in np.unique(bearings)}
        both_long, entire_condition = holdout(long), holdout(condition)

    summary = {"public": public_result["overall"]["metrics"], "short_bearing": individual[short[0]]["overall"]["metrics"],
               "both_long": both_long["overall"]["metrics"], "entire_40hz": entire_condition["overall"]["metrics"]}
    focal_checks = {name: both_lower(metrics) for name, metrics in summary.items()}
    report = {"dataset": "B", "units": "minutes", "scope": "offline_public_and_bearing_disjoint_holdouts",
              "train_sha256": train["sha256"], "validation_sha256": public["sha256"], "parent_sha256": digest(final),
              "script_sha256": digest(__file__), "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
              "parent_stage_names": [spec["name"] for spec, _, _ in stages], "short_bearing_name": short[0],
              "long_bearing_names": list(long), "summary": summary, "both_metrics_improved": focal_checks,
              "public_validation": public_result, "simultaneous_long_holdout": both_long,
              "entire_40hz_holdout": entire_condition, "individual_bearing_holdouts": individual,
              "pooled_individual_holdouts": pooled_individual(individual)}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    for name, methods in summary.items():
        before, after = methods["parent"], methods["upward_guard"]
        print(f"{name}: RMSE {before['rmse']:.2f} -> {after['rmse']:.2f}; "
              f"MAE {before['mae']:.2f} -> {after['mae']:.2f}", flush=True)


if __name__ == "__main__":
    main()
