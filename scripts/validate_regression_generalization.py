"""以整台设备留出评估 B/C 单点、组合、区间和离线历史输入。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
import numpy as np
from gas_a.regression.data import BEARING_FEATURES, load_model, schema, UNITS
from gas_a.regression.diagnostics import compact_metrics, groups
from gas_a.regression.evaluation import evaluate
from gas_a.regression.model import RegressionModel
from gas_a.regression.submission import validate_config
from gas_a.regression.training import continue_candidate
from gas_a.sensor_training.io import digest, write_json
from gas_a.sensor_training.reference.data import read_dataset
from scripts.validate_multitask import BASELINES


def subset(data, mask):
    ids = [identity for identity, keep in zip(data["ids"], mask) if keep]
    identity = hashlib.sha256(json.dumps([data["sha256"], ids], separators=(",", ":")).encode()).hexdigest()
    return {"x": data["x"][mask], "y": data["y"][mask], "ids": ids, "paths": data["paths"],
            "sha256": "sha256:" + identity,
            "metadata": {key: np.asarray(values)[mask].tolist() for key, values in data["metadata"].items()}}


def partitions(data, dataset):
    device = groups(data, dataset)["device"]
    if "unknown" in device:
        raise ValueError("Device-held-out validation requires complete public metadata")
    by_source = {}
    for name in np.unique(device):
        source = name.split("_")[0] if dataset == "B" else name.split("/")[0]
        by_source.setdefault(source, []).append(name)
    rng = np.random.default_rng(20261127)
    buckets = [np.array_split(rng.permutation(names), 3) for names in by_source.values()]
    for fold in range(3):
        split = {name: [] for name in ("fit", "tune", "calibrate", "holdout")}
        for index, bucket in enumerate(buckets):
            split["holdout"].extend(bucket[fold])
            remainder = np.concatenate([part for i, part in enumerate(bucket) if i != fold])
            if dataset == "B":
                # 官方训练每个工况三条轴承序列。每轮保留一个设备用于拟合。
                if len(remainder) != 2 or len(buckets) != 3:
                    raise ValueError("Expected three public training bearings in each of three conditions")
                split["fit"].append(remainder[0])
                split["tune" if index == fold else "calibrate"].append(remainder[1])
            else:
                size = max(1, len(remainder) // 6)
                split["tune"].extend(remainder[:size])
                split["calibrate"].extend(remainder[size:2 * size])
                split["fit"].extend(remainder[2 * size:])
        if any(not names for names in split.values()) or len(set(sum(split.values(), []))) != sum(map(len, split.values())):
            raise ValueError("Invalid or overlapping device partitions")
        yield split, {name: np.isin(device, names) for name, names in split.items()}


def intervals(cal_prediction, cal_target, cal_devices, prediction, target, devices, coverage=.90):
    """分组最大残差区间与分段经验区间并行报告。"""
    error = np.abs(cal_prediction - cal_target)
    measured = np.abs(prediction - target)
    group_scores = np.sort([error[cal_devices == name].max() for name in np.unique(cal_devices)])
    rank = int(np.ceil((len(group_scores) + 1) * coverage))
    group = None
    if rank <= len(group_scores):
        radius = float(group_scores[rank - 1])
        covered = measured <= radius
        group = {"radius": radius, "row_coverage": float(covered.mean()),
                 "complete_device_coverage": float(np.mean([covered[devices == name].all() for name in np.unique(devices)]))}

    # 经验区间按预测寿命分段，由独立校准设备生成。它用于评估覆盖率和区间宽度。
    cuts = np.unique(np.quantile(cal_prediction, [1 / 3, 2 / 3]))
    bins = np.searchsorted(cuts, cal_prediction)
    fallback = float(np.quantile(error, coverage, method="higher"))
    radii = []
    for idx in range(len(cuts) + 1):
        segment = error[bins == idx]
        radii.append(float(np.quantile(segment, coverage, method="higher")) if len(segment) >= 30 else fallback)
    widths = np.asarray(radii)[np.searchsorted(cuts, prediction)]
    return {"nominal_coverage": coverage, "calibration_devices": len(group_scores),
            "group_max_interval": group, "group_status": "calibrated" if group else "insufficient_calibration_devices",
            "empirical_interval": {"prediction_cuts": cuts.tolist(), "radii": radii,
                "row_coverage": float(np.mean(measured <= widths)),
                "mean_width_after_nonnegative_clip": float(np.mean(widths + np.minimum(widths, prediction)))}}


def history_features(x, device, time_index):
    """只读取同设备当前之前连续的最多五帧。断档时重新起窗。"""
    last = np.zeros_like(x)
    trend = np.zeros_like(x)
    for name in np.unique(device):
        indices = np.flatnonzero(device == name)
        ordered = indices[np.argsort(time_index[indices])]
        timestamps = time_index[ordered]
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Device timestamps must be unique")
        chunks = np.split(ordered, np.flatnonzero(np.diff(timestamps) != 1) + 1)
        for chunk in chunks:
            values = x[chunk]
            if len(values) < 2:
                continue
            last[chunk[1:]] = np.diff(values, axis=0)
            sums = np.vstack([np.zeros(x.shape[1]), np.cumsum(values, axis=0)])
            stop = np.arange(1, len(values))
            start = np.maximum(0, stop - 5)
            trend[chunk[1:]] = values[1:] - (sums[stop] - sums[start]) / (stop - start)[:, None]
    return np.column_stack([x, last, trend])


def ridge_prediction(features, y, fit, holdout, parent_prediction):
    origin = features[fit]
    mean = origin.mean(axis=0)
    scale = np.maximum(origin.std(axis=0), 1e-8)
    def design(x):
        return np.column_stack([np.ones(len(x)), np.clip((x - mean) / scale, -10, 10)])
    matrix = design(origin)
    ridge = .1 * np.eye(matrix.shape[1])
    ridge[0, 0] = 1e-8
    residual = y[fit] - parent_prediction[fit]
    coefficients = np.linalg.solve(matrix.T @ matrix / len(matrix) + ridge, matrix.T @ residual / len(matrix))
    return np.maximum(0., parent_prediction[holdout] + design(features[holdout]) @ coefficients)


def waveform_rows(data, root):
    """固定每十帧抽取一条已公开 B 样本，使用匹配原始双通道波形。"""
    conditions = {"1": "35Hz12kN", "2": "37.5Hz11kN", "3": "40Hz10kN"}
    index, features, identities = [], [], {}
    rms_column = sorted(BEARING_FEATURES).index("rms")
    for row, (name, snapshot) in enumerate(zip(data["metadata"]["bearing"], data["metadata"]["snapshot"])):
        match = re.fullmatch(r"Bearing([123])_\d+", name)
        if not match or type(snapshot) is not int or snapshot < 1:
            raise ValueError("Waveform trace metadata is invalid")
        if (snapshot - 1) % 10:
            continue
        path = root / conditions[match[1]] / name / f"{snapshot}.csv"
        if not path.is_file():
            continue
        signal = np.loadtxt(path, dtype=np.float64, delimiter=",", skiprows=1)
        if signal.shape != (32768, 2) or not np.isfinite(signal).all():
            raise ValueError(f"Invalid waveform: {path.name}")
        expected = data["x"][row, [rms_column, rms_column + len(BEARING_FEATURES)]]
        if not np.allclose(np.sqrt(np.mean(signal ** 2, axis=0)), expected, rtol=1e-8, atol=1e-9):
            raise ValueError(f"Waveform RMS differs from its public projection: {path}")
        power = np.abs(np.fft.rfft(signal - signal.mean(axis=0), axis=0)) ** 2
        power /= np.maximum(power.sum(axis=0), 1e-12)
        block_rms = np.sqrt(np.mean(signal.reshape(16, 2048, 2) ** 2, axis=1))
        # 频谱在等宽子频带的分布及 16 子窗口 RMS 的波动。
        bands = [part.sum(axis=0) for part in np.array_split(power, 16)]
        variation = block_rms.std(axis=0) / np.maximum(block_rms.mean(axis=0), 1e-12)
        features.append(np.concatenate([*bands, variation]))
        index.append(row)
        identities[path.relative_to(root).as_posix()] = digest(path)
    if not index:
        raise ValueError("No waveform samples matched the public projection")
    index = np.asarray(index)
    return index, np.column_stack([data["x"][index], np.asarray(features)]), identities


def scored(data, dataset, prediction):
    result = evaluate(None, data, dataset, prediction=prediction)
    result["scope"] = "local_device_holdout"
    return compact_metrics(result)


def run(base, output, dataset, config, waveforms=None, *, ablations=False):
    fields = ("bearing", "snapshot") if dataset == "B" else ("source_subset", "source_split", "unit", "cycle")
    data = read_dataset(base / "public/train.jsonl", "regression", [], schema(dataset), metadata_fields=fields)
    device = groups(data, dataset)["device"]
    time_index = np.asarray(data["metadata"][fields[-1]])
    if time_index.dtype.kind not in "iu" or np.any(time_index < 1):
        raise ValueError("History validation requires public positive integer timestamps")
    history = history_features(data["x"], device, time_index)
    raw = waveform_rows(data, waveforms) if dataset == "B" and waveforms else None
    menu = {s["name"]: s for s in config["experiments"]}
    additions = {}
    if ablations:
        if dataset == "B":
            ridge = menu["bearing_log_ridge"]
            additions = {
                "long_life_weight": {**menu["low_rate"], "datasets": ["B"], "tail_boost": 1.25},
                "short_overestimate": {**menu["low_rate"], "datasets": ["B"], "short_overestimate_boost": 1.25},
                "bearing_raw_ridge": {**ridge, "transform": "bearing_raw"},
                "bearing_energy_ridge": {**ridge, "transform": "bearing_energy"}}
        else:
            ridge = menu["cmapss_soft_ridge"]
            additions = {"cmapss_global_ridge": {**ridge, "transform": "cmapss_global"},
                         "cmapss_device_support_ridge": {**ridge, "transform": "condition_soft_device"}}
            for mix in (0, 5, 10):
                additions[f"cmapss_mlp_nasa_{mix}"] = {**menu["cmapss_asymmetric"], "nasa_mix": mix / 100}
        additions = {name: {**spec, "name": name} for name, spec in additions.items()}
        validate_config({**config, "experiments": [*config["experiments"], *additions.values()]}, config["limits"])
    linear, nonlinear, optimizer = (("bearing_log_ridge", "bearing_mlp", "bearing_balanced_half") if dataset == "B"
                                    else ("cmapss_soft_ridge", "cmapss_condition_mlp", "cmapss_asymmetric"))
    folds = []
    for index, (split, masks) in enumerate(partitions(data, dataset), 1):
        started = time.perf_counter()
        fold = output / dataset / f"fold-{index}"
        fold.mkdir(parents=True)
        parts = {name: subset(data, mask) for name, mask in masks.items()}
        fit, tune, calibration, holdout = [parts[name] for name in ("fit", "tune", "calibrate", "holdout")]
        if len({p["sha256"] for p in parts.values()}) != 4:
            raise ValueError("Duplicate fold identity")
        parent = RegressionModel("regression", schema(dataset), [], fit["x"], fit["y"],
                                 hidden=24 if dataset == "B" else 32, seed=42,
                                 learning_rate=.003 if dataset == "B" else .002, batch_size=128 if dataset == "B" else 256)
        parent.meta.update(profile=dataset, target_unit=UNITS[dataset])
        parent.fit(fit["x"], fit["y"], 15)
        checkpoint = fold / "fresh-baseline.npz"
        parent.save(checkpoint)
        parent_prediction = parent.predict(data["x"])
        scores = {"initial": scored(holdout, dataset, parent_prediction[masks["holdout"]])}
        eligible = []
        checkpoints = {}
        inner = {}
        def candidate(name, start, spec, *, fixed_epochs=False):
            dest = fold / name
            metrics = continue_candidate(start, fit, tune, dest, spec, config["limits"], config["controller"], dataset,
                                         digest(start), fixed_epochs=fixed_epochs)
            path = dest / "checkpoint.npz"
            checkpoints[name] = path
            candidate_model = load_model(path, dataset)
            inner[name] = {"rmse": metrics["validation_after"]["rmse"], "eligible": metrics["selected_epoch_eligible"],
                           "mae": metrics["validation_after"]["mae"], "nasa_score": metrics["validation_after"].get("nasa_score"),
                           "fixed_epochs": fixed_epochs}
            scores[name] = scored(holdout, dataset, candidate_model.predict(holdout["x"]))
            return path
        candidate("low_rate", checkpoint, menu["low_rate"], fixed_epochs=ablations)
        candidate(optimizer, checkpoint, menu[optimizer], fixed_epochs=ablations)
        if ablations and dataset == "B":
            candidate("bearing_balanced", checkpoint, menu["bearing_balanced"], fixed_epochs=True)
        prerequisites = {}
        for correction in (linear, nonlinear):
            corrected = candidate(correction, checkpoint, menu[correction])
            combined = f"{correction}_then_{optimizer}"
            candidate(combined, corrected, menu[optimizer])
            prerequisites[combined] = correction
        for name, spec in additions.items():
            from_condition_mlp = name.startswith("cmapss_mlp_nasa_")
            parent_path = checkpoints[nonlinear] if from_condition_mlp else checkpoint
            candidate(name, parent_path, spec, fixed_epochs=spec.get("method", "mlp") == "mlp")
            if from_condition_mlp:
                prerequisites[name] = nonlinear
        for name, value in inner.items():
            if value["eligible"] and (name not in prerequisites or inner[prerequisites[name]]["eligible"]):
                eligible.append((value["rmse"], name))
        selected = min(eligible)[1] if eligible else "initial"
        model = load_model(checkpoints.get(selected, checkpoint), dataset)
        uncertainty = intervals(model.predict(calibration["x"]), calibration["y"], device[masks["calibrate"]],
                                 model.predict(holdout["x"]), holdout["y"], device[masks["holdout"]])
        matched = {}
        for name, view in (("point_ridge", data["x"]), ("history_ridge", history)):
            predicted = ridge_prediction(view, data["y"], masks["fit"], masks["holdout"], parent_prediction)
            matched[name] = scored(holdout, dataset, predicted)
        if raw:
            rows, waveform, _ = raw
            train_mask, test_mask = masks["fit"][rows], masks["holdout"][rows]
            full = np.zeros(len(data["y"]), dtype=bool)
            full[rows[test_mask]] = True
            if np.any(train_mask) and np.any(test_mask):
                raw_scope = subset(data, full)
                matched["waveform_subset"] = {"fit_rows": int(train_mask.sum()), "holdout_rows": int(test_mask.sum()),
                    "matched_point": scored(raw_scope, dataset, ridge_prediction(data["x"][rows], data["y"][rows],
                        train_mask, test_mask, parent_prediction[rows])),
                    "waveform": scored(raw_scope, dataset, ridge_prediction(waveform, data["y"][rows], train_mask, test_mask, parent_prediction[rows]))}
        result = {"fold": index, "devices": split, "projection_sha256": {k: p["sha256"] for k, p in parts.items()},
                  "inner_model_selection": inner, "inner_selected": selected, "holdout_metrics": scores,
                  "prediction_intervals": uncertainty, "matched_input_experiments": matched,
                  "wall_seconds": time.perf_counter() - started}
        write_json(fold / "report.json", result)
        folds.append(result)
        print(f"{dataset} fold {index}: selected {selected}, held-out RMSE {scores[selected]['rmse']:.4f}", flush=True)
    if digest(base / "public/train.jsonl") != data["sha256"]:
        raise ValueError("Public training projection changed during validation")
    return {"dataset": dataset, "source_training_sha256": data["sha256"], "folds": folds,
            "matched_waveform_sha256": raw[2] if raw else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bearing-waveforms", type=Path)
    parser.add_argument("--ablations", action="store_true", help="Evaluate the next fixed-fold feature and loss comparisons")
    parser.add_argument("--datasets", choices="BC", nargs="+", default=list("BC"))
    args = parser.parse_args()
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]
    validate_config(config, config["limits"])
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "submission_config.json", *sorted((ROOT / "src").rglob("*.py"))]
    fingerprints = {str(p): digest(p) for p in sources}
    reports = [run(args.baseline_root / BASELINES[key], args.output, key, config, args.bearing_waveforms,
                   ablations=args.ablations)
               for key in args.datasets]
    if any(digest(path) != expected for path, expected in fingerprints.items()):
        raise ValueError("Source files changed during validation")
    write_json(args.output / "summary.json", {"scope": "public_training_device_holdouts_fresh_initialization",
        "source_sha256": fingerprints, "reports": reports, "real_model_calls": 0, "official_score_verified": False})


if __name__ == "__main__":
    main()
