"""Training-origin drift reference and public-validation diagnosis.

This module never receives hidden-stream labels or fits the model scaler.
KS is a quantile-bin approximation, not a significance-test p-value.
"""
import numpy as np

from .sensor_training.io import CLASSES


def fit_reference(train):
    x = train["x"]
    features = []
    for column in x.T:
        cuts = np.unique(np.quantile(column, np.arange(1, 10) / 10))
        counts = np.bincount(np.searchsorted(cuts, column, side="right"), minlength=len(cuts) + 1)
        features.append({"cuts": cuts.tolist(), "counts": counts.tolist(),
                         "mean": float(column.mean()), "scale": max(float(column.std()), 1e-12)})
    return {"format": "gas-a.monitor.v1", "origin": "authorized_public_train",
            "train_sha256": train["sha256"], "rows": len(x), "paths": train["paths"], "features": features}


def monitor(reference, current, thresholds):
    if (reference.get("format") != "gas-a.monitor.v1"
            or reference.get("origin") != "authorized_public_train"
            or reference.get("paths") != current["paths"] or len(reference.get("features", [])) != 128):
        raise ValueError("Invalid or incompatible training-origin monitoring reference")
    rows = len(current["x"])
    features = []
    for index, (column, ref) in enumerate(zip(current["x"].T, reference["features"])):
        cuts, counts = np.asarray(ref["cuts"]), np.asarray(ref["counts"], dtype=float)
        if (cuts.ndim != 1 or len(counts) != len(cuts) + 1 or not np.isfinite(cuts).all()
                or np.any(np.diff(cuts) <= 0) or not np.isfinite(counts).all() or np.any(counts < 0)
                or counts.sum() != reference["rows"] or not np.isfinite(ref["mean"])
                or not np.isfinite(ref["scale"]) or ref["scale"] <= 0):
            raise ValueError("Invalid monitoring reference statistics")
        measured = np.bincount(np.searchsorted(cuts, column, side="right"), minlength=len(counts))
        p, q = (counts + .5) / (counts.sum() + .5 * len(counts)), (measured + .5) / (rows + .5 * len(counts))
        features.append({"feature": index + 1, "sensor": index // 8 + 1,
                         "psi": float(np.sum((q - p) * np.log(q / p))),
                         "binned_ks": float(np.max(np.abs(np.cumsum(counts / counts.sum()) - np.cumsum(measured / rows)))),
                         "absolute_mean_z": float(abs(column.mean() - ref["mean"]) / ref["scale"])})
    sensors = []
    enough = rows >= thresholds["minimum_rows"]
    for index in range(16):
        group = features[index * 8:(index + 1) * 8]
        item = {"sensor": index + 1, **{key: float(np.mean([f[key] for f in group]))
                                        for key in ("psi", "binned_ks", "absolute_mean_z")}}
        item["drift"] = bool(enough and any(item[key] >= thresholds[key] for key in ("psi", "binned_ks", "absolute_mean_z")))
        sensors.append(item)
    return {"scope": "authorized_public_validation_features", "current_sha256": current["sha256"],
            "reference_train_sha256": reference["train_sha256"], "rows": rows,
            "sufficient_rows": enough, "drift": any(s["drift"] for s in sensors),
            "thresholds": thresholds, "sensors": sorted(sensors, key=lambda s: s["psi"], reverse=True),
            "features": features, "limitation": "Distribution shift is evidence of association, not a physical cause or hidden-stream measurement."}


def diagnose(metrics, drift, validation_sha256, previous=None):
    classes = [{"label": name, **item} for name, item in metrics["per_class"].items() if item["support"]]
    weakest = sorted(classes, key=lambda c: c["f1"])[:3]
    matrix = metrics["confusion_matrix"]
    confusions = sorted(({"actual": CLASSES[i], "predicted": CLASSES[j], "rows": matrix[i][j]}
                         for i in range(6) for j in range(6) if i != j and matrix[i][j]),
                        key=lambda item: item["rows"], reverse=True)[:5]
    same_scope = previous is not None and previous.get("validation_sha256") == validation_sha256
    return {"scope": "authorized_public_validation", "validation_sha256": validation_sha256,
            "weak_classes": weakest, "top_confusions": confusions, "drift_sensors": drift["sensors"][:4],
            "same_scope_previous_macro_f1_delta": metrics["macro_f1"] - previous["selected_macro_f1"] if same_scope else None,
            "hypothesis": "Test bounded sensor selection or head continuation against the measured confusions; sensor drift alone does not establish causation.",
            "hidden_feedback_available": False}
