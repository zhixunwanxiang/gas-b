"""Training-origin drift reference and public-validation diagnosis.

This module never receives hidden-stream labels or fits the model scaler.
KS is a quantile-bin approximation, not a significance-test p-value.
"""
# 漂移监测与统计诊断：固定参考来自公开训练特征，当前测量来自授权公开验证特征。
# PSI、分箱 KS 和均值偏移用于发现分布差异；混淆矩阵用于说明错误集中在哪里。
# 二者只能支持可检验假设，不构成传感器物理故障的因果证明，也不是隐藏流遥测。
import numpy as np

from .sensor_training.io import CLASSES


def fit_reference(train):
    # 逐特征保存训练集十分位切点、频数、均值和标准差，绑定训练数据摘要及特征顺序。
    # 重复分位点合并，常数列尺度设极小下限；这套监测统计不替换模型的标准化参数。
    x = train["x"]
    features = []
    for column in x.T:
        # 用训练分位数分箱以适应各特征不同量纲；unique 合并常数列/重复值造成的相同切点。
        cuts = np.unique(np.quantile(column, np.arange(1, 10) / 10))
        counts = np.bincount(np.searchsorted(cuts, column, side="right"), minlength=len(cuts) + 1)
        features.append({"cuts": cuts.tolist(), "counts": counts.tolist(),
                         "mean": float(column.mean()), "scale": max(float(column.std()), 1e-12)})
    return {"format": "gas-a.monitor.v1", "origin": "authorized_public_train",
            "train_sha256": train["sha256"], "rows": len(x), "paths": train["paths"], "features": features}


def monitor(reference, current, thresholds):
    # 当前样本始终放入同一组训练切点中，保证分布指标可比较。
    # 输出 128 个特征统计，并按每个传感器的 8 维取均值；样本数不足时保留统计但不触发告警。
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
        # 每箱加 0.5 平滑，防止 PSI 的 log(q/p) 遇到零频率；分母同步加上总平滑质量。
        # PSI=sum((q-p)*log(q/p)) 衡量占比分布差异；下面的 KS 比较未平滑箱频累计分布，不是检验 p 值。
        p, q = (counts + .5) / (counts.sum() + .5 * len(counts)), (measured + .5) / (rows + .5 * len(counts))
        features.append({"feature": index + 1, "sensor": index // 8 + 1,
                         "psi": float(np.sum((q - p) * np.log(q / p))),
                         "binned_ks": float(np.max(np.abs(np.cumsum(counts / counts.sum()) - np.cumsum(measured / rows)))),
                         "absolute_mean_z": float(abs(column.mean() - ref["mean"]) / ref["scale"])})
    sensors = []
    # 样本量门槛只决定是否触发告警，不阻止保留每个特征/传感器的实际统计。
    enough = rows >= thresholds["minimum_rows"]
    for index in range(16):
        group = features[index * 8:(index + 1) * 8]
        # 一个传感器对应连续八维特征；当前实现先对八维指标取均值，再用任意指标越界判漂移。
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
    # 选取 F1 最弱的三个有支持类别、最大的五种非对角混淆及漂移最强的传感器。
    # 只有验证集摘要相同才计算跨轮分数差，避免把数据变化误当成模型改善。
    classes = [{"label": name, **item} for name, item in metrics["per_class"].items() if item["support"]]
    weakest = sorted(classes, key=lambda c: c["f1"])[:3]
    matrix = metrics["confusion_matrix"]
    confusions = sorted(({"actual": CLASSES[i], "predicted": CLASSES[j], "rows": matrix[i][j]}
                         for i in range(6) for j in range(6) if i != j and matrix[i][j]),
                        key=lambda item: item["rows"], reverse=True)[:5]
    # 旧分数只有在验证文件摘要相同的情况下才可相减；数据范围改变时差值保持未知。
    same_scope = previous is not None and previous.get("validation_sha256") == validation_sha256
    return {"scope": "authorized_public_validation", "validation_sha256": validation_sha256,
            "weak_classes": weakest, "top_confusions": confusions, "drift_sensors": drift["sensors"][:4],
            "same_scope_previous_macro_f1_delta": metrics["macro_f1"] - previous["selected_macro_f1"] if same_scope else None,
            "hypothesis": "Test bounded sensor selection or head continuation against the measured confusions; sensor drift alone does not establish causation.",
            "hidden_feedback_available": False}
