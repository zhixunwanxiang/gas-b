"""B/C 的特征漂移与回归误差诊断，监测参考只来自公开训练集。"""
import numpy as np


def fit_reference(train, dataset):
    # 参考分布只拟合一次公开 train：每列保存十分位分箱、箱计数、均值和标准差。
    # 训练摘要与特征路径随参考一起保存，后续不会把验证数据反过来重拟合为“正常分布”。
    features = []
    for column in train["x"].T:
        cuts = np.unique(np.quantile(column, np.arange(1, 10) / 10))
        counts = np.bincount(np.searchsorted(cuts, column, side="right"), minlength=len(cuts) + 1)
        features.append({"cuts": cuts.tolist(), "counts": counts.tolist(), "mean": float(column.mean()),
                         "scale": max(float(column.std()), 1e-12)})
    return {"format": "rul.monitor.v1", "dataset": dataset, "train_sha256": train["sha256"],
            "paths": train["paths"], "rows": len(train["x"]), "features": features,
            "target_cuts": np.unique(np.quantile(train["y"], [.25, .5, .75])).tolist()}


def monitor(reference, current, train, dataset, thresholds):
    # SDK 允许本轮 train 投影更新；监测继续相对“最初公开 train”，不追随新分布移动参考。
    # 旧参考由 SDK 保存的状态提供完整性保护；这里核验数据集、来源摘要格式、维度和统计结构。
    reference_sha = reference.get("train_sha256")
    if (reference.get("format") != "rul.monitor.v1" or reference.get("dataset") != dataset
            or not isinstance(reference_sha, str) or not reference_sha.startswith("sha256:")
            or len(reference_sha) != 71 or any(c not in "0123456789abcdef" for c in reference_sha[7:])
            or type(reference.get("rows")) is not int or reference["rows"] < 1
            or reference.get("paths") != current["paths"] or train["paths"] != current["paths"]
            or len(reference.get("features", [])) != len(current["paths"])):
        raise ValueError("Regression monitoring reference differs from its original training projection")
    rows, features = len(current["x"]), []
    for index, (column, ref) in enumerate(zip(current["x"].T, reference["features"])):
        cuts, counts = np.asarray(ref["cuts"]), np.asarray(ref["counts"], dtype=float)
        if (cuts.ndim != 1 or len(counts) != len(cuts) + 1 or not np.isfinite(cuts).all()
                or np.any(np.diff(cuts) <= 0) or not np.isfinite(counts).all() or np.any(counts < 0)
                or counts.sum() != reference["rows"] or not np.isfinite(ref["mean"])
                or not np.isfinite(ref["scale"]) or ref["scale"] <= 0):
            raise ValueError("Invalid regression monitoring statistics")
        measured = np.bincount(np.searchsorted(cuts, column, side="right"), minlength=len(counts))
        # PSI 比较各箱占比；每箱加 0.5 平滑，防止 log 遇到零概率。
        # binned_ks 比较累计箱频，属于分箱近似距离，不是统计显著性的 p 值。
        # absolute_mean_z 把均值变化换算成“偏离训练均值多少个训练标准差”。
        p = (counts + .5) / (counts.sum() + .5 * len(counts))
        q = (measured + .5) / (rows + .5 * len(counts))
        item = {"feature": current["paths"][index], "psi": float(np.sum((q - p) * np.log(q / p))),
                "binned_ks": float(np.max(np.abs(np.cumsum(counts / counts.sum()) - np.cumsum(measured / rows)))),
                "absolute_mean_z": float(abs(column.mean() - ref["mean"]) / ref["scale"])}
        item["drift"] = bool(rows >= thresholds["minimum_rows"] and
                             any(item[key] >= thresholds[key] for key in ("psi", "binned_ks", "absolute_mean_z")))
        features.append(item)
    # 当前批达到样本数门槛后，一列的任一指标越界就报警；B/C 按列检查，不套 A 的八维传感器均值。
    # 展示排名按 PSI 排序；整体 drift 为任意列告警。这个结论只说明输入变了，不证明性能变差。
    ordered = sorted(features, key=lambda f: f["psi"], reverse=True)
    return {"scope": "authorized_public_validation_features", "dataset": dataset, "rows": rows,
            "drift": any(f["drift"] for f in features), "changed_features": sum(f["drift"] for f in features),
            "reference_policy": "fixed_initial_training_projection",
            "reference_train_sha256": reference_sha, "current_train_sha256": train["sha256"],
            "training_projection_changed": reference_sha != train["sha256"], "current_sha256": current["sha256"],
            "thresholds": thresholds, "top_features": ordered[:8], "features": ordered}


def diagnose(model, validation, reference, metrics, drift):
    # 只有授权公开 validation 的标签参与误差诊断，不向隐藏预测过程索取真实标签。
    # 寿命分组边界来自 train 的 25%/50%/75% 分位点，避免每轮更换分组后无法对比。
    error = model.predict(validation["x"]) - validation["y"]
    cuts = np.asarray(reference["target_cuts"], dtype=float)
    if cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(np.diff(cuts) <= 0):
        raise ValueError("Invalid training-origin RUL diagnosis cut points")
    groups = np.searchsorted(cuts, validation["y"], side="right")
    segments = []
    # 看误差集中在哪个寿命段、主要高估还是低估，再把输入变化一起交给控制器形成实验假设。
    # 分组误差和分布变化都只是线索，代码不会把它们直接断言为某个设备的物理故障原因。
    for group in range(len(cuts) + 1):
        selected = error[groups == group]
        if len(selected):
            segments.append({"rul_group": group, "rows": len(selected),
                             "rmse": float(np.sqrt(np.mean(selected ** 2))),
                             "mae": float(np.abs(selected).mean()), "bias": float(selected.mean())})
    return {"scope": "authorized_public_validation", "metrics": metrics,
            "training_target_cuts": cuts.tolist(), "rul_segments": segments,
            "drift_features": drift["top_features"],
            "interpretation": "Positive bias means overestimated remaining life; feature drift and error groups are hypotheses, not physical failure causes.",
            "hidden_feedback_available": False}
