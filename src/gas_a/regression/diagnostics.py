"""原单位误差、NASA 尾部及追踪分组；元数据始终不进入预测输入。"""
import math
import numpy as np

VERSION = "rul-group-tail-metrics-v2"
CENTERS = np.asarray([[0, 0, 100], [10, .25, 100], [20, .7, 100],
                      [25, .62, 60], [35, .84, 100], [42, .84, 100]], dtype=float)


def condition_distances(x):
    return np.sum(((x[:, None, 21:] - CENTERS[None]) / [42, .84, 100]) ** 2, axis=2)


def conditions(x):
    """只用合法 settings 匹配六个名义工况；超出邻域时回退到 unknown。"""
    distance = condition_distances(x)
    nearest = distance.argmin(axis=1)
    return np.where(distance[np.arange(len(x)), nearest] <= .01 ** 2, nearest, -1)


def groups(data, dataset):
    meta = data.get("metadata", {})
    n = len(data["y"])
    if any(len(values) != n for values in meta.values()):
        raise ValueError("Regression trace metadata length mismatch")
    def field(key):
        return meta.get(key, [None] * n)
    def label(value):
        return str(value) if value is not None else "unknown"
    if dataset == "B":
        return {"device": np.asarray([label(v) for v in field("bearing")])}
    # 不同 FD 的 unit 编号会重复，必须包含来源和 source_split。
    device = ["/".join(map(label, row)) if all(v is not None for v in row) else "unknown"
              for row in zip(field("source_subset"), field("source_split"), field("unit"))]
    return {"device": np.asarray(device), "source": np.asarray([label(v) for v in field("source_subset")]),
            "condition": np.asarray([str(i) if i >= 0 else "unknown" for i in conditions(data["x"])])}


def nasa_terms(error):
    exponent = np.where(error < 0, -error / 13., error / 10.)
    with np.errstate(over="ignore"):
        return np.expm1(exponent)


def error_metrics(error, dataset, *, detailed=True):
    if error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError("Regression errors must be a nonempty finite vector")
    absolute = np.abs(error)
    with np.errstate(over="ignore"):
        result = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(absolute.mean()),
                  "bias": float(error.mean()), "overestimate_fraction": float(np.mean(error > 0)),
                  "rows": len(error)}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite regression measurement")
    if detailed:
        result.update(absolute_error_p99=float(np.quantile(absolute, .99)), absolute_error_max=float(absolute.max()))
        for name, selected in (("underestimate", -error[error < 0]), ("overestimate", error[error > 0])):
            result[f"{name}_count"] = len(selected)
            result[f"{name}_mean"] = float(selected.mean()) if len(selected) else None
            p95, p99 = np.quantile(selected, [.95, .99]).tolist() if len(selected) else (None, None)
            result[f"{name}_p95"] = p95
            result[f"{name}_p99"] = p99
    if dataset == "C":
        terms = nasa_terms(error)
        with np.errstate(over="ignore"):
            score = float(terms.sum())
        result.update(nasa_score=score if math.isfinite(score) else None,
                      nasa_score_overflow=not math.isfinite(score))
        if not detailed:
            return result
        with np.errstate(over="ignore"):
            under, over = float(terms[error < 0].sum()), float(terms[error > 0].sum())
        result.update(nasa_score_mean=score / len(error) if math.isfinite(score) else None,
                      nasa_underestimate_score=under if math.isfinite(under) else None,
                      nasa_overestimate_score=over if math.isfinite(over) else None,
                      nasa_top_1pct_share=None)
        if math.isfinite(score):
            k = max(1, math.ceil(.01 * len(error)))
            result["nasa_top_1pct_share"] = float(np.partition(terms, len(terms) - k)[-k:].sum() / score) if score else 0.
    return result


def grouped_metrics(error, data, dataset, total_nasa=None):
    result = {}
    total_sse = float(np.sum(error ** 2))
    for axis, values in groups(data, dataset).items():
        rows = {}
        for name in np.unique(values):
            selected = error[values == name]
            metrics = error_metrics(selected, dataset)
            metrics["squared_error_share"] = float(np.sum(selected ** 2) / total_sse) if total_sse else 0.
            if dataset == "C":
                metrics["nasa_share"] = (metrics["nasa_score"] / total_nasa if total_nasa
                    and metrics["nasa_score"] is not None else 0. if total_nasa == 0 else None)
            rows[str(name)] = metrics
        result[axis] = rows
    return result


def compact_metrics(metrics):
    """保留 RMSE 最差组及 NASA 贡献最高的发动机，完整分组留在报告。"""
    result = {k: v for k, v in metrics.items() if k != "group_metrics"}
    result["group_metrics"] = {axis: dict(sorted(rows.items(), key=lambda item: item[1]["rmse"], reverse=True)[:8])
                               for axis, rows in metrics.get("group_metrics", {}).items()}
    if "nasa_score" in metrics:
        devices = metrics.get("group_metrics", {}).get("device", {})
        top = sorted(devices.items(), key=lambda item: item[1].get("nasa_share") or 0., reverse=True)[:4]
        result["largest_nasa_device_shares"] = {key: row.get("nasa_share") for key, row in top}
    return result
