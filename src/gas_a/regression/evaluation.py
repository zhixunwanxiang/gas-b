"""原单位回归指标和越低越好的上线门槛；不复用 A 的 F1/召回率规则。"""
import math
import numpy as np
from ..host_validation import SCORE_TOLERANCE, parse_measurement, metric_matches
from ..sensor_training.io import digest
from ..parent_checkpoint import verify_selected_checkpoint
from .diagnostics import VERSION as METRIC_VERSION, error_metrics, grouped_metrics, groups

PARENT_REFERENCE_VERSION = "current-parent-public-reevaluation-v2"
GROUP_GUARDS = {"B": ("bearing_device_guard", "device"), "C": ("cmapss_source_guard", "source")}


def guarded_masks(data, dataset):
    axis = GROUP_GUARDS[dataset][1]
    labels = groups(data, dataset)[axis]
    return {name: labels == name for name in np.unique(labels)}


def evaluate(model, data, dataset, *, prediction=None, detailed=True, group_masks=None):
    # Model.predict 已把标准化输出还原到原单位；这里直接计算分钟/循环上的误差。
    # 正 bias 表示寿命被高估，overestimate_fraction 表示高估样本占比，供诊断使用。
    prediction = model.predict(data["x"]) if prediction is None else prediction
    if prediction.shape != data["y"].shape or not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise ValueError("Regression predictions must be finite nonnegative values")
    error = prediction - data["y"]
    result = error_metrics(error, dataset, detailed=detailed)
    result.update(dataset=dataset, metric_version=METRIC_VERSION,
                  validation_sha256=data.get("sha256"), scope="authorized_public_validation")
    if not detailed:
        if group_masks is None:
            group_masks = guarded_masks(data, dataset)
        axis = GROUP_GUARDS[dataset][1]
        result["group_metrics"] = {axis: {name: {"rows": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error[mask] ** 2)))} for name, mask in group_masks.items()}}
        return result
    result["group_metrics"] = grouped_metrics(error, data, dataset, result.get("nasa_score"))
    devices = result["group_metrics"]["device"]
    known = [(key, row) for key, row in devices.items() if key != "unknown"]
    result["device_equal_mean_rmse"] = float(np.mean([v["rmse"] for _, v in known])) if known else None
    worst = max(known, key=lambda item: item[1]["rmse"], default=None)
    result["worst_device"] = ({"device": worst[0], "rmse": worst[1]["rmse"], "rows": worst[1]["rows"]} if worst else None)
    return result


def _guard_groups(reference, after, axis, bounds, *, stage=False):
    losses, comparable = {}, True
    candidate_groups = after.get("group_metrics", {}).get(axis, {})
    relative = bounds.get("stage_relative_increase", bounds["relative_increase"]) if stage else bounds["relative_increase"]
    for name, previous in reference.get("group_metrics", {}).get(axis, {}).items():
        if name == "unknown" or previous["rows"] < bounds["minimum_rows"]:
            continue
        current = candidate_groups.get(name)
        if current is None or current["rows"] != previous["rows"]:
            comparable = False
            continue
        allowance = max(bounds["absolute_increase"], previous["rmse"] * relative)
        if current["rmse"] > previous["rmse"] + allowance + 1e-12:
            losses[name] = {"rows": previous["rows"], "parent_rmse": previous["rmse"],
                            "candidate_rmse": current["rmse"], "allowed_increase": allowance}
    return losses, comparable


def promotion_gate(before, after, rules, *, require_all_improved=False, retention_reference=None):
    # 改善量为 父 RMSE - 新 RMSE；分钟和循环分别比较，不把它们转成分类百分点。
    gain = before["rmse"] - after["rmse"]
    required = max(rules["minimum_rmse_absolute_gain"], before["rmse"] * rules["minimum_rmse_relative_gain"])
    # 绝对收益和相对收益取较严者：默认相对 0.0005 即误差至少降低 0.05%。
    # NASA 保护为显式本地策略，不冒充比赛公式；缺失/溢出按不可核验处理。
    mae_limit = before["mae"] * (1 + rules["maximum_mae_relative_increase"])
    reasons = []
    if not all(math.isfinite(m[k]) and m[k] >= 0 for m in (before, after) for k in ("rmse", "mae")):
        reasons.append("invalid_primary_metrics")
    for key in ("validation_sha256", "rows", "dataset", "metric_version"):
        if (key in before or key in after) and before.get(key) != after.get(key):
            reasons.append("incomparable_" + key)
    if gain <= 1e-12 or gain < required:
        reasons.append("rmse_reduction_below_threshold")
    if after["mae"] > mae_limit + 1e-12:
        reasons.append("mae_regression")
    if require_all_improved and after["mae"] >= before["mae"]:
        reasons.append("mae_not_improved")
    group_losses, stage_losses = {}, {}
    guard = GROUP_GUARDS.get(before.get("dataset"))
    if guard and guard[0] in rules:
        name, axis = guard
        group_losses, comparable = _guard_groups(before, after, axis, rules[name])
        if not comparable:
            reasons.append(f"incomparable_{axis}_support")
        if group_losses:
            reasons.append("bearing_device_regression" if axis == "device" else "cmapss_source_regression")
        if retention_reference is not None and retention_reference is not before:
            # 两道护栏分别约束单轮退化与同阶段累计退化，避免小幅退化逐轮累积。
            if any(retention_reference.get(key) != after.get(key)
                   for key in ("validation_sha256", "rows", "dataset", "metric_version")):
                reasons.append("incomparable_stage_reference")
            else:
                stage_losses, comparable = _guard_groups(retention_reference, after, axis, rules[name], stage=True)
                if not comparable:
                    reasons.append("incomparable_stage_group_support")
                if stage_losses:
                    reasons.append("stage_group_regression")
    nasa_limit = None
    has_nasa = before.get("dataset") == "C" or "nasa_score" in before
    if has_nasa and (require_all_improved or "maximum_nasa_relative_increase" in rules):
        a, b = before.get("nasa_score"), after.get("nasa_score")
        if (a is None or b is None or not math.isfinite(a) or not math.isfinite(b)
                or before.get("nasa_score_overflow") or after.get("nasa_score_overflow")):
            reasons.append("nasa_measurement_unverified")
        else:
            nasa_limit = a if require_all_improved else a * (1 + rules["maximum_nasa_relative_increase"])
            if b > nasa_limit + 1e-12:
                reasons.append("nasa_regression")
            elif require_all_improved and b >= a:
                reasons.append("nasa_not_improved")
    return {"eligible": not reasons, "reasons": reasons,
            "rmse_reduction": gain, "required_rmse_reduction": required, "mae_limit": mae_limit,
            "nasa_limit": nasa_limit, "device_losses": group_losses if before.get("dataset") == "B" else {},
            "source_losses": group_losses if before.get("dataset") == "C" else {},
            "stage_group_losses": stage_losses}


def measurement(value, validation_sha256, *, binding=None):
    # SDK 候选回执和父分数帧都可能携带 summary；先统一形状，再核对指标和数据范围。
    # 数字看起来更好还不够：公开范围和文件身份必须可核验；缺省身份只接受本轮 SDK 候选绑定。
    return parse_measurement(value, validation_sha256, "rmse", binding=binding)


def parent_reference(host_value, before, sha):
    # 宿主没有给父分数时，使用本地同口径父评估；给了分数就必须核验，不能忽略冲突。
    if host_value is None or host_value == {}:
        return {"comparable": True, "rmse": before["rmse"], "validation_sha256": sha,
                "source": "local_parent_host_measurement_absent"}
    result = measurement(host_value, sha)
    if result["comparable"] and abs(result["rmse"] - before["rmse"]) > SCORE_TOLERANCE:
        return {**result, "comparable": False, "reason": "parent_host_local_measurement_conflict"}
    return result


def current_parent_reference(rnd, checkpoint, before, sha):
    """Use an independently remeasured parent when start metadata is incomplete.

    This does not certify the provenance of a compact host score. The reference
    comes from the actual SDK parent on this round's public validation file.
    Explicit conflicting metadata or scores still prevent promotion.
    """
    if (rnd.frame.get("protocol") != "contest.harness.workspace.v2"
            or rnd.dataset_key not in ("B", "C") or rnd.task.get("task") != "regression"
            or not metric_matches(rnd.task, "rmse")
            or digest(rnd.data.validation_path) != sha):
        raise ValueError("Current parent or public validation identity changed")
    verify_selected_checkpoint(rnd, checkpoint)
    raw = rnd.frame.get("current_validation")
    strict = parent_reference(raw, before, sha)
    if not isinstance(raw, dict) or not raw:
        return strict
    payload = raw.get("summary", raw)
    if not isinstance(payload, dict):
        return strict
    containers = [raw, payload] if payload is not raw else [raw]
    for value in containers:
        if (("artifact_id" in value and value["artifact_id"] != rnd.parent.artifact_id)
                or ("dataset_key" in value and value["dataset_key"] != rnd.dataset_key)
                or ("round_index" in value and value["round_index"] != rnd.round_index)
                or any(key in value and value[key] != rnd.parent.checkpoint_sha256
                       for key in ("checkpoint_sha256", "parent_checkpoint_sha256"))):
            return {"comparable": False, "reason": "parent_identity_conflict"}
    if strict["comparable"]:
        return strict
    missing = [key for key in ("scope", "validation_sha256") if not any(key in c for c in containers)]
    if not missing:
        return strict
    # Supply local identity only to check the remaining fields, including every
    # explicit host field. Never label this temporary normalization as host proof.
    check = {**raw, **{key: {"scope": "authorized_public_validation", "validation_sha256": sha}[key]
                            for key in missing}}
    checked = parent_reference(check, before, sha)
    if not checked["comparable"]:
        return {**strict, "reason": checked["reason"]}
    return {"comparable": True, "rmse": before["rmse"],
            "source": "local_current_parent_public_reevaluation",
            "scope": "authorized_public_validation", "validation_sha256": sha,
            "artifact_id": rnd.parent.artifact_id, "checkpoint_sha256": rnd.parent.checkpoint_sha256,
            "dataset_key": rnd.dataset_key, "round_index": rnd.round_index,
            "host_reference": strict, "host_score_matches_local": True,
            "missing_host_identity_fields": missing}


def host_gate(receipt, parent, after, sha, rules, *, binding=None):
    # 本地准入之外，再独立检查宿主回执：范围可比、真实改善、本地/宿主 RMSE 一致。
    # 此处不伪造宿主指标；MAE 保护已由 promotion_gate 检查，两道门槛最终取交集。
    candidate = measurement(receipt, sha, binding=binding)
    result = {"eligible": False, "parent": parent, "candidate": candidate}
    if not parent["comparable"] or not candidate["comparable"]:
        return {**result, "reason": "unverified_parent_or_candidate_measurement"}
    gain = parent["rmse"] - candidate["rmse"]
    required = max(rules["minimum_rmse_absolute_gain"], parent["rmse"] * rules["minimum_rmse_relative_gain"])
    if gain <= 1e-12 or gain + 1e-12 < required:
        return {**result, "reason": "host_rmse_reduction_below_threshold"}
    if abs(candidate["rmse"] - after["rmse"]) > SCORE_TOLERANCE:
        return {**result, "reason": "candidate_host_local_measurement_conflict"}
    return {**result, "eligible": True, "host_rmse_reduction": gain}
