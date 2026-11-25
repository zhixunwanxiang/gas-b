"""原单位回归指标和越低越好的上线门槛；不复用 A 的 F1/召回率规则。"""
import math
import numpy as np
from ..host_validation import SCORE_TOLERANCE, parse_measurement, metric_matches
from ..sensor_training.io import digest
from ..parent_checkpoint import verify_selected_checkpoint

PARENT_REFERENCE_VERSION = "current-parent-public-reevaluation-v2"


def evaluate(model, data, dataset):
    # Model.predict 已把标准化输出还原到原单位；这里直接计算分钟/循环上的误差。
    # 正 bias 表示寿命被高估，overestimate_fraction 表示高估样本占比，供诊断使用。
    prediction = model.predict(data["x"])
    if prediction.shape != data["y"].shape or not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise ValueError("Regression predictions must be finite nonnegative values")
    error = prediction - data["y"]
    result = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
              "bias": float(error.mean()), "overestimate_fraction": float(np.mean(error > 0)),
              "rows": len(error)}
    if not all(math.isfinite(result[k]) for k in ("rmse", "mae", "bias")):
        raise ValueError("Nonfinite regression measurement")
    if dataset == "C":
        # NASA：高估寿命除以 10，低估除以 13；逐样本求和，不冒充可跨样本数量比较的均值。
        exponent = np.where(error < 0, -error / 13.0, error / 10.0)
        with np.errstate(over="ignore"):
            score = float(np.expm1(exponent).sum())
        # 极端高估可能导致指数溢出：如实记录 None 和溢出标志，不截断后另报较小分数。
        result.update(nasa_score=score if math.isfinite(score) else None,
                      nasa_score_overflow=not math.isfinite(score))
    return result


def promotion_gate(before, after, rules):
    # 改善量为 父 RMSE - 新 RMSE；分钟和循环分别比较，不把它们转成分类百分点。
    gain = before["rmse"] - after["rmse"]
    required = max(rules["minimum_rmse_absolute_gain"], before["rmse"] * rules["minimum_rmse_relative_gain"])
    # 绝对收益和相对收益取较严者：默认相对 0.0005 即误差至少降低 0.05%。
    # 还要满足 MAE 最多相对增加 2% 的默认保护；NASA 只记录，不在此处作为硬门槛。
    mae_limit = before["mae"] * (1 + rules["maximum_mae_relative_increase"])
    return {"eligible": bool(gain > 1e-12 and gain >= required and after["mae"] <= mae_limit + 1e-12),
            "rmse_reduction": gain, "required_rmse_reduction": required, "mae_limit": mae_limit}


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
