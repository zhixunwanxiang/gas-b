"""宿主公开验证回执适配：显式身份或本轮 SDK 候选绑定，保留原始宿主测量。"""
# 宿主公开验证回执适配：先确认指标、数据摘要与公开范围，再决定能否同口径比较。
# 本地高分不能覆盖宿主冲突分数；缺省身份只接受本轮已绑定的 SDK 候选事务，未知来源仍拒绝。
import math
from dataclasses import dataclass
from .sensor_training.io import digest

PUBLIC_SCOPES = {"authorized_public_validation", "public_validation", "offline_public_validation"}
SCORE_TOLERANCE = 1e-6  # Permit ordinary metric rounding, not different measurements.
HOST_ADMISSION_VERSION = "sdk-bound-public-validation-v2"


@dataclass(frozen=True)
class CandidateValidationBinding:
    # 此对象只在实际 submit_candidate 返回后建立，不从模型回复或历史配置恢复。
    # SHA 来自本地授权投影，不能冒充宿主在回执中声明的摘要。
    artifact_id: str
    validation_sha256: str
    dataset_key: str
    round_index: int
    metric: str


def bind_candidate_validation(rnd, receipt, validation_sha256):
    """将 SDK 本轮候选回执与当前公开验证文件绑定，不修改原始回执。"""
    expected = {"A": ("classification", "macro_f1"), "B": ("regression", "rmse"),
                "C": ("regression", "rmse")}.get(rnd.dataset_key)
    if (rnd.frame.get("protocol") != "contest.harness.workspace.v2" or expected is None
            or rnd.task.get("task") != expected[0] or rnd.task.get("metric", expected[1]) != expected[1]
            or not isinstance(receipt, dict) or receipt.get("type") != "candidate_response"
            or not isinstance(receipt.get("artifact_id"), str) or not receipt["artifact_id"]):
        raise ValueError("Cannot bind an unrecognized current SDK candidate response")
    # SDK 文档规定此事务返回公开验证摘要。只有当前文件仍与训练前哈希相同才补充缺省身份。
    # 不能仅凭宿主分数与本地相等，就猜测任意历史/隐藏测量也属于这份数据。
    if digest(rnd.data.validation_path) != validation_sha256:
        raise ValueError("Public validation projection changed during candidate submission")
    return CandidateValidationBinding(receipt["artifact_id"], validation_sha256,
                                      rnd.dataset_key, rnd.round_index, expected[1])


def parse_measurement(value, validation_sha256, metric, *, binding=None):
    # 兼容具名指标的 metrics 对象、扁平摘要和 metric+value/score；不猜测无指标名的 score。
    payload = value.get("summary", value) if isinstance(value, dict) else None
    if not isinstance(payload, dict):
        return {"comparable": False, "reason": "unrecognized_summary"}
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return {"comparable": False, "reason": "unrecognized_metrics"}
    scores = [container[metric] for container in (metrics, payload) if metric in container]
    if "metric" in payload:
        if payload["metric"] != metric:
            return {"comparable": False, "reason": "host_metric_mismatch"}
        scores.extend(payload[key] for key in ("value", "score") if key in payload)
    if not scores:
        return {"comparable": False, "reason": "missing_or_invalid_" + metric}
    for score in scores:
        try:
            valid = (not isinstance(score, bool) and isinstance(score, (int, float))
                     and math.isfinite(score) and score >= 0 and (metric != "macro_f1" or score <= 1))
        except OverflowError:
            valid = False
        if not valid:
            return {"comparable": False, "reason": "missing_or_invalid_" + metric}
    if any(score != scores[0] for score in scores[1:]):
        return {"comparable": False, "reason": "conflicting_host_metrics"}
    bound = isinstance(binding, CandidateValidationBinding)
    if binding is not None and (not bound or binding.metric != metric
            or binding.validation_sha256 != validation_sha256
            or value.get("type") != "candidate_response" or value.get("artifact_id") != binding.artifact_id):
        return {"comparable": False, "reason": "candidate_binding_mismatch"}
    containers = [value, payload] if payload is not value else [payload]
    scopes = [c["scope"] for c in containers if "scope" in c]
    digests = [c["validation_sha256"] for c in containers if "validation_sha256" in c]
    # 显式范围/摘要始终优先校验。None、隐藏范围或冲突摘要不能因有本轮绑定而被忽略。
    if (not scopes and not bound) or any(not isinstance(s, str) or s not in PUBLIC_SCOPES for s in scopes):
        return {"comparable": False, "reason": "unknown_validation_scope"}
    if (not digests and not bound) or any(not isinstance(s, str) or
            s.removeprefix("sha256:") != validation_sha256.removeprefix("sha256:") for s in digests):
        return {"comparable": False, "reason": "validation_digest_missing_or_different"}
    if bound:
        for c in containers:
            if (("artifact_id" in c and c["artifact_id"] != binding.artifact_id)
                    or ("dataset_key" in c and c["dataset_key"] != binding.dataset_key)):
                return {"comparable": False, "reason": "candidate_binding_mismatch"}
    result = {"comparable": True, metric: float(scores[0]), "source": "host",
              "scope": scopes[0] if scopes else "authorized_public_validation",
              "validation_sha256": validation_sha256,
              "scope_source": "host" if scopes else "sdk_current_candidate_public_validation",
              "validation_sha256_source": "host" if digests else "locally_hashed_sdk_submission_projection"}
    if bound:
        result["submission_binding"] = {"artifact_id": binding.artifact_id, "dataset_key": binding.dataset_key,
                                        "round_index": binding.round_index}
    return result


def measurement(value, validation_sha256, *, binding=None):
    # 有效 Macro-F1 必须来自宿主；身份由显式回执字段或经过校验的当前提交绑定提供。
    return parse_measurement(value, validation_sha256, "macro_f1", binding=binding)


def parent_reference(host_value, local_before, validation_sha256):
    # 宿主完全未给父测量时可用本地同范围父分数；宿主已给但冲突时不能覆盖成可比。
    # 仅“没有提供测量”可回退本地父分数；已提供的未知或冲突回执必须保留拒绝原因。
    if host_value is None or host_value == {}:
        return {"comparable": True, "macro_f1": local_before["macro_f1"],
                "validation_sha256": validation_sha256, "scope": "authorized_public_validation",
                "source": "local_parent_host_measurement_absent"}
    result = measurement(host_value, validation_sha256)
    if result["comparable"] and abs(result["macro_f1"] - local_before["macro_f1"]) > SCORE_TOLERANCE:
        return {**result, "comparable": False, "reason": "parent_host_local_measurement_conflict"}
    return result


def candidate_gate(host_receipt, parent, local_after, validation_sha256, controller, *, binding=None):
    # 依次检查父/候选范围、宿主增益和本地测量一致性。
    # 此门槛与 evolution.promotion_gate 同时生效；逐类召回保护仍由后者负责。
    candidate = measurement(host_receipt, validation_sha256, binding=binding)
    result = {"eligible": False, "parent": parent, "candidate": candidate}
    if not parent["comparable"]:
        return {**result, "reason": "parent_measurement_unverified"}
    if not candidate["comparable"]:
        return {**result, "reason": candidate["reason"]}
    gain = candidate["macro_f1"] - parent["macro_f1"]
    result["host_macro_f1_gain"] = gain
    if gain <= 1e-12 or gain + 1e-12 < controller["minimum_macro_f1_gain"]:
        return {**result, "reason": "host_gain_below_promotion_threshold"}
    if abs(candidate["macro_f1"] - local_after["macro_f1"]) > SCORE_TOLERANCE:
        return {**result, "reason": "candidate_host_local_measurement_conflict"}
    return {**result, "eligible": True, "reason": "same_scope_host_measurement_verified"}
