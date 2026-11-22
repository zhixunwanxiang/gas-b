"""Conservative adapter for explicit public-validation host receipts.

Unknown schemas or scopes are not silently interpreted as candidate approval.
The organizer's actual receipt schema must be adapted here if it differs.
"""
import math

PUBLIC_SCOPES = {"authorized_public_validation", "public_validation", "offline_public_validation"}
SCORE_TOLERANCE = 1e-6  # Permit ordinary metric rounding, not different measurements.


def measurement(value, validation_sha256):
    if not isinstance(value, dict):
        return {"comparable": False, "reason": "unrecognized_receipt"}
    payload = value.get("summary", value)
    if not isinstance(payload, dict):
        return {"comparable": False, "reason": "unrecognized_summary"}
    metrics = payload.get("metrics", {})
    score = metrics.get("macro_f1") if isinstance(metrics, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1 or not math.isfinite(score):
        return {"comparable": False, "reason": "missing_or_invalid_macro_f1"}
    if not isinstance(payload.get("scope"), str) or payload["scope"] not in PUBLIC_SCOPES:
        return {"comparable": False, "reason": "unknown_validation_scope"}
    reported_sha = payload.get("validation_sha256")
    if (not isinstance(reported_sha, str)
            or reported_sha.removeprefix("sha256:") != validation_sha256.removeprefix("sha256:")):
        return {"comparable": False, "reason": "validation_digest_missing_or_different"}
    return {"comparable": True, "macro_f1": float(score), "scope": payload["scope"],
            "validation_sha256": validation_sha256, "source": "host"}


def parent_reference(host_value, local_before, validation_sha256):
    if host_value is None or host_value == {}:
        return {"comparable": True, "macro_f1": local_before["macro_f1"],
                "validation_sha256": validation_sha256, "scope": "authorized_public_validation",
                "source": "local_parent_host_measurement_absent"}
    result = measurement(host_value, validation_sha256)
    if result["comparable"] and abs(result["macro_f1"] - local_before["macro_f1"]) > SCORE_TOLERANCE:
        return {**result, "comparable": False, "reason": "parent_host_local_measurement_conflict"}
    return result


def candidate_gate(host_receipt, parent, local_after, validation_sha256, controller):
    candidate = measurement(host_receipt, validation_sha256)
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
