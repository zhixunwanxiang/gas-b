"""Bounded controller policy and comparable public-validation promotion gates."""
# 受控策略层：限制实验重试、执行上线门槛，并让策略状态与实际选中的模型版本配对。
# 模型层负责权重续训；这里仅更新实验偏好/策略版本，不允许大模型任意改代码或放宽预算。
from copy import deepcopy
import hashlib
import json
import math

EXPERIMENT_POLICY_VERSION = "classified-rejections-fresh-history-v2"
REPEAT_POLICY = "skip-completed-identical-performance-rejection-v1"
OUTCOMES = {"adopted", "performance_rejected", "measurement_unverified",
            "controller_guard", "not_selected", "local_training_failed", "unclassified"}


def experiment_outcome(*, adopted, failed, local_gate=None, host_gate=None, selection_source="model"):
    """Separate measured performance failures from adoption/controller failures."""
    if adopted:
        outcome, reason = "adopted", "candidate_selected"
    elif failed:
        outcome, reason = "local_training_failed", "local_training_failed"
    elif isinstance(local_gate, dict) and local_gate.get("eligible") is False:
        outcome = "performance_rejected"
        reason = "class_recall_regression" if local_gate.get("class_recall_losses") else "local_promotion_gate_rejected"
    elif isinstance(host_gate, dict) and host_gate.get("eligible") is False:
        reason = host_gate.get("reason") or "host_measurement_unverified"
        outcome = ("performance_rejected" if reason in (
            "host_gain_below_promotion_threshold", "host_rmse_reduction_below_threshold") else "measurement_unverified")
    elif selection_source == "program_guard":
        outcome, reason = "controller_guard", "selection_validation_exhausted"
    elif (isinstance(local_gate, dict) and local_gate.get("eligible") is True
          and isinstance(host_gate, dict) and host_gate.get("eligible") is True):
        outcome, reason = "not_selected", "eligible_candidate_not_selected"
    else:
        outcome, reason = "unclassified", "insufficient_rejection_evidence"
    return {"outcome": outcome, "reason": reason,
            "counts_toward_rejection_limit": outcome in ("performance_rejected", "local_training_failed")}


def counted_rejection(attempt):
    # Legacy measured/not-adopted entries contain no failure stage. Do not infer
    # a performance failure from them. Known legacy local failures stay bounded.
    return (not attempt["adopted"] and (attempt.get("outcome") in (
        "performance_rejected", "local_training_failed") or attempt.get("status") == "local_training_failed"))


def validate_attempts(attempts):
    for item in attempts:
        if (not isinstance(item, dict) or not isinstance(item.get("fingerprint"), str)
                or type(item.get("adopted")) is not bool
                or ("outcome" in item and (not isinstance(item["outcome"], str) or item["outcome"] not in OUTCOMES))
                or ("counts_toward_rejection_limit" in item and (
                    type(item["counts_toward_rejection_limit"]) is not bool
                    or item["counts_toward_rejection_limit"] != counted_rejection(item)))):
            raise ValueError("Invalid classified experiment history")


def eligible_history(restored, validation_sha256, parent_verified, authorized_ids):
    """Only actual current-round remeasurements can enter final selection."""
    if not parent_verified:
        return []
    return [identifier for identifier, value in restored.items()
            if identifier in authorized_ids and value.get("reevaluated_this_round") is True
            and value.get("validation_sha256") == validation_sha256
            and isinstance(value.get("gate"), dict) and value["gate"].get("eligible") is True]


def validate_controller(config):
    # 先校验控制器预算和阈值，防止布尔值冒充整数、NaN 或无界配置影响实验权限。
    for key, low, high in (("json_corrections", 0, 2), ("max_history_restores", 0, 2),
                           ("max_repeat_rejections", 1, 8), ("minimum_class_support", 1, 1000000)):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("Controller limit outside supported bounds: " + key)
    for key in ("minimum_macro_f1_gain", "maximum_class_recall_drop"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid promotion limit: " + key)
    thresholds = config["drift_thresholds"]
    if type(thresholds.get("minimum_rows")) is not int or thresholds["minimum_rows"] < 1:
        raise ValueError("Invalid monitoring sample floor")
    for key in ("psi", "binned_ks", "absolute_mean_z"):
        value = thresholds.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid drift threshold: " + key)


def fingerprint(parent_sha256, train_sha256, validation_sha256, spec, execution_policy=None):
    # 父模型、训练集、验证集、实验配置和执行策略共同确定实验身份。
    # 同名实验在不同父状态或修复后的执行策略上不必被误判为完全重复。
    identity = [parent_sha256, train_sha256, validation_sha256, spec]
    if execution_policy is not None:
        identity.append(execution_policy)
    return hashlib.sha256(json.dumps(identity,
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def initial_policy():
    # version 表示当前模型配套策略；revision 是单调递增计数，回滚不倒退该总计数。
    return {"format": "gas-a.strategy.v1", "version": 1, "revision": 1, "preferred_experiment": None,
            "attempts": [], "artifact_policies": {}, "last_measurement": None}


def load_policy(value, menu):
    # 校验跨轮策略与各产物配套快照，拒绝无法解释的版本号、偏好或结构。
    if value is None:
        return initial_policy()
    if (not isinstance(value, dict) or value.get("format") != "gas-a.strategy.v1"
            or type(value.get("version")) is not int or value["version"] < 1
            or value.get("preferred_experiment") not in (None, *menu)
            or not isinstance(value.get("attempts"), list)
            or not isinstance(value.get("artifact_policies"), dict)):
        raise ValueError("Invalid bounded strategy state")
    for snapshot in value["artifact_policies"].values():
        if (not isinstance(snapshot, dict) or set(snapshot) != {"version", "preferred_experiment"}
                or type(snapshot["version"]) is not int or snapshot["version"] < 1
                or snapshot["preferred_experiment"] not in (None, *menu)):
            raise ValueError("Invalid artifact-paired strategy")
    versions = [value["version"], *(item["version"] for item in value["artifact_policies"].values())]
    if type(value.get("revision")) is not int or value["revision"] < max(versions):
        raise ValueError("Invalid strategy revision counter")
    validate_attempts(value["attempts"])
    return value


def available_experiments(policy, menu, rnd, train, validation, config, execution_policy=None):
    # 只屏蔽完全相同实验身份达到拒绝上限的方案，保留其他允许菜单项给 AI 规划。
    allowed, blocked = {}, []
    for name, spec in menu.items():
        identity = fingerprint(rnd.parent.checkpoint_sha256, train["sha256"], validation["sha256"], spec, execution_policy)
        rejected = sum(a["fingerprint"] == identity and counted_rejection(a) for a in policy["attempts"])
        matching = [a for a in policy["attempts"] if a["fingerprint"] == identity]
        if rejected >= config["max_repeat_rejections"] or _completed_rejection(matching, execution_policy):
            blocked.append(name)
        else:
            allowed[name] = spec
    return allowed, blocked


def _completed_rejection(attempts, execution):
    # 固定 seed、父优化器、数据和参数的已完成数值实验可重放；同条件性能失败不重复付出计算。
    # 超时/本地异常、核验拒绝和未采用不触发此规则；换数据或父模型会生成新身份。
    return (isinstance(execution, dict) and execution.get("repeat_policy") == REPEAT_POLICY
            and any(not a["adopted"] and a.get("outcome") == "performance_rejected" for a in attempts))


def experiment_evidence(history, attempts, menu, parent_sha, train_sha, validation_sha, execution, rules,
                        *, gain_key="macro_f1_gain", blocked=()):
    """Expose exact-condition attempts separately from old or incomplete history."""
    def spec_digest(spec):
        return "sha256:" + hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False,
                                                     separators=(",", ":")).encode()).hexdigest()
    evidence = {}
    for name, spec in menu.items():
        identity = fingerprint(parent_sha, train_sha, validation_sha, spec, execution)
        matching = [a for a in attempts if a.get("fingerprint") == identity]
        gains = [a[gain_key] for a in matching if type(a.get(gain_key)) in (int, float) and math.isfinite(a[gain_key])]
        recent = []
        for item in history[-10:]:
            cfg = item.get("configuration", {})
            old_spec = cfg.get("experiment") if isinstance(cfg, dict) else None
            if not isinstance(old_spec, dict) or old_spec.get("name") != name:
                continue
            comparisons = {key: cfg[field] == current if cfg.get(field) is not None else None
                for key, field, current in (
                    ("parent", "parent_checkpoint_sha256", parent_sha), ("train", "train_sha256", train_sha),
                    ("validation", "validation_sha256", validation_sha), ("execution", "execution_policy", execution))}
            comparisons["experiment"] = cfg.get("experiment_config_sha256", spec_digest(old_spec)) == spec_digest(spec)
            status = ("different" if False in comparisons.values() else
                      "same" if all(value is True for value in comparisons.values()) else "unknown")
            gate = cfg.get("candidate_gate")
            outcome = cfg.get("experiment_outcome")
            gate = gate if isinstance(gate, dict) else {}
            outcome = outcome if isinstance(outcome, dict) else {}
            recent.append({"history_id": item["id"], "conditions": status,
                           "matches": comparisons, "recorded_gain": gate.get(gain_key),
                           "reason": outcome.get("reason") or ("local_training_failed" if cfg.get("experiment_failure")
                               else "class_recall_regression" if gate.get("class_recall_losses")
                               else "local_promotion_gate_rejected" if gate.get("eligible") is False
                               else "see_host_admission"),
                           "counts_as_current_condition_evidence": status == "same"})
        rejections = sum(counted_rejection(a) for a in matching)
        evidence[name] = {"available": name not in blocked, "current_condition_attempts": len(matching),
                          "current_condition_counted_rejections": rejections,
                          "remaining_rejection_budget": (0 if _completed_rejection(matching, execution)
                              else max(0, rules["max_repeat_rejections"] - rejections)),
                          "completed_identical_rejection": bool(_completed_rejection(matching, execution)),
                          "best_current_condition_gain": max(gains) if gains else None,
                          "gain_metric": gain_key,
                          "current_condition_outcomes": {outcome: sum(a.get("outcome") == outcome for a in matching)
                              for outcome in sorted(OUTCOMES) if any(a.get("outcome") == outcome for a in matching)},
                          "recent_history": recent[-3:],
                          "interpretation": "current_conditions_attempted" if matching or any(
                              row["conditions"] == "same" for row in recent) else "no_verified_attempt_for_current_conditions"}
    return evidence


def promotion_gate(before, after, config):
    # F1 必须严格提高且达到配置最小增益，同时满足受保护类别的召回下降限制。
    # 召回值使用 0～1 比例，0.20 表示 20 个百分点；比较基准是当轮父模型，不是累计初始基线。
    gain = after["macro_f1"] - before["macro_f1"]
    # 只有父验证支持数达到门槛的类别受单步召回保护；1e-12 只吸收浮点边界误差，不放宽百分点配置。
    losses = {name: item["recall"] - after["per_class"][name]["recall"]
              for name, item in before["per_class"].items()
              if item["support"] >= config["minimum_class_support"]
              and item["recall"] - after["per_class"][name]["recall"] > config["maximum_class_recall_drop"] + 1e-12}
    return {"eligible": bool(gain > 1e-12 and gain >= config["minimum_macro_f1_gain"] and not losses),
            "macro_f1_gain": gain, "class_recall_losses": losses}


def advance_policy(policy, *, selected, parent_id, candidate_id, plan, attempt_key, gain, validation_sha256, selected_f1,
                   experiment_failed=False, local_gate=None, host_gate=None, selection_source="model"):
    # 记录实际候选或本地失败尝试；只有被选中的新候选才能提升策略版本及实验偏好。
    # 回滚时恢复所选历史产物的配套策略，未知外部历史采用保守初值；日志长度受限以控制状态大小。
    result = deepcopy(policy)
    parent_snapshot = {k: policy[k] for k in ("version", "preferred_experiment")}
    result["artifact_policies"].setdefault(parent_id, parent_snapshot)
    # 即使本地训练失败、没有上传候选，也保留拒绝尝试，防止同一失败配置无限循环。
    if candidate_id or experiment_failed:
        adopted = candidate_id is not None and selected == candidate_id
        result["attempts"].append({"fingerprint": attempt_key, "experiment": plan["experiment"],
                                   "adopted": adopted, "macro_f1_gain": gain,
                                   "status": "local_training_failed" if experiment_failed else "measured",
                                   **experiment_outcome(adopted=adopted, failed=experiment_failed,
                                       local_gate=local_gate, host_gate=host_gate, selection_source=selection_source)})
        result["attempts"] = result["attempts"][-128:]
        # “执行过训练”不等于策略改善；只有实际采用新候选才提升版本并记录偏好。
        if adopted:
            result["revision"] += 1
            result["version"] = result["revision"]
            result["preferred_experiment"] = plan["experiment"]
    if selected not in (parent_id, candidate_id):
        # The model artifact is authoritative. Restore its paired strategy when
        # known; an external historical artifact gets the conservative default.
        saved = result["artifact_policies"].get(selected, {"version": 1, "preferred_experiment": None})
        result.update(saved)
    result["artifact_policies"][selected] = {k: result[k] for k in ("version", "preferred_experiment")}
    result["artifact_policies"] = dict(list(result["artifact_policies"].items())[-64:])
    result["last_measurement"] = {"validation_sha256": validation_sha256, "selected_macro_f1": selected_f1}
    return result
