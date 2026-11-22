"""提供显式启用的时间窗口选择和实测前向反馈，不使用目标窗口数据训练。"""
# 本地/历史回放的受限自适应策略工具，包含已授权窗口反馈与影子预测评估。
# 窗口标签必须来自被允许的公开反馈；这些函数的存在不代表 SDK 会提供隐藏标签。
# 正式入口的策略状态和晋升门槛在 gas_a.evolution，二者的配置字段不能混用。
import math
from .data import evaluate, load_model, score_predictions
from .io import CLASSES, digest, history, read_json, stamp
from .controller_policy import business_evidence, bind_plan

PARAMETER_BOUNDS = {"learning_rate": (1e-5, .003), "feature_lr_scale": (.05, 1), "anchor_l2": (0, .1),
                    "l2": (0, .01), "balance": (0, .75), "weak_boost": (1, 3)}


def focus_bounds(specs=(), architecture_choices=None, parent_shape=None):
    """SDK、联网和回放共同开放的训练范围与类别侧重点，沿用一次规划调用。"""
    result = {"freeze": {"none": "本轮不按层冻结", "features": "冻结全部隐藏层，只训练输出头",
                       "head": "冻结输出头，只训练隐藏层", "first_hidden": "冻结第一隐藏层"},
            "freeze_choices_by_experiment": {s["name"]: (["none", "head"] if "kept_sensors" in s else
                ["none", "features", "head", "first_hidden"]) for s in specs},
            "class_boosts": {"allowed_classes": CLASSES, "range": [1, 3], "max_classes": 2},
            "scope": "冻结只影响本轮；保留冻结权重和Adam矩，下轮可解冻；类别权重只用于授权训练。"}
    if architecture_choices:
        result.update(architecture_choice=architecture_choices, parent_architecture=parent_shape,
                      architecture_rule="architecture_choice只选所列名称；先保留父权重和Adam迁移，预测一致后续训，不重置基线。")
    return result


def parameter_bounds(specs=(), architecture_choices=None, parent_shape=None):
    """adaptive在共同训练侧重点之上开放受限数值参数。"""
    return {**focus_bounds(specs, architecture_choices, parent_shape), "bounds": {k: list(v) for k, v in PARAMETER_BOUNDS.items()},
            "learning_rate_menu_multiplier": [.5, 2]}


def policy(config):
    """组合基础 gate 与显式启用的 adaptive 策略，单独传递类别硬下限。"""
    return {**config["gate"], "adaptive": True,
            "hard_minimum_class_recall": config.get("adaptive", {}).get("hard_minimum_class_recall", {})}


def _valid(metric):
    return all(not isinstance(metric.get(k), bool) and isinstance(metric.get(k), (int, float))
               and math.isfinite(metric[k]) and 0 <= metric[k] <= 1 for k in ("macro_f1", "accuracy"))


def _class_checks(metric, reference, config):
    """分别返回类别退化警告与硬下限失败，保留软目标和硬约束的不同含义。"""
    warnings, failures = [], []
    for name, old in reference.get("per_class", {}).items():
        new = metric.get("per_class", {}).get(name, {})
        if old["support"] >= config["minimum_class_support"] and new.get("recall", 0) < old["recall"] - config["max_class_recall_drop"]:
            warnings.append("class_recall_regression:" + name)
        if old["support"] and new.get("recall", 0) < config.get("minimum_class_recall", {}).get(name, 0):
            warnings.append("class_target_unmet:" + name)
        if (config.get("reject_zero_target_recall") and config.get("minimum_class_recall", {}).get(name, 0) > 0
                and new.get("support", 0) >= config["minimum_class_support"] and new.get("recall", 0) <= 0):
            failures.append("unresolved_zero_target_recall:" + name)
        if (config.get("reject_zero_target_recall") and config.get("minimum_class_recall", {}).get(name, 0) > 0
                and min(old["support"], new.get("support", 0)) >= config["minimum_class_support"]
                and new.get("recall", 0) < old["recall"] - config.get("target_recall_max_drop", .02) - 1e-12):
            failures.append("target_recall_regression:" + name)
        if old["support"] and new.get("recall", 0) < config.get("hard_minimum_class_recall", {}).get(name, 0):
            failures.append("hard_class_minimum:" + name)
    return warnings, failures


def fixed_gate(metric, active, baseline, config):
    """以原始 validation 的 baseline 作为质量下限，不要求该固定验证分数逐轮提高。"""
    if not _valid(metric):
        return {"accepted": False, "reasons": ["invalid_metrics"], "warnings": []}
    reasons = []
    if metric["macro_f1"] + 1e-12 < baseline["macro_f1"]:
        reasons.append("below_static_baseline")
    if metric["accuracy"] < baseline["accuracy"] - config["max_accuracy_drop"]:
        reasons.append("accuracy_below_reference_floor")
    if (config.get("max_parent_fixed_f1_drop") is not None
            and metric["macro_f1"] < active["macro_f1"] - config["max_parent_fixed_f1_drop"] - 1e-12):
        reasons.append("fixed_f1_regression_from_parent")
    warnings, failures = _class_checks(metric, baseline, config)
    if config.get("reject_zero_target_recall"):
        other_warnings, other_failures = _class_checks(metric, active, config)
        warnings = list(dict.fromkeys([*warnings, *other_warnings]))
        failures = list(dict.fromkeys([*failures, *other_failures]))
    return {"accepted": not reasons and not failures, "reasons": reasons + failures or ["quality_gates_passed"],
            "warnings": warnings, "reference": "static_baseline"}


def window_gate(metric, active, baseline, config):
    """依据当前已揭示窗口的相对增益和 Accuracy 判断候选，并保留类别风险提示。"""
    if not _valid(metric):
        return {"accepted": False, "reasons": ["invalid_metrics"], "warnings": []}
    reasons = []
    if metric["macro_f1"] < active["macro_f1"] + config["min_macro_f1_gain"] - 1e-12:
        reasons.append("macro_f1_gain_below_minimum")
    if metric["accuracy"] < active["accuracy"] - config["max_accuracy_drop"]:
        reasons.append("accuracy_regression")
    warnings, failures = _class_checks(metric, active, config)
    return {"accepted": not reasons and not failures, "reasons": reasons + failures or ["quality_gates_passed"],
            "warnings": warnings, "unobserved_recall_targets": [name for name in config.get("minimum_class_recall", {})
                                                                if not baseline["per_class"][name]["support"]]}


def ranking(model, current, recent, weights):
    """仅使用当前及已揭示的历史窗口计算加权 Macro-F1，越新的窗口按配置赋予相应权重。"""
    windows = [current, *reversed(recent)][ :len(weights)]
    scores = [evaluate(model, dataset)["macro_f1"] for dataset in windows]
    norm = sum(weights[:len(scores)])
    return {"score": sum(s * w for s, w in zip(scores, weights)) / norm,
            "macro_f1_newest_first": scores, "weights": weights[:len(scores)],
            "scope": "only_current_and_previously_revealed_windows"}


def select_history(registry, current, recent, validation, config, past=(), current_feedback=()):
    """历史恢复仅允许选择曾经实际激活过的模型产物。"""
    active = registry.active_id()
    # 以 deployments 而非全部 registry 条目确定历史池，排除从未上线的实验产物。
    published = {"baseline", active, *[e["to"] for e in history(registry.root / "deployments.jsonl")]}
    baseline = evaluate(load_model(registry.checkpoint("baseline")), validation)
    weights = config["adaptive"]["recent_weights"]
    assessed, rejected = {}, {}
    reliability = feedback_stats(past, current_feedback, group_by="version") if config["adaptive"].get("forward_feedback") else {}
    for identity in sorted(published):
        try:
            observed = reliability.get(identity, {})
            if (identity not in (active, "baseline") and observed.get("windows", 0) >= 2
                    and observed.get("recent_mean_gain", 0) < 0):
                rejected[identity] = {"reason": "negative_forward_gain_vs_its_parent", "feedback": observed}
                continue
            model = load_model(registry.checkpoint(identity))
            fixed = evaluate(model, validation)
            decision = fixed_gate(fixed, fixed, baseline, policy(config))
            if not decision["accepted"] and identity != "baseline":
                rejected[identity] = decision
                continue
            metric = evaluate(model, current)
            warnings, failures = _class_checks(metric, metric, policy(config))
            assessed[identity] = {"ranking": ranking(model, current, recent, weights), "metric": metric,
                                  "warnings": warnings, "hard_failures": failures}
        except (ValueError, OSError, KeyError) as exc:
            rejected[identity] = {"error": str(exc)}
    if not assessed:
        raise RuntimeError("No intact historical model is available")
    safe = {name: value for name, value in assessed.items() if not value["hard_failures"]}
    # 全部版本均失守时明确报告降级，不能将基线假称为满足类别安全下限。
    pool = safe or assessed
    selected = max(pool, key=lambda name: (pool[name]["ranking"]["score"], name == active))
    margin = config["adaptive"]["switch_margin"]
    # 提升不足 switch_margin 时保留活动版本，避免近似同分造成频繁切换。
    if active in pool and assessed[selected]["ranking"]["score"] < assessed[active]["ranking"]["score"] + margin:
        selected = active
    review = {"selected": selected, "assessed": assessed, "rejected": rejected,
              "published_only": sorted(published), "selection_basis": "recent_window_macro_f1_with_static_validation_floor",
              "hard_targets_satisfied": bool(safe), "degraded": not bool(safe)}
    event = None
    if selected != active:
        event = {"from": active, "to": selected, "kind": "published_history_utility",
                 "before": assessed.get(active, {}).get("metric"), "after": assessed[selected]["metric"],
                 "time": stamp(), "review": review, "effective": "future_predictions_only"}
    return event, review


def capture_probes(registry, features, past):
    """在读取标签之前，保存待复评版本及其父模型对当前窗口的预测。"""
    identities = {registry.active_id(), *[e["to"] for e in history(registry.root / "deployments.jsonl")]}
    if past:
        identities.update(t["artifact_id"] for t in past[-1].get("experiments", []) if t.get("artifact_id"))
    entries = registry.entries()
    targets = []
    for identity in sorted(identities):
        parent = entries[identity].get("parent")
        if parent is None:
            continue
        identities_parent = registry.checkpoint(parent)
        path = registry.checkpoint(identity)
        report_path = path.parent / "experiment.json"
        report = read_json(report_path) if report_path.exists() else {}
        targets.append({"version": identity, "parent": parent, "family": report.get("name", "unknown"),
                        "checkpoint_sha256": digest(path), "parent_sha256": digest(identities_parent),
                        "prediction": load_model(path).predict(features["x"]),
                        "parent_prediction": load_model(identities_parent).predict(features["x"]),
                        "operational": identity == registry.active_id()})
    return targets


def score_probes(probes, expected, batch):
    """为标签揭示前冻结的候选及父模型预测计算配对增益，区分实际部署与影子探针。"""
    result = []
    for probe in probes:
        metric = score_predictions(probe["prediction"], expected)
        parent = score_predictions(probe["parent_prediction"], expected)
        result.append({"batch": batch, "family": probe["family"], "version": probe["version"], "parent": probe["parent"],
                       "parent_sha256": probe["parent_sha256"],
                       "macro_f1_gain": metric["macro_f1"] - parent["macro_f1"],
                       "recall_delta": [round(metric["per_class"][name]["recall"] - parent["per_class"][name]["recall"], 6) for name in CLASSES],
                       "scope": "operational_vs_parent" if probe["operational"] else "frozen_next_window_shadow_probe"})
    return result


def feedback_stats(past, current_feedback=(), *, group_by="family"):
    """按实验方向汇总后续窗口的实测反馈，供下一次规划判断方向是否可迁移。"""
    rows = [entry for row in past for entry in row.get("forward_feedback", [])] + list(current_feedback)
    groups = {}
    for row in rows:
        groups.setdefault(row[group_by], []).append(row)
    result = {}
    for name, entries in groups.items():
        windows = {}
        for entry in entries:
            windows.setdefault(entry["batch"], []).append(entry["macro_f1_gain"])
        means = [sum(values) / len(values) for _, values in sorted(windows.items())]
        result[name] = {"observations": len(entries), "windows": len(means),
                        "recent_mean_gain": sum(means[-3:]) / len(means[-3:]),
                        "positive_windows": sum(value > 0 for value in means), "last_batch": entries[-1]["batch"]}
        recall_windows = {}
        for entry in entries:
            if len(entry.get("recall_delta", [])) == len(CLASSES):
                recall_windows.setdefault(entry["batch"], []).append(entry["recall_delta"])
        recent_recalls = list(sorted(recall_windows.items()))[-3:]
        if recent_recalls:
            result[name]["recent_recall_delta"] = [round(sum(sum(v[i] for v in vectors) / len(vectors)
                for _, vectors in recent_recalls) / len(recent_recalls), 6) for i in range(len(CLASSES))]
    return result


def compact_evidence(evidence, menu, feedback):
    """兼容旧调用入口，业务投影统一在controller_policy生成。"""
    return business_evidence(evidence, menu, feedback)


def normalize_plan(plan):
    """兼容旧入口，SDK和回放现在使用相同的程序字段绑定。"""
    return bind_plan(plan)


def proposed_specs(specs, plan, limits, past, parent_sha, *, focus_only=False, architecture_choices=None):
    """对一个实验方向修改最多两个受限参数或命名架构，禁止重置状态及修改数据。"""
    from .optimize import bounded_experiment, FREEZE_MODES
    changes = plan.get("parameter_changes", {})
    if not isinstance(changes, dict) or len(changes) > 1:
        raise ValueError("At most one selected family may have parameter_changes")
    bounds = PARAMETER_BOUNDS
    architecture_choices = architecture_choices or {}
    # 架构只接受本轮已核验的命名选项，scaler、Adam状态和数据路径不接受模型修改。
    result = []
    for family, updates in changes.items():
        if family not in plan["experiment_names"] or not isinstance(updates, dict) or not 1 <= len(updates) <= 2:
            raise ValueError("Changes must target one selected family and contain one or two parameters")
        for name, value in updates.items():
            if focus_only and name not in ("freeze", "class_boosts", "architecture_choice"):
                raise ValueError("Only freezing, named class focus and checked architecture choices are enabled on this execution path")
            if name == "architecture_choice" and isinstance(value, str) and value in architecture_choices:
                continue
            if name == "freeze" and isinstance(value, str) and value in FREEZE_MODES:
                continue
            if name == "class_boosts" and isinstance(value, dict) and 1 <= len(value) <= 2 and all(
                    key in CLASSES and not isinstance(boost, bool) and isinstance(boost, (int, float))
                    and math.isfinite(boost) and 1 <= boost <= 3 for key, boost in value.items()):
                continue
            if (name not in bounds or isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not bounds[name][0] <= value <= bounds[name][1]):
                raise ValueError("Training change outside permitted bounds: " + name)
    for spec in specs:
        updates = dict(changes.get(spec["name"], {}))
        architecture_choice = updates.pop("architecture_choice", None)
        updated = {**spec, **updates}
        if architecture_choice is not None:
            updated["architecture"] = dict(architecture_choices[architecture_choice])
        # learning rate 同时受绝对边界和相对菜单值的倍率约束。
        if "learning_rate" in changes.get(spec["name"], {}) and not .5 <= updated["learning_rate"] / spec["learning_rate"] <= 2:
            raise ValueError("Learning-rate change must be between 0.5x and 2x the menu rate")
        bounded_experiment(updated, limits)
        if spec["name"] in changes:
            if any(t.get("report", {}).get("parent_checkpoint_sha256", row.get("parent_checkpoint_sha256")) == parent_sha
                   and t.get("report", {}).get("configuration", t.get("configuration")) == updated
                   for row in past for t in row.get("experiments", [])):
                raise ValueError("Exact parent/parameter experiment was already recorded")
        result.append(updated)
    return result


class RulePlanner:
    """用于 CPU 筛选的规则控制器，不调用远程模型，也不生成虚构的服务方回执。"""
    model = "local-rule-screen-no-llm"
    local_rules = True

    def decide(self, prompt):
        """以本地规则完成筛选或版本排名，输出明确标记为非 LLM 的决策。"""
        if "search_space" in prompt:
            enabled = prompt["budget"]["max_experiments"] > 0
            return {"action": "experiment" if enabled else "observe", "hypothesis": "规则配置参照，收益待实测。",
                    "reason": "明确标记为CPU规则筛选，不代表大模型建议。",
                    "experiments": [{"name": "rule_reference", "learning_rate": .0005, "epochs": 3,
                                     "batch_size": 128}] if enabled else []}
        if "allowed_experiments" in prompt:
            names = [s["name"] for s in prompt["allowed_experiments"][:prompt["budget"]["max_experiments"]]]
            return {"action": "experiment" if names else "observe", "experiment_names": names,
                    "hypothesis": "待验证候选能否改善当前及下一窗口。", "reason": "CPU规则筛选，非真实大模型建议。", "claims": []}
        eligible = prompt["eligible_artifacts"]
        selected = max(eligible, key=lambda name: eligible[name].get("selection_score", eligible[name]["current_window_shadow"]["macro_f1"]))
        return {"selected_artifact_id": selected, "reason": "CPU规则选择实测排名最高的合格版本，非大模型决策。"}
