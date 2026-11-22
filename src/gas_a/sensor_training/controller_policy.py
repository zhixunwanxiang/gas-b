"""整理精简的实测证据，并为控制器决策设置可执行的约束。

自然语言假设仍属于待验证假设；证据 ID 和明确的比较断言
必须与已记录的测量结果核对，不直接当作新增事实。"""
# 实验控制器的业务证据投影与校验：把指标转换为有限动作、训练侧重点和选择约束。
# 该模块支持本地/历史工具；正式平台的两阶段提示入口在 gas_a.controller_prompts。
import math
import re

from .io import CLASSES


def class_metrics(metrics):
    """按固定类别顺序压缩 support、recall、precision 和 F1，供控制器读取和比较。"""
    return {"macro_f1": round(metrics["macro_f1"], 6), "accuracy": round(metrics["accuracy"], 6),
            "classes": [[metrics["per_class"][name]["support"],
                         round(metrics["per_class"][name]["recall"], 6),
                         round(metrics["per_class"][name]["precision"], 6),
                         round(metrics["per_class"][name]["f1"], 6)] for name in CLASSES]}


def _accepted(trial):
    if "eligible" in trial:
        return trial["eligible"]
    return (trial.get("gate", trial.get("local_gate", {})).get("accepted", False)
            and trial.get("host_gate_passed", True))


def measured_evidence(current, fixed, training, diagnosis, drift, past, parent_sha, families, *, validation_sha=None,
                      fixed_scope="original_fixed_validation"):
    """绑定当前窗口、固定验证、训练和历史实验的测量范围，生成可引用的证据 ID。"""
    # C/F/T/D 分别表示当前评估、固定验证、授权训练和漂移诊断。
    # 显式保留 scope，避免将不同数据窗口上的绝对分数直接当作增益比较。
    facts = {"C": {"scope": "current_evaluation_after_label_reveal", **class_metrics(current)},
             "F": {"scope": fixed_scope, "sha256": validation_sha, **class_metrics(fixed)},
             "T": {"scope": "authorized_gradient_training_set", **class_metrics(training)},
             "D": {"flagged_features": drift["flagged_features"],
                   "class_prior_total_variation": diagnosis["class_prior_total_variation"],
                   "top_confusions": [{key: pair[key] for key in ("actual", "predicted", "count")}
                                      for pair in diagnosis.get("top_confusions", [])[:3]],
                   "largest_distribution_changes_not_importance": [
                       [s["sensor"], round(s["mean_psi"], 4)] for s in diagnosis["top_sensors"][:3]],
                   "meaning": "分布差异和训练评估差距不证明过拟合、传感器故障或物理原因。"}}
    by_family = {name: [] for name in families}
    for row in past:
        for trial in row.get("experiments", []):
            name = trial.get("name")
            if name:
                by_family.setdefault(name, []).append((row, trial))
    for name, attempts in by_family.items():
        def summarize(items):
            fixed_gains, window_gains = [], []
            rejected, failed = 0, 0
            cpu_seconds, unknown_resources = 0.0, 0
            for row, trial in items:
                report = trial.get("report", {})
                # 固定验证收益要求数据摘要一致；shadow 收益使用同一窗口内候选与父版本的差值。
                if report and (validation_sha is None or report.get("validation_sha256") == validation_sha):
                    fixed_gains.append(report["validation_after"]["macro_f1"] - report["validation_before"]["macro_f1"])
                shadow, before = trial.get("shadow_metrics"), row.get("shadow_before")
                if shadow and before:
                    window_gains.append(shadow["macro_f1"] - before["macro_f1"])
                failed += bool(trial.get("error"))
                rejected += not _accepted(trial)
                resources = trial.get("resources") or report.get("resources", {})
                cpu_seconds += resources.get("cpu_seconds") or 0
                unknown_resources += resources.get("cpu_seconds") is None
            return {"attempts": len(items), "rejected": rejected, "errors": failed,
                    "cpu_seconds_known": round(cpu_seconds, 6), "unknown_resource_attempts": unknown_resources,
                    "fixed_best_gain": round(max(fixed_gains), 6) if fixed_gains else None,
                    "window_best_paired_gain": round(max(window_gains), 6) if window_gains else None}
        same = [(row, trial) for row, trial in attempts
                if trial.get("report", {}).get("parent_checkpoint_sha256", row.get("parent_checkpoint_sha256")) == parent_sha]
        facts["H_" + name] = {"all_parents": summarize(attempts), "same_parent": summarize(same)}
    recent = []
    for row in past[-3:]:
        metric = row.get("operational", row.get("after", row.get("before", {})))
        if "per_class" not in metric:
            continue
        trials = []
        for trial in row.get("experiments", []):
            failures = []
            for key in ("fixed_gate", "window_gate", "gate", "local_gate"):
                if key in trial and not trial[key].get("accepted"):
                    failures.extend(trial[key].get("reasons", []))
            trials.append({"name": trial["name"], "accepted": _accepted(trial),
                           "rejection_reasons": list(dict.fromkeys(failures)), "error": trial.get("error")})
        recent.append({"window": row.get("batch", row.get("round")), "actual": class_metrics(metric),
                       "selected": row.get("selected_version", row.get("selected_artifact_id")), "trials": trials})
    return {"class_order": CLASSES, "class_columns": ["support", "recall", "precision", "f1"],
            "facts": facts, "recent_windows": recent,
            "history_scope": "fixed_best_gain只表示原固定验证收益；window_best_paired_gain是同批候选减父版本的影子差值。"
                             "不同时间批的绝对分数不可互相作为收益；历史最好值不保证未来收益。"}


def planning_contract(budget):
    """模型只输出业务意图；证据引用和执行门槛由程序绑定。"""
    return {"action": "experiment或observe",
            "fallback": "可选retain_parent：最终选版回复无效时，明确授权保留本轮合格父模型",
            "experiment_names": f"菜单名称列表，最多{budget}项；observe时[]",
            "hypothesis": "100字内：哪类错误、待验证机制、预期改善什么",
            "reason": "80字内：本轮实测依据和选择该实验的原因"}


def bind_plan(plan):
    """补齐可确定的执行字段，不改写模型的业务假设、理由或显式比较断言。"""
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    result = dict(plan)
    claims = result.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("Comparative claims must be a list")
    result.setdefault("claims", claims)
    result.setdefault("success_condition", "configured_quality_gates")
    result.setdefault("stop_condition", "observe_then_reassess" if result.get("action") == "observe" else "reject_if_gates_fail")
    if "evidence_ids" not in result:
        result["evidence_ids"] = ["C", "F", *dict.fromkeys("H_" + str(c.get("family")) for c in claims if isinstance(c, dict))]
        result["derived_fields"] = "program_bound_evidence_references_and_fixed_gate_names; raw_reply_unchanged"
    changes = result.get("parameter_changes")
    if changes is None or changes == []:
        result["parameter_changes"] = {}
    elif (isinstance(changes, dict) and changes and len(result.get("experiment_names", [])) == 1
          and all(not isinstance(value, dict) for value in changes.values())):
        result["parameter_changes"] = {result["experiment_names"][0]: changes}
    return result


def business_evidence(evidence, menu, feedback=None):
    """给27B一份业务视图：合并逐类成绩、保留误判及同父模型记录，不发送执行证据ID。"""
    facts, feedback = evidence["facts"], feedback or {}
    labels, columns = evidence["class_order"], evidence["class_columns"]
    classes = {key: {label: dict(zip(columns, row)) for label, row in zip(labels, facts[key]["classes"])}
               for key in ("C", "F", "T")}
    table = []
    for label in labels:
        c, f, t = [classes[key][label] for key in ("C", "F", "T")]
        table.append([label, c["support"], c["precision"], c["recall"], c["f1"],
                      t["support"], t["recall"], t["f1"], f["support"], f["recall"], f["f1"]])
    histories = []
    for spec in menu:
        value = facts.get("H_" + spec["name"], {})
        same, all_parents = value.get("same_parent", {}), value.get("all_parents", {})
        future = feedback.get(spec["name"], {})
        histories.append([spec["name"], same.get("attempts", 0), same.get("fixed_best_gain"), same.get("errors", 0),
                          all_parents.get("attempts", 0), all_parents.get("fixed_best_gain"),
                          future.get("recent_mean_gain"), future.get("windows", 0)])
    return {"overall": {name: {key: facts[ref][key] for key in ("macro_f1", "accuracy")}
                         for name, ref in (("current", "C"), ("fixed_validation", "F"), ("training", "T"))},
            "class_columns": ["gas", "current_support", "current_precision", "current_recall", "current_f1",
                              "train_support", "train_recall", "train_f1", "fixed_support", "fixed_recall", "fixed_f1"],
            "class_rows": table, "top_confusions": facts["D"].get("top_confusions", []),
            "distribution": {"flagged_features": facts["D"]["flagged_features"],
                             "class_prior_shift": facts["D"]["class_prior_total_variation"],
                             "changed_sensors": facts["D"]["largest_distribution_changes_not_importance"]},
            "history_columns": ["experiment", "same_parent_attempts", "same_parent_best_fixed_gain", "same_parent_errors",
                                "all_parent_attempts", "all_parent_best_fixed_gain", "next_window_mean_gain", "feedback_windows"],
            "history_rows": histories,
            "recent_trials": [{"window": r["window"], "trials": r["trials"]} for r in evidence.get("recent_windows", [])],
            "reading": "current是当前评估；fixed是固定验证；train是授权训练。历史增益按表中范围解读，null表示未知。"}


def build_planning_prompt(menu, evidence, quality_gate, budget, max_seconds, *, mode, index,
                          feedback=None, parameter_bounds=None):
    """统一SDK及回放的业务提示词，先给错误与实验选择，执行合同留在程序。"""
    if mode not in ("sdk", "temporal"):
        raise ValueError("Unknown planning mode")
    facts = evidence["facts"]
    labels = evidence["class_order"]
    columns = evidence["class_columns"]
    rows = [dict(zip(columns, values), label=label)
            for label, values in zip(labels, facts["C"]["classes"])]
    observed = [row for row in rows if row["support"] > 0]
    targets = quality_gate.get("hard_minimum_class_recall", quality_gate.get("minimum_class_recall", {}))
    shortfalls = [{"label": row["label"], "support": row["support"], "recall": row["recall"], "required": minimum}
                 for row in observed
                 if (minimum := targets.get(row["label"])) is not None
                 and row["recall"] < minimum]
    pause = bool(facts.get("S", {}).get("pause"))
    can_train = bool(menu) and budget > 0 and not pause
    metric_scope = ("改善宿主当前授权验证表现，重点减少弱类漏检和主要混淆。" if mode == "sdk" else
                    "改善下一批气体识别，重点减少弱类漏检和主要混淆；当前批已冻结，模拟收益还不是未来实绩。")
    result = {
        "prompt_version": "gas-a.business-planning.v1",
        "request": "你负责气体分类模型的下一步调优。根据实测错误选择最值得验证的菜单实验，只返回plan_contract要求的JSON。",
        "round" if mode == "sdk" else "batch": index,
        "objective": metric_scope,
        "decision_context": {"training_allowed": can_train,
                             "minimum_class_support": quality_gate.get("minimum_class_support", 20),
                             "low_f1_classes": sorted(observed, key=lambda row: row["f1"])[:2],
                             "recall_below_configured_minimum": shortfalls,
                             "recall_targets": [{"gas": name, "minimum": minimum,
                                                 "kind": "hard" if name in targets else "soft"}
                                                for name, minimum in {**quality_gate.get("minimum_class_recall", {}), **targets}.items()]},
        "decision_steps": [
            "先确定1至2个有足够样本的漏检或混淆问题。训练集也差时检验类别/难样本训练；训练集好而当前差时检验保守更新或鲁棒性。",
            "结合主要误判和同父模型实验记录选方法；分布变化不能证明传感器故障，增加容量也不保证更好。",
            "通常先做1项；第2项须检验不同机制或作为参照。只解释本轮依据，不宣称历史最佳、从未试过或效果已实现。",
        ],
        "observe_policy": "training_allowed=false时observe；允许训练且仍有可检验问题时做小实验。观察须说明实测依据，历史为空不等于禁止首次实验。",
        "allowed_experiments": menu,
        "experiment_guide": {spec["name"]: "旧格式配置参照；按实际参数验证，方案名称不保证收益。"
                             for spec in menu},
        "evidence": business_evidence(evidence, menu, feedback), "plan_contract": planning_contract(budget),
        "budget": {"max_experiments": budget, "max_seconds_per_candidate": max_seconds},
        "rules": "从实际父模型续训，保留权重、Adam、标准化和计数；仅授权train更新梯度、validation选epoch，固定128维输入和6类输出。程序负责迁移、证据绑定、预算和上线门槛，模型不得修改这些约束。",
    }
    if parameter_bounds:
        result["permitted_parameter_changes"] = parameter_bounds
        result["plan_contract"]["parameter_changes"] = "可省略；只改一个选中实验、最多两个白名单参数，格式{实验名称:{参数:值}}；freeze为模式字符串，class_boosts为{气体类别:倍数}。"
        if parameter_bounds.get("architecture_choice"):
            result["plan_contract"]["parameter_changes"] += "architecture_choice为已列出的架构名称；省略表示沿用方案架构。"
    return result


def build_selection_prompt(eligible, experiments, current_id, *, mode, index, history_ids=()):
    """按统一指标来源生成版本比较表，不把不同范围的分数或已花费成本混作未来收益。"""
    if mode not in ("sdk", "temporal") or not eligible:
        raise ValueError("Selection requires a supported mode and eligible artifacts")
    if mode == "sdk":
        basis = "host" if all("host" in value for value in eligible.values()) else "local"
        metric_key = basis
        scope = "host_current_validation" if basis == "host" else "same_local_validation_for_all_artifacts"
        comparison_rule = "所有版本都有宿主指标时共同使用宿主分数；缺项时共同使用本地复评。不能跨来源或数据范围排名。"
    else:
        basis = "selection_score" if all("selection_score" in value for value in eligible.values()) else "current_window_shadow"
        metric_key = "current_window_shadow"
        scope = "measured_recent_window_score" if basis == "selection_score" else "same_current_window_shadow"
        comparison_rule = "共同使用comparison的窗口评分；selection_score是配置的当前与最近窗口加权分数，不是未来实测成绩。"
    scores = {identity: value[basis] if basis == "selection_score" else value[metric_key]["macro_f1"]
              for identity, value in eligible.items()}
    best = max(scores.values())
    tied = [identity for identity in eligible if abs(scores[identity] - best) <= 1e-12]
    preferred = current_id if current_id in tied else max(tied, key=lambda identity: (
        eligible[identity][metric_key].get("accuracy", -1), identity))
    current_score = scores.get(current_id)
    detail_key = "local" if mode == "sdk" else "current_window_shadow"
    def rows(metrics):
        if "per_class" in metrics:
            return metrics["per_class"]
        return {name: dict(zip(("support", "recall", "precision", "f1"), values))
                for name, values in zip(CLASSES, metrics.get("classes", []))}
    current_classes = rows(eligible.get(current_id, {}).get(detail_key, {}))
    comparison, views = [], {}
    for identity, value in eligible.items():
        view = {key: {k: metric[k] for k in ("macro_f1", "accuracy") if k in metric} if isinstance(metric, dict) and "macro_f1" in metric else metric
                for key, metric in value.items()}
        view.setdefault("role", {"retain": "训练前已存在的父版本", "promote": "本轮通过门槛的新候选",
                                 "rollback": "已知历史产物，用于恢复"}.get(value["kind"], value["kind"]))
        views[identity] = view
        changes = []
        for name, new in rows(value.get(detail_key, {})).items():
            old = current_classes.get(name)
            if (old is not None and old["support"] and new["support"]
                    and (abs(new["recall"] - old["recall"]) > 1e-12 or abs(new["f1"] - old["f1"]) > 1e-12)):
                changes.append({"label": name, "support": new["support"],
                                "recall_delta": round(new["recall"] - old["recall"], 6),
                                "f1_delta": round(new["f1"] - old["f1"], 6)})
        comparison.append({"artifact_id": identity, "kind": value["kind"], "score": scores[identity],
                           "delta_vs_current_same_scope": scores[identity] - current_score if current_score is not None else None,
                           "class_change_scope": detail_key, "class_changes_vs_current": changes})
    trials = []
    for trial in experiments:
        report = trial.get("report", {})
        trials.append({"name": trial["name"], "new_candidate_artifact_id": trial.get("artifact_id") or trial.get("host_receipt", {}).get("artifact_id"),
                       "resources_already_spent": {k: v for k, v in (trial.get("resources") or report.get("resources", {})).items()
                                                   if k in ("cpu_seconds", "wall_seconds")},
                       "gate": trial.get("local_gate", trial.get("gate", trial.get("window_gate"))),
                       "fixed_gate": trial.get("fixed_gate"), "host_gate_passed": trial.get("host_gate_passed"),
                       "error": trial.get("error")})
    return {"prompt_version": "gas-a.business-selection.v1",
            "request": "决定气体分类模型采用、保留还是恢复哪个版本，并简述本轮业务效果。只返回plan_contract要求的JSON。",
            "round" if mode == "sdk" else "batch": index,
            "current_active_artifact_id": current_id, "eligible_artifacts": views,
            "comparison": {"scope": scope, "basis": basis, "rows": sorted(comparison, key=lambda row: row["score"], reverse=True),
                           "measured_preference": preferred, "tie_tolerance": 1e-12},
            "decision_policy": [
                comparison_rule,
                "从合格集合中选择comparison.measured_preference，同分优先当前版本。reason说明同口径分数和主要类别变化，不夸大改善。",
                "候选失败不代表同系列当前模型失效。已发生成本不能退回，未来成本未知，不据此改变实测排名。",
            ],
            "experiments": trials,
            "plan_contract": {"selected_artifact_id": "复制合格集合中的完整编号", "reason": "80字内中文业务依据"},
            "rules": "仅eligible_artifacts中的完整编号可返回。唯一合格版本也要明确决定和理由。"
                     + ("本地复评不能冒充宿主或隐藏集成绩。" if mode == "sdk" else "当前批预测已冻结，新版本只影响下一批，影子收益不能声称为线上收益。")}


def validate_plan(plan, menu, budget, evidence, *, text_limits=None):
    """在执行实验前校验菜单、预算和证据断言；模型回复不能自行降低质量门槛。"""
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    action, names = plan.get("action"), plan.get("experiment_names")
    if action not in ("experiment", "observe") or not isinstance(names, list):
        raise ValueError("Plan requires experiment/observe action and experiment_names")
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique strings")
    if action == "observe" and names or action == "experiment" and not 1 <= len(names) <= budget:
        raise ValueError("Observe cannot train; experiment must respect candidate budget")
    if any(name not in [s["name"] for s in menu] for name in names):
        raise ValueError("Experiment outside bounded menu")
    ids = plan.get("evidence_ids")
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 6 or any(not isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids)) or any(i not in evidence["facts"] for i in ids)
            or not set(ids) & {"C", "F"}):
        raise ValueError("Unknown or missing measured evidence ID")
    # 长度由调用程序决定；全部原文继续接受事实断言和归因检查。
    for field, maximum in (text_limits or {"hypothesis": 100, "reason": 80}).items():
        if not isinstance(plan.get(field), str) or not plan[field].strip() or len(plan[field]) > maximum:
            raise ValueError("Invalid concise " + field)
    if plan.get("success_condition") != "configured_quality_gates":
        raise ValueError("The model cannot weaken quality gates")
    stop = "reject_if_gates_fail" if action == "experiment" else "observe_then_reassess"
    if plan.get("stop_condition") != stop:
        raise ValueError("Invalid stop condition")
    claims = plan.get("claims")
    if not isinstance(claims, list) or len(claims) > 3:
        raise ValueError("Explicit comparative claims are required, or use []")
    # 自由文本不能新增历史事实；比较断言必须引用对应 H_方向 并通过实测数据校验。
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("scope") not in ("same_parent", "all_parents"):
            raise ValueError("Invalid claim scope")
        reference = "H_" + str(claim.get("family"))
        if reference not in ids or reference not in evidence["facts"]:
            raise ValueError(f"Claim for {claim.get('family')} must include {reference} in evidence_ids; cite that exact facts ID")
        stat = evidence["facts"][reference][claim["scope"]]
        if claim.get("kind") == "untried":
            if stat["attempts"] != 0:
                raise ValueError("Claimed untried family already has recorded attempts")
        elif claim.get("kind") == "best_fixed_gain":
            gain = stat["fixed_best_gain"]
            measured = [v[claim["scope"]]["fixed_best_gain"] for key, v in evidence["facts"].items()
                        if key.startswith("H_") and v[claim["scope"]]["fixed_best_gain"] is not None]
            if gain is None or not measured or gain < max(measured) - 1e-6:
                raise ValueError("Historical best-gain claim is unsupported")
        else:
            raise ValueError("Unsupported claim kind")
    text = plan["reason"] + " " + plan["hypothesis"]
    if re.search(r"历史.{0,12}(最高|最大|最好|最佳)", text) and not any(c.get("kind") == "best_fixed_gain" for c in claims):
        raise ValueError("Historical ranking assertion needs a checked best_fixed_gain claim")
    if "未试" in text and not any(c.get("kind") == "untried" for c in claims):
        raise ValueError("Untried assertion needs a scoped checked claim")
    for sentence in (plan["reason"], plan["hypothesis"]):
        if ("严重过拟合" in sentence or re.search(r"(确认|确定|证明|证实).{0,8}(过拟合|物理原因|传感器故障)", sentence)):
            if not any(word in sentence for word in ("可能", "假设", "待验证", "不能")):
                raise ValueError("Distribution evidence does not prove this causal diagnosis")
    return plan


def class_protection(candidate, active, baseline, config):
    """对新版本和历史恢复版本执行类别保护，覆盖满足支持数条件的类别。
    
    baseline 仍可作为明确标记的紧急兜底版本；
    当前窗口缺失类别的 recall 无法估计，不用虚构的零值替代。"""
    reasons = []
    for reference in (active, baseline):
        for name, old in reference.get("per_class", {}).items():
            if old["support"] < config["minimum_class_support"]:
                continue
            new = candidate.get("per_class", {}).get(name)
            if new is None or new["recall"] < old["recall"] - config["max_class_recall_drop"] - 1e-12:
                reasons.append("class_recall_regression:" + name)
    for name, minimum in config.get("minimum_class_recall", {}).items():
        if not baseline.get("per_class", {}).get(name, {}).get("support", 0):
            continue
        value = candidate.get("per_class", {}).get(name, {}).get("recall")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value + 1e-12 < minimum:
            reasons.append("class_recall_below_minimum:" + name)
    return {"accepted": not reasons, "reasons": list(dict.fromkeys(reasons))}


def stopping_evidence(past, parent_sha, train_sha, validation_sha, config):
    """仅在父模型和数据摘要均未变化时累计连续拒绝窗口，达到阈值后暂停新增训练一窗。"""
    threshold = config.get("controller", {}).get("rejected_windows_before_pause", 3)
    rejected = 0
    for row in reversed(past):
        if row.get("decision_source") == "budget_pause":
            break  # 观察一个窗口后，再依据新增证据重新判断。
        # 更换父模型、数据或已获得合格候选后，旧的连续拒绝计数不再适用。
        if (row.get("parent_checkpoint_sha256") != parent_sha or row.get("action") == "promote"
                or row.get("train_sha256") != train_sha or row.get("validation_sha256") != validation_sha):
            break
        trials = row.get("experiments", [])
        if not trials or any(_accepted(trial) for trial in trials):
            break
        rejected += 1
    return {"pause": rejected >= threshold, "consecutive_rejected_windows": rejected,
            "threshold": threshold, "policy": "训练与固定验证数据未变、同父版本连续无合格实验时观察一窗；继续监测和回滚。"}


def protect_decision(decision, candidate, active, baseline, config):
    """将类别保护结果合并到基础 gate 决策中，保留去重后的全部拒绝理由。"""
    protection = class_protection(candidate, active, baseline, config)
    if protection["accepted"]:
        return decision
    return {**decision, "accepted": False,
            "reasons": list(dict.fromkeys([r for r in decision["reasons"] if r != "quality_gates_passed"]
                                          + protection["reasons"]))}
