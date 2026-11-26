"""B/C 原生回归闭环：读历史、监测诊断、模型规划、真实续训、测量选版和回滚。"""
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from contest_harness.errors import HarnessError
from .data import UNITS, input_values, load_model, load_pair
from .evaluation import evaluate, promotion_gate, current_parent_reference, host_gate, PARENT_REFERENCE_VERSION, EVALUATION_VERSION
from .assessment import Measurements, StageEvidence
from .monitoring import fit_reference, monitor, diagnose
from .training import continue_candidate
from .prompts import VERSION as PROMPT_VERSION, plan_contract, PLAN_INSTRUCTION, SELECT_INSTRUCTION
from ..evolution import (available_experiments, fingerprint, experiment_outcome,
                         validate_attempts, eligible_history, experiment_evidence, EXPERIMENT_POLICY_VERSION)
from ..host_validation import HOST_ADMISSION_VERSION, bind_candidate_validation
from ..progress import progress, stage
from ..sensor_training.io import append_jsonl, digest, write_json
from ..submission import ROOT, PREDICT_ARGV, _controller_decision, _parent_checkpoint, _prediction_checkpoint, _snapshot
from ..sdk_records import bounded_configuration, portable_record


@lru_cache(maxsize=2)
def _prediction_model(path, dataset, identity):
    # 单个预测进程内按文件身份缓存 B/C 模型；identity 的时间和大小变化会使缓存失效。
    # 它只减少重复读权重，不保存标签，也不跨评测进程共享可写状态。
    return load_model(Path(path), dataset)


def predict(sample, model_dir):
    # 先核对原生输入结构，再读取 SDK 实际选中的检查点；这里没有训练和标签反馈分支。
    dataset, paths, values = input_values(sample.inputs, sample.dataset_key)
    checkpoint, identity = _prediction_checkpoint(sample, model_dir)
    model = _prediction_model(str(checkpoint), dataset, identity)
    if paths != model.meta["paths"]:
        raise ValueError("Regression prediction schema differs from its checkpoint")
    result = float(model.predict(np.asarray([values], dtype=np.float64))[0])
    # 官方模型负责还原目标尺度并限制为非负值；回调仍再次拒绝 NaN/Inf 或负数。
    if not np.isfinite(result) or result < 0:
        raise ValueError("Regression output must be a finite nonnegative number")
    return {"value": result}  # SDK 原样包装为 outputs:{value:...}；不伪装成分类概率。


def dataset_config(full_config, dataset):
    """Keep B's shared menu and append only the selected dataset's experiments."""
    shared = full_config["regression"]
    extras = shared.get("dataset_experiments", {})
    if not isinstance(extras, dict) or set(extras) - {"B", "C"} or any(not isinstance(v, list) for v in extras.values()):
        raise ValueError("Invalid dataset-specific regression menu")
    specs = [*shared["experiments"], *extras.get(dataset, [])]
    for spec in specs:
        if (spec.get("weighting") == "bearing_sqrt_inverse" and dataset != "B"
                or spec.get("weighting") == "parent_tail" and dataset != "C"):
            raise ValueError("Regression weighting belongs to a different dataset")
    return {**shared, "controller": {**shared["controller"], "guards": shared.get("guards", {}).get(dataset)},
            "experiments": specs}


def validate_config(config, limits):
    # 将可执行方案限制在预先配置的范围：AI 只能选菜单，不能临时增加轮数、学习率或放宽门槛。
    # Python 的 bool 是 int 子类，因此预算字段特意用 type(...) is int 排除 true/false。
    rules = config["controller"]
    for key in ("enable_research_experiments", "single_choice_observe"):
        if key in config and type(config[key]) is not bool:
            raise ValueError("Invalid regression runtime flag: " + key)
    for key in ("minimum_rmse_absolute_gain", "minimum_rmse_relative_gain", "maximum_mae_relative_increase"):
        value = rules[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid regression promotion threshold")
    for key, low, high in (("json_corrections", 0, 2), ("max_history_restores", 0, 2), ("max_repeat_rejections", 1, 8)):
        if type(rules[key]) is not int or not low <= rules[key] <= high:
            raise ValueError("Invalid regression controller bound")
    if (type(limits["max_epochs"]) is not int or not 1 <= limits["max_epochs"] <= 32
            or type(limits["early_stopping_patience"]) is not int or not 1 <= limits["early_stopping_patience"] <= 32
            or isinstance(limits["max_seconds_per_candidate"], bool)
            or not 0 < limits["max_seconds_per_candidate"] <= 300):
        raise ValueError("Invalid regression training limits")
    names = set()
    for spec in config["experiments"]:
        if (not isinstance(spec["name"], str) or not spec["name"] or spec["name"] in names
                or type(spec["epochs"]) is not int or not 1 <= spec["epochs"] <= limits["max_epochs"]
                or isinstance(spec["learning_rate"], bool) or not 0 < spec["learning_rate"] <= .01
                or type(spec["freeze_features"]) is not bool or spec["loss"] not in ("mse", "huber")):
            raise ValueError("Invalid bounded regression experiment")
        weight = spec.get("underestimate_weight", 1.0)
        if type(weight) not in (int, float) or not np.isfinite(weight) or not 1 <= weight <= 1.5:
            raise ValueError("Invalid regression bias correction")
        names.add(spec["name"])
        if spec.get("weighting", "uniform") not in ("uniform", "bearing_sqrt_inverse", "parent_tail"):
            raise ValueError("Invalid regression weighting")
        if spec.get("weighting") == "parent_tail":
            for key, low, high in (("tail_quantile", .5, .99), ("tail_weight", 1., 2.)):
                value = spec.get(key)
                if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
                    raise ValueError("Invalid tail weighting")
        if "research_only" in spec and type(spec["research_only"]) is not bool:
            raise ValueError("Invalid research experiment flag")
    guards = rules.get("guards")
    if guards is not None:
        if (not isinstance(guards, dict) or type(guards.get("minimum_group_rows")) is not int
                or not 1 <= guards["minimum_group_rows"] <= 1000000
                or type(guards.get("require_stage_reference")) is not bool
                or type(guards.get("maximum_group_relative_increase")) not in (int, float)
                or not np.isfinite(guards["maximum_group_relative_increase"])
                or not 0 <= guards["maximum_group_relative_increase"] <= .02):
            raise ValueError("Invalid regression group guards")


def _policy(state, dataset, menu):
    # /state 复制到工作区后读取策略；首轮没有文件才初始化，损坏状态不会被悄悄清空。
    # version 是当前模型配套版本，revision 是只增不减的总修订计数，两者在回滚时可能不同。
    path = state / "regression_strategy.json"
    if not path.exists():
        return {"format": "rul.strategy.v1", "dataset": dataset, "version": 1, "revision": 1,
                "preferred_experiment": None, "attempts": [], "artifact_policies": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("format") != "rul.strategy.v1" or value.get("dataset") != dataset
            or type(value.get("version")) is not int or value["version"] < 1
            or type(value.get("revision")) is not int or value["revision"] < value["version"]
            or value.get("preferred_experiment") not in (None, *menu)
            or not isinstance(value.get("attempts"), list) or not isinstance(value.get("artifact_policies"), dict)):
        raise ValueError("Invalid regression strategy state")
    validate_attempts(value["attempts"])
    for item in value["artifact_policies"].values():
        if (type(item.get("version")) is not int or not 1 <= item["version"] <= value["revision"]
                or item.get("preferred_experiment") not in (None, *menu)):
            raise ValueError("Invalid regression artifact-paired policy")
    return value


def _advance(policy, selected, parent, candidate, plan, identity, reduction, failed,
             local_gate=None, host_gate=None, selection_source="model"):
    # 依据已经发生的结果更新策略，而不是依据模型在 reason 里声称“有效”。
    # 拒绝/本地失败也计入 attempts，后续用完整实验指纹限制同一个失败方案重复尝试。
    result = deepcopy(policy)
    result["artifact_policies"].setdefault(parent, {k: policy[k] for k in ("version", "preferred_experiment")})
    if candidate is not None or failed or isinstance(local_gate, dict):
        adopted = candidate is not None and selected == candidate
        result["attempts"].append({"fingerprint": identity, "experiment": plan["experiment"],
                                   "adopted": adopted, "rmse_reduction": reduction,
                                   "status": "local_training_failed" if failed else "host_measured" if candidate else "local_measured_not_submitted",
                                   **experiment_outcome(adopted=adopted, failed=failed,
                                       local_gate=local_gate, host_gate=host_gate, selection_source=selection_source)})
        result["attempts"] = result["attempts"][-128:]
        if adopted:
            # 只有实际选中新候选才增加策略版本，并将这个方案记为实验偏好。
            result["revision"] += 1
            result.update(version=result["revision"], preferred_experiment=plan["experiment"])
    if selected not in (parent, candidate):
        # 选回历史模型时也恢复它的配套策略；不能用故障父模型的偏好替代历史状态。
        # 无配套记录的外部历史采用保守初值，revision 总计数仍保留。
        result.update(result["artifact_policies"].get(selected, {"version": 1, "preferred_experiment": None}))
    result["artifact_policies"][selected] = {k: result[k] for k in ("version", "preferred_experiment")}
    result["artifact_policies"] = dict(list(result["artifact_policies"].items())[-64:])
    return result


def _history_view(history):
    # 发给模型的上下文只展示最近十条关键证据，防止历史无限增长撑满 token 预算。
    # finish 时仍引用实际 history() 读到的全部 ID，不能把摘要裁剪说成没有读过旧历史。
    result = []
    for item in history[-10:]:
        record = item.get("record", item)
        if not isinstance(record, dict):
            record = {}
        config = record.get("configuration", {})
        if not isinstance(config, dict):
            config = {}
        result.append(portable_record({"id": item["id"], "selected_artifact_id": record.get("selected_artifact_id"),
                       "reason": str(record.get("reason", ""))[:600],
                       "configuration": {key: config[key] for key in (
                           "dataset", "experiment", "parent_checkpoint_sha256", "train_sha256", "validation_sha256",
                           "validation_before", "validation_after", "candidate_gate", "host_candidate_gate",
                           "experiment_failure", "experiment_outcome", "execution_policy", "selected_local_rmse") if key in config}}, context="regression_history_view"))
    # Full measurements remain in decisions/state. The model sees compact history,
    # while exact execution policies remain available for condition comparisons.
    for row in result:
        cfg = row["configuration"]
        for key in ("validation_before", "validation_after"):
            value = cfg.get(key)
            if isinstance(value, dict):
                cfg[key] = {k: value[k] for k in ("rmse", "mae", "nasa_score", "nasa_score_overflow",
                            "absolute_error_p99", "rows", "bias") if k in value}
                groups = value.get("group_evaluation")
                if isinstance(groups, dict):
                    cfg[key]["group_summary"] = {k: groups[k] for k in ("source", "mean_rmse", "worst_rmse") if k in groups}
        host_gate = cfg.get("host_candidate_gate")
        if isinstance(host_gate, dict):
            cfg["host_candidate_gate"] = {k: host_gate[k] for k in ("eligible", "reason", "host_rmse_reduction") if k in host_gate}
    return result


def _restore_history(rnd, history, run, validation, before, rules, dataset, *, measurements=None, priority=None):
    # 允许恢复的范围只来自当前 start 帧 artifacts；优先最近选中过的版本，跳过实际父版本。
    # 历史旧分数不能直接用于本轮选版：取回后必须在本轮同一 validation 上重新预测并过门槛。
    receipts = {item["artifact_id"]: item for item in rnd.artifacts}
    order = [priority] if priority else []
    order += [row.get("selected_artifact_id") for row in reversed(_history_view(history))]
    order.extend(reversed(list(receipts)))
    restored, seen = {}, {rnd.parent.artifact_id}
    for identifier in order:
        # 历史记录本身合法，不代表可选 selected_artifact_id 一定是字符串。
        # 不把列表/对象放进集合判断；忽略坏提示后仍按本帧授权清单寻找可用历史。
        if not isinstance(identifier, str) or not identifier.strip():
            if identifier is not None:
                progress("history_selection_hint_rejected", dataset=dataset, value_type=type(identifier).__name__)
            continue
        if identifier in seen or identifier not in receipts:
            continue
        seen.add(identifier)
        if len(restored) >= rules["max_history_restores"]:
            break
        descriptor = receipts[identifier].get("artifact", {})
        if descriptor.get("predict_argv") != PREDICT_ARGV or len(descriptor.get("checkpoint_paths", [])) != 1:
            continue
        # SDK 恢复与摘要校验属于 RPC，必须放在本地数值异常捕获之外；不执行历史源码。
        directory = rnd.restore_artifact(identifier, run / f"restored-{len(restored) + 1}")
        try:
            path = directory / descriptor["checkpoint_paths"][0]
            model = load_model(path, dataset, validation["paths"])
            metrics, source = (measurements.measure(path, model) if measurements else
                               (evaluate(model, validation, dataset), {"measured_this_round": True}))
            restored[identifier] = {"metrics": metrics, "gate": promotion_gate(before, metrics, rules),
                                    "checkpoint_sha256": digest(path), "seed": model.meta["seed"],
                                    "validation_sha256": validation["sha256"],
                                    "reevaluated_this_round": source["measured_this_round"],
                                    "measurement_source": source}
        except (ValueError, OSError, KeyError) as error:
            # 文件/模型不兼容只使这个历史版本失去资格；restore_artifact 的 RPC 异常不在这里捕获。
            # 加载器把坏 NPZ 和非对象元数据统一为 ValueError，保留具体原因供下一轮复盘。
            restored[identifier] = {"error": str(error)[:1000], "error_type": type(error).__name__,
                                    "stage": "historical_model_evaluation", "gate": {"eligible": False}}
    return restored


def train(rnd):
    # 阶段 1：检查冻结合同。B 是分钟，C 是循环，两者都是 RMSE 越低越好，不能套 A 的分类门槛。
    dataset = rnd.dataset_key
    progress("train_started", dataset=dataset, round_index=rnd.round_index)
    if (dataset not in UNITS or rnd.task.get("task") != "regression"
            or str(rnd.task.get("metric", "rmse")).lower() != "rmse"
            or rnd.task.get("metric_direction", "lower") != "lower"
            or rnd.task.get("target_unit", UNITS[dataset]) != UNITS[dataset]):
        raise ValueError("B/C requires the native RMSE/lower regression contract and original target unit")
    if rnd.limits.get("artifact_upload_protocol") != "chunked-v1":
        raise HarnessError("CHUNKED_UPLOAD_UNSUPPORTED", "SDK 1.0.4 requires chunked-v1 uploads")
    full_config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
    config = dataset_config(full_config, dataset)
    limits, rules = config["limits"], config["controller"]
    validate_config(config, limits)
    work = Path(os.environ.get("CONTEST_WORK", "/workspace")).resolve()
    # /app、数据、父产物及原始状态都只读；所有候选与状态副本写到新的工作区目录。
    protected = [ROOT, rnd.parent.root, Path(rnd.data.train_path).parent, Path(rnd.data.validation_path).parent,
                 *(Path(p) for p in ("/data", "/previous", "/state", "/model"))]
    if rnd.state_path:
        protected.append(Path(rnd.state_path))
    if any(work.is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError("Regression workspace must be outside read-only mounts")
    work.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"round-{rnd.round_index:02d}-", dir=work))
    checkpoint = _parent_checkpoint(rnd)
    state = rnd.state_workspace(run / "state")
    # 阶段 2：从宿主实际父产物加载权重、Adam、scaler、seed，先测父模型，再读历史与状态副本。
    # parent_host 会核对宿主父分数和本地分数是否同一范围，冲突时不开放新版本上线。
    with stage("load_public_data", dataset=dataset):
        train_data, validation = load_pair(rnd.data.train_path, rnd.data.validation_path, dataset)
    with stage("evaluate_parent", dataset=dataset):
        parent = load_model(checkpoint, dataset, train_data["paths"])
        measurements = Measurements(state, dataset, validation, rnd.round_index)
        before, parent_measurement_source = measurements.measure(checkpoint, parent)
    with stage("validate_parent_measurement", dataset=dataset):
        parent_host = current_parent_reference(rnd, checkpoint, before, validation["sha256"])
        if not parent_measurement_source["measured_this_round"] and str(parent_host.get("source", "")).startswith("local_"):
            parent_host["source"] = "local_cached_current_parent_public_evaluation"
        parent_host["local_measurement_source"] = parent_measurement_source
    with stage("history_rpc", dataset=dataset):
        history = rnd.history()
    stage_evidence = StageEvidence(state, rnd, train_data, validation, before, measurements.implementation)
    reference_path = state / "regression_monitor_reference.json"
    # 阶段 3：首轮只从 train 建立监测参考；后续校验其来源和结构，保持最初的比较口径。
    # 平台可以更新公开 train，不能因此重拟合父模型 scaler，也不能把合法新投影判成状态损坏。
    # monitor 看输入变没变，diagnose 看公开标签上的误差分段；二者一起给出可验证的实验线索。
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else fit_reference(train_data, dataset)
    drift = monitor(reference, validation, train_data, dataset, full_config["controller"]["drift_thresholds"])
    diagnosis = diagnose(parent, validation, reference, before, drift)
    menu = {spec["name"]: spec for spec in config["experiments"]}
    policy = _policy(state, dataset, menu)
    # 阶段 4：策略必须与本轮实际父模型配对；换了父产物却没有对应策略时，偏好恢复为初值。
    paired = policy["artifact_policies"].get(rnd.parent.artifact_id)
    if paired:
        policy.update(paired)
    elif policy["artifact_policies"]:
        policy.update(version=1, preferred_experiment=None)
    from ..evolution import REPEAT_POLICY
    execution = {"format": "rul.execution.v1", "rules": rules, "limits": limits, "prompt_version": PROMPT_VERSION,
                 "evaluation_version": EVALUATION_VERSION, "implementation_sha256": measurements.implementation,
                 "training_implementation_sha256": digest(Path(__file__).with_name("training.py")),
                 "resource_policy": "skip-local-rejections-observe-single-choice-v1",
                 "repeat_policy": REPEAT_POLICY,
                 "experiment_policy_version": EXPERIMENT_POLICY_VERSION,
                 "host_admission_policy_version": HOST_ADMISSION_VERSION,
                 "parent_reference_policy_version": PARENT_REFERENCE_VERSION,
                 "monitor_reference_policy": "fixed_initial_training_projection-v1"}
    allowed, blocked = available_experiments(policy, menu, rnd, train_data, validation, rules, execution)
    research_disabled = [name for name, spec in allowed.items()
                         if spec.get("research_only") and not config.get("enable_research_experiments", False)]
    allowed = {name: spec for name, spec in allowed.items() if name not in research_disabled}
    admission_blocker = None
    if not parent_host["comparable"]:
        # 不可核验的父测量使本轮任何候选都不能上线；直接观察并留证，避免重复消耗训练预算。
        admission_blocker = parent_host.get("reason", "parent_measurement_unverified")
        allowed = {}
    # 指纹包含父模型、train、validation、方案和执行规则；只屏蔽同一身份下反复失败的方案。
    with stage("restore_and_evaluate_history", dataset=dataset):
        # 父测量不可比时历史也不能采用，避免白白传输整份历史产物。
        restored = {} if admission_blocker else _restore_history(rnd, history, run, validation, before, rules, dataset,
            measurements=measurements, priority=stage_evidence.stage["reference_artifact_id"])
    stage_reference = stage_evidence.bind_reference(before, rnd.parent.artifact_id, restored)
    for item in restored.values():
        if "metrics" in item:
            item["gate"] = promotion_gate(before, item["metrics"], rules, reference=stage_reference)
    if stage_reference is None and (rules.get("guards") or {}).get("require_stage_reference"):
        admission_blocker = "stage_reference_unavailable_on_current_validation"
        allowed = {}
    rollback_ids = [identifier for identifier, item in restored.items()
                    if parent_host["comparable"] and not admission_blocker and item["gate"]["eligible"]
                    and item.get("validation_sha256") == validation["sha256"]
                    and (item.get("reevaluated_this_round") or
                         item.get("measurement_source", {}).get("identity_verified_this_round"))]
    # observe 永远保留；只有存在可执行菜单/合格历史时才开放 experiment/rollback。
    actions = ["observe"] + (["experiment"] if allowed else []) + (["rollback"] if rollback_ids else [])

    def validate_plan(value):
        # 阶段 5：检查完整“动作—方案—目标”组合，而不只分别检查每个字段是不是合法名字。
        # observe/experiment 必须指向真实父版本；使用历史模型必须先 rollback，本轮不直接续训历史。
        if value["action"] not in actions:
            raise ValueError("Unavailable regression action")
        if value["action"] == "experiment" and value["experiment"] not in allowed:
            raise ValueError("Regression experiment outside the configured menu")
        if value["action"] != "experiment" and value["experiment"] != "none":
            raise ValueError("Observe/rollback must use experiment=none")
        targets = rollback_ids if value["action"] == "rollback" else [rnd.parent.artifact_id]
        if value["target_artifact_id"] not in targets:
            raise ValueError("Action %s must target %s. The actual current parent is %s (RMSE %.9f). "
                             "To use any different historical artifact, select action=rollback and experiment=none; "
                             "observe/experiment cannot restore or train a historical model." % (
                                 value["action"], targets, rnd.parent.artifact_id, before["rmse"]))

    request = {"phase": "plan", "round_index": rnd.round_index, "dataset": dataset,
               "task": "regression", "metric": "rmse", "metric_direction": "lower", "target_unit": UNITS[dataset],
               "parent_artifact_id": rnd.parent.artifact_id, "validation_before": before,
               "parent_host_reference": parent_host,
               "admission_blocker": admission_blocker,
               "stage_reference": {"scope_kind": stage_evidence.stage["scope_kind"],
                   "reference_artifact_id": stage_evidence.stage["reference_artifact_id"], "metrics": stage_reference},
               "research_candidates_disabled": research_disabled,
               "history": _history_view(history), "experiments": allowed, "strategy": {
                   "version": policy["version"], "preferred_experiment": policy["preferred_experiment"] if policy["preferred_experiment"] in allowed else None,
                   "blocked_repeated_experiments": blocked},
               "experiment_evidence": experiment_evidence(_history_view(history), policy["attempts"], menu,
                   rnd.parent.checkpoint_sha256, train_data["sha256"], validation["sha256"], execution, rules,
                   gain_key="rmse_reduction", blocked=blocked if not admission_blocker else list(menu)),
               "monitoring": {k: v for k, v in drift.items() if k != "features"}, "diagnosis": diagnosis,
               "historical_validation": restored, "promotion_rules": rules,
               "plan_contract": plan_contract(rnd.parent.artifact_id, before, restored, rollback_ids, allowed),
               "controller_prompt": {"version": PROMPT_VERSION, "instruction": PLAN_INSTRUCTION},
               "response_schema": {"action": actions, "experiment": ["none", *allowed],
                   "target_artifact_id": [rnd.parent.artifact_id, *rollback_ids], "hypothesis": "非空实验假设", "reason": "非空证据理由"}}
    plan, planning_calls = _controller_decision(rnd, request, ("action", "experiment", "target_artifact_id", "hypothesis", "reason"),
                                     full_config["max_output_tokens"], validate_plan, rules["json_corrections"], state)
    spec = allowed[plan["experiment"]] if plan["action"] == "experiment" else None
    # 保存实际展开的配置，而非只写菜单名称；模型输出不能覆写 spec 中的学习率、epoch 或损失。
    cfg = {"dataset": dataset, "task": "regression", "metric": "rmse", "metric_direction": "lower",
           "target_unit": UNITS[dataset], "action": plan["action"], "experiment": spec,
           "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256, "train_sha256": train_data["sha256"],
           "parent_primary_file_sha256": digest(checkpoint),
           "validation_sha256": validation["sha256"], "validation_before": before,
           "parent_host_reference": parent_host,
           "stage_reference": {"artifact_id": stage_evidence.stage["reference_artifact_id"],
                               "scope_kind": stage_evidence.stage["scope_kind"], "metrics": stage_reference},
           "admission_blocker": admission_blocker,
           "controller_prompt_version": PROMPT_VERSION, "execution_policy": execution}
    submitted, report, gate, admission, failure = None, None, None, None, None
    identity = fingerprint(rnd.parent.checkpoint_sha256, train_data["sha256"], validation["sha256"], spec, execution) if spec else None
    change = "Observed public regression data and retained the actual parent without training."
    if spec:
        # 阶段 6：先落盘“已开始”，再做真实数值续训。只捕获已知本地计算失败并保留拒绝记录。
        # 模型调用、上传、恢复、保存状态等 SDK 操作都不放进这个 catch，避免结果未知后继续 RPC。
        append_jsonl(state / "experiment_attempts.jsonl", {"round_index": rnd.round_index, "configuration": spec, "status": "started"})
        candidate = run / "candidate"
        try:
            with stage("local_training", dataset=dataset, experiment=spec["name"]):
                report = continue_candidate(checkpoint, train_data, validation, candidate, spec, limits, rules, dataset,
                                            digest(checkpoint), reference=stage_reference)
        except (TimeoutError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            failure = {"stage": "local_training", "error_type": type(error).__name__, "message": str(error)[:1000]}
            append_jsonl(state / "experiment_attempts.jsonl", {"round_index": rnd.round_index, "status": "failed", **failure})
            cfg["experiment_failure"] = failure
            change = "Rejected a failed local regression experiment and retained the actual parent."
        else:
            append_jsonl(state / "experiment_attempts.jsonl", {"round_index": rnd.round_index, "status": "trained"})
    if report is not None:
        # 阶段 7：本地算准入、写诊断证据，并把候选交给宿主真实测量；上传成功不等于准许上线。
        gate = promotion_gate(before, report["validation_after"], rules, reference=stage_reference)
        cfg.update(validation_after=report["validation_after"], candidate_gate=gate)
        write_json(candidate / "monitoring.json", drift)
        write_json(candidate / "diagnosis.json", diagnosis)
        write_json(candidate / "planning.json", {"decision": plan, **planning_calls})
        change = "Continued the actual %s regression parent using %s; preserved Adam/scalers/seed and measured RMSE in %s." % (dataset, spec["name"], UNITS[dataset])
        # SDK 上传在计算异常捕获之外；任何结果未知的 RPC 立即结束会话，不重试或伪造成功。
        if gate["eligible"]:
            code_paths = _snapshot(candidate)
            from ..progress import record_transfer
            transfer = record_transfer("candidate", candidate)
            with stage("submit_candidate_rpc", dataset=dataset):
                submitted = rnd.submit_candidate(candidate, hypothesis=plan["hypothesis"], change_summary=change,
                    configuration=bounded_configuration(cfg, state, rnd.round_index, "candidate"), seed=parent.meta["seed"],
                    checkpoint_paths=["checkpoint.npz"], code_paths=code_paths, predict_argv=PREDICT_ARGV)
            transfer["acknowledged"] = True
            admission = host_gate(submitted, parent_host, report["validation_after"], validation["sha256"], rules,
                                  binding=bind_candidate_validation(rnd, submitted, validation["sha256"]))
        else:
            admission = {"eligible": False, "reason": "local_performance_rejected_not_submitted", "host_measured": False}
            cfg["submission_skipped"] = True
            append_jsonl(state / "experiment_attempts.jsonl", {"round_index": rnd.round_index,
                "status": "local_performance_rejected_not_submitted", "gate": gate,
                "checkpoint_sha256": report["checkpoint_sha256"]})
        cfg["host_candidate_gate"] = admission
    eligible = [rnd.parent.artifact_id]
    # 阶段 8：最终白名单取本地和宿主两道门槛的交集；不合格候选即使模型坚持也不能选。
    # 历史资格由当前 SDK 授权及本轮复评决定，独立于是否在规划中预先声明 rollback。
    if submitted and gate["eligible"] and admission["eligible"]:
        eligible.append(submitted["artifact_id"])
    selection_history_ids = rollback_ids if planning_calls["decision_source"] == "model" else []
    eligible.extend(selection_history_ids)

    def validate_selection(value):
        if value["selected_artifact_id"] not in eligible:
            raise ValueError("Selected ID is outside current measured choices. Actual parent=%s; action=%s; allowed IDs=%s. "
                             "History requires SDK authorization and a qualified current-scope reevaluation." % (
                                 rnd.parent.artifact_id, plan["action"], eligible))

    selection_request = {"phase": "select", "dataset": dataset, "action": plan["action"],
        "parent_artifact_id": rnd.parent.artifact_id, "candidate_artifact_id": submitted["artifact_id"] if submitted else None,
        "same_local_validation": {"parent": before, "candidate": report["validation_after"] if report else None, "historical": restored},
        "historical_admission": {key: {"eligible": key in selection_history_ids,
            "reevaluated_this_round": value.get("reevaluated_this_round") is True,
            "measurement_source": value.get("measurement_source"),
            "validation_sha256": value.get("validation_sha256"), "gate": value["gate"],
            "fresh_metrics": value.get("metrics")} for key, value in restored.items()},
        "host_candidate_receipt": submitted, "host_candidate_gate": admission, "candidate_gate": gate,
        "parent_host_reference": parent_host, "experiment_failure": failure, "promotion_rules": rules,
        "controller_prompt": {"version": PROMPT_VERSION, "instruction": SELECT_INSTRUCTION},
        "response_schema": {"selected_artifact_id": eligible, "reason": "依据实际回归测量的非空理由"}}
    if (config.get("single_choice_observe", True) and plan["action"] == "observe"
            and planning_calls["decision_source"] == "model" and eligible == [rnd.parent.artifact_id]):
        selection = {"selected_artifact_id": rnd.parent.artifact_id, "reason": plan["reason"]}
        selection_calls = {**planning_calls, "selection_reused_plan": True,
                           "reuse_reason": "validated_observe_and_unique_actual_parent_choice"}
        validate_selection(selection)
    else:
        selection, selection_calls = _controller_decision(rnd, selection_request, ("selected_artifact_id", "reason"),
            full_config["max_output_tokens"], validate_selection, rules["json_corrections"], state)
    selected, seed = selection["selected_artifact_id"], parent.meta["seed"]
    # 阶段 9：依据最终 ID 记录真实分数和种子，不把未采用候选的好看分数写成当前成绩。
    candidate_id = submitted["artifact_id"] if submitted else None
    selected_metrics = before
    if selected == candidate_id:
        selected_metrics = report["validation_after"]
    elif selected != rnd.parent.artifact_id:
        selected_metrics, seed = restored[selected]["metrics"], restored[selected]["seed"]
        change = "Restored historical regression artifact %s after current-scope RMSE reevaluation." % selected
    elif submitted:
        change = "Measured the regression candidate but retained the actual parent artifact."
    next_policy = _advance(policy, selected, rnd.parent.artifact_id, candidate_id, plan, identity,
                           gate["rmse_reduction"] if gate else None, failure is not None,
                           local_gate=gate, host_gate=admission, selection_source=selection_calls["decision_source"])
    if candidate_id or failure or report is not None:
        cfg["experiment_outcome"] = {key: next_policy["attempts"][-1][key] for key in (
            "outcome", "reason", "counts_toward_rejection_limit")}
    cfg["selection_action"] = ("adopt_candidate" if selected == candidate_id else
                                "retain_parent" if selected == rnd.parent.artifact_id else "rollback")
    cfg.update(selected_local_rmse=selected_metrics["rmse"], strategy_version_after=next_policy["version"],
               measurement_events=measurements.events)
    cfg["controller_decisions"] = {"plan": planning_calls["decision_source"], "select": selection_calls["decision_source"]}
    if "program_guard" in cfg["controller_decisions"].values():
        change = "Controller decision validation exhausted; program guard retained the actual parent without adopting an unapproved candidate."
    finish_configuration = bounded_configuration(cfg, state, rnd.round_index, "finish")
    record = {"round_index": rnd.round_index, "dataset": dataset, "plan": plan, "selection": selection,
              "configuration": cfg, "selected_artifact_id": selected, "model_call_ids": list(rnd.call_ids),
              "planning_calls": planning_calls, "selection_calls": selection_calls,
              "historical_validation": restored, "host_candidate_receipt": submitted}
    record["host_current_validation"] = rnd.frame.get("current_validation")
    write_json(reference_path, reference)
    # 阶段 10：持久化监测参考、配套策略、诊断及两次决策的真实调用号，供下一轮“读历史”使用。
    write_json(state / "regression_strategy.json", next_policy)
    write_json(state / "monitoring.json", drift)
    write_json(state / "diagnosis.json", diagnosis)
    record = portable_record(record, context="regression_round_record")
    append_jsonl(state / "decisions.jsonl", record)
    write_json(run / "round_summary.json", record)
    measurements.save()
    from ..progress import save_resources, resource_snapshot
    stage_row = stage_evidence.record(rnd, selected, selected_metrics, resource_snapshot(), cfg["selection_action"])
    write_json(state / "regression_round_assessment.json", stage_row)
    save_resources(state)
    with stage("save_state_rpc", dataset=dataset):
        rnd.save_state(state)
    # 阶段 11：保存的是可写状态副本；finish 引用实际读过的历史和本轮真实回执，等待宿主确认后退出。
    with stage("finish_rpc", dataset=dataset):
        rnd.finish(selected_artifact_id=selected, hypothesis=plan["hypothesis"], change_summary=change,
                   reason=selection["reason"], configuration=finish_configuration, seed=seed,
                   history_ids=[item["id"] for item in history], model_call_ids=list(rnd.call_ids))
