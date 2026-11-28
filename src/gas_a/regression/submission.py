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
from .evaluation import evaluate, promotion_gate, current_parent_reference, host_gate, GROUP_GUARDS, PARENT_REFERENCE_VERSION
from .monitoring import fit_reference, monitor, diagnose
from .training import continue_candidate
from .prompts import (VERSION as PROMPT_VERSION, plan_contract, metric_evidence, history_evidence,
                      stage_evidence, PLAN_INSTRUCTION, SELECT_INSTRUCTION)
from .diagnostics import compact_metrics
from .stage_reference import measure_stage, compare_stage
from .schedule import ordered_experiments, stage_experiment_preference
from .model import BOUNDED_HEADS, BOUNDED_METHODS, CORRECTION_METHODS, MAX_CORRECTIONS, TRANSFORMS
from .bearing_guard import MAX_PROTOTYPES
from .residual_basis import BASIS_KINDS
from ..evolution import (available_experiments, fingerprint, experiment_outcome,
                         validate_attempts, eligible_history, EXPERIMENT_POLICY_VERSION, REPEAT_POLICY, experiment_evidence)
from ..host_validation import HOST_ADMISSION_VERSION, bind_candidate_validation
from ..progress import progress, stage, record_transfer, save_resources
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


def validate_config(config, limits):
    # 将可执行方案限制在预先配置的范围：AI 只能选菜单，不能临时增加轮数、学习率或放宽门槛。
    # Python 的 bool 是 int 子类，因此预算字段特意用 type(...) is int 排除 true/false。
    rules = config["controller"]
    for key in ("minimum_rmse_absolute_gain", "minimum_rmse_relative_gain", "maximum_mae_relative_increase"):
        value = rules[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid regression promotion threshold")
    if "maximum_nasa_relative_increase" in rules:
        value = rules["maximum_nasa_relative_increase"]
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid NASA protection threshold")
    for guard_name, _ in GROUP_GUARDS.values():
        if guard_name in rules:
            guard = rules[guard_name]
            if (not isinstance(guard, dict) or type(guard.get("minimum_rows")) is not int or guard["minimum_rows"] < 1
                    or any(type(guard.get(key)) not in (int, float) or not np.isfinite(guard[key])
                           or not 0 <= guard[key] <= maximum
                           for key, maximum in (("relative_increase", 1), ("absolute_increase", 1000)))
                    or ("stage_relative_increase" in guard and (
                        type(guard["stage_relative_increase"]) not in (int, float)
                        or not np.isfinite(guard["stage_relative_increase"])
                        or not guard["relative_increase"] <= guard["stage_relative_increase"] <= 1))):
                raise ValueError(f"Invalid regression group protection: {guard_name}")
    if type(config.get("skip_locally_rejected_uploads", True)) is not bool:
        raise ValueError("Invalid local candidate upload policy")
    preferences = config.get("default_first_experiment", {})
    if (not isinstance(preferences, dict)
            or any(dataset not in UNITS or not isinstance(name, str) for dataset, name in preferences.items())):
        raise ValueError("Invalid regression first experiment defaults")
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
                or type(spec["freeze_features"]) is not bool or spec["loss"] not in ("mse", "huber", "nasa_proxy")):
            raise ValueError("Invalid bounded regression experiment")
        names.add(spec["name"])
        for key, low, high in (("device_balance", 0, .5), ("tail_boost", 1, 2), ("huber_quantile", .5, .95),
                               ("nasa_mix", 0, .35), ("short_overestimate_boost", 1, 1.5)):
            if key in spec and (type(spec[key]) not in (int, float) or not np.isfinite(spec[key]) or not low <= spec[key] <= high):
                raise ValueError("Invalid regression weighting or loss parameter")
        if "datasets" in spec and (not isinstance(spec["datasets"], list) or not spec["datasets"] or any(k not in ("B", "C") for k in spec["datasets"])):
            raise ValueError("Invalid regression experiment datasets")
        if spec.get("sampling", "shuffle") not in ("shuffle", "device_condition"):
            raise ValueError("Invalid regression sampling policy")
        if (spec["loss"] == "nasa_proxy" or spec.get("sampling") == "device_condition") and spec.get("datasets") != ["C"]:
            raise ValueError("NASA and condition sampling experiments require C")
        if "nasa_mix" in spec and spec["loss"] != "nasa_proxy":
            raise ValueError("NASA mix requires the NASA proxy loss")
        if "short_overestimate_boost" in spec and (spec["loss"] != "mse" or spec.get("datasets") != ["B"]
                or spec.get("method", "mlp") != "mlp"):
            raise ValueError("Short-life overestimate protection requires B MSE continuation")
        method = spec.get("method", "mlp")
        if method not in ("mlp", *CORRECTION_METHODS, *BOUNDED_METHODS):
            raise ValueError("Invalid regression training method")
        if method != "mlp":
            strengths = spec.get("strengths")
            if (not spec["freeze_features"] or spec.get("sampling", "shuffle") != "shuffle"
                    or not isinstance(strengths, list) or not 1 <= len(strengths) <= 8
                    or any(type(s) not in (int, float) or not np.isfinite(s) or not 0 < s <= 1 for s in strengths)
                    or spec.get("device_balance", 0) or spec.get("tail_boost", 1) != 1 or spec["loss"] != "mse"):
                raise ValueError("Invalid independent correction experiment")
        if method in BOUNDED_METHODS and (spec["epochs"] != 1 or not any(
                spec.get("datasets") == [dataset] and method == bounded for dataset, (_, bounded) in BOUNDED_HEADS.items())):
            raise ValueError("Bounded residual requires one frozen fit for its configured dataset")
        if method in CORRECTION_METHODS:
            datasets = spec.get("datasets", [])
            if (len(datasets) != 1 or spec.get("transform") not in TRANSFORMS[datasets[0]]
                    or spec.get("basis", "linear") not in BASIS_KINDS
                    or type(spec.get("ridge")) not in (int, float) or not np.isfinite(spec["ridge"]) or not 0 < spec["ridge"] <= 10):
                raise ValueError("Invalid independent correction experiment")
    sequences = config.get("preferred_experiment_sequence", {})
    if not isinstance(sequences, dict):
        raise ValueError("Invalid regression experiment sequences")
    for dataset, sequence in sequences.items():
        if (dataset not in UNITS or dataset in preferences or not isinstance(sequence, list)
                or not 1 <= len(sequence) <= 10 or any(not isinstance(name, str) for name in sequence)
                or len(set(sequence)) != len(sequence)):
            raise ValueError("Invalid regression experiment sequence")
    for dataset in (*preferences, *sequences):
        ordered_experiments(config, dataset)


def experiment_menu(config, dataset, parent, train):
    menu = ordered_experiments(config, dataset, mode="configured")
    bounded_key, bounded_method = BOUNDED_HEADS[dataset]
    if dataset == "B":
        bearings = set(train.get("metadata", {}).get("bearing", [None]))
        if None in bearings or "unknown" in bearings:
            menu = {name: s for name, s in menu.items() if not s.get("device_balance", 0)}
        if None in bearings or "unknown" in bearings or not 2 <= len(bearings) <= MAX_PROTOTYPES:
            menu = {name: s for name, s in menu.items() if s.get("method") != bounded_method}
    bounded = parent.meta.get(bounded_key)
    if bounded is not None:
        # 向上限制取决于完整父预测。后续只在训练投影更新时重拟合修正头。
        return {name: s for name, s in menu.items()
                if s.get("method") == bounded_method and bounded["train_sha256"] != train["sha256"]}
    if len(parent.meta.get("corrections", [])) >= MAX_CORRECTIONS:
        menu = {name: s for name, s in menu.items() if s.get("method") not in CORRECTION_METHODS}
    else:
        menu = {name: s for name, s in menu.items() if s.get("method") != bounded_method}
    return menu


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
             local_gate=None, host_gate=None, selection_source="model", completed=False):
    # 依据已经发生的结果更新策略，而不是依据模型在 reason 里声称“有效”。
    # 拒绝/本地失败也计入 attempts，后续用完整实验指纹限制同一个失败方案重复尝试。
    result = deepcopy(policy)
    result["artifact_policies"].setdefault(parent, {k: policy[k] for k in ("version", "preferred_experiment")})
    if candidate is not None or failed or completed:
        adopted = candidate is not None and selected == candidate
        result["attempts"].append({"fingerprint": identity, "experiment": plan["experiment"],
                                   "adopted": adopted, "rmse_reduction": reduction,
                                   "status": "local_training_failed" if failed else "measured",
                                   "candidate_artifact_id": candidate,
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
                           "candidate_gate",
                           "experiment_failure", "selected_local_rmse", "execution_policy", "experiment_outcome") if key in config}}, context="regression_history_view"))
    return result


def _restore_history(rnd, history, run, validation, before, rules, dataset, *, retention_reference=None):
    # 允许恢复的范围只来自当前 start 帧 artifacts；优先最近选中过的版本，跳过实际父版本。
    # 历史旧分数不能直接用于本轮选版：取回后必须在本轮同一 validation 上重新预测并过门槛。
    receipts = {item["artifact_id"]: item for item in rnd.artifacts}
    order = [row.get("selected_artifact_id") for row in reversed(_history_view(history))]
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
            metrics = evaluate(model, validation, dataset)
            restored[identifier] = {"metrics": metrics, "gate": promotion_gate(
                                        before, metrics, rules, retention_reference=retention_reference),
                                    "checkpoint_sha256": digest(path), "seed": model.meta["seed"],
                                    "validation_sha256": validation["sha256"], "reevaluated_this_round": True}
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
        raise HarnessError("CHUNKED_UPLOAD_UNSUPPORTED", "SDK 1.0.5 requires chunked-v1 uploads")
    full_config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
    config = full_config["regression"]
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
    # 阶段 2：从宿主实际父产物加载权重、Adam、scaler、seed，先测父模型，再读历史与状态副本。
    # parent_host 会核对宿主父分数和本地分数是否同一范围，冲突时不开放新版本上线。
    with stage("load_public_data", dataset=dataset):
        train_data, validation = load_pair(rnd.data.train_path, rnd.data.validation_path, dataset)
    with stage("evaluate_parent", dataset=dataset):
        primary_sha = digest(checkpoint)
        parent = load_model(checkpoint, dataset, train_data["paths"])
        parent_prediction = parent.predict(validation["x"])
        before = evaluate(parent, validation, dataset, prediction=parent_prediction)
    with stage("validate_parent_measurement", dataset=dataset):
        parent_host = current_parent_reference(rnd, checkpoint, before, validation["sha256"])
    with stage("history_rpc", dataset=dataset):
        history = rnd.history()
    state = rnd.state_workspace(run / "state")
    with stage("evaluate_stage_start", dataset=dataset):
        stage_identity, stage_metrics = measure_stage(rnd, state, checkpoint, primary_sha, before, validation)
    stage_comparison = {"identity": stage_identity, "validation_sha256": validation["sha256"],
                        "start_metrics": compact_metrics(stage_metrics),
                        "parent_delta": compare_stage(stage_metrics, before)}
    reference_path = state / "regression_monitor_reference.json"
    # 阶段 3：首轮只从 train 建立监测参考；后续校验其来源和结构，保持最初的比较口径。
    # 平台可以更新公开 train，不能因此重拟合父模型 scaler，也不能把合法新投影判成状态损坏。
    # monitor 看输入变没变，diagnose 看公开标签上的误差分段；二者一起给出可验证的实验线索。
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else fit_reference(train_data, dataset)
    drift = monitor(reference, validation, train_data, dataset, full_config["controller"]["drift_thresholds"])
    diagnosis = diagnose(parent, validation, reference, before, drift, prediction=parent_prediction)
    menu = experiment_menu(config, dataset, parent, train_data)
    policy = _policy(state, dataset, {s["name"]: s for s in config["experiments"]})
    # 阶段 4：策略必须与本轮实际父模型配对；换了父产物却没有对应策略时，偏好恢复为初值。
    paired = policy["artifact_policies"].get(rnd.parent.artifact_id)
    if paired:
        policy.update(paired)
    elif policy["artifact_policies"]:
        policy.update(version=1, preferred_experiment=None)
    execution = {"format": "rul.execution.v2", "rules": rules, "limits": limits, "prompt_version": PROMPT_VERSION,
                 "repeat_policy": REPEAT_POLICY, "training_version": "weighted-composite-rul-v3",
                 "skip_locally_rejected_uploads": config.get("skip_locally_rejected_uploads", True),
                 "experiment_policy_version": EXPERIMENT_POLICY_VERSION,
                 "host_admission_policy_version": HOST_ADMISSION_VERSION,
                 "parent_reference_policy_version": PARENT_REFERENCE_VERSION,
                 "monitor_reference_policy": "fixed_initial_training_projection-v1"}
    stage_round = rnd.round_index - stage_identity["round_index"] + 1
    first_experiment = config.get("default_first_experiment", {}).get(dataset) if stage_round == 1 else None
    if first_experiment in menu:
        execution["default_first_experiment"] = first_experiment
    allowed, blocked = available_experiments(policy, menu, rnd, train_data, validation, rules, execution)
    bounded_key, bounded_method = BOUNDED_HEADS[dataset]
    def unavailable_reason(spec):
        if spec.get("method") == bounded_method:
            if dataset == "C":
                return "bounded_head_already_fitted" if parent.meta.get(bounded_key) else "requires_frozen_c_residuals"
            if len(parent.meta.get("corrections", [])) < MAX_CORRECTIONS:
                return "requires_frozen_b_residuals"
            return "bounded_head_already_fitted" if parent.meta.get(bounded_key) else "trace_metadata"
        if parent.meta.get(bounded_key):
            return "bounded_head_requires_frozen_anchor"
        return "correction_capacity" if spec.get("method") in CORRECTION_METHODS else "trace_metadata"

    unavailable = {s["name"]: unavailable_reason(s)
                   for s in config["experiments"] if dataset in s.get("datasets", ["B", "C"]) and s["name"] not in menu}
    unavailable.update(dict.fromkeys(blocked, "repeat_policy"))
    admission_blocker = None
    if not parent_host["comparable"]:
        # 不可核验的父测量使本轮任何候选都不能上线；直接观察并留证，避免重复消耗训练预算。
        admission_blocker = parent_host.get("reason", "parent_measurement_unverified")
        unavailable.update(dict.fromkeys(allowed, "parent_measurement_unverified"))
        allowed = {}
    first_experiment = first_experiment if first_experiment in allowed else None
    scheduled = stage_experiment_preference(config, dataset, stage_round, allowed)
    preferred = scheduled or first_experiment
    if preferred:
        allowed = {preferred: allowed[preferred], **allowed}
    # 指纹包含父模型、train、validation、方案和执行规则；只屏蔽同一身份下反复失败的方案。
    with stage("restore_and_evaluate_history", dataset=dataset):
        # 父测量不可比时历史也不能采用，避免白白传输整份历史产物。
        restored = {} if admission_blocker else _restore_history(
            rnd, history, run, validation, before, rules, dataset, retention_reference=stage_metrics)
    rollback_ids = eligible_history(restored, validation["sha256"], parent_host["comparable"],
                                    {item["artifact_id"] for item in rnd.artifacts})
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

    history_rows = _history_view(history)
    request = {"phase": "plan", "round_index": rnd.round_index, "stage_round": stage_round, "dataset": dataset,
               "task": "regression", "metric": "rmse", "metric_direction": "lower", "target_unit": UNITS[dataset],
               "parent_artifact_id": rnd.parent.artifact_id, "validation_before": metric_evidence(before),
               "parent_host_reference": parent_host,
               "stage_reference": stage_evidence(stage_comparison),
               "admission_blocker": admission_blocker,
               "history": history_evidence(history_rows, validation["sha256"]), "experiments": allowed, "strategy": {
                   "version": policy["version"], "preferred_experiment": policy["preferred_experiment"],
                   "blocked_repeated_experiments": blocked, "unavailable_experiments": unavailable},
               "monitoring": {k: drift[k] for k in ("drift", "changed_features", "training_projection_changed", "rows")},
               "diagnosis": {"rul_segments": [{"rul_group": row["rul_group"], **metric_evidence(row)}
                                             for row in diagnosis["rul_segments"]],
                             "drift_features": drift["top_features"][:4]},
               "promotion_rules": rules,
               "plan_contract": plan_contract(rnd.parent.artifact_id, {"rmse": before["rmse"], "mae": before["mae"]},
                                              restored, rollback_ids, allowed),
               "controller_prompt": {"version": PROMPT_VERSION, "instruction": PLAN_INSTRUCTION},
               "response_schema": {"action": actions, "experiment": ["none", *allowed],
                   "target_artifact_id": [rnd.parent.artifact_id, *rollback_ids], "hypothesis": "非空实验假设", "reason": "非空证据理由"}}
    # 模型需要看到零尝试记录，才能区分未试过的方案与已失败的方案。
    # 可用性与本轮实际菜单一致，旧结果仍保留逐项范围核验。
    request["experiment_evidence"] = experiment_evidence(history_rows, policy["attempts"], menu,
        rnd.parent.checkpoint_sha256, train_data["sha256"], validation["sha256"], execution, rules,
        gain_key="rmse_reduction", blocked=menu.keys() - allowed.keys())
    if first_experiment:
        request["strategy"]["default_first_experiment"] = first_experiment
        request["controller_prompt"]["instruction"] += (
                                "阶段首轮决定做实验时，默认选择strategy.default_first_experiment。"
            "有本轮已核验的证据支持其他合法动作时，可以按证据选择并说明理由。")
    if scheduled:
        request["strategy"]["scheduled_experiment_preference"] = scheduled
        request["controller_prompt"]["instruction"] += (
            "为尽早获得收益，本轮默认优先评估 strategy.scheduled_experiment_preference。"
            "如当轮可比证据支持其他合法动作，按证据决策并说明理由。")
    bounded_name = next((name for name, s in allowed.items() if s.get("method") == bounded_method), None)
    if bounded_name:
        if dataset == "C":
            request["strategy"]["preferred_bounded_experiment"] = bounded_name
            request["controller_prompt"]["instruction"] += (
                "C 已有完整残差父模型时，优先评估 preferred_bounded_experiment 的限幅修正。"
                "该候选必须同时改善 RMSE、MAE 和 NASA。")
        else:
            request["controller_prompt"]["instruction"] += (
                "B 的 bearing_upward_guard 可用时，可依据轴承误差方向选择它。"
                "它冻结完整父模型，对高估风险的上调限幅，且须同时降低 RMSE、MAE 并通过分轴承保护。")
    request["data_scope"] = {"train_sha256": train_data["sha256"], "validation_sha256": validation["sha256"],
                             "train_rows": len(train_data["y"]), "validation_rows": len(validation["y"]), "hidden_labels_available": False}
    request["experiment_guide"] = {name: {"actual_change": s.get("method", "mlp"),
        "freeze_mlp_and_existing_corrections": s.get("method") in (*CORRECTION_METHODS, *BOUNDED_METHODS),
        "device_balance": s.get("device_balance", 0), "tail_boost": s.get("tail_boost", 1),
        "basis": s.get("basis"), "sampling": s.get("sampling", "shuffle"),
        "loss": s["loss"], "transform": s.get("transform"), "huber_quantile": s.get("huber_quantile")}
        for name, s in allowed.items()}
    plan, planning_calls = _controller_decision(rnd, request, ("action", "experiment", "target_artifact_id", "hypothesis", "reason"),
                                     full_config["max_output_tokens"], validate_plan, rules["json_corrections"], state)
    spec = allowed[plan["experiment"]] if plan["action"] == "experiment" else None
    # 保存实际展开的配置，而非只写菜单名称；模型输出不能覆写 spec 中的学习率、epoch 或损失。
    cfg = {"dataset": dataset, "task": "regression", "metric": "rmse", "metric_direction": "lower",
           "target_unit": UNITS[dataset], "action": plan["action"], "experiment": spec,
           "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256, "train_sha256": train_data["sha256"],
           "parent_primary_file_sha256": primary_sha,
           "validation_sha256": validation["sha256"], "validation_before": compact_metrics(before),
           "parent_host_reference": parent_host,
           "stage_reference": stage_comparison,
           **({"scheduled_experiment_preference": scheduled} if scheduled else {}),
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
                                            primary_sha, parent_measurement=(primary_sha, before), retention_reference=stage_metrics)
        except (TimeoutError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            failure = {"stage": "local_training", "error_type": type(error).__name__, "message": str(error)[:1000]}
            append_jsonl(state / "experiment_attempts.jsonl", {"round_index": rnd.round_index, "status": "failed", **failure})
            cfg["experiment_failure"] = failure
            change = "Rejected a failed local regression experiment and retained the actual parent."
        else:
            append_jsonl(state / "experiment_attempts.jsonl", {"round_index": rnd.round_index, "status": "trained"})
    if report is not None:
        # 阶段 7：本地算准入、写诊断证据，并把候选交给宿主真实测量；上传成功不等于准许上线。
        gate = promotion_gate(before, report["validation_after"], rules,
                              require_all_improved=spec.get("method") == bounded_method, retention_reference=stage_metrics)
        cfg.update(validation_after=compact_metrics(report["validation_after"]), candidate_gate=gate)
        stage_comparison["candidate_delta"] = compare_stage(stage_metrics, report["validation_after"])
        write_json(candidate / "monitoring.json", drift)
        write_json(candidate / "diagnosis.json", diagnosis)
        write_json(candidate / "planning.json", {"decision": plan, **planning_calls})
        change = (("Fitted a new correction on the frozen complete parent" if spec.get("method") in (*CORRECTION_METHODS, *BOUNDED_METHODS)
                   else "Continued the complete regression parent") + " using %s; preserved parent Adam/scalers/seed and measured RMSE in %s." % (spec["name"], UNITS[dataset]))
        # SDK 上传在计算异常捕获之外；任何结果未知的 RPC 立即结束会话，不重试或伪造成功。
        if gate["eligible"] or not config.get("skip_locally_rejected_uploads", True):
            code_paths = _snapshot(candidate)
            transfer = record_transfer("candidate", candidate)
            with stage("submit_candidate_rpc", dataset=dataset):
                submitted = rnd.submit_candidate(candidate, hypothesis=plan["hypothesis"], change_summary=change,
                    configuration=bounded_configuration(cfg, state, rnd.round_index, "candidate"), seed=parent.meta["seed"], checkpoint_paths=["checkpoint.npz"],
                    code_paths=code_paths, predict_argv=PREDICT_ARGV)
            transfer["acknowledged"] = True
            admission = host_gate(submitted, parent_host, report["validation_after"], validation["sha256"], rules,
                                  binding=bind_candidate_validation(rnd, submitted, validation["sha256"]))
            cfg["host_candidate_gate"] = admission
        else:
            cfg["candidate_upload_skipped"] = "local_promotion_gate_rejected"
            # 保存真实本地报告而不伪造 artifact_id 或宿主测量回执。
            write_json(state / f"local-training-{rnd.round_index}.json", report)
            change = "Completed and locally rejected the regression experiment; retained the actual parent."
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
        "same_local_validation": {"parent": metric_evidence(before), "candidate": metric_evidence(report["validation_after"]) if report else None},
        "historical_admission": {key: {"eligible": key in selection_history_ids,
            "reevaluated_this_round": value.get("reevaluated_this_round") is True,
            "validation_sha256": value.get("validation_sha256"), "gate": value["gate"],
            "fresh_metrics": metric_evidence(value.get("metrics"))} for key, value in restored.items()},
        "host_candidate_gate": admission, "candidate_gate": gate,
        "parent_host_reference": parent_host, "experiment_failure": failure, "promotion_rules": rules,
        "stage_reference": stage_evidence(stage_comparison),
        "controller_prompt": {"version": PROMPT_VERSION, "instruction": SELECT_INSTRUCTION},
        "response_schema": {"selected_artifact_id": eligible, "reason": "依据实际回归测量的非空理由"}}
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
                           local_gate=gate, host_gate=admission, selection_source=selection_calls["decision_source"], completed=report is not None)
    if report is not None or failure:
        cfg["experiment_outcome"] = {key: next_policy["attempts"][-1][key] for key in (
            "outcome", "reason", "counts_toward_rejection_limit")}
    cfg["selection_action"] = ("adopt_candidate" if selected == candidate_id else
                                "retain_parent" if selected == rnd.parent.artifact_id else "rollback")
    cfg.update(selected_local_rmse=selected_metrics["rmse"], strategy_version_after=next_policy["version"])
    stage_comparison["selected_delta"] = compare_stage(stage_metrics, selected_metrics)
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
    write_json(state / "parent-validation-metrics.json", before)
    record = portable_record(record, context="regression_round_record")
    append_jsonl(state / "decisions.jsonl", record)
    write_json(run / "round_summary.json", record)
    save_resources(state)
    with stage("save_state_rpc", dataset=dataset):
        rnd.save_state(state)
    # 阶段 11：保存的是可写状态副本；finish 引用实际读过的历史和本轮真实回执，等待宿主确认后退出。
    with stage("finish_rpc", dataset=dataset):
        rnd.finish(selected_artifact_id=selected, hypothesis=plan["hypothesis"], change_summary=change,
                   reason=selection["reason"], configuration=finish_configuration, seed=seed,
                   history_ids=[item["id"] for item in history], model_call_ids=list(rnd.call_ids))
