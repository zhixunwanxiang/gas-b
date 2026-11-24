"""Workspace-v2 training and feature-only JSONL inference for task A.

The platform supplies the checkpoint and public projection. The local replay
router, archived batches, and packaged experiment weights are not used here.
"""
# 平台闭环主控：校验合同与父产物 → 读历史和状态 → 监测诊断 → AI 规划 → 训练/回滚 → 测量选版 → 保存并结束。
# train 只使用宿主授权的公开数据；predict 独立接收特征，不能依赖训练端标签反馈。
# 模型/上传等 SDK RPC 一旦结果未知必须终止，不能把它们当成本地数值失败继续处理。
from functools import lru_cache
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import numpy as np
from contest_harness.errors import HarnessError

from .sensor_training.data import evaluate, load_pair
from .sensor_training.io import CLASSES, append_jsonl, digest, write_json
from .sensor_training.reference.data import flatten, json_value
from .continuation import continue_candidate
from .frozen_router import load_predictor, checkpoint_paths as artifact_checkpoint_paths
from .monitoring import fit_reference, monitor, diagnose
from .evolution import (load_policy, available_experiments, promotion_gate,
                        fingerprint, advance_policy, validate_controller)
from .controller_prompts import (VERSION as CONTROLLER_PROMPT_VERSION, execution_policy,
                                 enrich_plan, enrich_selection, request_digest)
from .host_validation import parent_reference, candidate_gate as host_candidate_gate, bind_candidate_validation
from .progress import progress, stage

ROOT = Path(__file__).resolve().parents[2]
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
# 冻结的 128 维路径只构造一次；数值与每条输入仍独立校验，不缓存样本或预测结果。
_PREDICTION_PATHS = flatten({"features": [0.0] * 128})[0]
_READONLY_CHECKPOINTS = {}


def _check_task(task, dataset_key):
    # 分别核对任务类型与数据集代号，拒绝把其他赛题按六分类合同处理。
    if task != "classification" or dataset_key not in (None, "A"):
        raise ValueError("This submission supports task A classification only")


@lru_cache(maxsize=1)
def _prediction_model(path, file_identity=None):
    # Cache only within this prediction process. Each measurement gets a new one.
    # 每个预测进程内缓存一次完整预测器；文件时间/大小参与键，避免同路径文件变化后误用旧对象。
    return load_predictor(Path(path), _PREDICTION_PATHS)


def _selected_checkpoint(sample, model_dir):
    # 优先使用宿主指定的产物路径；否则按 SDK 描述选择唯一相对检查点。
    # 多文件产物必须明确主文件，不能依赖目录扫描顺序；描述中的文件名必须为无穿越的相对路径。
    explicit = os.environ.get("CONTEST_ARTIFACT_PATH")
    if explicit:
        return Path(explicit).resolve()
    context = getattr(sample, "context", None) or {}
    paths = context.get("artifact", {}).get("checkpoint_paths", [])
    if not paths:
        return (Path(model_dir) / "checkpoint.npz").resolve()
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
        raise ValueError("CONTEST_ARTIFACT_PATH is required to select among multiple checkpoint files")
    relative = PurePosixPath(paths[0])
    if (relative.is_absolute() or "\\" in paths[0] or not relative.parts
            or any(part in ("..", ".") for part in paths[0].split("/"))):
        raise ValueError("Prediction checkpoint path must be relative to the model artifact")
    return Path(model_dir).joinpath(*relative.parts).resolve()


def _prediction_checkpoint(sample, model_dir):
    # 平台每个预测进程挂载一份不可变产物。只在 /model 且 OS 确认只读时复用文件身份，
    # 避免每条样本反复 resolve/lstat/stat；本地可写模型仍逐条检查，以检测同路径文件更新。
    context = getattr(sample, "context", None) or {}
    explicit = os.environ.get("CONTEST_ARTIFACT_PATH")
    key = (str(model_dir), explicit, context.get("artifact_id"),
           None if explicit else json.dumps(context.get("artifact", {}).get("checkpoint_paths", [])))
    cached = _READONLY_CHECKPOINTS.get(key)
    if cached is not None:
        return cached
    checkpoint = _selected_checkpoint(sample, model_dir)
    info = checkpoint.stat()
    value = checkpoint, (info.st_mtime_ns, info.st_ctime_ns, info.st_size)
    if str(model_dir) == "/model" and checkpoint.is_relative_to(Path("/model")) and hasattr(os, "statvfs"):
        try:
            readonly = bool(os.statvfs(checkpoint).f_flag & getattr(os, "ST_RDONLY", 1))
        except OSError:
            readonly = False
        if readonly:
            if len(_READONLY_CHECKPOINTS) >= 4:
                _READONLY_CHECKPOINTS.pop(next(iter(_READONLY_CHECKPOINTS)))
            _READONLY_CHECKPOINTS[key] = value
    return value


def predict(sample, model_dir):
    # 仅接受 features，校验维度、数值和模型特征顺序后返回六类分数。
    # 这里不读取 answer、历史标签或训练数据；同一预测进程中的特征流状态由模型内部维护。
    _check_task(sample.task, sample.dataset_key)
    inputs = sample.inputs
    if not isinstance(inputs, dict) or set(inputs) != {"features"}:
        raise ValueError("Task A requires only inputs.features")
    features = inputs["features"]
    if (not isinstance(features, list) or len(features) != 128
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in features)):
        raise ValueError("Task A requires 128 finite numeric features")
    checkpoint, identity = _prediction_checkpoint(sample, model_dir)
    model = _prediction_model(str(checkpoint), identity)
    # 结构/标量已在上方检查，路径已在缓存模型首次加载时验证；保持官方 float 转换。
    # 不为每条样本重复生成 128 个 JSON 路径，预测顺序和路由流状态均不改变。
    values = [float(value) for value in features]
    inference = getattr(model, "predict_one", model.predict)
    scores = inference(np.asarray([values], dtype=np.float64))[0].copy()
    if scores.shape != (6,) or not np.isfinite(scores).all() or np.any(scores < 0) or scores.sum() <= 0:
        raise ValueError("Invalid classifier probabilities")
    scores /= scores.sum()
    # The SDK requires a unique top score, including for degenerate checkpoints.
    top = int(scores.argmax())
    # 仅在最高分并列时施加极小扰动，满足 SDK 唯一最高分约束；不会由 id 或标签参与破除平分。
    if np.count_nonzero(scores == scores[top]) > 1:
        scores *= 1 - 1e-12
        scores[top] += 1e-12
    return [{"label": label, "score": float(score)} for label, score in zip(CLASSES, scores)]


def _parent_checkpoint(rnd):
    # 以宿主授权的主检查点摘要定位续训起点；文件还需通过 SDK 声明和摘要校验。
    # 不能用本地默认权重或旧状态记录替代平台选中的父产物。
    expected = rnd.parent.checkpoint_sha256.removeprefix("sha256:")
    for relative in rnd.parent.descriptor["checkpoint_paths"]:
        checkpoint = rnd.parent.path(relative)  # SDK validates declaration and file digest.
        if digest(checkpoint).removeprefix("sha256:") == expected:
            return checkpoint
    raise ValueError("No declared parent checkpoint matches parent_checkpoint_sha256")


def _decision(rnd, request, fields, token_limit, validate, corrections=1):
    # 一次完整返回后才解析并校验 JSON，格式错误允许有限次数的纠正请求。
    # 纠正请求也是真实调用，必须记录其 call_id；RPC 异常发生在 try 之外，不能吞掉后继续会话。
    call_ids, failures, request_hashes = [], [], []
    for attempt in range(corrections + 1):
        # request 只含本轮证据和自有提示；不重复加入 rnd.prompt，官方提示由宿主在请求前附加。
        # Never catch an RPC failure: its outcome may be unknown to the SDK.
        request_hashes.append(request_digest(request))
        with stage("model_rpc", phase=request.get("phase"), attempt=attempt + 1):
            receipt = rnd.model(
                [{"role": "user", "content": json.dumps(request, ensure_ascii=False, allow_nan=False)}],
                # Structured controller decisions must leave room for final JSON.
                # Qwen can otherwise spend the entire authorized budget reasoning.
                max_tokens=min(token_limit, rnd.max_output_tokens), temperature=0, enable_thinking=False,
            )
        call_ids.append(receipt.call_id)
        try:
            raw = receipt.text.strip()
            if raw.startswith("```json\n") and raw.endswith("```"):
                raw = raw[8:-3].strip()
            value = json_value(raw)
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ValueError("Model decision does not match the requested JSON schema")
            if any(not isinstance(value[k], str) or not value[k].strip()
                   or len(value[k].encode("utf-8")) > 4000 for k in fields):
                raise ValueError("Model decision fields must be nonempty text <=4000 bytes")
            validate(value)
            return value, {"model_call_ids": call_ids, "corrections": failures,
                           "prompt_version": request.get("controller_prompt", {}).get("version", CONTROLLER_PROMPT_VERSION),
                           "request_sha256": request_hashes}
        except ValueError as error:
            # 只有拿到完整响应后发现 JSON/选择不合法才允许纠正；它也是一次需要付费和留回执的模型调用。
            # 这与超时重试不同：rnd.model 抛异常时根本不会进入本分支。
            failures.append({"call_id": receipt.call_id, "error": str(error), "response_excerpt": receipt.text[:2000]})
            if attempt == corrections:
                raise
            request = {**request, "correction": {"attempt": attempt + 1, "error": str(error),
                       "previous_completed_response": receipt.text[:2000],
                       "instruction": "The previous completed response failed validation. Return a corrected JSON object using only the supplied schema and choices."}}


def _read_state(state, name):
    # 可选状态不存在时返回 None，由调用方建立首轮默认状态；文件存在但内容损坏则显式报错。
    path = state / name
    return json_value(path.read_text(encoding="utf-8")) if path.is_file() else None


def _restore_history(rnd, history, run, validation, before, controller):
    # 只恢复 start 帧允许的历史 ID，限制数量，跳过当前父模型并优先检查最近版本。
    # 恢复后在本轮公开 validation 重跑完整预测器，再计算回滚资格；不直接信任历史分数。
    receipts = {item["artifact_id"]: item for item in rnd.artifacts}
    order = [item.get("selected_artifact_id") for item in reversed(_history_view(history))]
    order.extend(reversed(list(receipts)))
    restored, seen = {}, {rnd.parent.artifact_id}
    for identifier in order:
        if identifier in seen or identifier not in receipts:
            continue
        seen.add(identifier)
        if len(restored) >= controller["max_history_restores"]:
            break
        receipt = receipts[identifier]
        # Only this submission's feature-only predictor format can be evaluated
        # locally. No historical source is imported or executed.
        if receipt.get("artifact", {}).get("predict_argv") != PREDICT_ARGV:
            continue
        directory = rnd.restore_artifact(identifier, run / f"restored-{len(restored) + 1}")
        try:
            # 历史产物可包含源码，但这里只把它当作数据文件保存，不导入或执行历史 Python 代码。
            names = set(receipt["artifact"]["checkpoint_paths"] + receipt["artifact"]["code_paths"])
            def resolve_file(path):
                relative = Path(path).resolve().relative_to(directory.resolve()).as_posix()
                if relative not in names:
                    raise ValueError("Historical predictor references an undeclared file")
                return directory / relative
            # Our artifact contract declares the primary checkpoint first.
            primary = directory / receipt["artifact"]["checkpoint_paths"][0]
            model = load_predictor(primary, validation["paths"], resolve_file=resolve_file)
            metric = evaluate(model, validation)
            restored[identifier] = {"metrics": metric, "gate": promotion_gate(before, metric, controller),
                                    "checkpoint_sha256": digest(primary), "seed": model.meta["seed"],
                                    "reevaluated_this_round": True, "validation_sha256": validation["sha256"]}
        except (ValueError, OSError, KeyError) as error:
            restored[identifier] = {"error": str(error), "gate": {"eligible": False}}
    return restored


def _history_view(records):
    # 压缩最近十条历史，保留父/数据/配置摘要、门槛和失败证据以支持可比实验判断。
    # 只是缩减模型上下文；最终 finish 仍引用本轮实际 history() 读取的全部记录 ID。
    result = []
    for item in records[-10:]:
        record = item.get("record", item)
        if not isinstance(record, dict):
            record = item
        compact = {"id": item["id"], **{k: record[k] for k in (
            "round_index", "selected_artifact_id") if k in record}}
        if "reason" in record:
            compact["reason"] = str(record["reason"])[:600]
        cfg = record.get("configuration")
        if isinstance(cfg, dict):
            compact["configuration"] = {k: cfg[k] for k in (
                "round_index", "parent_checkpoint_sha256", "train_sha256", "validation_sha256",
                "local_validation_before", "local_validation_after", "selected_local_macro_f1",
                "candidate_gate", "host_candidate_gate", "experiment_failure",
                "execution_policy", "controller_prompt_version") if k in cfg}
            spec = cfg.get("experiment")
            if isinstance(spec, dict):
                compact["configuration"]["experiment"] = {k: spec[k] for k in (
                    "name", "learning_rate", "epochs", "kept_sensors", "freeze") if k in spec}
                compact["configuration"]["experiment_config_sha256"] = request_digest(spec)
            if isinstance(cfg.get("checkpoint_selection"), dict):
                compact["configuration"]["checkpoint_selection"] = {k: cfg["checkpoint_selection"][k] for k in (
                    "mode", "selected_epoch", "selected_epoch_eligible", "eligible_epoch_count", "executed_epoch_count")
                    if k in cfg["checkpoint_selection"]}
        result.append(compact)
    return result


def _snapshot(candidate):
    # Explicit code sources only: no local models, data, replay outputs, or secrets.
    # 按明确源码清单复制可复现代码，避免把本机权重、回放数据、运行目录或密钥混入候选。
    paths = [ROOT / name for name in ("harness.py", "contest_submission.py", "submission_config.json")]
    paths.extend(sorted((ROOT / "src" / "gas_a").rglob("*.py")))
    names = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        names.append(relative.as_posix())
    return names


def train(rnd):
    # 本轮只执行一个经过真实模型规划的动作：observe、experiment 或 rollback。
    # 即使保留父模型也需要真实模型决策、保存记录并等待 SDK 的完成确认；每个回合不等于整个程序只能训练一次。
    _check_task(rnd.task.get("task"), rnd.dataset_key)
    progress("train_started", dataset="A", round_index=rnd.round_index)
    # 1.0.4 has no inline fallback. Fail before data loading or model calls.
    if rnd.limits.get("artifact_upload_protocol") != "chunked-v1":
        raise HarnessError("CHUNKED_UPLOAD_UNSUPPORTED", "SDK 1.0.4 requires a host with chunked-v1 uploads")
    # 阶段 1：加载固定实验菜单与门槛。大模型只能选择允许项，不能在回复里扩大配置权限。
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
    validate_controller(config["controller"])
    # 所有新候选和状态先写独立工作目录；/app、父模型、数据和旧状态挂载均按只读对待。
    work = Path(os.environ.get("CONTEST_WORK", "/workspace")).resolve()
    protected = [ROOT, rnd.parent.root, Path(rnd.data.train_path).parent,
                 Path(rnd.data.validation_path).parent]
    if rnd.state_path:
        protected.append(Path(rnd.state_path))
    protected.extend(Path(p) for p in ("/data", "/previous", "/state", "/model"))
    if any(work.is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError("CONTEST_WORK must be outside read-only code, data, model, and state mounts")
    work.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"round-{rnd.round_index:02d}-", dir=work))
    # 阶段 2：读取宿主实际父产物并在本轮公开验证集测量，后续所有增益都相对这一份完整预测器。
    checkpoint = _parent_checkpoint(rnd)
    with stage("load_public_data", dataset="A"):
        train_data, validation = load_pair(rnd.data.train_path, rnd.data.validation_path)
    def parent_file(path):
        return rnd.parent.path(Path(path).resolve().relative_to(rnd.parent.root.resolve()).as_posix())
    with stage("evaluate_parent", dataset="A"):
        parent = load_predictor(checkpoint, train_data["paths"], resolve_file=parent_file)
        before = evaluate(parent, validation)
    host_parent = parent_reference(rnd.frame.get("current_validation"), before, validation["sha256"])
    # 阶段 3：真正读取平台历史。保存引用的 id 是证明“读过历史”的证据，不能自行编造。
    with stage("history_rpc", dataset="A"):
        history = rnd.history()
    menu = {spec["name"]: spec for spec in config["experiments"]}
    controller = config["controller"]
    execution = execution_policy(controller, config["limits"])
    training_limits = {**config["limits"], "controller_checkpoint_gate": execution["promotion_gate"]}
    # SDK 把只读旧状态复制到可写目录；下面只修改副本，最终通过 save_state 显式提交。
    state = rnd.state_workspace(run / "state")
    # 阶段 4：首轮仅用 train 拟合监测参考，后续沿用，避免参考跟随漂移一起移动而掩盖变化。
    reference = _read_state(state, "monitor_reference.json")
    if reference is None:
        reference = fit_reference(train_data)
    write_json(state / "monitor_reference.json", reference)
    drift = monitor(reference, validation, controller["drift_thresholds"])
    policy = load_policy(_read_state(state, "strategy.json"), menu)
    # State can describe a rejected candidate or a different artifact. It never
    # overrides the host-selected model or carries its strategy across blindly.
    # 旧状态不一定对应本轮父模型：先恢复模型配套策略，无法配对时退回保守偏好。
    paired = policy["artifact_policies"].get(rnd.parent.artifact_id)
    if paired:
        policy.update(paired)
    elif policy["artifact_policies"]:
        policy.update(version=1, preferred_experiment=None)
    diagnosis = diagnose(before, drift, validation["sha256"], policy.get("last_measurement"))
    allowed_menu, blocked = available_experiments(policy, menu, rnd, train_data, validation, controller, execution)
    # 阶段 5：历史版本也必须在当前验证数据上复评；父测量口径不可比时不开放回滚动作。
    with stage("restore_and_evaluate_history", dataset="A"):
        restored = _restore_history(rnd, history, run, validation, before, controller)
    rollback_ids = [identifier for identifier, item in restored.items()
                    if item["gate"]["eligible"] and host_parent["comparable"]]
    actions = ["observe"] + (["experiment"] if allowed_menu else []) + (["rollback"] if rollback_ids else [])

    def validate_plan(value):
        if value["action"] not in actions:
            raise ValueError("Model selected an unavailable action")
        if value["action"] == "experiment" and value["experiment"] not in allowed_menu:
            raise ValueError("Model selected an experiment outside the configured menu or repetition budget")
        if value["action"] != "experiment" and value["experiment"] != "none":
            raise ValueError("Observe and rollback must use experiment=none")
        if value["action"] == "rollback" and value["target_artifact_id"] not in rollback_ids:
            raise ValueError("Rollback target has not passed current-scope validation")
        if value["action"] != "rollback" and value["target_artifact_id"] != rnd.parent.artifact_id:
            raise ValueError("Observe and experiment must refer to the actual parent")

    # 阶段 6：把监测、诊断、历史、已测量回滚目标和允许菜单交给真实模型，先提出可验证假设。
    plan_request = enrich_plan({
        "phase": "plan", "round_index": rnd.round_index,
        "request": "Use drift, diagnosis, history and bounded strategy evidence to choose observation, one continuation experiment, or a validated historical rollback. Observation is appropriate when no useful experiment remains.",
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "parent_artifact_id": rnd.parent.artifact_id,
        "validation_before": before, "history": _history_view(history), "experiments": allowed_menu,
        "monitoring": {key: value for key, value in drift.items() if key != "features"},
        "diagnosis": diagnosis, "historical_validation": restored,
        "strategy": {"version": policy["version"], "preferred_experiment": policy["preferred_experiment"],
                     "blocked_repeated_experiments": blocked},
        "response_schema": {"action": actions, "experiment": ["none", *allowed_menu],
                            "target_artifact_id": [rnd.parent.artifact_id, *rollback_ids],
                            "hypothesis": "testable hypothesis", "reason": "cite measured evidence and uncertainty"},
        "rules": "Return only JSON. Only public training rows update gradients; validation selects epochs and versions. Preserve parent Adam/scaler/steps. Hidden-stream labels or telemetry are unavailable. Do not infer physical causes from PSI alone.",
    }, controller, train_data, validation, execution)
    plan, planning_calls = _decision(rnd, plan_request, ("action", "experiment", "target_artifact_id", "hypothesis", "reason"),
       config["max_output_tokens"], validate_plan, controller["json_corrections"])
    spec = allowed_menu[plan["experiment"]] if plan["action"] == "experiment" else None
    report, submitted, candidate_gate, host_gate, experiment_failure = None, None, None, None, None
    configuration = {"action": plan["action"], "experiment": spec, "round_index": rnd.round_index,
                     "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
                     "train_sha256": train_data["sha256"], "validation_sha256": validation["sha256"],
                     "local_validation_before": before["macro_f1"],
                     "controller_model_name": rnd.model_name, "strategy_version_before": policy["version"],
                     "controller_prompt_version": CONTROLLER_PROMPT_VERSION, "execution_policy": execution,
                     "drift_detected": drift["drift"], "monitor_reference_train_sha256": reference["train_sha256"],
                     "validation_scope": "authorized_public_validation", "promotion_rules": controller,
                     "host_parent_reference": host_parent}
    change = "Observed authorized public features and validation; retained the actual parent without training."
    # 阶段 7：仅 experiment 才执行梯度训练。先持久化 started 记录，失败尝试也不能从实验历史中消失。
    if spec is not None:
        weak_class = diagnosis["weak_classes"][0]["label"]
        candidate = run / "candidate"
        attempt = {"round_index": rnd.round_index, "experiment": spec,
                   "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
                   "train_sha256": train_data["sha256"], "validation_sha256": validation["sha256"],
                   "planning_model_call_ids": planning_calls["model_call_ids"]}
        append_jsonl(state / "experiment_attempts.jsonl", {**attempt, "status": "started"})
        try:
            # Only this CPU/filesystem computation is recoverable here. Never
            # include rnd.model/submit/save/restore RPCs in this exception scope.
            with stage("local_training", dataset="A", experiment=spec["name"]):
                report = continue_candidate(checkpoint, train_data, validation, candidate, spec,
                                            {"weakest_class": weak_class}, training_limits,
                                            expected_parent_sha256=rnd.parent.checkpoint_sha256,
                                            parent_artifact_id=rnd.parent.artifact_id)
        except (TimeoutError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            experiment_failure = {"stage": "local_training", "error_type": type(error).__name__,
                                  "message": str(error)[:1000], "candidate_submitted": False}
            configuration["experiment_failure"] = experiment_failure
            change = "Local continuation experiment failed (%s); rejected partial files and retained the actual parent." % type(error).__name__
            append_jsonl(state / "experiment_attempts.jsonl", {**attempt, "status": "failed", **experiment_failure})
        else:
            append_jsonl(state / "experiment_attempts.jsonl", {**attempt, "status": "trained"})
    # 阶段 8：训练成功才组装候选；路由包必须声明所有专家检查点和清单，不能只上传主权重。
    if report is not None:
        code_paths = _snapshot(candidate)
        if (candidate / "routing.json").is_file():
            code_paths.append("routing.json")
            manifest = json.loads((candidate / "routing.json").read_text(encoding="utf-8"))
            configuration.update(routing_policy=manifest["policy"], routing_manifest_sha256=digest(candidate / "routing.json"))
        configuration["local_validation_after"] = report["validation_after"]["macro_f1"]
        candidate_gate = promotion_gate(before, report["validation_after"], controller)
        configuration["candidate_gate"] = candidate_gate
        configuration["checkpoint_selection"] = report["checkpoint_selection"]
        change = "Continued the actual parent using %s; preserved Adam/scaler and exported epoch %d, step %d." % (
            spec["name"], report["final_epochs"], report["final_steps"])
        write_json(candidate / "planning.json", {"decision": plan, **planning_calls})
        write_json(candidate / "diagnosis.json", diagnosis)
        write_json(candidate / "monitoring.json", drift)
        # 上传后宿主会重新测量并返回回执。此 RPC 位于本地训练异常捕获范围之外，失败时不继续发请求。
        with stage("submit_candidate_rpc", dataset="A"):
            submitted = rnd.submit_candidate(
                candidate, hypothesis=plan["hypothesis"], change_summary=change,
                configuration=configuration, seed=parent.meta["seed"],
                checkpoint_paths=artifact_checkpoint_paths(candidate / "checkpoint.npz"),
                code_paths=code_paths, predict_argv=PREDICT_ARGV,
            )
        host_gate = host_candidate_gate(submitted, host_parent, report["validation_after"],
                                        validation["sha256"], controller,
                                        binding=bind_candidate_validation(rnd, submitted, validation["sha256"]))
        configuration["host_candidate_gate"] = host_gate
    # 阶段 9：程序先收紧可选 ID，父产物始终可保留；新候选必须同时通过本地逐类保护与宿主测量门槛。
    eligible = [rnd.parent.artifact_id]
    if submitted and candidate_gate["eligible"] and host_gate["eligible"]:
        eligible.append(submitted["artifact_id"])
    if plan["action"] == "rollback":
        eligible.append(plan["target_artifact_id"])

    def validate_selection(value):
        if value["selected_artifact_id"] not in eligible:
            raise ValueError("Model selected an unknown or regressing artifact")

    # 阶段 10：再次调用真实模型，依据已发生的测量选择最终版本；不能把规划中的预期收益当作实验结果。
    selection_request = enrich_selection({
        "phase": "select", "round_index": rnd.round_index,
        "request": "Confirm a guarded experiment promotion, observation, or actual historical rollback using the current public validation. Retain the parent if evidence is insufficient.",
        "action": plan["action"], "parent_artifact_id": rnd.parent.artifact_id,
        "candidate_artifact_id": submitted["artifact_id"] if submitted else None,
        "same_local_validation": {"parent": before, "candidate": report["validation_after"] if report else None,
                                  "historical": restored},
        "host_parent_validation": rnd.frame.get("current_validation", {}),
        "host_candidate_receipt": submitted,
        "host_admission": host_gate, "experiment_failure": experiment_failure,
        "response_schema": {"selected_artifact_id": eligible,
                            "reason": "explain measured results and choice"},
        "rules": "Return only JSON. Compare the same metric and dataset scope. Never call local metrics hidden-test results. Prefer retaining the parent on a tie.",
    }, controller, validation["sha256"], candidate_gate, report["checkpoint_selection"] if report else None)
    selection, selection_calls = _decision(rnd, selection_request, ("selected_artifact_id", "reason"),
                                         config["max_output_tokens"], validate_selection, controller["json_corrections"])
    selected = selection["selected_artifact_id"]
    candidate_id = submitted["artifact_id"] if submitted else None
    selected_metric, selected_sha256 = before, rnd.parent.checkpoint_sha256
    selected_seed = parent.meta["seed"]
    if selected == candidate_id:
        selected_metric, selected_sha256 = report["validation_after"], digest(candidate / "checkpoint.npz")
    elif selected != rnd.parent.artifact_id:
        selected_metric, selected_sha256 = restored[selected]["metrics"], restored[selected]["checkpoint_sha256"]
        selected_seed = restored[selected]["seed"]
        change = "Restored and selected historical artifact %s after evaluating its complete predictor on the current public validation." % selected
    elif submitted:
        change = "Retained the actual parent checkpoint; the tested candidate was not adopted."
    # 阶段 11：依据实际采用/拒绝/回滚结果更新策略，实验偏好必须跟最终选中模型配套。
    next_policy = advance_policy(policy, selected=selected, parent_id=rnd.parent.artifact_id,
                                 candidate_id=candidate_id, plan=plan,
                                 attempt_key=fingerprint(rnd.parent.checkpoint_sha256, train_data["sha256"], validation["sha256"], spec, execution) if spec else None,
                                 gain=candidate_gate["macro_f1_gain"] if candidate_gate else None,
                                  validation_sha256=validation["sha256"], selected_f1=selected_metric["macro_f1"],
                                  experiment_failed=experiment_failure is not None)
    configuration.update(selected_checkpoint_sha256=selected_sha256,
                         selected_local_macro_f1=selected_metric["macro_f1"], strategy_version_after=next_policy["version"])
    write_json(state / "strategy.json", next_policy)
    write_json(state / "monitoring.json", drift)
    write_json(state / "diagnosis.json", diagnosis)
    record = {
        "round_index": rnd.round_index, "parent_artifact_id": rnd.parent.artifact_id,
        "candidate_artifact_id": candidate_id, "selected_artifact_id": selected,
        "plan": plan, "selection": selection, "configuration": configuration,
        "model_call_ids": list(rnd.call_ids), "planning_calls": planning_calls, "selection_calls": selection_calls,
        "host_candidate_receipt": submitted, "historical_validation": restored,
        "experiment_failure": experiment_failure, "host_admission": host_gate,
    }
    append_jsonl(state / "decisions.jsonl", record)
    write_json(run / "round_summary.json", record)
    # 阶段 12：先提交完整原生状态，再结束本轮。下一轮父模型仍以宿主 start 为准，不由 state 擅自指定。
    with stage("save_state_rpc", dataset="A"):
        rnd.save_state(state)
    # finish 引用本轮实际读取的历史和真实模型回执，等待宿主确认后由 SDK 结束进程。
    with stage("finish_rpc", dataset="A"):
        rnd.finish(
            selected_artifact_id=selected, hypothesis=plan["hypothesis"],
            change_summary=change,
            reason=selection["reason"], configuration=configuration, seed=selected_seed,
            history_ids=[item["id"] for item in history], model_call_ids=list(rnd.call_ids),
        )
