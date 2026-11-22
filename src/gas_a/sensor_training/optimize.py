"""基于原始 NumPy MLP 的受限续训：保留 Adam 状态，记录逐 epoch 证据并执行质量门槛。"""
# 数值训练核心：所有梯度只从 train 计算，validation 仅用于逐 epoch 测量和检查点选择。
# 候选继承父权重、Adam 一二阶矩、scaler、seed 和计数器；冻结参数连同其动量一起保持不变。
# 平台路径在每个 epoch 使用与最终晋升相同的类别保护门槛，避免导出最高分却不合格的版本。
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
from .data import load_model, evaluate
from .io import CLASSES, append_jsonl, digest, stamp, write_json
from .reference.model import _THREADS
from .adaptive import fixed_gate
from .architecture import MigratedModel, validate_request, migrate_parent, shape as architecture_shape


FREEZE_MODES = ("none", "features", "head", "first_hidden")


def validate_freezing(spec):
    """冻结范围只描述本次训练，不修改检查点格式或父模型状态。"""
    mode, names = spec.get("freeze", "none"), spec.get("frozen_parameters", [])
    if not isinstance(mode, str) or mode not in FREEZE_MODES:
        raise ValueError("Invalid freeze mode")
    allowed = {f"{kind}{layer}" for kind in ("w", "b") for layer in range(1, 6)}
    if (not isinstance(names, list) or any(not isinstance(n, str) or n not in allowed for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("Invalid frozen_parameters")
    freezes_first = mode in ("features", "first_hidden") or "w1" in names
    if freezes_first and "kept_sensors" in spec:
        raise ValueError("Frozen w1 cannot also be changed by kept_sensors projection")


def frozen_parameters(model, spec):
    """按实际父网络解析冻结张量；w2/b2始终是输出头，w3及以后为残差层。"""
    validate_freezing(spec)
    names = set(spec.get("frozen_parameters", []))
    if names - model.params.keys():
        raise ValueError("Frozen parameter does not exist in the actual parent architecture")
    mode = spec.get("freeze", "none")
    if mode == "features":
        names.update(model.params.keys() - {"w2", "b2"})
    elif mode == "head":
        names.update(("w2", "b2"))
    elif mode == "first_hidden":
        names.update(("w1", "b1"))
    if names == model.params.keys():
        raise ValueError("Freezing every parameter leaves no trainable parameters")
    return names


def frozen_state_hashes(model, names):
    """逐字节核验冻结权重与Adam一、二阶矩，不能只检查梯度是否为零。"""
    return {group: {name: "sha256:" + hashlib.sha256(getattr(model, group)[name].tobytes()).hexdigest()
                    for name in sorted(names)} for group in ("params", "first", "second")}


def bounded_experiment(spec, limits):
    """校验候选配置的数值范围、epoch 预算和权重约束，拒绝改变默认预测预处理的方案。"""
    result = dict(spec)
    if "loss" in result:
        if result["loss"] not in ("cross_entropy", "focal"):
            raise ValueError("Unsupported loss")
        if result["loss"] == "cross_entropy" and result.get("focal_gamma", 0) != 0:
            raise ValueError("cross_entropy conflicts with nonzero focal_gamma")
        if result["loss"] == "focal":
            result.setdefault("focal_gamma", 1.)
    validate_freezing(result)
    if "architecture" in result:
        validate_request(result["architecture"])
    if (result.get("input_transform", "identity") != "identity" or any(
            name in result for name in ("active_sensors", "active_feature_types", "sensor_weights"))):
        raise ValueError("Custom prediction preprocessing is disabled; use kept_sensors as an explicit weight constraint")
    for name, lo, hi in [("learning_rate", 1e-6, .01), ("balance", 0, 1), ("weak_boost", .1, 5), ("l2", 0, .01)]:
        v = result.get(name)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"Invalid {name}")
    if type(result.get("epochs")) is not int or not 1 <= result["epochs"] <= limits["max_epochs"]:
        raise ValueError("Invalid candidate epoch budget")
    if type(result.get("minimum_epochs", 1)) is not int or not 1 <= result.get("minimum_epochs", 1) <= result["epochs"]:
        raise ValueError("Invalid minimum training epochs")
    if not isinstance(result.get("name"), str) or not result["name"].replace("_", "").isalnum():
        raise ValueError("Invalid experiment name")
    for name, default, lo, hi in [("focal_gamma", 0, 0, 2), ("anchor_l2", 0, 0, 1),
                                  ("feature_lr_scale", 1, .05, 1), ("final_lr_ratio", 1, .05, 1),
                                  ("input_noise", 0, 0, 1), ("sensor_dropout", 0, 0, .5),
                                  ("sensor_gain", 0, 0, 1.5), ("sensor_shift", 0, 0, 2),
                                  ("response_gain", 0, 0, 1.5), ("response_sensor_gain", 0, 0, 1),
                                  ("pair_weight", 0, 0, 5), ("pair_margin", 0, 0, 4),
                                  ("adversarial_epsilon", 0, 0, 2)]:
        value = result.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"Invalid {name}")
    if result.get("schedule", "constant") not in ("constant", "cosine"):
        raise ValueError("Invalid learning-rate schedule")
    if result.get("input_transform", "identity") not in ("identity", "signed_log", "signed_sqrt", "signed_fourth_root",
                                                        "relative_response", "relative_sensor", "relative_sample"):
        raise ValueError("Invalid input transform")
    if "batch_size" in result and (type(result["batch_size"]) is not int or not 16 <= result["batch_size"] <= 256):
        raise ValueError("Invalid batch size")
    if result.get("pair_other", "Ethanol") not in CLASSES:
        raise ValueError("Invalid confusion-pair class")
    boosts = result.get("class_boosts", {})
    if not isinstance(boosts, dict) or any(name not in CLASSES or isinstance(value, bool)
           or not isinstance(value, (int, float)) or not math.isfinite(value) or not .1 <= value <= 5
           for name, value in boosts.items()):
        raise ValueError("Invalid class boosts")
    for name, lo, hi in (("kept_sensors", 1, 16),):
        if name in result:
            values = result[name]
            if (not isinstance(values, list) or not values or any(type(v) is not int or not lo <= v <= hi for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"Invalid {name}")
    if "sensor_weights" in result:
        weights = result["sensor_weights"]
        if (not isinstance(weights, list) or len(weights) != 16 or not any(weights)
                or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w)
                       or not 0 <= w <= 2 for w in weights)):
            raise ValueError("Invalid sensor weights")
    return result


def classification_delta(probability, target, sample_weights, gamma=0):
    """计算加权 softmax Focal Loss 及其关于 logits 的精确导数。"""
    # gamma=0 退化为加权交叉熵；gamma>0 调低容易样本的贡献，delta 是对 logits 的梯度。
    pt = np.maximum(probability[np.arange(len(target)), target], 1e-300)
    log_pt = np.log(pt)
    remaining = np.maximum(1 - pt, 1e-12)
    modulation = remaining ** gamma
    loss = -float(np.sum(sample_weights * modulation * log_pt))
    factor = modulation * (1 - gamma * pt * log_pt / remaining)
    delta = probability.copy()
    delta[np.arange(len(target)), target] -= 1
    delta *= (sample_weights * factor)[:, None] / len(target)
    return loss, delta


def confusion_pair_delta(logits, target, weights, weak, other, strength, margin):
    """仅对两个易混淆类别的训练样本计算辅助 pair loss。"""
    selected = (target == weak) | (target == other)
    sign = np.where(target == weak, 1., -1.)
    residual = margin - sign * (logits[:, weak] - logits[:, other])
    factor = strength * weights * selected
    loss = float(np.sum(factor * np.logaddexp(0, residual)))
    derivative = -sign * factor * np.exp(-np.logaddexp(0, -residual)) / len(target)
    delta = np.zeros_like(logits)
    delta[:, weak], delta[:, other] = derivative, -derivative
    return loss, delta


def augment_training_batch(batch, spec, rng):
    """仅在训练时施加扰动；标准化后的零值对应原始训练均值。"""
    if not any(spec.get(k, 0) for k in ("input_noise", "sensor_dropout", "sensor_gain", "sensor_shift")):
        return batch
    shaped = batch.copy().reshape(-1, 16, 8)
    shape = (len(batch), 16, 1)
    if spec.get("sensor_gain", 0):
        sigma = spec["sensor_gain"]
        shaped *= np.exp(rng.normal(-sigma**2 / 2, sigma, size=shape))
    if spec.get("sensor_shift", 0):
        shaped += rng.normal(0, spec["sensor_shift"], size=shape)
    if spec.get("sensor_dropout", 0):
        shaped *= rng.random(shape) >= spec["sensor_dropout"]
    result = shaped.reshape(batch.shape)
    if spec.get("input_noise", 0):
        result += rng.normal(0, spec["input_noise"], size=result.shape)
    return np.clip(result, -10, 10)


def augment_response(raw, model, spec, rng):
    """对训练响应施加扰动，再使用保持不变的父模型 scaler。
    
    UCI 数据中每个传感器的第二维特征为 Rmax/R0，无响应时取值为 1；
    delta-R 和 EMA 特征的无响应值则为 0。
    此操作用于验证鲁棒性假设，不表示已拟合出物理漂移模型。"""
    shaped = raw.copy().reshape(-1, 16, 8)
    gain = np.ones((len(raw), 16, 1))
    for name, shape in (("response_gain", (len(raw), 1, 1)), ("response_sensor_gain", gain.shape)):
        sigma = spec.get(name, 0)
        if sigma:
            gain *= np.exp(rng.normal(-sigma**2 / 2, sigma, size=shape))
    shaped[:, :, 1] -= 1
    shaped *= gain
    shaped[:, :, 1] += 1
    return model.normalized(shaped.reshape(raw.shape))


def recall_target_score(metrics, config):
    """将各目标类别的 recall 按配置下限归一化，取最弱目标用于 epoch 排序。"""
    targets = config.get("minimum_class_recall", {})
    return min((min(1., metrics.get("per_class", {}).get(name, {}).get("recall", 0) / value)
                for name, value in targets.items() if value > 0), default=1.)


def _mask_batch(batch, model):
    if model.meta.get("active_features") is not None:
        batch = batch.copy()
        mask = np.zeros(128, dtype=bool)
        mask[model.meta["active_features"]] = True
        batch[:, ~mask] = 0
    return batch


def fit_weighted(model, train, spec, weak_class, deadline, *, on_epoch=None,
                 epoch_offset=0, epochs_to_run=None, anchor_params=None):
    """仅 train.x/train.y 参与梯度计算，保留父模型的标准化参数和 Adam 状态。"""
    # 先处理受限传感器权重投影，再按累计 epoch 派生随机序列，逐 batch 更新可训练参数。
    # on_epoch 只负责评估和是否早停；验证标签不会流回反向传播或 Adam 更新。
    if (spec.get("input_transform", "identity") != "identity" or any(
            name in spec for name in ("active_sensors", "active_feature_types", "sensor_weights"))):
        raise ValueError("Legacy input transformations are unsupported")
    if (model.meta.get("input_transform", "identity") != "identity" or model.meta.get("active_features") is not None
            or model.meta.get("sensor_weights") is not None):
        raise ValueError("Training requires the original reference preprocessing contract")
    frozen = frozen_parameters(model, spec)  # 所有冻结配置在第一次状态修改之前校验。
    kept = spec.get("kept_sensors")
    fixed_rows = np.zeros(128, dtype=bool)
    if kept is not None:
        fixed_rows[:] = True
        fixed_rows[np.array([(sensor - 1) * 8 + kind for sensor in kept for kind in range(8)])] = False
        # 对权重实施显式投影，输入保持不变；父模型的 Adam 矩、
        # seed、计数器和 scaler 均完整保留。
        model.params["w1"][fixed_rows] = 0
    x, y = model.normalized(train["x"]), train["y"]
    counts = np.bincount(y, minlength=6)
    # balance 控制逆频率加权强度，weak_boost 额外关注诊断出的弱类。
    # 最后将训练样本的平均权重归一化为 1，避免整体改变梯度尺度。
    weights = (len(y) / (6 * np.maximum(counts, 1))) ** spec["balance"]
    weights[CLASSES.index(weak_class)] *= spec["weak_boost"]
    for name, boost in spec.get("class_boosts", {}).items():
        weights[CLASSES.index(name)] *= boost
    weights /= float(weights[y].mean())
    anchor = anchor_params if anchor_params is not None else {k: v.copy() for k, v in model.params.items()}
    model.meta["batch_size"] = spec.get("batch_size", model.meta["batch_size"])
    model.meta["learning_rate"] = spec["learning_rate"]
    epoch_records = []
    for local_epoch in range(spec["epochs"] if epochs_to_run is None else epochs_to_run):
        if time.monotonic() >= deadline:
            raise TimeoutError("Candidate exceeded CPU wall-time budget")
        # 按累计 epoch 派生随机序列，使分段续训可重放；增强使用独立随机流。
        rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
        aug_rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"], 2718]))
        order = rng.permutation(len(x))
        loss_sum = 0.0
        progress = (epoch_offset + local_epoch) / max(1, spec["epochs"] - 1)
        ratio = spec.get("final_lr_ratio", 1)
        rate = spec["learning_rate"] * (ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2
                                        if spec.get("schedule", "constant") == "cosine" else 1)
        model.meta["learning_rate"] = rate
        for start in range(0, len(order), model.meta["batch_size"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("Candidate exceeded CPU wall-time budget")
            ids = order[start:start + model.meta["batch_size"]]
            batch, target, sample_weights = x[ids], y[ids], weights[y[ids]]
            if spec.get("response_gain", 0) or spec.get("response_sensor_gain", 0):
                batch = augment_response(train["x"][ids], model, spec, aug_rng)
            batch = _mask_batch(augment_training_batch(batch, spec, aug_rng), model)
            if spec.get("adversarial_epsilon", 0):
                adv_hidden, adv_logits = model.forward(batch)
                adv_probability = np.exp(adv_logits - adv_logits.max(axis=1, keepdims=True))
                adv_probability /= adv_probability.sum(axis=1, keepdims=True)
                _, adv_delta = classification_delta(adv_probability, target, sample_weights)
                input_gradient = (model.backward(batch, adv_delta)[1] if isinstance(model, MigratedModel) else
                                  ((adv_delta @ model.params["w2"].T) * (1 - adv_hidden**2)) @ model.params["w1"].T)
                adversarial = np.clip(batch + spec["adversarial_epsilon"] * np.sign(input_gradient), -10, 10)
                batch = _mask_batch(np.where(aug_rng.random((len(batch), 1)) < .5, adversarial, batch), model)
            hidden, logits = model.forward(batch)
            # softmax 先减去行最大值，避免指数溢出；概率分布保持不变。
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
            loss, delta = classification_delta(probability, target, sample_weights, spec.get("focal_gamma", 0))
            if spec.get("pair_weight", 0):
                if weak_class == spec.get("pair_other", "Ethanol"):
                    raise ValueError("Confusion pair requires distinct classes")
                pair_loss, pair_delta = confusion_pair_delta(logits, target, sample_weights, CLASSES.index(weak_class),
                                                             CLASSES.index(spec.get("pair_other", "Ethanol")),
                                                             spec["pair_weight"], spec.get("pair_margin", 0))
                loss += pair_loss
                delta += pair_delta
            loss_sum += loss
            if isinstance(model, MigratedModel):
                gradient, _ = model.backward(batch, delta)
                for name in gradient:
                    if name.startswith("w"):
                        gradient[name] += spec["l2"] * model.params[name]
            else:
                gradient = {"w2": hidden.T @ delta + spec["l2"] * model.params["w2"], "b2": delta.sum(axis=0)}
                dh = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
                gradient.update(w1=batch.T @ dh + spec["l2"] * model.params["w1"], b1=dh.sum(axis=0))
            # l2 约束权重幅度；anchor_l2 约束偏离本轮父参数的程度。
            for name in gradient:
                gradient[name] += spec.get("anchor_l2", 0) * (model.params[name] - anchor[name])
            gradient["w1"][fixed_rows] = 0  # 固定系数位置的梯度投影为零。
            # 冻结张量不参与裁剪、Adam矩更新或参数更新。仅把梯度置零仍会因旧动量移动权重。
            for name in frozen:
                del gradient[name]
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in gradient.values()))
            if not math.isfinite(norm):
                raise ValueError("Non-finite candidate gradient")
            # Adam 的 bias correction 沿用父模型累计 steps，不能在候选开始时重置。
            model.meta["steps"] += 1
            step = model.meta["steps"]
            for name, value in gradient.items():
                value *= min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = .9 * model.first[name] + .1 * value
                model.second[name] = .999 * model.second[name] + .001 * value**2
                m = model.first[name] / (1 - .9**step)
                v = model.second[name] / (1 - .999**step)
                layer_scale = spec.get("feature_lr_scale", 1) if name not in ("w2", "b2") else 1
                model.params[name] -= rate * layer_scale * m / (np.sqrt(v) + 1e-8)
            model.params["w1"][fixed_rows] = 0  # 每次 Adam 更新后重新施加权重约束。
        model.meta["epochs"] += 1
        model.validate()
        record = {"epoch": model.meta["epochs"], "steps": step, "loss": loss_sum / len(y), "learning_rate": rate}
        if frozen:
            record["frozen_parameters"] = sorted(frozen)
        keep_going = on_epoch(record) if on_epoch is not None else None
        epoch_records.append(record)
        if keep_going is False:
            break
    return epoch_records, weights.tolist()


def train_candidate(parent, train, validation, directory, spec, diagnosis, limits):
    """为单次候选创建独立目录；任何训练异常均写入 failure.json 并保留已消耗资源。"""
    # 独立新目录保证候选不会覆盖父产物或其他实验；失败报告保留已发生的资源和配置证据。
    spec = bounded_experiment(spec, limits)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        return _train_candidate(parent, train, validation, directory, spec, diagnosis, limits, started, cpu_started)
    except Exception as exc:
        write_json(directory / "failure.json", {"format": "gas-a.failed-experiment.v1", "configuration": spec,
                   "error": f"{type(exc).__name__}: {exc}", "resources": _resources(started, cpu_started),
                   "created_at": stamp(), "official_scoring_ready": False})
        raise


def _resources(started, cpu_started):
    """记录当前进程已发生的 wall time 和 CPU time；平台未提供的训练额度保留为未知。"""
    return {"wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu_started,
            "device": "cpu", "local_gpu_training_seconds": 0, "platform_training_credits": None,
            "new_label_requests": 0, "scope": "本进程候选加载、评估、续训及导出，计至报告写入前；平台额度另行计量"}


def _train_candidate(parent, train, validation, directory, spec, diagnosis, limits, attempt_started, cpu_started):
    """从父模型完整状态续训，保存源码快照、逐 epoch 证据和最终选定的 checkpoint。"""
    # 训练前先快照父模型和源码；逐 epoch 记录门槛及完整状态，选择后导出、重载复评。
    # 已执行的训练量与最终导出 epoch 分开记录，早停或回选最佳 epoch 不会抹去实际成本。
    parent_sha = digest(parent)
    model = load_model(parent, train["paths"])
    # Continue the primary neural state; a new bundle is explicitly rebuilt later.
    model.meta.pop("prediction_bundle", None)
    mean, scale = model.mean.copy(), model.scale.copy()
    initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
    initial_transform = model.meta.get("input_transform", "identity")
    source_root = Path(__file__).parent
    source_names = tuple(sorted(p.relative_to(source_root).as_posix() for p in source_root.rglob("*.py")))
    for name in source_names:
        target = directory / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / name).read_bytes())
    stage_id = directory.parent.name + "/" + directory.name
    before = evaluate(model, validation)
    train_before = evaluate(model, train)
    migration = None
    # 可选架构变化先保函数迁移，并在 train/validation 上验证续训前预测一致，不能重新初始化后冒称续训。
    if "architecture" in spec:
        parent_model = model
        model, migration = migrate_parent(parent_model, spec["architecture"], parent_sha)
        probes = []
        for name, dataset in (("train", train), ("validation", validation)):
            old_scores, new_scores = parent_model.predict(dataset["x"]), model.predict(dataset["x"])
            error = float(np.max(np.abs(old_scores - new_scores)))
            identical_labels = bool(np.array_equal(old_scores.argmax(axis=1), new_scores.argmax(axis=1)))
            if not np.allclose(old_scores, new_scores, rtol=0, atol=1e-12) or not identical_labels:
                raise ValueError("Architecture migration changed parent predictions before continuation")
            probes.append({"dataset": name, "rows": len(dataset["x"]), "max_probability_error": error,
                           "labels_identical": identical_labels})
        migration["prediction_checks"] = probes
        model.save(directory / "migrated_parent.npz")
        migration["migrated_parent_sha256"] = digest(directory / "migrated_parent.npz")
        write_json(directory / "migration.json", migration)
    frozen = frozen_parameters(model, spec)
    frozen_before = frozen_state_hashes(model, frozen)
    best, best_key, best_record = None, None, None
    stale = 0
    select_best = limits.get("select_best_epoch", False)
    patience = limits.get("early_stopping_patience", 0)
    if type(select_best) is not bool or type(patience) is not int or patience < 0:
        raise ValueError("Invalid checkpoint selection settings")

    def record_epoch(record):
        # 平台 controller_checkpoint_gate 优先于本地其他 gate，先按是否合格排序，再比较 F1/Accuracy。
        # best 保存整个模型副本，包含 Adam 和计数器；不能只把最佳权重替换进最后一轮优化器状态。
        nonlocal best, best_key, best_record, stale
        # 在下一次参数更新前，评估当前 epoch 对应的模型。
        record.update(stage_id=stage_id, seed=model.meta["seed"], step=record["steps"],
                      validation=evaluate(model, validation))
        controller_policy = limits.get("controller_checkpoint_gate")
        policy = limits.get("checkpoint_gate")
        # 平台训练路径使用正式控制器门槛；即使某 epoch 的 F1 更高，只要类别保护不合格就不能排在合格 epoch 前。
        if controller_policy is not None:
            from ..evolution import promotion_gate
            measured_gate = promotion_gate(limits.get("controller_checkpoint_reference", before),
                                            record["validation"], controller_policy)
            eligible = measured_gate["eligible"]
            record["controller_gate"] = measured_gate
            key = (eligible, record["validation"]["macro_f1"], record["validation"]["accuracy"])
        else:
            decision = (fixed_gate(record["validation"], before, limits["checkpoint_reference"], policy)
                        if policy and policy.get("adaptive") else
                        gate(record["validation"], before, before, policy) if policy else {"accepted": True, "reasons": []})
            eligible = decision["accepted"]
            hard_passed = not any(reason != "macro_f1_gain_below_minimum" and not reason.startswith("class_recall_below_minimum:")
                                  for reason in decision["reasons"] if reason != "quality_gates_passed")
            key = (eligible, hard_passed, recall_target_score(record["validation"], policy or {}),
                   record["validation"]["macro_f1"], record["validation"]["accuracy"])
            if policy and policy.get("adaptive"):
                key = (eligible, record["validation"]["macro_f1"], record["validation"]["accuracy"])
        record["checkpoint_eligible"] = eligible
        append_jsonl(directory / "history.jsonl", record)
        # 先比较门槛与目标 recall，再比较 Macro-F1 和 Accuracy。
        # 仅在严格更优时替换，因此同分保留更早的完整 checkpoint。
        if best_key is None or key > best_key:
            best_key, best_record, stale = key, record, 0
            if select_best:
                best = copy.deepcopy(model)  # 同时包含 Adam m/v、计数器、seed 和 scaler。
        else:
            stale += 1
        # 早停同时受 minimum_epochs 约束，保证需要较长收敛过程的方案得到最低训练量。
        return not (select_best and patience and stale >= patience
                    and record["epoch"] - initial_epochs >= spec.get("minimum_epochs", 1))

    started = time.monotonic()
    epoch_records, weights = fit_weighted(model, train, spec, diagnosis["weakest_class"],
                                        started + limits["max_seconds_per_candidate"], on_epoch=record_epoch)
    executed_steps, executed_epochs = model.meta["steps"], model.meta["epochs"]
    # 导出最佳 epoch 时恢复权重、Adam 矩与计数器；实际执行量单独保留用于成本核算。
    # 最佳 epoch 选择只改变导出状态；实际训练已执行的 epoch/step 另记，不回退成本记录。
    if select_best:
        model = best
    if not np.array_equal(mean, model.mean) or not np.array_equal(scale, model.scale):
        raise ValueError("Parent standardizer changed")
    if model.meta["steps"] <= initial_steps:
        raise ValueError("No optimizer steps were performed")
    after = best_record["validation"] if select_best else epoch_records[-1]["validation"]
    model.save(directory / "checkpoint.npz")
    # 选择版本之前，核验实际导出的 checkpoint。
    reloaded = load_model(directory / "checkpoint.npz", train["paths"])
    frozen_after = frozen_state_hashes(reloaded, frozen)
    if frozen_before != frozen_after:
        raise ValueError("Frozen weights or Adam state changed")
    if abs(evaluate(reloaded, validation)["macro_f1"] - after["macro_f1"]) > 1e-12:
        raise ValueError("Reloaded checkpoint prediction changed")
    projected_count = ((16 - len(spec["kept_sensors"])) * 8 * model.params["w1"].shape[1]
                       if "kept_sensors" in spec else 0)
    report = {"format": "gas-a.experiment.v1", "name": spec["name"], "configuration": spec,
              "stage_id": stage_id, "initial_epochs": initial_epochs, "final_epochs": model.meta["epochs"],
              "parent_checkpoint_sha256": parent_sha, "checkpoint_sha256": digest(directory / "checkpoint.npz"),
              "seed": model.meta["seed"], "initial_steps": initial_steps, "final_steps": model.meta["steps"],
              "architecture": architecture_shape(model), "architecture_migration": migration,
              "freezing": {"mode": spec.get("freeze", "none"), "scope": "this_experiment_only",
                           "frozen_parameters": sorted(frozen), "trainable_parameters": sorted(model.params.keys() - frozen),
                           "frozen_parameter_count": sum(model.params[n].size for n in frozen),
                           "trainable_parameter_count": sum(v.size for n, v in model.params.items() if n not in frozen) - projected_count,
                           "projected_parameter_count": projected_count,
                           "partly_projected_parameters": ["w1"] if projected_count else [],
                           "state_origin": "actual_parent_after_optional_architecture_migration",
                           "before_sha256": frozen_before, "exported_sha256": frozen_after,
                           "verified_unchanged": True if frozen else None,
                           "optimizer_step_policy": "Global steps advance; frozen m/v remain exact; unfreezing resumes them without reset."},
              "executed_steps": executed_steps, "executed_epochs": executed_epochs,
              "checkpoint_selection": {"mode": ("best_controller_eligible_epoch" if "controller_checkpoint_gate" in limits
                                                else "best_guarded_validation_epoch") if select_best else "last_epoch",
                                       "reference_metrics": limits.get("controller_checkpoint_reference", limits.get("checkpoint_reference")),
                                       "reference_sha256": limits.get("controller_checkpoint_reference_sha256", limits.get("checkpoint_reference_sha256")),
                                       "selected_epoch": model.meta["epochs"], "executed_epoch_count": len(epoch_records),
                                       "requested_epoch_count": spec["epochs"], "patience": patience,
                                       "quality_gate": limits.get("controller_checkpoint_gate", limits.get("checkpoint_gate")),
                                       "selected_epoch_eligible": (best_record if select_best else epoch_records[-1])["checkpoint_eligible"],
                                       "eligible_epoch_count": sum(r["checkpoint_eligible"] for r in epoch_records),
                                       "stopped_early": len(epoch_records) < spec["epochs"]},
              "input_transform_before": initial_transform, "input_transform_after": model.meta.get("input_transform", "identity"),
              "parameter_constraint": {"kind": "projected_first_layer_rows", "kept_sensors": spec.get("kept_sensors"),
                                       "input_preprocessing": "unchanged_reference_mean_std_clip", "adam_reset": False,
                                       "projection": "Set excluded w1 rows to zero before training and after every Adam update; their gradients are zero, moments continue to decay."},
              "active_features": model.meta.get("active_features"), "source_snapshot": "source/",
              "train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
              "weak_class": diagnosis["weakest_class"], "class_weights": weights,
              "validation_before": before, "validation_after": after, "epoch_history": epoch_records,
              "train_before": train_before, "train_after": evaluate(model, train),
              "terminal_validation": epoch_records[-1]["validation"],
              "training_seconds": time.monotonic() - started, "official_scoring_ready": False}
    report["resources"] = _resources(attempt_started, cpu_started)
    configuration = {**spec, **architecture_shape(model),
                     "activation": "tanh_with_identity_residual_layers" if isinstance(model, MigratedModel) else "tanh", "optimizer": "Adam",
                     "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-8, "gradient_clip_norm": 5.0,
                     "feature_clip": [-10.0, 10.0], "normalization": "initial_training_mean_std",
                     "input_transform": model.meta.get("input_transform", "identity"),
                     "batch_size": model.meta["batch_size"], "requested_epochs": spec["epochs"],
                     "checkpoint_selection": report["checkpoint_selection"],
                     "classes": model.meta["classes"]}
    configuration_raw = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    training = {**report, "format": "contest.cpu-baseline-report.v1", "profile": "A", "task": "classification",
                "purpose": "development_reference", "device": "cpu", "cpu_threads_requested": _THREADS,
                "numpy_version": np.__version__, "python_version": platform.python_version(), "machine": platform.machine(),
                "parameter_count": model.meta["parameter_count"], "input_features": len(model.meta["paths"]),
                "train_rows": len(train["ids"]), "validation_rows": len(validation["ids"]),
                "learning_rate": model.meta["learning_rate"], "batch_size": model.meta["batch_size"],
                "configuration": configuration, "configuration_sha256": "sha256:" + hashlib.sha256(configuration_raw).hexdigest(),
                "source_sha256": {name: digest(directory / "source" / name) for name in source_names},
                "label_provenance": None, "created_at": stamp()}
    write_json(directory / "training.json", training)
    write_json(directory / "experiment.json", report)
    return report


def gate(candidate, active, baseline, config):
    """判断新候选是否满足 Macro-F1 增益、静态 baseline 下限、Accuracy 和类别 recall 约束。"""
    reasons = []
    for key in ("macro_f1", "accuracy"):
        if isinstance(candidate.get(key), bool) or not isinstance(candidate.get(key), (float, int)) or not math.isfinite(candidate[key]) or not 0 <= candidate[key] <= 1:
            return {"accepted": False, "reasons": ["invalid_metrics"]}
    if candidate["macro_f1"] < active["macro_f1"] + config["min_macro_f1_gain"]:
        reasons.append("macro_f1_gain_below_minimum")
    if candidate["macro_f1"] + 1e-12 < baseline["macro_f1"]:
        reasons.append("below_static_baseline")
    if candidate["accuracy"] < active["accuracy"] - config["max_accuracy_drop"]:
        reasons.append("accuracy_regression")
    for name, old in active.get("per_class", {}).items():
        if old["support"] >= config["minimum_class_support"]:
            new = candidate.get("per_class", {}).get(name)
            if new is None or new["recall"] < old["recall"] - config["max_class_recall_drop"]:
                reasons.append("class_recall_regression:" + name)
    for name, minimum in config.get("minimum_class_recall", {}).items():
        value = candidate.get("per_class", {}).get(name, {}).get("recall")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value + 1e-12 < minimum:
            reasons.append("class_recall_below_minimum:" + name)
    return {"accepted": not reasons, "reasons": reasons or ["quality_gates_passed"]}
