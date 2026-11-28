"""从宿主父回归检查点续训；保存 Adam、缩放和种子，只用 train 更新梯度。"""
from copy import deepcopy
import math
from pathlib import Path
import time
import numpy as np
from .data import load_model
from .evaluation import evaluate, guarded_masks, promotion_gate
from .model import BOUNDED_HEADS, BOUNDED_METHODS, CORRECTION_METHODS, fit_bounded_cmapss, fit_guarded_bearing, frozen_correction
from . import bearing_guard, bounded_residual
from .diagnostics import groups
from .weighting import sample_weights, resolve_training_spec, stratified_order, nasa_proxy
from ..sensor_training.io import append_jsonl, digest, write_json


def prepare_training(model, train, spec):
    x = model.normalized(train["x"])
    target = (train["y"] - model.meta["target_mean"]) / model.meta["target_scale"]
    correction = model.correction(train["x"]) / model.meta["target_scale"]
    strata = None
    if spec.get("sampling") == "device_condition":
        axes = groups(train, model.meta["profile"])
        pairs = np.column_stack([axes["device"], axes["condition"]])
        _, strata = np.unique(pairs, axis=0, return_inverse=True)
    return x, target, correction, strata


def fit_epoch(model, train, spec, deadline, *, weights=None, prepared=None):
    # 输入与目标都使用父检查点已有的缩放，绝不在验证集上重新拟合 scaler。
    # 随机顺序由父 seed 和累计 epoch 决定：从同一父状态重放时得到同一批次顺序。
    x, target, correction, strata = prepare_training(model, train, spec) if prepared is None else prepared
    weights = sample_weights(train, spec, model.meta["profile"]) if weights is None else weights
    if weights.shape != target.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid regression sample weights")
    # 修正层保留冻结；NN 的梯度按 f(x)+g(x)-y 计算，不让修正层失配。
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
    order = rng.permutation(len(x)) if strata is None else stratified_order(rng, strata)
    active = ("w2", "b2") if spec["freeze_features"] else ("w1", "b1", "w2", "b2")
    # head_only 只训练最后一层；其余方案更新两层。菜单参数已经在主控中校验。
    total = 0.0
    for start in range(0, len(order), model.meta["batch_size"]):
        if time.monotonic() >= deadline:
            # 这是本地数值超时，由主控记录失败并拒绝候选；不是允许重试 SDK 请求的信号。
            raise TimeoutError("Regression numerical training budget exhausted")
        indices = order[start:start + model.meta["batch_size"]]
        batch, expected = x[indices], target[indices]
        hidden, output = model.forward(batch)
        error = output[:, 0] + correction[indices] - expected
        weight = weights[indices]
        if spec["loss"] == "huber":
            # delta 由 train 残差标定，原始标签和比赛误差单位保持不变。
            absolute = np.abs(error)
            threshold = spec.get("huber_delta", 1.)
            total += float(np.sum(weight * np.where(absolute <= threshold, .5 * error ** 2, threshold * (absolute - .5 * threshold))))
            delta = (weight * np.clip(error, -threshold, threshold) / len(indices))[:, None]
        elif spec["loss"] == "nasa_proxy" and spec.get("nasa_mix", .35) > 0:
            losses, gradient = nasa_proxy(error, model.meta["target_scale"], spec.get("nasa_mix", .35))
            total += float(np.sum(weight * losses))
            delta = (weight * gradient / len(indices))[:, None]
        else:
            if "short_overestimate_boost" in spec:
                short_over = (train["y"][indices] <= spec["short_rul_cut"]) & (error > 0)
                weight = weight * np.where(short_over, spec["short_overestimate_boost"], 1.)
            total += float(np.sum(weight * error ** 2))
            delta = (2 * weight * error / len(indices))[:, None]
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
        # tanh 隐藏层的导数为 1-hidden**2；整个反向传播只访问 train 的 batch 与标签。
        if not spec["freeze_features"]:
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if not math.isfinite(norm):
            raise ValueError("Nonfinite regression gradient")
        model.meta["steps"] += 1
        # 先按总梯度范数 5 做裁剪，再在原有 Adam 一二阶矩上续接；steps 也从父状态递增。
        # 偏差修正必须使用累计 steps，重置为 1 会改变续训轨迹。
        for name in active:
            gradient = gradients[name] * min(1.0, 5.0 / max(norm, 1e-12))
            model.first[name] = .9 * model.first[name] + .1 * gradient
            model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
            first = model.first[name] / (1 - .9 ** model.meta["steps"])
            second = model.second[name] / (1 - .999 ** model.meta["steps"])
            model.params[name] -= spec["learning_rate"] * first / (np.sqrt(second) + 1e-8)
        # 冻结隐藏层时，其参数和一二阶矩都保持不变，不能只在更新后把权重覆盖回去。
    model.meta["epochs"] += 1
    model.validate()
    return total / len(x)


def continue_candidate(checkpoint, train, validation, output, spec, limits, rules, dataset, expected_sha,
                       *, parent_measurement=None, detailed_epochs=False, fixed_epochs=False, retention_reference=None):
    # 再次核对实际父文件，防止“声明从父模型续训，实际读了固定本机权重”。
    # 训练在加载后的内存副本上进行，original 留作继承关系核验，不修改父文件。
    if digest(checkpoint) != expected_sha:
        raise ValueError("Regression parent checkpoint digest changed")
    deadline = time.monotonic() + limits["max_seconds_per_candidate"]
    model = load_model(checkpoint, dataset, train["paths"])
    original = deepcopy(model)
    numerical_spec = resolve_training_spec(model, train, spec)
    weights = sample_weights(train, spec, dataset)
    bounded_key, bounded_method = BOUNDED_HEADS[dataset]
    method = spec.get("method")
    is_bounded = method == bounded_method
    is_correction = spec.get("method") in CORRECTION_METHODS
    frozen = is_bounded or is_correction
    if method in BOUNDED_METHODS and not is_bounded:
        raise ValueError("Regression head does not match the dataset")
    if bounded_key in model.meta and not is_bounded:
        raise ValueError("A calibrated regression parent requires a bounded head refit")
    if fixed_epochs and frozen:
        raise ValueError("Fixed epoch comparison requires MLP continuation")
    if not frozen:
        model.meta["learning_rate"] = spec["learning_rate"]
    model.validate()
    if (parent_measurement is not None and parent_measurement[0] == expected_sha
            and parent_measurement[1].get("validation_sha256") == validation["sha256"]
            and parent_measurement[1].get("dataset") == dataset):
        before = parent_measurement[1]
    else:
        before = evaluate(model, validation, dataset)
    output = Path(output)
    output.mkdir()  # 候选目录必须全新，不覆盖父产物或之前实验。
    best, best_metrics, best_eligible, stale = None, None, False, 0
    group_masks = None if detailed_epochs else guarded_masks(validation, dataset)
    gate_options = {"require_all_improved": is_bounded, "retention_reference": retention_reference}
    executed = 0
    if is_bounded:
        fit_head = fit_guarded_bearing if dataset == "B" else fit_bounded_cmapss
        model = fit_head(model, train, deadline, learning_rate=spec["learning_rate"])
        fitted_head = model.meta[bounded_key]
    elif is_correction:
        model = frozen_correction(model, train, spec["transform"], spec["ridge"], deadline,
                                  basis=spec.get("basis", "linear"))
        fitted_head = model.meta["corrections"][-1]
    else:
        # 缓存仅在本次训练中使用，输入、目标和冻结修正层在各 epoch 间固定。
        prepared = prepare_training(model, train, numerical_spec)
    choices = spec["strengths"] if frozen else range(min(spec["epochs"], limits["max_epochs"]))
    for choice in choices:
        if time.monotonic() >= deadline:
            raise TimeoutError("Regression numerical training budget exhausted")
        if frozen:
            fitted_head["strength"] = choice
            loss = None  # 这些是固定收缩系数的验证，不伪装成 NN epoch。
        else:
            loss = fit_epoch(model, train, numerical_spec, deadline, weights=weights, prepared=prepared)
        metric = evaluate(model, validation, dataset, detailed=detailed_epochs, group_masks=group_masks)
        gate = promotion_gate(before, metric, rules, **gate_options)
        # 先在可上线的 epoch 中选 RMSE 最低者；全部不合格时仍保留真实训练结果供拒绝和归档。
        better = (fixed_epochs or best is None or (gate["eligible"] and not best_eligible)
                  or (gate["eligible"] == best_eligible and metric["rmse"] < best_metrics["rmse"]))
        if better:
            # 必须深拷贝权重和优化器状态；否则后续 epoch 会继续改动所谓“最佳”版本。
            best, best_metrics, best_eligible, stale = deepcopy(model), metric, gate["eligible"], 0
        else:
            stale += 1
        executed += 1
        append_jsonl(output / "history.jsonl", {"epoch": model.meta["epochs"], "step": model.meta["steps"],
                     "correction_strength": choice if frozen else None,
                     "loss": loss, "validation": metric, "gate": gate})
        if not frozen and not fixed_epochs and stale >= limits["early_stopping_patience"]:
            # 连续若干 epoch 没有更好的候选就停止，避免无收益的计算；并不强制最后一轮上线。
            break
    if best is None:
        raise ValueError("No completed regression epoch")
    best.save(output / "checkpoint.npz")
    restored = load_model(output / "checkpoint.npz", dataset, train["paths"])
    # 导出后加载完整预测器，校验实际预测；防止只存 NN、漏掉新增层。
    best_prediction = best.predict(validation["x"])
    if not np.array_equal(restored.predict(validation["x"]), best_prediction):
        raise ValueError("Full regression predictor changed after serialization")
    best_metrics = evaluate(best, validation, dataset, prediction=best_prediction)
    if promotion_gate(before, best_metrics, rules, **gate_options)["eligible"] != best_eligible:
        raise ValueError("Full diagnostic measurement changed candidate eligibility")
    head = best.meta.get(bounded_key, {})
    widths = bearing_guard.WIDTHS if dataset == "B" else bounded_residual.WIDTHS
    # selected_epoch/final_steps 属于导出的最佳状态；executed_* 属于实际尝试到的末尾状态。
    # 两者可能不同，因此同时留档，不能把“执行过更多轮”写成选中模型的累计轮数。
    report = {"dataset": dataset, "task": "regression", "parent_checkpoint_sha256": expected_sha,
              "checkpoint_sha256": digest(output / "checkpoint.npz"), "train_sha256": train["sha256"],
              "validation_sha256": validation["sha256"], "configuration": spec,
              "resolved_training_configuration": numerical_spec,
              "fixed_epoch_comparison": fixed_epochs,
              "weight_summary": {"mean": float(weights.mean()), "minimum": float(weights.min()), "maximum": float(weights.max())},
              "training_mode": ("frozen_parent_bounded_residual" if is_bounded else
                                "frozen_parent_new_correction" if is_correction else "complete_prediction_mlp_continuation"),
              "bounded_head_epochs": head.get("epochs", bounded_residual.EPOCHS) if is_bounded else 0,
              "bounded_head_steps": head.get("steps", 0),
              "correction_layers_before": len(original.meta.get("corrections", [])),
              "correction_layers_after": len(best.meta.get("corrections", [])),
              "initial_steps": original.meta["steps"], "final_steps": best.meta["steps"],
              "initial_epochs": original.meta["epochs"], "selected_epoch": best.meta["epochs"],
              "executed_epochs": 0 if frozen else executed,
              "evaluated_correction_strengths": executed if frozen else 0,
              "executed_final_steps": model.meta["steps"],
              "total_prediction_parameters": best.meta["parameter_count"] + sum(
                  len(layer["coefficients"]) + sum(np.size(layer.get("basis", {}).get(key, []))
                                                   for key in ("weight", "bias", "centers", "thresholds"))
                  for layer in best.meta.get("corrections", [])) + (
                      sum((a + 1) * b for a, b in zip(widths, widths[1:]))
                      if is_bounded else 0),
              "validation_before": before, "validation_after": best_metrics,
              "selected_epoch_eligible": best_eligible,
              "checks": {"seed_preserved": best.meta["seed"] == original.meta["seed"],
                         "input_scaler_preserved": bool(np.array_equal(best.mean, original.mean) and np.array_equal(best.scale, original.scale)),
                         "target_scaler_preserved": all(best.meta[k] == original.meta[k] for k in ("target_mean", "target_scale")),
                         "actual_continuation": (head["steps"] > 0 if is_bounded else
                                                 len(best.meta.get("corrections", [])) > len(original.meta.get("corrections", []))
                                                 if is_correction else best.meta["steps"] > original.meta["steps"]),
                         "full_predictor_reload_verified": True,
                         "frozen_parent_preserved": (all(np.array_equal(getattr(best, group)[key], getattr(original, group)[key])
                             for group in ("params", "first", "second") for key in original.params)
                             and (best.meta.get("corrections", []) if is_bounded else best.meta.get("corrections", [])[:-1])
                             == original.meta.get("corrections", [])) if frozen else True}}
    if not all(report["checks"].values()):
        raise ValueError("Regression continuation state was not preserved")
    # 与三份官方续训指南的结果目录保持一致；两份报告来自同一个对象，不另算或美化成绩。
    write_json(output / "training.json", report)
    write_json(output / "continuation.json", report)
    return report
