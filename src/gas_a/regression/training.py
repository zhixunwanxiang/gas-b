"""从宿主父回归检查点续训；保存 Adam、缩放和种子，只用 train 更新梯度。"""
from copy import deepcopy
import math
from pathlib import Path
import time
import numpy as np
from .data import load_model
from .evaluation import evaluate, promotion_gate
from .model import component_digest
from . import fast_component
from .timing import measure as measure_prediction_time
from ..sensor_training.io import append_jsonl, digest, write_json


def prepare_training(model, train, spec):
    x = model.normalized(train["x"])
    target = (train["y"] - model.meta["target_mean"]) / model.meta["target_scale"]
    weights = np.ones(len(x))
    evidence = {"mode": spec.get("weighting", "uniform"), "fit_scope": "authorized_train_only"}
    if spec.get("weighting") == "bearing_sqrt_inverse":
        labels = train.get("groups")
        if labels is None:
            raise ValueError("Balanced training requires documented train groups")
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        raw = np.sqrt(len(x) / (len(counts) * counts))
        lo, hi = 0., 100.
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.sum(counts * np.clip(raw * mid, .5, 2.)) / len(x) < 1:
                lo = mid
            else:
                hi = mid
        weights = np.clip(raw * ((lo + hi) / 2), .5, 2.)[inverse]
        strength = spec.get("balance_strength", 1.)
        weights = (1 - strength) + strength * weights
        evidence.update(group_source=train["grouping"]["source"], groups=len(counts),
                        group_counts=counts.tolist(), balance_strength=strength)
    elif spec.get("weighting") == "parent_tail":
        residual = np.abs(model.predict(train["x"]) - train["y"])
        threshold = float(np.quantile(residual, spec.get("tail_quantile", .9)))
        weights = np.where(residual >= threshold, spec.get("tail_weight", 2.), 1.)
        weights /= weights.mean()
        evidence.update(threshold=threshold, quantile=spec.get("tail_quantile", .9),
                        tail_rows=int(np.count_nonzero(residual >= threshold)),
                        parent_steps=model.meta["steps"])
    elif spec.get("weighting", "uniform") != "uniform":
        raise ValueError("Unknown regression weighting")
    evidence.update(minimum_weight=float(weights.min()), maximum_weight=float(weights.max()),
                    mean_weight=float(weights.mean()))
    return {"x": x, "target": target, "weights": weights, "evidence": evidence}


def fit_epoch(model, train, spec, deadline, *, prepared=None):
    # 输入与目标都使用父检查点已有的缩放，绝不在验证集上重新拟合 scaler。
    # 随机顺序由父 seed 和累计 epoch 决定：从同一父状态重放时得到同一批次顺序。
    prepared = prepare_training(model, train, spec) if prepared is None else prepared
    x, target = prepared["x"], prepared["target"]
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
    order = rng.permutation(len(x))
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
        error = output[:, 0] - expected
        if spec["loss"] == "huber":
            # Huber 在目标标准化后使用固定 delta=1，降低大残差对梯度的支配；不修改原始寿命标签。
            absolute = np.abs(error)
            losses = np.where(absolute <= 1, .5 * error ** 2, absolute - .5)
            delta = (np.clip(error, -1, 1) / len(indices))[:, None]
        else:
            losses = error ** 2
            delta = (2 * error / len(indices))[:, None]
        weights = prepared["weights"][indices]
        if spec.get("underestimate_weight", 1.0) != 1.0:
            # 仅根据 train 残差方向加权，预测和验证计算保持原始 RMSE/MAE 定义。
            weights = weights * np.where(error < 0, spec["underestimate_weight"], 1.0)
        losses = losses * weights
        delta = delta * weights[:, None]
        derivative = getattr(model, "neural_output_derivative", None)
        if derivative is not None:
            jacobian = derivative(batch)
            if jacobian is not None:
                delta *= jacobian[:, None]
        total += float(losses.sum())
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
                       *, reference=None):
    # 再次核对实际父文件，防止“声明从父模型续训，实际读了固定本机权重”。
    # 训练在加载后的内存副本上进行，original 留作继承关系核验，不修改父文件。
    if digest(checkpoint) != expected_sha:
        raise ValueError("Regression parent checkpoint digest changed")
    started = time.monotonic()
    deadline = started + limits["max_seconds_per_candidate"]
    model = load_model(checkpoint, dataset, train["paths"])
    original = deepcopy(model)
    if spec.get("component_update"):
        return continue_component(checkpoint, model, original, train, validation, output, spec, rules,
                                  dataset, expected_sha, deadline, reference)
    model.meta["learning_rate"] = spec["learning_rate"]
    model.validate()
    before = evaluate(model, validation, dataset)
    output = Path(output)
    output.mkdir()  # 候选目录必须全新，不覆盖父产物或之前实验。
    prepared = prepare_training(model, train, spec)
    best, best_metrics, best_eligible, stale = None, None, False, 0
    executed = 0
    for _ in range(min(spec["epochs"], limits["max_epochs"])):
        loss = fit_epoch(model, train, spec, deadline, prepared=prepared)
        metric = evaluate(model, validation, dataset)
        if time.monotonic() >= deadline:
            raise TimeoutError("Regression candidate exceeded its training and measurement budget")
        gate = promotion_gate(before, metric, rules, reference=reference)
        # 先在可上线的 epoch 中选 RMSE 最低者；全部不合格时仍保留真实训练结果供拒绝和归档。
        better = (best is None or (gate["eligible"] and not best_eligible)
                  or (gate["eligible"] == best_eligible and metric["rmse"] < best_metrics["rmse"]))
        if better:
            # 必须深拷贝权重和优化器状态；否则后续 epoch 会继续改动所谓“最佳”版本。
            best, best_metrics, best_eligible, stale = deepcopy(model), metric, gate["eligible"], 0
        else:
            stale += 1
        executed += 1
        append_jsonl(output / "history.jsonl", {"epoch": model.meta["epochs"], "step": model.meta["steps"],
                     "loss": loss, "validation": metric, "gate": gate})
        if stale >= limits["early_stopping_patience"]:
            # 连续若干 epoch 没有更好的候选就停止，避免无收益的计算；并不强制最后一轮上线。
            break
    if best is None:
        raise ValueError("No completed regression epoch")
    best.save(output / "checkpoint.npz")
    # selected_epoch/final_steps 属于导出的最佳状态；executed_* 属于实际尝试到的末尾状态。
    # 两者可能不同，因此同时留档，不能把“执行过更多轮”写成选中模型的累计轮数。
    report = {"dataset": dataset, "task": "regression", "parent_checkpoint_sha256": expected_sha,
              "checkpoint_sha256": digest(output / "checkpoint.npz"), "train_sha256": train["sha256"],
              "validation_sha256": validation["sha256"], "configuration": spec,
              "weighting_evidence": prepared["evidence"],
              "initial_steps": original.meta["steps"], "final_steps": best.meta["steps"],
              "initial_epochs": original.meta["epochs"], "selected_epoch": best.meta["epochs"],
              "executed_epochs": executed, "executed_final_steps": model.meta["steps"],
              "validation_before": before, "validation_after": best_metrics,
              "selected_epoch_eligible": best_eligible,
              "checks": {"seed_preserved": best.meta["seed"] == original.meta["seed"],
                         "input_scaler_preserved": bool(np.array_equal(best.mean, original.mean) and np.array_equal(best.scale, original.scale)),
                         "target_scaler_preserved": all(best.meta[k] == original.meta[k] for k in ("target_mean", "target_scale")),
                         "actual_continuation": best.meta["steps"] > original.meta["steps"]}}
    if original.meta.get("regression_components") is not None:
        report["checks"]["existing_component_preserved"] = component_digest(best.meta["regression_components"]) == component_digest(original.meta["regression_components"])
        report["prediction_parameter_count"] = best.meta["prediction_parameter_count"]
        report["predictor_scope"] = "complete_regression_predictor"
    if original.meta.get(fast_component.KEY) is not None:
        report["checks"]["existing_fast_component_preserved"] = component_digest(best.meta[fast_component.KEY]) == component_digest(original.meta[fast_component.KEY])
        report["prediction_timing"] = measure_prediction_time(original, best, validation["x"], rules, deadline)
        if report["prediction_timing"] is None:
            raise ValueError("Fast predictor continuation requires a prediction budget")
        report["selected_epoch_eligible"] &= report["prediction_timing"]["eligible"]
    if not all(report["checks"].values()):
        raise ValueError("Regression continuation state was not preserved")
    # 与三份官方续训指南的结果目录保持一致；两份报告来自同一个对象，不另算或美化成绩。
    write_json(output / "training.json", report)
    write_json(output / "continuation.json", report)
    return report


def continue_component(checkpoint, model, original, train, validation, output, spec, rules,
                       dataset, expected_sha, deadline, reference):
    """Update one fixed component; never invent MLP epochs or Adam steps."""
    fast = spec["component_update"] == "fixed_rbf"
    if not fast and (dataset != "C" or spec["component_update"] != "condition_ridge"):
        raise ValueError("Unsupported regression component update")
    if fast and dataset not in ("B", "C"):
        raise ValueError("Fixed RBF supports B/C only")
    if (fast or original.meta.get(fast_component.KEY) is not None) and not rules.get("prediction_budget"):
        raise ValueError("Fast component requires a prediction budget")
    before = evaluate(original, validation, dataset)
    key = fast_component.KEY if fast else "regression_components"
    other_key = "regression_components" if fast else fast_component.KEY
    old = deepcopy(original.meta.get(key))
    initialized = model.initialize_fast_component(train, spec) if fast else model.initialize_component(train)
    if (not np.array_equal(model.predict(train["x"]), original.predict(train["x"]))
            or not np.array_equal(model.predict(validation["x"]), original.predict(validation["x"]))):
        raise ValueError("Zero component migration changed parent predictions")
    if fast:
        prepared = prepare_training(model, train, spec)
        model.update_fast_component(train, prepared["weights"], deadline)
    else:
        model.update_component(train)
    if time.monotonic() >= deadline:
        raise TimeoutError("Regression component fitting budget exhausted")
    output = Path(output)
    output.mkdir()
    model.save(output / "checkpoint.npz")
    loaded = load_model(output / "checkpoint.npz", dataset, train["paths"])
    after = evaluate(loaded, validation, dataset)
    if time.monotonic() >= deadline:
        raise TimeoutError("Complete regression component export and measurement budget exhausted")
    head = loaded.meta[key]
    base_preserved = (all(np.array_equal(getattr(original, group)[name], getattr(loaded, group)[name])
                          for group in ("params", "first", "second") for name in original.params)
                      and all(loaded.meta[k] == original.meta[k] for k in
                              ("seed", "epochs", "steps", "learning_rate", "target_mean", "target_scale"))
                      and np.array_equal(loaded.mean, original.mean) and np.array_equal(loaded.scale, original.scale))
    stats = (("format", "cap", "initial_train_sha256", "profile", "width", "space", "bandwidth", "centers") if fast else
             ("format", "cap", "initial_train_sha256", "setting_mean", "setting_scale", "centers", "means", "scales"))
    inherited = old is None or all(head[name] == old[name] for name in stats)
    checks = {"seed_preserved": loaded.meta["seed"] == original.meta["seed"],
              "input_scaler_preserved": bool(np.array_equal(loaded.mean, original.mean) and np.array_equal(loaded.scale, original.scale)),
              "target_scaler_preserved": all(loaded.meta[k] == original.meta[k] for k in ("target_mean", "target_scale")),
              "frozen_mlp_and_adam_preserved": base_preserved, "component_statistics_inherited": inherited,
              "other_component_preserved": original.meta.get(other_key) == loaded.meta.get(other_key),
              "actual_component_continuation": head["updates"] == (old["updates"] if old else 0) + 1,
              "parent_file_unchanged": digest(checkpoint) == expected_sha}
    if not all(checks.values()):
        raise ValueError("Complete regression component continuation invariants failed")
    gate = promotion_gate(before, after, rules, reference=reference)
    timing = measure_prediction_time(original, loaded, validation["x"], rules, deadline) if fast or original.meta.get(fast_component.KEY) is not None else None
    if timing is not None:
        gate["eligible"] &= timing["eligible"]
    record = {"epoch": loaded.meta["epochs"], "step": loaded.meta["steps"], "component_updates": head["updates"],
              "validation": after, "gate": gate, "scope": "complete_predictor_with_frozen_mlp"}
    append_jsonl(output / "history.jsonl", record)
    report = {"dataset": dataset, "task": "regression", "parent_checkpoint_sha256": expected_sha,
        "checkpoint_sha256": digest(output / "checkpoint.npz"), "train_sha256": train["sha256"],
        "validation_sha256": validation["sha256"], "configuration": spec,
        "initial_steps": original.meta["steps"], "final_steps": loaded.meta["steps"],
        "initial_epochs": original.meta["epochs"], "selected_epoch": loaded.meta["epochs"],
        "executed_epochs": 0, "executed_final_steps": loaded.meta["steps"], "executed_component_updates": 1,
        "component_updates_before": old["updates"] if old else 0, "component_updates_after": head["updates"],
        "component_parent_sha256": component_digest(old) if old else None,
        "component_sha256": component_digest(head), "component_format": head["format"], "component_key": key,
        "component_initialized_from_zero": initialized, "prediction_parameter_count": loaded.meta["prediction_parameter_count"],
        "format_migration": {"from": original.meta["format"], "to": loaded.meta["format"], "zero_prediction_change_verified": True},
        "weighting_evidence": {**(prepared["evidence"] if fast else {"mode": "condition_ridge", "fit_scope": "authorized_train_only"}), "solver": head.get("solver", "ridge_closed_form"),
                               "frozen_mlp_epochs_added": 0, "frozen_mlp_steps_added": 0},
        "prediction_timing": timing,
        "validation_before": before, "validation_after": after, "selected_epoch_eligible": gate["eligible"],
        "checks": checks}
    write_json(output / "training.json", report)
    write_json(output / "continuation.json", report)
    return report
