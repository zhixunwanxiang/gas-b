"""从宿主父回归检查点续训；保存 Adam、缩放和种子，只用 train 更新梯度。"""
from copy import deepcopy
import math
from pathlib import Path
import time
import numpy as np
from .data import load_model
from .evaluation import evaluate, promotion_gate
from ..sensor_training.io import append_jsonl, digest, write_json


def fit_epoch(model, train, spec, deadline):
    # 输入与目标都使用父检查点已有的缩放，绝不在验证集上重新拟合 scaler。
    # 随机顺序由父 seed 和累计 epoch 决定：从同一父状态重放时得到同一批次顺序。
    x = model.normalized(train["x"])
    target = (train["y"] - model.meta["target_mean"]) / model.meta["target_scale"]
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
            total += float(np.where(absolute <= 1, .5 * error ** 2, absolute - .5).sum())
            delta = (np.clip(error, -1, 1) / len(indices))[:, None]
        else:
            total += float(np.sum(error ** 2))
            delta = (2 * error / len(indices))[:, None]
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


def continue_candidate(checkpoint, train, validation, output, spec, limits, rules, dataset, expected_sha):
    # 再次核对实际父文件，防止“声明从父模型续训，实际读了固定本机权重”。
    # 训练在加载后的内存副本上进行，original 留作继承关系核验，不修改父文件。
    if digest(checkpoint) != expected_sha:
        raise ValueError("Regression parent checkpoint digest changed")
    deadline = time.monotonic() + limits["max_seconds_per_candidate"]
    model = load_model(checkpoint, dataset, train["paths"])
    original = deepcopy(model)
    model.meta["learning_rate"] = spec["learning_rate"]
    model.validate()
    before = evaluate(model, validation, dataset)
    if "baseline_groups" in validation:
        before["baseline_groups"] = validation["baseline_groups"]
    output = Path(output)
    output.mkdir()  # 候选目录必须全新，不覆盖父产物或之前实验。
    best, best_metrics, best_eligible, stale = None, None, False, 0
    executed = 0
    for _ in range(min(spec["epochs"], limits["max_epochs"])):
        loss = fit_epoch(model, train, spec, deadline)
        if spec.get("tree_update"):
            from .tree_training import fit_layer
            fit_layer(model, train, dataset, deadline, spec.get("positive_scale", 1.))
        if spec.get("residual_mode") and not executed:
            model.update_residual(train, spec["residual_mode"])
        metric = evaluate(model, validation, dataset)
        if time.monotonic() >= deadline:
            raise TimeoutError("Regression candidate exceeded its training and measurement budget")
        gate = promotion_gate(before, metric, rules)
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
    roundtrip = load_model(output / "checkpoint.npz", dataset, train["paths"])
    probe = validation["x"][:min(32,len(validation["x"]))]
    if not np.allclose(roundtrip.predict(probe), best.predict(probe), rtol=1e-10, atol=1e-9):
        raise ValueError("Regression artifact round-trip prediction mismatch")
    if time.monotonic() >= deadline:
        raise TimeoutError("Regression export and round-trip check exceeded candidate budget")
    # selected_epoch/final_steps 属于导出的最佳状态；executed_* 属于实际尝试到的末尾状态。
    # 两者可能不同，因此同时留档，不能把“执行过更多轮”写成选中模型的累计轮数。
    report = {"dataset": dataset, "task": "regression", "parent_checkpoint_sha256": expected_sha,
              "checkpoint_sha256": digest(output / "checkpoint.npz"), "train_sha256": train["sha256"],
              "validation_sha256": validation["sha256"], "configuration": spec,
              "initial_steps": original.meta["steps"], "final_steps": best.meta["steps"],
              "initial_epochs": original.meta["epochs"], "selected_epoch": best.meta["epochs"],
              "executed_epochs": executed, "executed_final_steps": model.meta["steps"],
              "validation_before": before, "validation_after": best_metrics,
              "selected_epoch_eligible": best_eligible,
              "checks": {"seed_preserved": best.meta["seed"] == original.meta["seed"],
                         "input_scaler_preserved": bool(np.array_equal(best.mean, original.mean) and np.array_equal(best.scale, original.scale)),
                         "target_scaler_preserved": all(best.meta[k] == original.meta[k] for k in ("target_mean", "target_scale")),
                         "actual_continuation": best.meta["steps"] > original.meta["steps"]}}
    old_bundle, new_bundle = original.meta.get("rul_tree_bundle_v1"), roundtrip.meta.get("rul_tree_bundle_v1")
    old_layers = old_bundle["layers"] if old_bundle else []
    new_layers = new_bundle["layers"] if new_bundle else []
    expected_added = executed if spec.get("tree_update") else 0
    # selected_epoch may precede the final executed epoch. Tree menus are one epoch.
    report["checks"]["existing_tree_layers_preserved"] = new_layers[:len(old_layers)] == old_layers
    report["checks"]["bounded_tree_continuation"] = len(new_layers) == len(old_layers) + expected_added
    if old_bundle and dataset == "C":
        report["checks"]["cumulative_anchor_and_radius_preserved"] = all(new_bundle[k] == old_bundle[k] for k in ("anchor", "radius"))
    if not spec.get("residual_mode"):
        report["checks"]["existing_engine_residual_preserved"] = roundtrip.meta.get("rul_residual_v1") == original.meta.get("rul_residual_v1")
    report["checks"]["parent_file_unchanged"] = digest(checkpoint) == expected_sha
    report.update(predictor_format=roundtrip.meta["format"], tree_layers_before=len(old_layers), tree_layers_after=len(new_layers),
                  predictor_scope="complete_xiaoyang_regression_predictor",
                  format_migration={"from": original.meta["format"], "to": roundtrip.meta["format"]})
    if not all(report["checks"].values()):
        raise ValueError("Regression continuation state was not preserved")
    # 与三份官方续训指南的结果目录保持一致；两份报告来自同一个对象，不另算或美化成绩。
    write_json(output / "training.json", report)
    write_json(output / "continuation.json", report)
    return report
