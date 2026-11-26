"""Record and verify the actual parent state used by every continuation path."""
# 续训证明与完整预测器衔接：核对宿主父检查点，记录权重、Adam、标准化和累计训练状态。
# 单个神经组件的 epoch 分数与完整路由包分数分别记录，最终选版比较完整预测器。
import hashlib
from pathlib import Path
import json
import time

import numpy as np

from .sensor_training.data import load_model
from .sensor_training.io import digest, write_json
from .sensor_training.optimize import train_candidate


def _arrays(values):
    # 为每个数值张量保存原始字节摘要、dtype 和形状，使权重与优化器继承关系可独立核验。
    return {name: {"sha256": "sha256:" + hashlib.sha256(value.tobytes()).hexdigest(),
                   "dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(values.items())}


def state_identity(model):
    # 同时记录权重、Adam 一二阶矩、scaler、seed、目标尺度和累计 epoch/step，避免只证明权重被读过。
    return {"epochs": model.meta["epochs"], "steps": model.meta["steps"],
            "seed": model.meta["seed"], "target_mean": model.meta["target_mean"],
            "target_scale": model.meta["target_scale"],
            "parameters": _arrays(model.params), "adam_first": _arrays(model.first),
            "adam_second": _arrays(model.second),
            "normalization": _arrays({"mean": model.mean, "scale": model.scale})}


def continue_candidate(parent_path, train, validation, directory, spec, diagnosis, limits,
                       *, expected_parent_sha256=None, parent_artifact_id=None):
    """Load the supplied parent, continue it, and persist a verifiable parent/child record."""
    # 先核对父文件，再训练主组件；如需路由则从各实际父组件构建完整新包。
    # 导出后重新加载并核对父文件未改、scaler 未改、计数增加，最后写入父子状态证据。
    from .frozen_router import FrozenRouter, build_bundle, load_predictor, retain_bundle
    started, cpu_started = time.monotonic(), time.process_time()
    parent_path = Path(parent_path)
    parent_sha = digest(parent_path)
    if (expected_parent_sha256 is not None and parent_sha.removeprefix("sha256:")
            != expected_parent_sha256.removeprefix("sha256:")):
        raise ValueError("Actual parent checkpoint digest differs from the authorized parent")
    parent = load_model(parent_path, train["paths"])
    parent_predictor = load_predictor(parent_path, train["paths"])
    initial = state_identity(parent)
    primary_limits = dict(limits)
    if "controller_checkpoint_gate" in limits:
        from .sensor_training.data import evaluate
        # A routed parent is compared as a complete predictor, never silently
        # replaced by its primary component's weaker validation score.
        # 逐 epoch 参考仍是完整父预测器；不能为了让新主组件容易过关而换成较弱的主组件父分数。
        primary_limits["controller_checkpoint_reference"] = evaluate(parent_predictor, validation)
        primary_limits["controller_checkpoint_reference_sha256"] = parent_sha
    retain = isinstance(parent_predictor, FrozenRouter) and not spec.get("routing")
    def epoch_predictor(primary):
        return FrozenRouter({**parent_predictor.models, "four_sensor": primary}, parent_predictor.manifest,
                            parent_predictor.component_paths, parent_predictor.manifest_path)
    report = train_candidate(parent_path, train, validation, directory, spec, diagnosis, primary_limits,
                             validation_predictor=epoch_predictor if retain else None)
    if spec.get("routing") or isinstance(parent_predictor, FrozenRouter):
        from .sensor_training.data import evaluate
        # 把神经主组件指标另存 anchor 字段，完整路由的最终收益不得混用逐 epoch 主组件曲线。
        if not retain:
            report["anchor_validation_before"] = report["validation_before"]
            report["anchor_validation_after"] = report["validation_after"]
            report["validation_before"] = evaluate(parent_predictor, validation)
            report["epoch_history_scope"] = "primary_neural_component_only"
            report["training_metrics_scope"] = "primary_neural_component_only"
        if spec.get("routing"):
            report["validation_after"], manifest = build_bundle(
                parent_path, parent_predictor, train, validation, Path(directory), spec, diagnosis, limits, started)
        else:
            measured, manifest = retain_bundle(parent_predictor, Path(directory), validation)
            if abs(measured["macro_f1"] - report["validation_after"]["macro_f1"]) > 1e-12:
                raise ValueError("Retained bundle does not match selected complete epoch")
            report["validation_after"] = measured
            report["retained_components"] = manifest["calibration"]["retained_components"]
        if time.monotonic() - started > limits["max_seconds_per_candidate"]:
            raise TimeoutError("Complete candidate exceeded its wall-time budget")
        report["prediction_bundle"] = {"format": manifest["format"], "manifest": "routing.json"}
        report["prediction_parameter_count"] = manifest["prediction_parameter_count"]
        report["parameter_count_scope"] = "primary_neural_component"
        report["checkpoint_sha256"] = digest(Path(directory) / "checkpoint.npz")
        report["training_seconds"] = time.monotonic() - started
        report["resources"].update(wall_seconds=report["training_seconds"], cpu_seconds=time.process_time() - cpu_started)
        training_path = Path(directory) / "training.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training.update(report)
        write_json(training_path, training)
        write_json(Path(directory) / "experiment.json", report)
    child_path = Path(directory) / "checkpoint.npz"
    child = load_model(child_path, train["paths"])
    # 比较导出后的真实文件状态，核对 scaler/seed/目标尺度未改及累计训练量增加，而非只相信报告文字。
    preserved = (
        np.array_equal(parent.mean, child.mean) and np.array_equal(parent.scale, child.scale)
        and all(parent.meta[k] == child.meta[k] for k in ("seed", "target_mean", "target_scale"))
    )
    if (digest(parent_path) != parent_sha or report["parent_checkpoint_sha256"] != parent_sha
            or report["initial_steps"] != initial["steps"] or report["initial_epochs"] != initial["epochs"]
            or not preserved or child.meta["steps"] <= initial["steps"]
            or child.meta["epochs"] <= initial["epochs"]):
        raise ValueError("Parent continuation invariants failed")
    write_json(Path(directory) / "continuation.json", {
        "format": "gas-a.continuation.v1", "dataset_key": "A",
        "parent": {"artifact_id": parent_artifact_id, "checkpoint_sha256": parent_sha, "state": initial},
        "child": {"checkpoint_path": "checkpoint.npz", "checkpoint_sha256": digest(child_path),
                  "state": state_identity(child)},
        "public_projection": {"train_sha256": train["sha256"], "train_rows": len(train["y"]),
                              "validation_sha256": validation["sha256"], "validation_rows": len(validation["y"])},
        "checks": {"parent_file_unchanged": True, "scaler_seed_target_scale_preserved": True,
                   "steps_increased": True, "epochs_increased": True},
    })
    return report
