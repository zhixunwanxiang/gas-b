"""Record and verify the actual parent state used by every continuation path."""
import hashlib
from pathlib import Path
import json
import time

import numpy as np

from .sensor_training.data import load_model
from .sensor_training.io import digest, write_json
from .sensor_training.optimize import train_candidate


def _arrays(values):
    return {name: {"sha256": "sha256:" + hashlib.sha256(value.tobytes()).hexdigest(),
                   "dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(values.items())}


def state_identity(model):
    return {"epochs": model.meta["epochs"], "steps": model.meta["steps"],
            "seed": model.meta["seed"], "target_mean": model.meta["target_mean"],
            "target_scale": model.meta["target_scale"],
            "parameters": _arrays(model.params), "adam_first": _arrays(model.first),
            "adam_second": _arrays(model.second),
            "normalization": _arrays({"mean": model.mean, "scale": model.scale})}


def continue_candidate(parent_path, train, validation, directory, spec, diagnosis, limits,
                       *, expected_parent_sha256=None, parent_artifact_id=None):
    """Load the supplied parent, continue it, and persist a verifiable parent/child record."""
    from .frozen_router import FrozenRouter, build_bundle, load_predictor
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
        primary_limits["controller_checkpoint_reference"] = evaluate(parent_predictor, validation)
        primary_limits["controller_checkpoint_reference_sha256"] = parent_sha
    report = train_candidate(parent_path, train, validation, directory, spec, diagnosis, primary_limits)
    if spec.get("routing") or isinstance(parent_predictor, FrozenRouter):
        from .sensor_training.data import evaluate
        report["anchor_validation_before"] = report["validation_before"]
        report["anchor_validation_after"] = report["validation_after"]
        report["validation_before"] = evaluate(parent_predictor, validation)
        report["epoch_history_scope"] = "primary_neural_component_only"
        report["training_metrics_scope"] = "primary_neural_component_only"
        if spec.get("routing"):
            report["validation_after"], manifest = build_bundle(
                parent_path, parent_predictor, train, validation, Path(directory), spec, diagnosis, limits, started)
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
