"""在当轮公开数据上复评阶段起点，累计比较独立于晋级门槛。"""
import json
import shutil
from .data import load_model
from .evaluation import evaluate
from ..sensor_training.io import digest, write_json


def measure_stage(rnd, state, checkpoint, primary_sha, before, validation):
    stage_id = getattr(rnd, "stage_id", "") or ""
    manifest = state / "regression-stage-start.json"
    snapshot = state / "regression-stage-start.npz"
    identity = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    if identity is not None:
        if (identity.get("dataset") != rnd.dataset_key or not snapshot.is_file()
                or digest(snapshot) != identity.get("checkpoint_sha256")):
            raise ValueError("Regression stage start checkpoint identity mismatch")
    if identity is None or identity.get("stage_id") != stage_id:
        provenance = "first_round" if rnd.round_index == 1 else (
            "sdk_stage_transition" if identity is not None else "first_observed_round")
        identity = {"stage_id": stage_id, "dataset": rnd.dataset_key, "round_index": rnd.round_index,
                    "artifact_id": rnd.parent.artifact_id, "checkpoint_sha256": primary_sha,
                    "provenance": provenance}
        shutil.copyfile(checkpoint, snapshot)
        if digest(snapshot) != primary_sha:
            raise ValueError("Regression stage start snapshot differs from current parent")
        write_json(manifest, identity)
    metrics = before if identity["checkpoint_sha256"] == primary_sha else evaluate(
        load_model(snapshot, rnd.dataset_key, validation["paths"]), validation, rnd.dataset_key)
    return identity, metrics


def compare_stage(origin, measured):
    """报告相同验证投影上的累计变化，所有增量均为新值减旧值。"""
    if any(origin.get(key) != measured.get(key) for key in ("validation_sha256", "rows", "dataset", "metric_version")):
        raise ValueError("Stage measurements must use the same validation projection")
    def delta(before, after, key):
        a, b = before.get(key), after.get(key)
        return b - a if a is not None and b is not None else None

    result = {"rmse_delta": delta(origin, measured, "rmse"), "mae_delta": delta(origin, measured, "mae")}
    axis, keys = ("device", ("rmse", "bias")) if origin["dataset"] == "B" else (
        "source", ("nasa_score", "nasa_underestimate_score", "nasa_overestimate_score"))
    if origin["dataset"] == "C":
        result["nasa_delta"] = delta(origin, measured, "nasa_score")
    old = origin["group_metrics"][axis]
    new = measured["group_metrics"][axis]
    if old.keys() != new.keys() or any(old[k]["rows"] != new[k]["rows"] for k in old):
        raise ValueError("Stage measurements must use the same group support")
    result[axis] = {name: {"rows": row["rows"], **{f"{key}_delta": delta(row, new[name], key) for key in keys}}
                    for name, row in old.items()}
    return result
