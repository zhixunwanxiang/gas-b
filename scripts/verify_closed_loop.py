"""只读核对正常实验、真实故障回滚、下一轮完整继承及云端归档；不调用模型或云端。"""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from scripts.verify_multitask_cloud_run import read, rows, verify_dataset
from scripts.test_multitask_recovery_cloud import model_manifest
from scripts.test_real_sdk import MODEL
from gas_a.sensor_training.reference.model import Model
from gas_a.sensor_training.io import CLASSES, digest, write_json


def recompute(directory, dataset):
    expected = np.array([CLASSES.index(r["answer"]) if dataset == "A" else r["answer"]
                         for r in rows(directory / "data/validation.jsonl")])
    groups, count = 0, 0
    for folder in sorted(directory.glob("measurement-*")):
        outputs = [r["outputs"] for r in rows(folder / "predictions.jsonl")]
        saved = read(folder / "metrics.json")
        assert len(outputs) == len(expected)
        if dataset == "A":
            assert all([p["label"] for p in row] == CLASSES for row in outputs)
            probabilities = np.asarray([[p["score"] for p in row] for row in outputs])
            assert np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
            labels = probabilities.argmax(axis=1)
            f1 = np.mean([2 * np.sum((labels == k) & (expected == k)) /
                          (np.sum(labels == k) + np.sum(expected == k)) for k in set(labels) | set(expected)])
            assert abs(f1 - saved["macro_f1"]) < 1e-12
        else:
            assert all(set(row) == {"value"} for row in outputs)
            values = np.array([row["value"] for row in outputs])
            assert np.isfinite(values).all() and np.all(values >= 0)
            error = values - expected
            assert abs(np.sqrt(np.mean(error ** 2)) - saved["rmse"]) < 1e-9
            assert abs(np.abs(error).mean() - saved["mae"]) < 1e-9
            if dataset == "C":
                nasa = np.expm1(np.where(error < 0, -error / 13, error / 10)).sum()
                assert np.isclose(nasa, saved["nasa_score"], rtol=1e-12)
        groups += 1
        count += len(outputs)
    return groups, count


def verify_recovery(normal, root, dataset):
    suite, cloud, report = read(root / "suite.json"), read(root / "cloud-verification.json"), read(root / "recovery/report.json")
    assert suite["status"] == "completed" and suite["cloud_verified"] and suite["cloud_state"] == "FINISHED"
    assert all(suite["checks"].values()) and all(report["checks"].values())
    assert cloud["verified"] and cloud["state"] == "FINISHED"
    probe = read(root / "image-preflight.json")
    assert probe["image_id"] == suite["image_id"] and probe["runtime_probe"]["verified"]
    assert suite["runtime"]["container_policy_verified"] and suite["runtime"]["train_containers"] == 2
    source = read(root / "source-sha256.json")
    assert all(digest(root / "reproduction" / name) == sha for name, sha in source.items())
    recovery = root / "recovery"
    for name in ("train.jsonl", "validation.jsonl"):
        assert digest(recovery / "data" / name) == digest(normal / "official-baseline/public" / name)
    original, fault = Model.load(normal / "official-baseline/checkpoint.npz"), Model.load(recovery / "fault-parent/checkpoint.npz")
    assert original.meta == fault.meta
    for group in ("first", "second"):
        assert all(np.array_equal(getattr(original, group)[k], getattr(fault, group)[k]) for k in original.params)
    assert np.array_equal(original.mean, fault.mean) and np.array_equal(original.scale, fault.scale)
    assert all(np.array_equal(original.params[k], fault.params[k]) for k in ("w1", "b1"))
    if dataset == "A":
        assert np.array_equal(fault.params["w2"], np.roll(original.params["w2"], 1, axis=1))
        assert np.array_equal(fault.params["b2"], np.roll(original.params["b2"], 1))
    else:
        assert np.array_equal(fault.params["w2"], original.params["w2"])
        assert np.array_equal(fault.params["b2"], original.params["b2"] + 10)
    calls = {}
    tokens = 0
    for folder in sorted((recovery / "provider-calls").iterdir()):
        receipt, response = read(folder / "receipt.json"), read(folder / "response.json")
        assert receipt["status"] == "verified_response" and response["model"] == MODEL
        assert response["id"] == receipt["call_id"] and receipt["call_id"] not in calls
        calls[receipt["call_id"]] = (response, read(folder / "request.json"))
        tokens += response.get("usage", {}).get("total_tokens", 0)
    source_report = read(normal / "normal/report.json")
    expected_history = [{"id": f"real-local-{dataset}-history-{row['round']}", **row["record"]} for row in source_report["rounds"]]
    sdk_ids, corrections = [], 0
    first, second = report["rounds"]
    target = first["selected_artifact_id"]
    assert first["actual_historical_rollback"] and first["plan"]["action"] == "rollback" and target != "fault-parent"
    historical = recovery / "history-artifacts" / target
    assert model_manifest(historical) == report["restored_model_manifest"]
    if target == "initial":
        source_target = normal / "normal/initial"
    else:
        source_target = normal / f"normal/host-{int(target.removeprefix('candidate-'))}/candidate"
    assert model_manifest(historical) == model_manifest(source_target)
    for index, row in enumerate(report["rounds"]):
        host = recovery / f"host-{row['round']}"
        frames = [e["frame"] for e in rows(host / "protocol.jsonl")]
        start = frames[0]
        finish = next(f["record"] for f in frames if f.get("type") == "round_complete")
        assert any(f.get("type") == "round_ack" and f.get("accepted") for f in frames)
        history = next(f["records"] for f in frames if f.get("type") == "history_response")
        assert history == expected_history and finish["history_ids"] == [h["id"] for h in history]
        responses = [f for f in frames if f.get("type") == "model_response"]
        assert finish["model_call_ids"] == [r["call_id"] for r in responses]
        for response in responses:
            assert response["response"] == calls[response["call_id"]][0]
            sdk_ids.append(response["call_id"])
        for call in rows(host / "model-calls.jsonl"):
            assert call["real_remote_model_call"] and call["response"] == calls[call["call_id"]][0]
            assert call["bound_messages"] == calls[call["call_id"]][1]["messages"]
        decision = rows(host / "state/decisions.jsonl")[-1]
        corrections += sum(len(decision[k]["corrections"]) for k in ("planning_calls", "selection_calls"))
        if dataset != "A":
            assert decision["planning_calls"]["prompt_version"] == "rul-bc-controller-zh-v2"
        if index == 0:
            validated = decision["historical_validation"][target]
            assert validated["reevaluated_this_round"] and validated["gate"]["eligible"]
            assert validated["validation_sha256"] == digest(recovery / "data/validation.jsonl")
            assert any(f.get("type") == "artifact_read_request" and f.get("artifact_id") == target for f in frames)
        else:
            assert start["parent_artifact"]["artifact_id"] == target
            mounted_files = {f["path"]: f["sha256"] for f in start["parent_artifact"]["files"]}
            assert mounted_files == report["restored_model_manifest"]
            assert start["parent_checkpoint_sha256"] == first["record"]["configuration"].get(
                "selected_checkpoint_sha256", decision["configuration"]["parent_checkpoint_sha256"])
        expected_history.append({"id": f"real-recovery-{dataset}-history-{row['round']}", **finish})
    assert len(calls) == len(sdk_ids) == suite["real_model_calls"] and set(calls) == set(sdk_ids)
    assert tokens == suite["provider_reported_total_tokens"]
    assert digest(root / "evidence-bundle.zip") == digest(root / "cloud-downloaded-evidence.zip") == "sha256:" + cloud["evidence_zip_sha256"]
    with zipfile.ZipFile(root / "evidence-bundle.zip") as archive:
        assert archive.testzip() is None and not any(Path(p).name.startswith(".env") for p in archive.namelist())
    groups, predictions = recompute(recovery, dataset)
    return {"dataset": dataset, "verified": True, "image_id": suite["image_id"], "real_model_calls": len(calls),
        "provider_reported_total_tokens": tokens, "automatic_response_corrections": corrections,
        "actual_historical_rollback": True, "next_round_complete_parent_verified": True,
        "measurement_groups_recalculated": groups, "prediction_rows_recalculated": predictions,
        "fault_metrics": report["fault_metrics"], "restored_metrics": report["restored_metrics"],
        "cloud_url": suite["cloud_url"], "cloud_state": suite["cloud_state"],
        "cloud_text_records_verified": cloud["text_records_downloaded_and_hash_checked"],
        "cloud_score_points_verified": cloud["score_points_checked"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    args = parser.parse_args()
    results = []
    for key in args.datasets:
        normal = verify_dataset(args.normal.resolve() / key, key)
        restored = verify_recovery(args.normal.resolve() / key, args.recovery.resolve() / key, key)
        results.append({"dataset": key, "normal": normal, "recovery": restored})
    result = {"verified": True, "datasets": results,
        "successful_acceptance_real_calls": sum(r[phase]["real_model_calls"] for r in results for phase in ("normal", "recovery")),
        "official_host_or_hidden_score_verified": False}
    write_json(args.output, result)
    print(json.dumps({"verified": True, "datasets": args.datasets,
                      "successful_acceptance_real_calls": result["successful_acceptance_real_calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
