"""独立重算真实A/B/C实验，核对提供方原始回复、SDK、历史引用、逐条预测与云端归档。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from scripts.test_real_sdk import MODEL
from gas_a.sensor_training.io import CLASSES, digest, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_dataset(root, dataset):
    suite, cloud = read(root / "suite.json"), read(root / "cloud-verification.json")
    assert suite["status"] == "completed" and suite["cloud_verified"] and suite["cloud_state"] == "FINISHED"
    assert all(suite["checks"].values()) and cloud["verified"] and cloud["state"] == "FINISHED"
    assert suite["dataset"] == dataset and suite["model"] == MODEL
    source = read(root / "source-sha256.json")
    assert all(digest(root / "reproduction" / name) == sha for name, sha in source.items())
    normal = root / "normal"
    assert all(digest(normal / "data" / name) == digest(root / "official-baseline/public" / name)
               for name in ("train.jsonl", "validation.jsonl"))
    raw_calls = {}
    total_tokens = 0
    for folder in sorted((normal / "provider-calls").iterdir()):
        receipt, response = read(folder / "receipt.json"), read(folder / "response.json")
        assert receipt["status"] == "verified_response" and response["model"] == MODEL
        assert response["id"] == receipt["call_id"] and receipt["call_id"] not in raw_calls
        raw_calls[receipt["call_id"]] = (response, read(folder / "request.json"))
        total_tokens += response.get("usage", {}).get("total_tokens", 0)
    report = read(normal / "report.json")
    complete, sdk_calls, corrections, adoptions = 0, [], 0, 0
    for index, record in enumerate(report["rounds"], 1):
        host = normal / f"host-{index}"
        frames = [r["frame"] for r in rows(host / "protocol.jsonl")]
        start = frames[0]
        assert start["dataset_key"] == dataset and start["model_name"] == MODEL
        assert start["task_contract"]["task"] == ("classification" if dataset == "A" else "regression")
        finish = next(f["record"] for f in frames if f.get("type") == "round_complete")
        assert any(f.get("type") == "round_ack" and f.get("accepted") for f in frames)
        history = next(f["records"] for f in frames if f.get("type") == "history_response")
        assert finish["history_ids"] == [h["id"] for h in history] and len(history) == index - 1
        calls = [f for f in frames if f.get("type") == "model_response"]
        assert finish["model_call_ids"] == [f["call_id"] for f in calls] and len(calls) >= 2
        for call in calls:
            assert call["response"] == raw_calls[call["call_id"]][0]
            sdk_calls.append(call["call_id"])
        for call in rows(host / "model-calls.jsonl"):
            response, request = raw_calls[call["call_id"]]
            assert call["real_remote_model_call"] and call["response"] == response
            assert call["bound_messages"] == request["messages"]
        decision = rows(host / "state/decisions.jsonl")[-1]
        corrections += sum(len(decision[k]["corrections"]) for k in ("planning_calls", "selection_calls"))
        complete += bool(history) and any(f.get("type") == "candidate_request" for f in frames)
        adoptions += finish["selected_artifact_id"] == f"candidate-{index}"
        assert finish["configuration"]["parent_checkpoint_sha256"] == start["parent_checkpoint_sha256"]
    assert len(sdk_calls) == len(raw_calls) == suite["real_model_calls"] and set(sdk_calls) == set(raw_calls)
    assert complete >= 3 and complete == report["complete_history_experiments"]
    assert total_tokens == suite["provider_reported_total_tokens"]
    expected = np.asarray([CLASSES.index(r["answer"]) if dataset == "A" else r["answer"]
                           for r in rows(normal / "data/validation.jsonl")])
    groups, count = 0, 0
    for directory in sorted(normal.glob("measurement-*")):
        outputs = [r["outputs"] for r in rows(directory / "predictions.jsonl")]
        metric = read(directory / "metrics.json")
        assert len(outputs) == len(expected)
        if dataset == "A":
            assert all([p["label"] for p in row] == CLASSES for row in outputs)
            probabilities = np.asarray([[p["score"] for p in row] for row in outputs])
            assert np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
            labels = probabilities.argmax(axis=1)
            score = np.mean([2 * np.sum((labels == k) & (expected == k)) /
                             (np.sum(labels == k) + np.sum(expected == k)) for k in set(labels) | set(expected)])
            assert abs(score - metric["macro_f1"]) < 1e-12
        else:
            assert all(set(row) == {"value"} for row in outputs)
            prediction = np.asarray([row["value"] for row in outputs])
            assert np.isfinite(prediction).all() and np.all(prediction >= 0)
            error = prediction - expected
            assert abs(np.sqrt(np.mean(error ** 2)) - metric["rmse"]) < 1e-9
            assert abs(np.mean(abs(error)) - metric["mae"]) < 1e-9
            if dataset == "C":
                score = np.expm1(np.where(error < 0, -error / 13, error / 10)).sum()
                assert np.isclose(score, metric["nasa_score"], rtol=1e-12)
        groups += 1
        count += len(expected)
    key = "macro_f1" if dataset == "A" else "rmse"
    assert (report["final"][key] >= report["baseline"][key] if dataset == "A" else report["final"][key] <= report["baseline"][key])
    bundle, downloaded = root / "evidence-bundle.zip", root / "cloud-downloaded-evidence.zip"
    assert digest(bundle) == digest(downloaded) == "sha256:" + cloud["evidence_zip_sha256"]
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert not any(Path(p).name.startswith(".env") for p in archive.namelist())
    return {"dataset": dataset, "verified": True, "real_model_calls": len(raw_calls), "provider_reported_total_tokens": total_tokens,
            "rounds": len(report["rounds"]), "complete_history_experiments": complete, "adoptions": adoptions,
            "automatic_response_corrections": corrections, "measurement_groups_recalculated": groups,
            "prediction_rows_recalculated": count, "baseline": report["baseline"], "final": report["final"],
            "cloud_url": suite["cloud_url"], "cloud_state": suite["cloud_state"],
            "cloud_text_records_verified": cloud["text_records_downloaded_and_hash_checked"],
            "cloud_score_points_verified": cloud["score_points_checked"], "evidence_zip_sha256": cloud["evidence_zip_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", choices=("A", "B", "C"), nargs="+", default=["A", "B", "C"])
    args = parser.parse_args()
    results = [verify_dataset(args.run.resolve() / key, key) for key in args.datasets]
    result = {"verified": True, "datasets": results, "real_model_calls": sum(r["real_model_calls"] for r in results),
              "official_host_or_hidden_score_verified": False}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
