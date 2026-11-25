"""只读重算A/B/C离线镜像测试证据，不调用模型、云端或Docker。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from gas_a.sensor_training.io import CLASSES, write_json
from scripts.validate_multitask import complete_experiment


def verify(root):
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    image = json.loads((root / "image-preflight.json").read_text(encoding="utf-8"))
    assert image["describe"]["tasks"] == ["classification", "regression"]
    assert image["runtime_probe"]["verified"]
    assert summary["real_model_calls"] == 0
    assert summary["runtime"]["container_policy_verified"]
    datasets = [report["dataset"] for report in summary["reports"]]
    assert datasets and len(set(datasets)) == len(datasets) and set(datasets) <= {"A", "B", "C"}
    groups, rows_checked, complete = 0, 0, {}
    for dataset in datasets:
        directory = root / dataset
        records = [json.loads(line) for line in (directory / "data/validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = np.array([CLASSES.index(row["answer"]) if dataset == "A" else row["answer"] for row in records])
        for measured in sorted(directory.glob("measurement-*")):
            outputs = [json.loads(line)["outputs"] for line in (measured / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            assert len(outputs) == len(records)
            saved = json.loads((measured / "metrics.json").read_text(encoding="utf-8"))
            if dataset == "A":
                assert all([item["label"] for item in output] == CLASSES for output in outputs)
                scores = np.array([[item["score"] for item in output] for output in outputs])
                assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
                labels = scores.argmax(axis=1)
                observed = set(labels) | set(expected)
                score = np.mean([2 * np.sum((labels == k) & (expected == k)) /
                                 (np.sum(labels == k) + np.sum(expected == k)) for k in observed])
                assert abs(score - saved["macro_f1"]) < 1e-12
                assert abs(np.mean(labels == expected) - saved["accuracy"]) < 1e-12
            else:
                assert all(set(output) == {"value"} and not isinstance(output["value"], bool) for output in outputs)
                values = np.array([output["value"] for output in outputs])
                assert np.isfinite(values).all() and np.all(values >= 0)
                error = values - expected
                assert abs(np.sqrt(np.mean(error ** 2)) - saved["rmse"]) < 1e-9
                assert abs(np.mean(abs(error)) - saved["mae"]) < 1e-9
                if dataset == "C":
                    nasa = np.expm1(np.where(error < 0, -error / 13, error / 10)).sum()
                    assert np.isclose(nasa, saved["nasa_score"], rtol=1e-12)
            groups += 1
            rows_checked += len(records)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        assert report["not_worse_than_public_baseline"]
        metric = "macro_f1" if dataset == "A" else "rmse"
        if dataset == "A":
            assert report["final"][metric] >= report["baseline"][metric]
        else:
            assert report["final"][metric] <= report["baseline"][metric]
        complete[dataset] = 0
        for index, row in enumerate(report["rounds"], 1):
            transcript = [json.loads(line) for line in (directory / f"host-{index}/protocol.jsonl").read_text(encoding="utf-8").splitlines()]
            assert any(e["direction"] == "host" and e["frame"].get("type") == "round_ack" and e["frame"].get("accepted") for e in transcript)
            finish = next(e["frame"]["record"] for e in transcript if e["frame"].get("type") == "round_complete")
            history = next(e["frame"]["records"] for e in transcript if e["frame"].get("type") == "history_response")
            call_ids = [e["frame"]["call_id"] for e in transcript if e["frame"].get("type") == "model_response"]
            assert finish["history_ids"] == [h["id"] for h in history]
            assert finish["model_call_ids"] == call_ids and len(call_ids) >= 2
            assert len(history) == index - 1
            candidate = any(e["frame"].get("type") == "candidate_request" for e in transcript)
            complete[dataset] += complete_experiment({"history_ids": finish["history_ids"],
                "candidate_submitted": candidate, "record": finish})
        assert complete[dataset] >= 3
    return {"verified": True, "image_id": image["image_id"], "datasets": datasets,
            "measurement_groups_recalculated": groups, "prediction_rows_recalculated": rows_checked,
            "complete_history_experiments": complete, "real_model_calls": 0,
            "official_host_or_hidden_score_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run.resolve())
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
