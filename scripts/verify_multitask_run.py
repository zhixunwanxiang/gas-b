"""只读重算A/B/C离线镜像测试证据，不调用模型、云端或Docker。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from gas_a.sensor_training.io import CLASSES, write_json, digest
from gas_a.regression.data import load_model
from scripts.validate_multitask import complete_experiment


def verify(root):
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    image = json.loads((root / "image-preflight.json").read_text(encoding="utf-8"))
    assert image["describe"]["tasks"] == ["classification", "regression"]
    assert image["runtime_probe"]["verified"]
    assert summary["real_model_calls"] == 0
    assert summary["runtime"]["container_policy_verified"]
    datasets = [report["dataset"] for report in (summary.get("reports") or summary.get("datasets", []))]
    assert datasets and len(set(datasets)) == len(datasets) and set(datasets) <= {"A", "B", "C"}
    groups, rows_checked, complete = 0, 0, {}
    optimized = summary.get("scope") in ("fixed_public_sdk_test", "public_proxy_projection_sdk_test")
    for dataset in datasets:
        directory = root / dataset
        records = [json.loads(line) for line in (directory / "data/validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["id"]: row for row in records}
        for measured in sorted(directory.glob("measurement-*")):
            scope = measured / "validation-scope.json"
            scope_record = json.loads(scope.read_text(encoding="utf-8")) if scope.exists() else None
            selected_rows = [by_id[key] for key in scope_record["sample_ids"]] if scope_record else records
            if scope_record and "canonical_rows_sha256" in scope_record:
                assert hashlib.sha256(json.dumps(selected_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == scope_record["canonical_rows_sha256"]
            expected = np.array([CLASSES.index(row["answer"]) if dataset == "A" else row["answer"] for row in selected_rows])
            outputs = [json.loads(line)["outputs"] for line in (measured / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            assert len(outputs) == len(selected_rows)
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
            rows_checked += len(selected_rows)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        assert report["not_worse_than_public_baseline"]
        metric = "macro_f1" if dataset == "A" else "rmse"
        if dataset == "A":
            assert report["final"][metric] >= report["baseline"][metric]
        else:
            assert report["final"][metric] <= report["baseline"][metric]
        complete[dataset] = 0
        registry = {"initial": directory / "initial"}
        selected_before = "initial"
        for index, row in enumerate(report["rounds"], 1):
            transcript = [json.loads(line) for line in (directory / f"host-{index}/protocol.jsonl").read_text(encoding="utf-8").splitlines()]
            assert any(e["direction"] == "host" and e["frame"].get("type") == "round_ack" and e["frame"].get("accepted") for e in transcript)
            finish = next(e["frame"]["record"] for e in transcript if e["frame"].get("type") == "round_complete")
            history = next(e["frame"]["records"] for e in transcript if e["frame"].get("type") == "history_response")
            call_ids = [e["frame"]["call_id"] for e in transcript if e["frame"].get("type") == "model_response"]
            assert finish["history_ids"] == [h["id"] for h in history]
            assert finish["model_call_ids"] == call_ids and len(call_ids) >= (1 if optimized else 2)
            if len(call_ids) == 1:
                full = json.loads((directory / f"host-{index}/state/decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
                assert full["selection_calls"].get("selection_reused_plan") is True
                assert full["planning_calls"]["decision_source"] == "model"
                assert full["plan"]["action"] == "observe" and full["plan"]["experiment"] == "none"
                assert full["selected_artifact_id"] == full["plan"]["target_artifact_id"]
                assert full["selected_artifact_id"] == selected_before
                assert not any(e["frame"].get("type") == "candidate_request" for e in transcript)
                response = next(e["frame"]["response"] for e in transcript if e["frame"].get("type") == "model_response")
                text = response["choices"][0]["message"]["content"].strip()
                lines = text.splitlines()
                if len(lines) >= 3 and lines[0].strip().lower() in ("```json", "```") and lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1]).strip()
                assert json.loads(text) == full["plan"]
            assert len(history) == index - 1
            candidate = any(e["frame"].get("type") == "candidate_request" for e in transcript)
            if candidate:
                child_dir = directory / f"host-{index}/candidate"
                registry[f"candidate-{index}"] = child_dir
                if dataset != "A":
                    original = load_model(registry[selected_before] / "checkpoint.npz", dataset)
                    child = load_model(child_dir / "checkpoint.npz", dataset)
                    training = json.loads((child_dir / "training.json").read_text(encoding="utf-8"))
                    assert training["parent_checkpoint_sha256"] == digest(registry[selected_before] / "checkpoint.npz")
                    assert training["checkpoint_sha256"] == digest(child_dir / "checkpoint.npz")
                    assert np.array_equal(original.mean, child.mean) and np.array_equal(original.scale, child.scale)
                    assert all(original.meta[k] == child.meta[k] for k in ("seed", "target_mean", "target_scale"))
                    old_head, new_head = original.meta.get("regression_components"), child.meta.get("regression_components")
                    old_fast, new_fast = original.meta.get("regression_fast_component"), child.meta.get("regression_fast_component")
                    if training["configuration"].get("component_update"):
                        assert all(np.array_equal(getattr(original, group)[name], getattr(child, group)[name])
                                   for group in ("params", "first", "second") for name in original.params)
                        assert all(original.meta[k] == child.meta[k] for k in ("epochs", "steps", "learning_rate"))
                        fast_update = training["configuration"]["component_update"] == "fixed_rbf"
                        prior, updated = (old_fast, new_fast) if fast_update else (old_head, new_head)
                        assert updated["updates"] == (prior["updates"] if prior else 0) + 1
                        assert (old_head == new_head) if fast_update else (old_fast == new_fast)
                        fields = (("cap", "bandwidth", "centers", "width", "space") if fast_update else
                                  ("cap", "means", "scales", "centers", "setting_mean", "setting_scale"))
                        if prior:
                            assert all(prior[k] == updated[k] for k in fields)
                    else:
                        assert child.meta["steps"] > original.meta["steps"] and child.meta["epochs"] > original.meta["epochs"]
                        assert old_head == new_head
                        assert old_fast == new_fast
                    if new_fast is not None:
                        timing = training["prediction_timing"]
                        assert timing["eligible"] and timing["median_cpu_ratio"] <= timing["maximum_cpu_ratio"] <= 2
            assert finish["selected_artifact_id"] in registry
            selected_before = finish["selected_artifact_id"]
            complete[dataset] += complete_experiment({"history_ids": finish["history_ids"],
                "candidate_submitted": candidate, "record": finish})
        if not optimized:
            assert complete[dataset] >= 3
        else:
            assert len(report["rounds"]) == 10
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
