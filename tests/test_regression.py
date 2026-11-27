"""B/C native SDK contracts, numerical continuation and recovery; no paid model calls."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost, writes_only_in
import contest_submission
import numpy as np
from contest_harness.round import Round
from contest_harness.errors import HarnessError
from gas_a.regression import submission as regression
from gas_a.regression.data import BEARING_FEATURES, input_values, load_pair, load_model, schema
from gas_a.regression.evaluation import evaluate, promotion_gate, host_gate, parent_reference
from gas_a.regression.training import continue_candidate, fit_epoch
from gas_a.regression.monitoring import fit_reference, monitor
from gas_a.regression.schedule import ordered_experiments
from gas_a.sensor_training.io import digest
from gas_a.sensor_training.reference.model import Model


def make_inputs(dataset, values):
    if dataset == "C":
        return {"sensors": list(values[:21]), "settings": list(values[21:])}
    return {"features": {channel: dict(zip(sorted(BEARING_FEATURES), values[i * 17:(i + 1) * 17]))
                          for i, channel in enumerate(("horizontal", "vertical"))}}


def make_fixture(root, dataset):
    (root / "data").mkdir(parents=True)
    (root / "previous").mkdir()
    rng = np.random.default_rng(91)
    size = len(schema(dataset))
    for name, n in (("train", 80), ("validation", 40)):
        x = rng.normal(size=(n, size))
        rows = [{"id": f"{name}-{i}", "inputs": make_inputs(dataset, row.tolist()),
                 "answer": float(100 + 10 * row[0]), "meta": {"ignored": "not a feature"}}
                for i, row in enumerate(x)]
        (root / f"data/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    train, _ = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", dataset)
    model = Model("regression", train["paths"], [], train["x"], train["y"], hidden=8, seed=42)
    model.meta.update(profile=dataset, target_unit="minutes" if dataset == "B" else "cycles")
    model.fit(train["x"], train["y"], 2)
    path = root / "previous/checkpoint.npz"
    model.save(path)
    return path


class RegressionHost(OfflineHost):
    """Reuse the SDK's real chunked peer; only the regression host measurement differs."""
    def __init__(self, root, parent, dataset, *args, **kwargs):
        super().__init__(root, parent, *args, experiment="low_rate", **kwargs)
        self.dataset = dataset
        self.frame.update(dataset_key=dataset)
        self.frame["task_contract"].update(task="regression", metric="rmse", metric_direction="lower",
                                          target_unit="minutes" if dataset == "B" else "cycles")

    def checkpoint_paths(self, directory, checkpoint="checkpoint.npz"):
        return [checkpoint]

    def measure_candidate(self):
        _, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", self.dataset)
        return evaluate(load_model(self.candidate / "checkpoint.npz", self.dataset), validation, self.dataset)


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="regression-sdk-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]

    def run_host(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = host.root / f"work-{rnd.round_index}"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertFalse(host.pending)
        return host.saved_state

    def test_native_feature_contracts_outputs_and_no_label_dependency(self):
        for dataset in ("B", "C"):
            root = self.root / dataset
            checkpoint = make_fixture(root, dataset)
            row = json.loads((root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
            sample = SimpleNamespace(task="regression", dataset_key=dataset, inputs=row["inputs"])
            expected = load_model(checkpoint, dataset).predict(np.asarray([input_values(row["inputs"], dataset)[2]]))[0]
            with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(checkpoint)}), writes_only_in():
                result = contest_submission.predict(sample, root)
            self.assertEqual(set(result), {"value"})
            self.assertEqual(result["value"], expected)
            self.assertGreaterEqual(result["value"], 0)
            bad = deepcopy(row["inputs"])
            bad["id"] = {}  # Even empty extra inputs must not sneak past feature flattening.
            with self.assertRaises(ValueError):
                input_values(bad, dataset)
            with self.assertRaises(ValueError):
                load_model(checkpoint, "C" if dataset == "B" else "B")

    def test_dataset_defaults_are_shared_by_schedule_and_first_round_plan(self):
        for dataset, first in (("B", "bearing_balanced"), ("C", "cmapss_condition_mlp")):
            with self.subTest(dataset=dataset):
                names = list(ordered_experiments(self.config, dataset))
                original = list(ordered_experiments(self.config, dataset, mode="configured"))
                sequence = self.config.get("preferred_experiment_sequence", {}).get(dataset, [first])
                self.assertEqual(names, [*sequence, *(name for name in original if name not in sequence)])
                root = self.root / dataset
                host = RegressionHost(root, make_fixture(root, dataset), dataset)
                if dataset == "B":
                    train_path = root / "data/train.jsonl"
                    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
                    for idx, row in enumerate(rows):
                        row["meta"]["bearing"] = f"Bearing1_{1 + idx % 2}"
                    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                host.experiment = None  # The offline reply follows the first available option.
                self.run_host(host)
                request = json.loads(host.messages[0][0]["content"])
                self.assertEqual(next(iter(request["experiments"])), first)
                self.assertEqual(host.record["configuration"]["experiment"]["name"], first)
                if dataset == "C":
                    self.assertEqual(request["strategy"]["default_first_experiment"], first)
                else:
                    self.assertEqual(request["strategy"]["scheduled_experiment_preference"], first)

    def test_b_schedule_falls_back_when_bearing_metadata_is_unavailable(self):
        host = RegressionHost(self.root, make_fixture(self.root, "B"), "B")
        host.experiment = None
        self.run_host(host)
        request = json.loads(host.messages[0][0]["content"])
        self.assertNotIn("scheduled_experiment_preference", request["strategy"])
        self.assertNotIn("bearing_balanced", request["experiments"])
        self.assertEqual(host.record["configuration"]["experiment"]["name"], "low_rate")

    def test_four_rounds_each_dataset_read_history_train_and_finish(self):
        for dataset in ("B", "C"):
            root = self.root / dataset
            parent = make_fixture(root, dataset)
            histories, state = [], None
            for index in range(1, 5):
                host = RegressionHost(root, parent, dataset, index, history=histories, state=state)
                state = self.run_host(host)
                self.assertEqual(host.record["history_ids"], [r["id"] for r in histories])
                self.assertEqual(host.record["model_call_ids"], host.calls)
                self.assertEqual(len(host.calls), 2)
                self.assertIsNotNone(host.candidate)
                candidate = load_model(host.candidate / "checkpoint.npz", dataset)
                previous = load_model(parent, dataset)
                self.assertGreater(candidate.meta["steps"], previous.meta["steps"])
                np.testing.assert_array_equal(previous.mean, candidate.mean)
                np.testing.assert_array_equal(previous.scale, candidate.scale)
                self.assertEqual(candidate.meta["target_mean"], previous.meta["target_mean"])
                self.assertEqual(candidate.meta["target_scale"], previous.meta["target_scale"])
                report = json.loads((host.candidate / "training.json").read_text(encoding="utf-8"))
                self.assertEqual(report,json.loads((host.candidate / "continuation.json").read_text(encoding="utf-8")))
                self.assertEqual(report["parent_checkpoint_sha256"],digest(parent))
                self.assertEqual(report["checkpoint_sha256"],digest(host.candidate / "checkpoint.npz"))
                record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
                self.assertEqual(record["planning_calls"]["prompt_version"], regression.PROMPT_VERSION)
                self.assertNotIn("OFFICIAL_PROMPT_SENTINEL", json.dumps(host.messages))
                histories.append({"id": f"history-{index}", **host.record})
                if host.record["selected_artifact_id"] == f"candidate-{index}":
                    parent = host.candidate / "checkpoint.npz"

    def test_head_only_preserves_hidden_weights_and_adam(self):
        parent = make_fixture(self.root, "C")
        train, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", "C")
        original = load_model(parent, "C")
        spec = next(s for s in self.config["experiments"] if s["name"] == "head_only")
        result = continue_candidate(parent, train, validation, self.root / "candidate", spec,
                                    self.config["limits"], self.config["controller"], "C", digest(parent))
        after = load_model(self.root / "candidate/checkpoint.npz", "C")
        self.assertTrue(all(result["checks"].values()))
        for group in ("params", "first", "second"):
            for key in ("w1", "b1"):
                np.testing.assert_array_equal(getattr(original, group)[key], getattr(after, group)[key])
        self.assertFalse(np.array_equal(original.params["w2"], after.params["w2"]))

    def test_authorized_training_projection_can_change_between_sdk_rounds(self):
        for dataset in ("B","C"):
            root=self.root/dataset
            checkpoint=make_fixture(root,dataset)
            first=RegressionHost(root,checkpoint,dataset,1)
            state=self.run_host(first)
            reference=json.loads((state/"regression_monitor_reference.json").read_text(encoding="utf-8"))
            parent=first.candidate/"checkpoint.npz" if first.record["selected_artifact_id"]=="candidate-1" else checkpoint
            original=load_model(parent,dataset)
            path=root/"data/train.jsonl"
            extra=json.loads(path.read_text(encoding="utf-8").splitlines()[0]);extra["id"]="new-authorized-train-row"
            with path.open("a",encoding="utf-8") as stream: stream.write(json.dumps(extra)+"\n")
            second=RegressionHost(root,parent,dataset,2,history=[{"id":"history-1",**first.record}],state=state)
            next_state=self.run_host(second)
            after=load_model(second.candidate/"checkpoint.npz",dataset)
            np.testing.assert_array_equal(after.mean,original.mean)
            np.testing.assert_array_equal(after.scale,original.scale)
            for key in ("target_mean","target_scale","seed"): self.assertEqual(after.meta[key],original.meta[key])
            self.assertGreater(after.meta["steps"],original.meta["steps"])
            self.assertEqual(json.loads((next_state/"regression_monitor_reference.json").read_text(encoding="utf-8")),reference)
            drift=json.loads((next_state/"monitoring.json").read_text(encoding="utf-8"))
            self.assertTrue(drift["training_projection_changed"])
            self.assertEqual(drift["reference_train_sha256"],reference["train_sha256"])
            self.assertEqual(drift["current_train_sha256"],digest(path))
            self.assertEqual(second.record["history_ids"],["history-1"])

    def test_changed_train_does_not_allow_corrupt_or_cross_dataset_monitor_state(self):
        checkpoint=make_fixture(self.root,"B")
        train,validation=load_pair(self.root/"data/train.jsonl",self.root/"data/validation.jsonl","B")
        reference=fit_reference(train,"B")
        thresholds=json.loads((ROOT/"submission_config.json").read_text(encoding="utf-8"))["controller"]["drift_thresholds"]
        for change in ({"train_sha256":"not-a-digest"},{"dataset":"C"},{"rows":0},{"paths":[]}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                monitor({**reference,**change},validation,train,"B",thresholds)

    def test_sdk_compact_regression_candidate_summary_can_be_adopted(self):
        for dataset in ("B","C"):
            root=self.root/dataset;checkpoint=make_fixture(root,dataset)
            host=RegressionHost(root,checkpoint,dataset)
            original=host.write
            def compact(text):
                result=original(text)
                if json.loads(text)["type"]=="candidate_request":
                    score=host.pending[-1]["summary"]["metrics"]["rmse"]
                    host.pending[-1]["summary"]={"rmse":score}
                return result
            host.write=compact
            self.run_host(host)
            cfg=host.record["configuration"]
            self.assertTrue(cfg["candidate_gate"]["eligible"])
            self.assertTrue(cfg["host_candidate_gate"]["eligible"])
            self.assertEqual(host.record["selected_artifact_id"],"candidate-1")
            decision=json.loads((host.saved_state/"decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertNotIn("scope",decision["host_candidate_receipt"]["summary"])
            self.assertEqual(cfg["host_candidate_gate"]["candidate"]["scope_source"],"sdk_current_candidate_public_validation")

    def test_lower_rmse_and_comparable_host_required(self):
        before = {"rmse": 100., "mae": 80.}
        better = {"rmse": 90., "mae": 79.}
        rules = self.config["controller"]
        self.assertTrue(promotion_gate(before, better, rules)["eligible"])
        self.assertFalse(promotion_gate(before, {"rmse": 101., "mae": 79.}, rules)["eligible"])
        self.assertFalse(promotion_gate(before, {"rmse": 90., "mae": 90.}, rules)["eligible"])
        sha = "sha256:abc"
        parent = parent_reference({}, before, sha)
        receipt = {"summary": {"metrics": {"rmse": 90.}, "scope": "public_validation", "validation_sha256": sha}}
        self.assertTrue(host_gate(receipt, parent, better, sha, rules)["eligible"])
        receipt["summary"]["metrics"]["rmse"] = 89.
        self.assertFalse(host_gate(receipt, parent, better, sha, rules)["eligible"])
        receipt["summary"]["validation_sha256"] = "wrong"
        self.assertFalse(host_gate(receipt, parent, better, sha, rules)["eligible"])

    def test_local_failure_retains_parent_and_rpc_failure_stops(self):
        checkpoint = make_fixture(self.root, "B")
        host = RegressionHost(self.root, checkpoint, "B")
        with patch.object(regression, "continue_candidate", side_effect=TimeoutError("injected local deadline")):
            self.run_host(host)
        self.assertEqual(host.record["selected_artifact_id"], "parent-1")
        self.assertIsNone(host.candidate)
        self.assertEqual(len(host.calls), 2)
        host = RegressionHost(self.root, checkpoint, "B", 2)
        with patch.object(regression, "continue_candidate", side_effect=HarnessError("RPC_OUTCOME_UNKNOWN")), self.assertRaises(HarnessError):
            self.run_host(host)
        self.assertEqual(len(host.calls), 1)
        self.assertIsNone(host.record)

    def test_historical_regression_rollback_reevaluated_with_lower_rmse(self):
        checkpoint = make_fixture(self.root, "C")
        good = self.root / "good"
        good.mkdir()
        shutil.copyfile(checkpoint, good / "checkpoint.npz")
        damaged = load_model(checkpoint, "C")
        damaged.params["b2"] += 100
        fault = self.root / "previous/fault.npz"
        damaged.save(fault)
        host = RegressionHost(self.root, fault, "C", history=[{"id": "past", "selected_artifact_id": "good"}])
        host.action = "rollback"
        host.add_artifact("good", good)
        self.run_host(host)
        self.assertEqual(host.record["selected_artifact_id"], "good")
        self.assertEqual(host.record["history_ids"], ["past"])
        self.assertIsNone(host.candidate)
        record = json.loads((host.saved_state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertTrue(record["historical_validation"]["good"]["reevaluated_this_round"])

    def test_sdk_prediction_jsonl_is_native_and_label_free(self):
        for dataset in ("B", "C"):
            root = self.root / dataset
            checkpoint = make_fixture(root, dataset)
            row = json.loads((root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
            env = {k: v for k, v in os.environ.items() if not k.startswith("CONTEST_")}
            context = {"protocol": "contest.predict-context.v1", "dataset_key": dataset, "task": "regression",
                       "artifact_id": "selected", "artifact": {"checkpoint_paths": [checkpoint.name]}}
            env.update(CONTEST_ARTIFACT_PATH=str(checkpoint), CONTEST_PREDICTION_CONTEXT=json.dumps(context))
            result = subprocess.run([sys.executable, "-B", str(ROOT / "harness.py"), "predict"],
                input=(json.dumps({"task": "regression", "inputs": row["inputs"]}) + "\n") * 2,
                capture_output=True, text=True, encoding="utf-8", env=env, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(set(line) == {"outputs"} and set(line["outputs"]) == {"value"}
                                and line["outputs"]["value"] >= 0 for line in lines))

    def test_mse_epoch_matches_official_numpy_continuation(self):
        checkpoint = make_fixture(self.root, "B")
        train, _ = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", "B")
        official, actual = load_model(checkpoint, "B"), load_model(checkpoint, "B")
        spec = self.config["experiments"][0]
        official.meta["learning_rate"] = actual.meta["learning_rate"] = spec["learning_rate"]
        official.fit(train["x"], train["y"], 1)
        fit_epoch(actual, train, spec, time.monotonic() + 30)
        self.assertEqual(official.meta, actual.meta)
        for group in ("params", "first", "second"):
            for name in official.params:
                np.testing.assert_array_equal(getattr(official, group)[name], getattr(actual, group)[name])

    def test_invalid_model_decisions_get_bounded_correction_without_training(self):
        checkpoint = make_fixture(self.root, "C")
        host = RegressionHost(self.root, checkpoint, "C")
        host.bad_plan = True
        state = self.run_host(host)
        self.assertEqual(len(host.calls), 3)
        self.assertIsNone(host.candidate)
        self.assertEqual(host.record["selected_artifact_id"], "parent-1")
        record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(record["planning_calls"]["decision_source"], "program_guard")

    def test_identical_local_failures_are_blocked_on_next_round(self):
        checkpoint = make_fixture(self.root, "B")
        state = None
        for index in (1, 2):
            host = RegressionHost(self.root, checkpoint, "B", index, state=state)
            with patch.object(regression, "continue_candidate", side_effect=TimeoutError("local failure")):
                state = self.run_host(host)
        host = RegressionHost(self.root, checkpoint, "B", 3, state=state)
        host.action = "observe"
        self.run_host(host)
        request = json.loads(host.messages[0][0]["content"])
        self.assertNotIn("low_rate", request["experiments"])
        self.assertIn("low_rate", request["strategy"]["blocked_repeated_experiments"])

    def test_nasa_score_uses_asymmetric_error_and_nonnegative_outputs(self):
        class Measured:
            def predict(self, _):
                return np.array([87., 110.])
        data = {"x": np.zeros((2, 24)), "y": np.array([100., 100.])}
        measured = evaluate(Measured(), data, "C")
        self.assertAlmostEqual(measured["nasa_score"], 2 * np.expm1(1.0))
        for dataset in ("B", "C"):
            size = len(schema(dataset))
            values = [0.] * size
            values[0] = True
            with self.assertRaises(ValueError):
                input_values(make_inputs(dataset, values), dataset)
            values[0] = float("nan")
            with self.assertRaises(ValueError):
                input_values(make_inputs(dataset, values), dataset)


if __name__ == "__main__":
    unittest.main()
