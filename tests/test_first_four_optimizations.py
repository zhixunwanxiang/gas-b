"""Public-data continuation, exact-condition evidence and schema compatibility."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost, simulated_checkpoint_identity, writes_only_in
from tests.test_regression import make_fixture, RegressionHost
from contest_harness.round import Round
from gas_a.continuation import continue_candidate
from gas_a.frozen_router import FrozenRouter, load_predictor, _expert_scores, _new_stream
from gas_a.evolution import experiment_evidence, fingerprint, promotion_gate
from gas_a.sensor_training.data import load_pair, evaluate
from gas_a.sensor_training.reference.data import flatten, read_dataset
from gas_a.sensor_training.io import digest
from gas_a.regression.submission import dataset_config, train as regression_train
from gas_a import submission


class FirstFourTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="first-four-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
        cls.train, cls.validation = load_pair(ROOT / "data/public/train.jsonl", ROOT / "data/public/validation.jsonl")
        cls.limits = {**cls.config["limits"], "controller_checkpoint_gate": {
            key: cls.config["controller"][key] for key in (
                "minimum_macro_f1_gain", "minimum_class_support", "maximum_class_recall_drop")}}
        parent = ROOT / "models/reference.npz"
        for index in (1, 2):
            before = evaluate(load_predictor(parent), cls.validation)
            weak = min(before["per_class"], key=lambda name: before["per_class"][name]["f1"])
            child = cls.root / f"route-{index}"
            continue_candidate(parent, cls.train, cls.validation, child, cls.config["experiments"][0],
                               {"weakest_class": weak}, cls.limits)
            parent = child / "checkpoint.npz"
        cls.parent = parent

    def test_complete_epoch_export_improves_report_parent_and_preserves_auxiliaries(self):
        child = self.root / "continued-head"
        parent = load_predictor(self.parent)
        before = evaluate(parent, self.validation)
        spec = next(s for s in self.config["experiments"] if s["name"] == "conservative_head")
        report = continue_candidate(self.parent, self.train, self.validation, child, spec,
                                   {"weakest_class": "Toluene"}, self.limits)
        after = load_predictor(child / "checkpoint.npz")
        metric = evaluate(after, self.validation)
        self.assertTrue(promotion_gate(before, metric, self.config["controller"])["eligible"])
        self.assertGreater(metric["macro_f1"], .874)
        for role in ("five_sensor", "home_trunk", "home_router"):
            self.assertEqual(digest(after.component_paths[role]), digest(parent.component_paths[role]))
        expected = max(report["epoch_history"], key=lambda e: (
            e["checkpoint_eligible"], e["validation"]["macro_f1"], e["validation"]["accuracy"]))
        self.assertEqual(after.meta["epochs"], expected["epoch"])
        self.assertAlmostEqual(metric["macro_f1"], expected["validation"]["macro_f1"])
        self.assertLess(report["anchor_validation_after"]["macro_f1"], metric["macro_f1"])

    def test_sdk_accepts_retained_bundle_then_continues_every_actual_component(self):
        root = self.root / "sdk-retained"
        root.mkdir()
        shutil.copytree(ROOT / "data/public", root / "data")
        parent = self.parent
        state, history = None, []
        for index, experiment in ((3, "conservative_head"), (4, "frozen_router")):
            host = OfflineHost(root, parent, index, history, state, experiment=experiment)
            receipt = host.add_artifact(f"parent-{index}", parent.parent)
            host.frame["parent_artifact"] = receipt
            host.frame["parent_checkpoint_sha256"] = simulated_checkpoint_identity(receipt)
            rnd = Round.start(host, host)
            rnd.parent.root = parent.parent
            work = root / f"work-{index}"
            before = load_predictor(parent)
            with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store):
                submission.train(rnd)
            self.assertTrue(rnd._finished)
            after = load_predictor(host.candidate / "checkpoint.npz")
            self.assertIsInstance(after, FrozenRouter)
            descriptor = next(f["artifact"] for f in host.frames if f["type"] == "candidate_request")
            self.assertEqual(len(descriptor["checkpoint_paths"]), 4)
            self.assertIn("routing.json", descriptor["code_paths"])
            self.assertEqual(len(rnd.call_ids), 2)
            if index == 3:
                self.assertEqual(host.record["selected_artifact_id"], "candidate-3")
                for role in ("five_sensor", "home_trunk", "home_router"):
                    self.assertEqual(digest(before.component_paths[role]), digest(after.component_paths[role]))
                parent = host.candidate / "checkpoint.npz"
            else:
                for role in before.models:
                    self.assertGreater(after.models[role].meta["steps"], before.models[role].meta["steps"])
            state = host.saved_state
            history.append({"id": f"retained-{index}", **host.record})

    def test_no_correction_reuses_scores_and_preserves_stream_values(self):
        model = load_predictor(self.parent)
        x = self.validation["x"][:160]
        # Compare the former redundant formula against the optimized output.
        four = model.models["four_sensor"]
        five = model.models["five_sensor"]
        original = _expert_scores(model.models, dict(model.policy, feature_correction=False), x, _new_stream(four))
        with patch.object(four, "predict", wraps=four.predict) as a, patch.object(five, "predict", wraps=five.predict) as b:
            scores = _expert_scores(model.models, dict(model.policy, feature_correction=False), x, _new_stream(four))
            self.assertEqual(a.call_count, 1)
            self.assertEqual(b.call_count, 1)
        import numpy as np
        np.testing.assert_array_equal(scores[0], (four.predict(x) + four.predict(x)) * .5)
        np.testing.assert_array_equal(scores[1], (five.predict(x) + five.predict(x)) * .5)
        for left, right in zip(original, scores):
            np.testing.assert_array_equal(left, right)

    def test_evidence_separates_old_data_unknown_identity_and_current_rejections(self):
        spec = self.config["experiments"][0]
        execution = {"policy": "current"}
        identity = fingerprint("parent", "train", "validation", spec, execution)
        attempts = [{"fingerprint": identity, "adopted": False, "outcome": "performance_rejected", "macro_f1_gain": -.01}] * 2
        history = [{"id": "old-data", "configuration": {"experiment": spec,
            "parent_checkpoint_sha256": "parent", "train_sha256": "train", "validation_sha256": "old-validation",
            "execution_policy": execution, "candidate_gate": {"eligible": False, "macro_f1_gain": -.2}}},
            {"id": "missing-identity", "configuration": {"experiment": spec, "candidate_gate": []}}]
        def evidence(validation):
            return experiment_evidence(history, attempts, {spec["name"]: spec}, "parent", "train", validation,
                                       execution, self.config["controller"])[spec["name"]]
        current = evidence("validation")
        self.assertEqual(current["current_condition_counted_rejections"], 2)
        self.assertEqual(current["best_current_condition_gain"], -.01)
        self.assertEqual([r["conditions"] for r in current["recent_history"]], ["different", "unknown"])
        changed = evidence("new-validation")
        self.assertEqual(changed["current_condition_attempts"], 0)
        self.assertEqual(changed["remaining_rejection_budget"], 2)
        self.assertIsNone(changed["best_current_condition_gain"])

    def test_c_fine_rate_menu_is_isolated_from_b_and_reaches_sdk_training(self):
        self.assertNotIn("fine_rate", [s["name"] for s in dataset_config(self.config, "B")["experiments"]])
        self.assertIn("fine_rate", [s["name"] for s in dataset_config(self.config, "C")["experiments"]])
        root = self.root / "c-fine-sdk"
        parent = make_fixture(root, "C")
        host = RegressionHost(root, parent, "C")
        host.experiment = "fine_rate"
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(root / "work")}):
            regression_train(rnd)
        self.assertTrue(rnd._finished)
        self.assertEqual(host.record["configuration"]["experiment"]["learning_rate"], .00003)
        plan = json.loads(host.messages[0][0]["content"])
        self.assertIn("fine_rate", plan["experiment_evidence"])
        self.assertEqual(len(rnd.call_ids), 2)

    def test_schema_reuse_keeps_feature_order_and_rejects_later_row_changes(self):
        import numpy as np
        inputs = {"settings": [1., 2., 3.], "sensors": [4., 5.]}
        rows = [{"id": str(i), "inputs": deepcopy(inputs), "answer": float(i)} for i in range(2)]
        path = self.root / "schema.jsonl"
        def write():
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        write()
        data = read_dataset(path, "regression")
        paths, expected = flatten(inputs)
        self.assertEqual(data["paths"], paths)
        np.testing.assert_array_equal(data["x"], [expected, expected])
        for bad in ({"settings": [1., 2.], "sensors": [3., 4., 5.]},
                    {"settings": [1., True, 3.], "sensors": [4., 5.]},
                    {"settings": [1., 2., 3.], "rul": [4., 5.]}):
            rows[1]["inputs"] = bad
            write()
            with self.assertRaises(ValueError):
                read_dataset(path, "regression")


if __name__ == "__main__":
    unittest.main()
