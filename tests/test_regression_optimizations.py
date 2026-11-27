"""B/C 加权、复合预测、NASA 准入及本地拒绝去重的行为验证。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import contest_submission
from contest_harness.round import Round
from gas_a.evolution import available_experiments, REPEAT_POLICY
from gas_a.regression import submission
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.diagnostics import conditions, error_metrics, groups
from gas_a.regression.evaluation import evaluate, promotion_gate
from gas_a.regression.model import frozen_correction, MAX_CORRECTIONS
from gas_a.regression.training import continue_candidate, fit_epoch
from gas_a.regression.weighting import sample_weights, resolve_training_spec
from gas_a.sensor_training.io import digest
from gas_a.sensor_training.reference.model import Model
from tests.test_regression import make_fixture, RegressionHost
from tests.test_submission import ROOT, writes_only_in


class RegressionOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bc-optimizations-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]

    def fixture(self, dataset="C"):
        parent = make_fixture(self.root, dataset)
        train, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", dataset)
        return parent, load_model(parent, dataset), train, validation

    def test_nasa_direction_tail_contribution_and_overflow_are_reported(self):
        metrics = error_metrics(np.asarray([-20., 20., 0., 0.]), "C")
        self.assertAlmostEqual(metrics["nasa_score"], np.expm1(20 / 13) + np.expm1(2))
        self.assertGreater(metrics["nasa_overestimate_score"], metrics["nasa_underestimate_score"])
        self.assertAlmostEqual(metrics["nasa_top_1pct_share"], np.expm1(2) / metrics["nasa_score"])
        extreme = error_metrics(np.asarray([10000.]), "C")
        self.assertTrue(extreme["nasa_score_overflow"])
        self.assertIsNone(extreme["nasa_score"])
        json.dumps(extreme, allow_nan=False)

    def test_nasa_guard_rejects_rmse_gain_and_incomparable_scope(self):
        before = {"dataset": "C", "rows": 10, "validation_sha256": "sha256:a", "rmse": 100., "mae": 80., "nasa_score": 100.}
        after = {**before, "rmse": 90., "mae": 79., "nasa_score": 120.}
        rules = self.config["controller"]
        self.assertIn("nasa_regression", promotion_gate(before, after, rules)["reasons"])
        after["nasa_score"] = 104.
        self.assertTrue(promotion_gate(before, after, rules)["eligible"])
        self.assertFalse(promotion_gate(before, {**after, "rows": 20}, rules)["eligible"])
        self.assertFalse(promotion_gate(before, {**after, "nasa_score": None}, rules)["eligible"])

    def test_condition_is_inputs_only_and_engine_ids_include_subset(self):
        x = np.zeros((3, 24))
        x[:, 21:] = [[-.0007, -.0004, 100], [25, .62, 60], [90, 2, 200]]
        np.testing.assert_array_equal(conditions(x), [0, 3, -1])
        data = {"x": x, "y": np.zeros(3), "metadata": {"source_subset": ["FD001", "FD003", None],
            "source_split": ["train", "train", None], "unit": [1, 1, None]}}
        self.assertEqual(groups(data, "C")["device"].tolist(), ["FD001/train/1", "FD003/train/1", "unknown"])

    def test_weight_normalization_train_only_and_huber_calibration(self):
        _, model, train, _ = self.fixture("B")
        train["metadata"]["bearing"] = ["short"] * 10 + ["long"] * 70
        weights = sample_weights(train, {"device_balance": .25}, "B")
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertAlmostEqual(weights[0] / weights[-1], 7 ** .25)
        tail = sample_weights(train, {"tail_boost": 1.5}, "B")
        self.assertAlmostEqual(tail.max() / tail.min(), 1.5)
        config = resolve_training_spec(model, train, {"huber_quantile": .8})
        self.assertTrue(.1 <= config["huber_delta"] <= 2)
        with self.assertRaises(ValueError):
            sample_weights({**train, "metadata": {}}, {"device_balance": .25}, "B")

    def test_zero_layer_is_exact_and_serialized_composite_cannot_be_silently_truncated(self):
        _, parent, train, validation = self.fixture()
        corrected = frozen_correction(parent, train, "condition", .1, time.monotonic() + 10)
        np.testing.assert_array_equal(parent.predict(validation["x"]), corrected.predict(validation["x"]))
        corrected.meta["corrections"][-1]["strength"] = .5
        checkpoint = self.root / "composite.npz"
        corrected.save(checkpoint)
        restored = load_model(checkpoint, "C")
        np.testing.assert_array_equal(restored.predict(validation["x"]), corrected.predict(validation["x"]))
        with self.assertRaises(ValueError):
            Model.load(checkpoint)
        bad = deepcopy(corrected)
        bad.meta["corrections"][0]["statistics"]["scale"][0] = 0.
        with self.assertRaises(ValueError):
            bad.validate()

    def test_correction_fit_preserves_parent_and_train_statistics_ignore_validation(self):
        path, parent, train, validation = self.fixture("B")
        spec = next(s for s in self.config["experiments"] if s["name"] == "bearing_log_ridge")
        original_sha = digest(path)
        report = continue_candidate(path, train, validation, self.root / "candidate", spec,
            self.config["limits"], self.config["controller"], "B", original_sha)
        after = load_model(self.root / "candidate/checkpoint.npz", "B")
        self.assertEqual(digest(path), original_sha)
        self.assertTrue(all(report["checks"].values()))
        for group in ("params", "first", "second"):
            for key in parent.params:
                np.testing.assert_array_equal(getattr(parent, group)[key], getattr(after, group)[key])
        reference = frozen_correction(parent, train, "bearing_log", .1, time.monotonic() + 10)
        self.assertEqual(reference.meta["corrections"][0]["statistics"], after.meta["corrections"][0]["statistics"])
        self.assertEqual(after.meta["steps"], parent.meta["steps"])

    def test_mlp_continuation_uses_complete_prediction_with_frozen_correction(self):
        _, parent, train, _ = self.fixture()
        corrected = frozen_correction(parent, train, "condition", .1, time.monotonic() + 10)
        layer = corrected.meta["corrections"][-1]
        layer["coefficients"] = [3.] + [0.] * (len(layer["coefficients"]) - 1)
        layer["strength"] = 1.
        equivalent = deepcopy(parent)
        equivalent.params["b2"] += 3 / parent.meta["target_scale"]
        spec = self.config["experiments"][0]
        original_layer = deepcopy(layer)
        fit_epoch(corrected, train, spec, time.monotonic() + 10)
        fit_epoch(equivalent, train, spec, time.monotonic() + 10)
        self.assertEqual(original_layer, corrected.meta["corrections"][-1])
        for key in corrected.params:
            expected = equivalent.params[key] - (3 / parent.meta["target_scale"] if key == "b2" else 0)
            np.testing.assert_allclose(corrected.params[key], expected, rtol=1e-9, atol=1e-10)

    def test_completed_local_rejection_records_attempt_without_artifact_and_blocks_exact_repeat(self):
        path, _, _, _ = self.fixture("B")
        host = RegressionHost(self.root, path, "B")
        real_continue = submission.continue_candidate
        def rejected(*args, **kwargs):
            report = real_continue(*args, **kwargs)
            report["validation_after"] = report["validation_before"]
            return report
        rnd = Round.start(host, host)
        rnd.parent.root = path.parent
        work = self.root / "work"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), patch.object(submission, "continue_candidate", side_effect=rejected), writes_only_in(work, host.store):
            contest_submission.train(rnd)
        self.assertIsNone(host.candidate)
        self.assertEqual(host.record["selected_artifact_id"], "parent-1")
        policy = json.loads((host.saved_state / "regression_strategy.json").read_text(encoding="utf-8"))
        self.assertEqual(len(policy["attempts"]), 1)
        attempt = policy["attempts"][0]
        self.assertEqual(attempt["outcome"], "performance_rejected")
        self.assertIsNone(attempt["candidate_artifact_id"])
        cfg = host.record["configuration"]
        self.assertEqual(cfg["execution_policy"]["repeat_policy"], REPEAT_POLICY)
        train, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", "B")
        allowed, blocked = available_experiments(policy, {"low_rate": self.config["experiments"][0]}, rnd,
            train, validation, self.config["controller"], cfg["execution_policy"])
        self.assertEqual(blocked, ["low_rate"])
        validation = {**validation, "sha256": "changed"}
        allowed, _ = available_experiments(policy, {"low_rate": self.config["experiments"][0]}, rnd,
            train, validation, self.config["controller"], cfg["execution_policy"])
        self.assertIn("low_rate", allowed)

    def test_layer_budget_and_config_limits(self):
        _, parent, train, _ = self.fixture()
        for _ in range(MAX_CORRECTIONS):
            parent = frozen_correction(parent, train, "condition", .1, time.monotonic() + 10)
        menu = submission.experiment_menu(self.config, "C", parent, train)
        self.assertNotIn("cmapss_condition_ridge", menu)
        with self.assertRaises(ValueError):
            frozen_correction(parent, train, "condition", .1, time.monotonic() + 10)
        config = deepcopy(self.config)
        next(spec for spec in config["experiments"] if spec["name"] == "cmapss_condition_ridge")["strengths"] = [float("nan")]
        with self.assertRaises(ValueError):
            submission.validate_config(config, config["limits"])

    def test_historical_composite_rollback_and_next_round_keep_all_components(self):
        path, parent, train, validation = self.fixture()
        corrected = frozen_correction(parent, train, "condition", .1, time.monotonic() + 10)
        corrected.meta["corrections"][-1]["strength"] = .25
        historical = self.root / "historical"
        historical.mkdir()
        corrected.save(historical / "checkpoint.npz")
        damaged = deepcopy(parent)
        damaged.params["b2"] += 10
        fault = self.root / "previous/fault.npz"
        damaged.save(fault)
        first = RegressionHost(self.root, fault, "C", history=[{"id": "past", "selected_artifact_id": "composite-good"}])
        first.add_artifact("composite-good", historical)
        first.action = "rollback"
        rnd = Round.start(first, first)
        rnd.parent.root = fault.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "work-1")}), writes_only_in(self.root / "work-1", first.store):
            contest_submission.train(rnd)
        self.assertEqual(first.record["selected_artifact_id"], "composite-good")
        second = RegressionHost(self.root, historical / "checkpoint.npz", "C", 2,
            history=[{"id": "rollback", **first.record}], state=first.saved_state)
        receipt = second.add_artifact("composite-good", historical)
        second.frame["parent_artifact"] = receipt
        rnd = Round.start(second, second)
        rnd.parent.root = historical
        work = self.root / "work-2"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, second.store):
            contest_submission.train(rnd)
        exported = next(work.glob("round-*/candidate/checkpoint.npz"))
        child = load_model(exported, "C")
        report = json.loads((exported.parent / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["parent_checkpoint_sha256"], digest(historical / "checkpoint.npz"))
        self.assertEqual(child.meta["corrections"], corrected.meta["corrections"])
        self.assertGreater(child.meta["steps"], corrected.meta["steps"])
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
