"""回归保护、非线性层与设备留出评估的边界测试。"""
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np

from tests.test_regression import make_fixture
from tests.test_submission import ROOT
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.evaluation import evaluate, promotion_gate
from gas_a.regression.model import fit_bounded_cmapss, frozen_correction
from gas_a.regression.stage_reference import compare_stage, measure_stage
from gas_a.sensor_training.io import digest
from gas_a.regression.submission import experiment_menu
from gas_a.regression.weighting import nasa_proxy, stratified_order
from scripts.validate_regression_generalization import history_features, intervals


class RegressionFollowUpTest(unittest.TestCase):
    def test_bounded_head_preserves_anchor_round_trips_and_respects_current_train(self):
        config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = make_fixture(root, "C")
            train, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", "C")
            parent = load_model(original, "C")
            self.assertNotIn("cmapss_bounded_residual", experiment_menu(config, "C", parent, train))
            for _ in range(2):
                parent = frozen_correction(parent, train, "condition", .1, time.monotonic() + 10)
            self.assertIn("cmapss_bounded_residual", experiment_menu(config, "C", parent, train))
            child = fit_bounded_cmapss(parent, train, time.monotonic() + 10, learning_rate=.0015)
            self.assertEqual(parent.meta["corrections"], child.meta["corrections"])
            np.testing.assert_array_equal(child.predict(validation["x"]), parent.predict(validation["x"]))
            child.meta["bounded_cmapss"]["strength"] = 1.
            checkpoint = root / "bounded.npz"
            child.save(checkpoint)
            restored = load_model(checkpoint, "C")
            self.assertEqual(experiment_menu(config, "C", restored, train), {})
            menu = experiment_menu(config, "C", restored, {**train, "sha256": "new projection"})
            self.assertEqual(list(menu), ["cmapss_bounded_residual"])
            expected = child.predict(validation["x"])
            np.testing.assert_array_equal(restored.predict(validation["x"]), expected)
            np.testing.assert_allclose([restored.predict(x[None])[0] for x in validation["x"]], expected, atol=1e-10)

    def test_bounded_candidate_requires_improvement_on_all_three_metrics(self):
        rules = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]["controller"]
        before = {"dataset": "C", "rmse": 50., "mae": 35., "nasa_score": 300.}
        after = {**before, "rmse": 48., "mae": 34., "nasa_score": 301.}
        self.assertTrue(promotion_gate(before, after, rules)["eligible"])
        self.assertFalse(promotion_gate(before, after, rules, require_all_improved=True)["eligible"])
        after.update(nasa_score=299., mae=35.)
        self.assertIn("mae_not_improved", promotion_gate(before, after, rules, require_all_improved=True)["reasons"])
        after.update(mae=34.)
        self.assertTrue(promotion_gate(before, after, rules, require_all_improved=True)["eligible"])

    def test_device_protection_can_block_overall_gain(self):
        rules = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]["controller"]
        def report(overall, rmse):
            return {"dataset": "B", "rmse": overall, "mae": 75.,
                    "group_metrics": {"device": {"weak": {"rows": 30, "rmse": rmse}}}}
        before = report(100., 100.)
        blocked = promotion_gate(before, report(90., 111.), rules)
        self.assertFalse(blocked["eligible"])
        self.assertEqual(blocked["device_losses"]["weak"]["allowed_increase"], 10.)
        self.assertTrue(promotion_gate(before, report(90., 110.), rules)["eligible"])

    def test_proxy_gradient_and_stratified_epoch(self):
        error = np.array([-70., -1., -.3, 0., .3, 1., 70.])
        delta = 1e-6
        plus, _ = nasa_proxy(error + delta, 84.)
        minus, _ = nasa_proxy(error - delta, 84.)
        loss, gradient = nasa_proxy(error, 84.)
        np.testing.assert_allclose(gradient, (plus - minus) / (2 * delta), atol=1e-7)
        self.assertGreater(loss[-2], loss[1])
        strata = np.array([0] * 50 + [1] * 25 + [2] * 25)
        order = stratified_order(np.random.default_rng(1), strata)
        np.testing.assert_array_equal(np.sort(order), np.arange(100))
        for part in np.array_split(order, 5):
            self.assertEqual(set(strata[part]), {0, 1, 2})

    def test_nonlinear_layers_round_trip_and_preserve_base(self):
        for dataset, transform, bases in (("B", "bearing_log", ("rbf", "tanh", "stumps")),
                                           ("C", "condition_soft", ("linear", "tanh"))):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = make_fixture(root, dataset)
                parent = load_model(path, dataset)
                train, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", dataset)
                for basis in bases:
                    child = frozen_correction(parent, train, transform, .1, time.monotonic() + 30, basis=basis)
                    np.testing.assert_array_equal(parent.predict(validation["x"]), child.predict(validation["x"]))
                    child.meta["corrections"][-1]["strength"] = .5
                    out = root / f"{basis}.npz"
                    child.save(out)
                    restored = load_model(out, dataset)
                    np.testing.assert_array_equal(restored.predict(validation["x"]), child.predict(validation["x"]))
                    for name, tensor in parent.params.items():
                        np.testing.assert_array_equal(restored.params[name], tensor)

    def test_temporal_history_resets_and_intervals_flag_small_group_counts(self):
        x = np.array([[30.], [10.], [20.], [100.], [80.]])
        device = np.array(["a", "a", "a", "b", "a"])
        timestamp = np.array([3, 1, 2, 1, 8])
        got = history_features(x, device, timestamp)
        np.testing.assert_array_equal(got, [[30, 10, 15], [10, 0, 0], [20, 10, 10], [100, 0, 0], [80, 0, 0]])
        predicted = np.arange(10, dtype=float) + 10
        target = predicted + 2
        inadequate = intervals(predicted, target, np.repeat("a", 10), predicted, target, devices=np.repeat("b", 10))
        self.assertIsNone(inadequate["group_max_interval"])
        adequate = intervals(predicted, target, np.arange(10), predicted, target, np.repeat("b", 10))
        self.assertEqual(adequate["group_max_interval"]["radius"], 2.)

    def test_directional_tails_and_fast_epoch_metrics_use_same_gate(self):
        rules = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]["controller"]
        for dataset in ("B", "C"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = make_fixture(root, dataset)
                _, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", dataset)
                if dataset == "B":
                    validation["metadata"]["bearing"] = ["supported"] * len(validation["y"])
                target = validation["y"]
                pred = np.maximum(0., target + np.resize([-25., 1., 0., 10., -1.], target.shape))
                model = load_model(path, dataset)
                parent = evaluate(model, validation, dataset)
                full = evaluate(model, validation, dataset, prediction=pred)
                fast = evaluate(model, validation, dataset, prediction=pred, detailed=False)
                self.assertEqual(promotion_gate(parent, full, rules), promotion_gate(parent, fast, rules))
                self.assertEqual(full["underestimate_count"], int((pred < target).sum()))
                self.assertEqual(full["overestimate_count"], int((pred > target).sum()))
                self.assertGreater(full["underestimate_p99"], full["overestimate_p99"])
                self.assertNotIn("underestimate_p99", fast)
                if dataset == "B":
                    self.assertEqual(full["worst_device"]["device"], "supported")
                else:
                    self.assertEqual(full["nasa_score"], fast["nasa_score"])

    def test_stage_checkpoint_is_reevaluated_on_changed_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_fixture(root, "C")
            _, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", "C")
            parent = load_model(first, "C")
            rnd = SimpleNamespace(dataset_key="C", stage_id="initial", round_index=1,
                                  parent=SimpleNamespace(artifact_id="a"))
            state = root / "state"
            state.mkdir()
            first_metrics = evaluate(parent, validation, "C")
            _, measured = measure_stage(rnd, state, first, digest(first), first_metrics, validation)
            self.assertEqual(compare_stage(measured, first_metrics)["nasa_delta"], 0.)
            next_parent = load_model(first, "C")
            next_parent.params["b2"] += 1
            changed = {**validation, "y": validation["y"] + 12, "sha256": "test-changed"}
            second = root / "second.npz"
            next_parent.save(second)
            rnd.round_index, rnd.parent.artifact_id = 2, "b"
            second_metrics = evaluate(next_parent, changed, "C")
            identity, measured = measure_stage(rnd, state, second, digest(second), second_metrics, changed)
            self.assertEqual(identity["artifact_id"], "a")
            self.assertEqual(measured, evaluate(parent, changed, "C"))
            with self.assertRaises(ValueError):
                compare_stage(first_metrics, second_metrics)
            rnd.stage_id = "next-stage"
            identity, measured = measure_stage(rnd, state, second, digest(second), second_metrics, changed)
            self.assertEqual(identity["provenance"], "sdk_stage_transition")
            self.assertEqual(measured, second_metrics)


if __name__ == "__main__":
    unittest.main()
