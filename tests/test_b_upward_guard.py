import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_b_upward_guard import compare
from tests.test_regression import make_fixture
from tests.test_submission import ROOT
from gas_a.regression.bearing_guard import METHOD, guarded_delta
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.evaluation import promotion_gate
from gas_a.regression.model import fit_guarded_bearing, frozen_correction
from gas_a.regression.submission import experiment_menu
from gas_a.sensor_training.io import read_json


class BUpwardGuardTests(unittest.TestCase):
    def test_positive_correction_tapers_to_zero_at_230_minutes(self):
        original = np.array([0., 115., 172.5, 230., 759., 859., 1000.])
        correction = guarded_delta(original, np.full_like(original, 100.))
        np.testing.assert_allclose(correction, [100., 100., 50., 0., 0., -20., -30.])

    def test_high_value_reduction_does_not_weaken_a_stronger_downward_correction(self):
        original = np.array([200., 494.5, 759., 859.])
        correction = guarded_delta(original, np.full_like(original, -100.))
        np.testing.assert_allclose(correction, [0., -50., -100., -100.])

    def test_validation_refuses_bearing_or_sample_overlap(self):
        fit = {"ids": ["fit-row"], "y": np.array([0.]), "metadata": {"bearing": ["Bearing3_1"]}}
        for identity, bearing in (("check-row", "Bearing3_1"), ("fit-row", "Bearing3_2"), ("check-row", None)):
            with self.subTest(identity=identity, bearing=bearing):
                check = {"ids": [identity], "y": np.array([0.]), "metadata": {"bearing": [bearing]}}
                with self.assertRaisesRegex(ValueError, "disjoint bearing metadata and sample IDs"):
                    compare(fit, check, None)

    def test_guard_is_offered_on_a_complete_parent_and_round_trips(self):
        config = read_json(ROOT / "submission_config.json")["regression"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = load_model(make_fixture(root, "B"), "B")
            train, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", "B")
            train["metadata"]["bearing"] = ["Bearing1_1", "Bearing1_2"] * (len(train["y"]) // 2)
            self.assertNotIn(METHOD, experiment_menu(config, "B", parent, train))
            for _ in range(2):
                parent = frozen_correction(parent, train, "bearing_log", .1, time.monotonic() + 10.)
            self.assertIn(METHOD, experiment_menu(config, "B", parent, train))
            self.assertNotIn(METHOD, experiment_menu(config, "B", parent, {**train, "metadata": {}}))
            child = fit_guarded_bearing(parent, train, time.monotonic() + 10., learning_rate=.0015)
            features = validation["x"]
            np.testing.assert_array_equal(child.predict(features), parent.predict(features))
            self.assertEqual(child.meta["corrections"], parent.meta["corrections"])
            for group in ("params", "first", "second"):
                for key, value in getattr(parent, group).items():
                    np.testing.assert_array_equal(value, getattr(child, group)[key])
            child.meta["bounded_bearing"]["strength"] = 1.
            path = root / "guard.npz"
            child.save(path)
            restored = load_model(path, "B")
            expected = child.predict(features)
            np.testing.assert_array_equal(restored.predict(features), expected)
            np.testing.assert_allclose([restored.predict(row[None])[0] for row in features], expected, atol=1e-10)
            self.assertEqual(experiment_menu(config, "B", restored, train), {})
            self.assertEqual(list(experiment_menu(config, "B", restored, {**train, "sha256": "changed"})), [METHOD])

    def test_b_requires_both_errors_to_improve_with_no_nasa_metric(self):
        rules = read_json(ROOT / "submission_config.json")["regression"]["controller"]
        before = {"dataset": "B", "rmse": 100., "mae": 80.}
        after = {**before, "rmse": 97.}
        self.assertEqual(promotion_gate(before, after, rules, require_all_improved=True)["reasons"], ["mae_not_improved"])
        self.assertTrue(promotion_gate(before, {**after, "mae": 79.}, rules, require_all_improved=True)["eligible"])


if __name__ == "__main__":
    unittest.main()
