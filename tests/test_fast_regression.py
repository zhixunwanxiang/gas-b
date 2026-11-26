"""Fixed component inheritance, complete prediction, timeout and speed admission."""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import contest_submission
from contest_harness.round import Round
from gas_a.regression import submission
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.evaluation import evaluate
from gas_a.regression.fast_component import KEY, PREDICTOR_FORMAT
from gas_a.regression.model import PREDICTOR_FORMAT as CONDITION_PREDICTOR_FORMAT
from gas_a.regression.training import continue_candidate
from gas_a.sensor_training.io import digest
from gas_a.sensor_training.reference.model import Model as OriginalModel
from scripts.validate_fast_bc import run_bounded, worker
from tests.test_regression import make_fixture, RegressionHost
from tests.test_submission import ROOT, writes_only_in

TIMING = {"eligible": True, "median_cpu_ratio": 1.25, "maximum_cpu_ratio": 2.}


class FastRegressionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="fast-regression-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))

    def fixture(self, dataset):
        root = self.root / dataset
        parent = make_fixture(root, dataset)
        train, val = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", dataset)
        cfg = submission.dataset_config(self.config, dataset)
        spec = next(s for s in cfg["experiments"] if s.get("component_update") == "fixed_rbf")
        return root, parent, train, val, cfg, spec

    def fit(self, parent, train, val, cfg, spec, output):
        dataset = "C" if len(train["paths"]) == 24 else "B"
        before = evaluate(load_model(parent, dataset), val, dataset)
        with patch("gas_a.regression.training.measure_prediction_time", return_value=deepcopy(TIMING)):
            return continue_candidate(parent, train, val, output, spec, cfg["limits"], cfg["controller"], dataset,
                                      digest(parent), reference=before)

    def test_zero_migration_reloads_and_old_loader_refuses_new_format(self):
        for dataset in "BC":
            with self.subTest(dataset=dataset):
                root, parent, train, val, _, spec = self.fixture(dataset)
                model = load_model(parent, dataset)
                if dataset == "C":
                    model.initialize_component(train)
                    model.update_component(train)
                expected = model.predict(val["x"])
                self.assertTrue(model.initialize_fast_component(train, spec))
                np.testing.assert_array_equal(model.predict(val["x"]), expected)
                saved = root / "complete.npz"
                model.save(saved)
                loaded = load_model(saved, dataset)
                np.testing.assert_array_equal(loaded.predict(val["x"]), expected)
                self.assertEqual(loaded.meta["format"], PREDICTOR_FORMAT)
                with self.assertRaises(ValueError):
                    OriginalModel.load(saved)
                model.meta["format"] = CONDITION_PREDICTOR_FORMAT
                with self.assertRaises(ValueError):
                    model.validate()

    def test_resuming_updates_same_basis_and_preserves_neural_adam_and_scalers(self):
        root, parent, train, val, cfg, spec = self.fixture("B")
        original = load_model(parent, "B")
        source = parent
        for i in range(1, 3):
            output = root / f"update-{i}"
            report = self.fit(source, train, val, cfg, spec, output)
            child = load_model(output / "checkpoint.npz", "B")
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["executed_epochs"], 0)
            self.assertEqual(child.meta[KEY]["updates"], i)
            self.assertEqual(child.meta["steps"], original.meta["steps"])
            for group in ("params", "first", "second"):
                for name in original.params:
                    np.testing.assert_array_equal(getattr(original, group)[name], getattr(child, group)[name])
            np.testing.assert_array_equal(child.scale, original.scale)
            if i == 2:
                self.assertEqual(child.meta[KEY]["centers"], previous.meta[KEY]["centers"])
                self.assertEqual(child.meta[KEY]["cap"], previous.meta[KEY]["cap"])
            previous, source = child, output / "checkpoint.npz"

    def test_both_c_heads_survive_component_and_neural_updates(self):
        root, parent, train, val, cfg, spec = self.fixture("C")
        model = load_model(parent, "C")
        model.initialize_component(train)
        model.update_component(train)
        combined_parent = root / "condition.npz"
        model.save(combined_parent)
        first = root / "fast"
        self.fit(combined_parent, train, val, cfg, spec, first)
        fitted = load_model(first / "checkpoint.npz", "C")
        self.assertEqual(fitted.meta["regression_components"], model.meta["regression_components"])
        ridge = next(s for s in cfg["experiments"] if s["name"] == "condition_ridge")
        second = root / "ridge"
        self.fit(first / "checkpoint.npz", train, val, cfg, ridge, second)
        updated = load_model(second / "checkpoint.npz", "C")
        self.assertEqual(updated.meta[KEY], fitted.meta[KEY])
        self.assertEqual(updated.meta["regression_components"]["updates"], 2)
        neural = {**next(s for s in cfg["experiments"] if s["name"] == "head_short"), "epochs": 1}
        third = root / "neural"
        self.fit(second / "checkpoint.npz", train, val, cfg, neural, third)
        resumed = load_model(third / "checkpoint.npz", "C")
        self.assertEqual(resumed.meta[KEY], updated.meta[KEY])
        self.assertEqual(resumed.meta["regression_components"], updated.meta["regression_components"])
        self.assertGreater(resumed.meta["steps"], updated.meta["steps"])

    def test_validation_labels_do_not_change_fit_and_single_row_matches_batch(self):
        root, parent, train, val, cfg, spec = self.fixture("B")
        spec = next(s for s in cfg["experiments"] if s["name"] == "fast_rbf_log")
        first, second = root / "first", root / "second"
        self.fit(parent, train, val, cfg, spec, first)
        changed = {**val, "y": val["y"] + 300}
        self.fit(parent, train, changed, cfg, spec, second)
        self.assertEqual(digest(first / "checkpoint.npz"), digest(second / "checkpoint.npz"))
        model = load_model(first / "checkpoint.npz", "B")
        x = np.vstack((val["x"][:4], np.full((1, 34), 1e100)))
        np.testing.assert_allclose(model.predict(x), np.array([model.predict(row[None, :])[0] for row in x]), rtol=1e-12, atol=1e-10)

    def test_invalid_component_scales_counts_and_inherited_basis_are_rejected(self):
        _, parent, train, _, _, spec = self.fixture("B")
        original = load_model(parent, "B")
        original.initialize_fast_component(train, spec)
        for key, value in (("bandwidth", 0), ("cap", float("inf")), ("parameter_count", 1), ("weights", [0])):
            model = deepcopy(original)
            model.meta[KEY][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                model.validate()
        with self.assertRaises(ValueError):
            original.initialize_fast_component(train, {**spec, "basis_width": 16})

    def test_speed_failure_blocks_sdk_upload_even_with_accepted_numerical_gate(self):
        root, parent, _, _, cfg, spec = self.fixture("B")
        host = RegressionHost(root, parent, "B")
        host.experiment = spec["name"]
        cfg["enable_research_experiments"] = True
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        work = root / "work"
        with patch.object(submission, "dataset_config", return_value=cfg), \
                patch.object(submission, "promotion_gate", return_value={"eligible": True, "rmse_reduction": 1.}), \
                patch("gas_a.regression.training.measure_prediction_time", return_value={**TIMING, "eligible": False, "median_cpu_ratio": 3.}), \
                patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertIsNone(host.candidate)
        record = json.loads((host.saved_state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertIn("prediction_cpu_budget", record["configuration"]["candidate_gate"]["guard_failures"])

    def test_worker_timeout_kills_and_reaps_process(self):
        child = unittest.mock.Mock()
        child.wait.side_effect = [__import__("subprocess").TimeoutExpired("child", .1), 0]
        child.returncode = -9
        with patch("scripts.validate_fast_bc.subprocess.Popen", return_value=child):
            code, expired = run_bounded(self.root / "job.json", self.root / "timeout.log", .1)
        self.assertEqual(code, -9)
        self.assertTrue(expired)
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)

    def test_changed_registered_data_stops_before_fitting(self):
        root, parent, train, _, _, _ = self.fixture("B")
        path = root / "job.json"
        path.write_text(json.dumps({"dataset": "B", "parent": str(parent), "sources": {},
            "train": str(root / "data/train.jsonl"), "validation": str(root / "data/validation.jsonl"),
            "train_sha256": train["sha256"], "validation_sha256": "wrong"}), encoding="utf-8")
        with patch("gas_a.regression.training.continue_candidate") as fitting:
            with self.assertRaisesRegex(ValueError, "Dataset changed"):
                worker(path)
            fitting.assert_not_called()

    def test_log_component_neural_derivative_matches_finite_differences(self):
        _, parent, train, val, cfg, _ = self.fixture("B")
        spec = next(s for s in cfg["experiments"] if s["name"] == "fast_rbf_log")
        model = load_model(parent, "B")
        model.initialize_fast_component(train, spec)
        model.meta[KEY]["weights"][0] = model.meta[KEY]["cap"] * .5
        model.validate()
        x = model.normalized(val["x"][:1])
        target = (val["y"][0] - model.meta["target_mean"]) / model.meta["target_scale"]
        for bias in (0., -100.):
            tested = deepcopy(model)
            tested.params["b2"] += bias
            error = tested.forward(x)[1][0, 0] - target
            analytic = 2 * error * tested.neural_output_derivative(x)[0]
            epsilon = 1e-5
            tested.params["b2"] += epsilon
            upper = (tested.forward(x)[1][0, 0] - target) ** 2
            tested.params["b2"] -= 2 * epsilon
            lower = (tested.forward(x)[1][0, 0] - target) ** 2
            self.assertAlmostEqual((upper - lower) / (2 * epsilon), analytic, delta=abs(analytic) * 1e-8 + 1e-8)

    def test_sdk_submits_and_independently_predicts_complete_fast_checkpoint(self):
        root, parent, _, val, cfg, spec = self.fixture("B")
        model = load_model(parent, "B")
        model.params["b2"] += 2
        parent.unlink()
        model.save(parent)
        host = RegressionHost(root, parent, "B")
        host.experiment = spec["name"]
        cfg["enable_research_experiments"] = True
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        work = root / "work"
        with patch.object(submission, "dataset_config", return_value=cfg), \
                patch("gas_a.regression.training.measure_prediction_time", return_value=deepcopy(TIMING)), \
                patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertIsNotNone(host.candidate)
        child = load_model(host.candidate / "checkpoint.npz", "B")
        sample = SimpleNamespace(dataset_key="B", task="regression", inputs=json.loads(
            (root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()[0])["inputs"])
        with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(host.candidate / "checkpoint.npz")}):
            result = contest_submission.predict(sample, host.candidate)
        self.assertAlmostEqual(result["value"], child.predict(val["x"][:1])[0])
        self.assertEqual(child.meta[KEY]["updates"], 1)
        self.assertEqual(child.meta["steps"], model.meta["steps"])


if __name__ == "__main__":
    unittest.main()
