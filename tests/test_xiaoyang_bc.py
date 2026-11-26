"""Complete xiaoyang components, inheritance, actual SDK export and train-only fitting."""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import contest_submission
from contest_harness.round import Round
from gas_a.regression import submission
from gas_a.regression.composite import KEY, MAX_LAYERS, read_layer, unpack
from gas_a.regression.data import load_pair, load_model
from gas_a.regression.evaluation import evaluate
from gas_a.regression.forests import Forest
from gas_a.regression.training import continue_candidate
from gas_a.regression.reference_model import COMPLETE_FORMAT
from gas_a.sensor_training.io import digest
from gas_a.sensor_training.reference.model import Model as TaskAModel
from tests.test_regression import make_fixture, RegressionHost
from tests.test_submission import ROOT, writes_only_in


class XiaoyangRegressionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="xiaoyang-bc-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))

    def fixture(self, dataset):
        root = self.root / dataset
        parent = make_fixture(root, dataset)
        if dataset == "B":
            path = root / "data/validation.jsonl"
            rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
            for index, row in enumerate(rows): row["meta"]["bearing"] = "bearing-" + str(index % 2)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        train, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", dataset)
        return root, parent, train, validation

    def spec(self, name):
        return next(s for s in self.config["regression"]["experiments"] if s["name"] == name)

    def fit(self, parent, train, validation, output, spec, dataset):
        cfg = self.config["regression"]
        if dataset == "B" and "baseline_groups" not in validation:
            validation["baseline_groups"] = evaluate(load_model(parent, dataset), validation, dataset)["groups"]
        return continue_candidate(parent, train, validation, output, spec, cfg["limits"], cfg["controller"], dataset, digest(parent))

    def test_bearing_two_layer_resume_keeps_prefix_optimizer_and_complete_format(self):
        root, parent, train, val = self.fixture("B")
        original = load_model(parent, "B")
        source, previous = parent, None
        for index in (1, 2):
            out = root / f"tree-{index}"
            report = self.fit(source, train, val, out, self.spec("bearing_tree_mix"), "B")
            child = load_model(out / "checkpoint.npz", "B")
            self.assertEqual(child.meta["format"], COMPLETE_FORMAT)
            self.assertEqual(len(child.tree_layers), index)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(child.meta["epochs"], original.meta["epochs"] + index)
            self.assertGreater(child.meta["steps"], original.meta["steps"])
            np.testing.assert_array_equal(child.params["w1"], original.params["w1"])
            np.testing.assert_array_equal(child.first["w1"], original.first["w1"])
            np.testing.assert_array_equal(child.scale, original.scale)
            if previous:
                self.assertEqual(child.meta[KEY]["layers"][:-1], previous.meta[KEY]["layers"])
            with self.assertRaises(ValueError): TaskAModel.load(out / "checkpoint.npz")
            source, previous = out / "checkpoint.npz", child
        after = root / "neural"
        self.fit(source, train, val, after, self.spec("low_rate"), "B")
        resumed = load_model(after / "checkpoint.npz", "B")
        self.assertEqual(resumed.meta[KEY], previous.meta[KEY])
        self.assertGreater(resumed.meta["steps"], previous.meta["steps"])
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            resumed.append_layer(unpack(resumed.meta[KEY]["layers"][0]))

    def test_engine_anchor_radius_and_layers_survive_condition_residual_update(self):
        root, parent, train, val = self.fixture("C")
        first = root / "first"
        self.fit(parent, train, val, first, self.spec("engine_tail_tree"), "C")
        one = load_model(first / "checkpoint.npz", "C")
        second = root / "second"
        self.fit(first / "checkpoint.npz", train, val, second, self.spec("engine_cumulative_tree"), "C")
        two = load_model(second / "checkpoint.npz", "C")
        self.assertEqual(two.meta[KEY]["anchor"], one.meta[KEY]["anchor"])
        self.assertEqual(two.meta[KEY]["radius"], one.meta[KEY]["radius"])
        self.assertEqual(two.meta[KEY]["layers"][0], one.meta[KEY]["layers"][0])
        x = np.vstack((val["x"], np.full((1, 24), 1e6), np.full((1, 24), -1e6)))
        anchor = np.maximum(0, two.anchor.predict(x) + two.tree_layers[0].correction(x))
        values = two.predict(x)
        self.assertTrue(np.all(values >= np.maximum(0, anchor - two.radius) - 1e-10))
        self.assertTrue(np.all(values <= anchor + two.radius + 1e-10))
        third = root / "residual"
        report = self.fit(second / "checkpoint.npz", train, val, third, self.spec("condition_residual"), "C")
        residual = load_model(third / "checkpoint.npz", "C")
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(residual.meta[KEY], two.meta[KEY])
        self.assertIn("rul_residual_v1", residual.meta)
        np.testing.assert_allclose(residual.predict(x), np.array([residual.predict(r[None, :])[0] for r in x]), rtol=1e-10, atol=1e-9)

    def test_validation_answers_do_not_change_single_epoch_tree_fit(self):
        root, parent, train, val = self.fixture("C")
        first, second = root / "a", root / "b"
        self.fit(parent, train, val, first, self.spec("engine_tail_tree"), "C")
        self.fit(parent, train, {**val, "y": val["y"] + 300}, second, self.spec("engine_tail_tree"), "C")
        self.assertEqual(digest(first / "checkpoint.npz"), digest(second / "checkpoint.npz"))

    def test_embedded_digest_and_tree_structure_fail_closed(self):
        root, parent, train, val = self.fixture("B")
        out = root / "tree"
        self.fit(parent, train, val, out, self.spec("bearing_tree_mix"), "B")
        model = load_model(out / "checkpoint.npz", "B")
        bad = deepcopy(model.meta[KEY]["layers"][0]); bad["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"): unpack(bad)
        malformed = {"method": np.array("extra_trees"), "intercept": np.array(0.), "value": np.array([1., 2.]),
            "threshold": np.zeros(2), "feature": np.zeros(2, dtype=int), "left": np.array([0, -1]),
            "right": np.array([1, -1]), "starts": np.array([0])}
        with self.assertRaises(ValueError): Forest(malformed, 51)
        with self.assertRaises(ValueError): read_layer(b"not-an-npz", "B")

    def test_candidate_deadline_includes_load_and_initial_measurement(self):
        root, parent, train, val = self.fixture("B")
        cfg = self.config["regression"]
        with self.assertRaises(TimeoutError):
            continue_candidate(parent, train, val, root / "timeout", self.spec("bearing_tree_mix"),
                {**cfg["limits"], "max_seconds_per_candidate": 1e-9}, cfg["controller"], "B", digest(parent))

    def test_sdk_tree_export_independent_prediction_and_origin_resume(self):
        root, parent, train, val = self.fixture("B")
        model = load_model(parent, "B"); model.params["b2"] += 3
        parent.unlink(); model.save(parent)
        first = RegressionHost(root, parent, "B")
        first.experiment = "bearing_tree_mix"
        def run(host):
            rnd = Round.start(host, host); rnd.parent.root = host.parent.parent
            work = root / ("work-" + str(rnd.round_index))
            with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), contextlib.redirect_stderr(io.StringIO()):
                contest_submission.train(rnd)
            self.assertTrue(rnd._finished)
        run(first)
        self.assertIsNotNone(first.candidate)
        child = load_model(first.candidate / "checkpoint.npz", "B")
        self.assertEqual(len(child.tree_layers), 1)
        sample = SimpleNamespace(dataset_key="B", task="regression", inputs=json.loads(
            (root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()[0])["inputs"])
        with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(first.candidate / "checkpoint.npz")}):
            value = contest_submission.predict(sample, first.candidate)["value"]
        self.assertAlmostEqual(value, child.predict(val["x"][:1])[0])
        second = RegressionHost(root, first.candidate / "checkpoint.npz", "B", round_index=2,
            state=first.saved_state, history=[{"id": "first", **first.record}])
        second.action = "observe"
        run(second)
        self.assertEqual(digest(first.saved_state / "bearing_origin.npz"), digest(second.saved_state / "bearing_origin.npz"))

    def test_legacy_plain_parent_can_enter_xiaoyang_after_round_one(self):
        root, parent, _, _ = self.fixture("B")
        host = RegressionHost(root, parent, "B", round_index=4)
        host.action = "observe"
        rnd = Round.start(host, host); rnd.parent.root = parent.parent
        work = root / "work"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertEqual(digest(host.saved_state / "bearing_origin.npz"), digest(parent))
        origin = json.loads((host.saved_state / "bearing_origin_context.json").read_text(encoding="utf-8"))
        self.assertEqual(origin["round_index"], 4)
        plan = json.loads(host.messages[0][0]["content"])
        self.assertIn("bearing_tree_mix", plan["experiments"])

    def test_sdk_restores_complete_engine_history(self):
        root, parent, train, val = self.fixture("C")
        history_dir = root / "historical"
        self.fit(parent, train, val, history_dir, self.spec("engine_tail_tree"), "C")
        expected = load_model(history_dir / "checkpoint.npz", "C").predict(val["x"])
        model = load_model(parent, "C"); model.params["b2"] += 3
        parent.unlink(); model.save(parent)
        host = RegressionHost(root, parent, "C")
        host.add_artifact("engine-history", history_dir)
        host.action = "rollback"
        rnd = Round.start(host, host); rnd.parent.root = parent.parent
        work = root / "work"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertEqual(host.record["selected_artifact_id"], "engine-history")
        record = json.loads((host.saved_state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertTrue(record["historical_validation"]["engine-history"]["gate"]["eligible"])
        np.testing.assert_array_equal(load_model(history_dir / "checkpoint.npz", "C").predict(val["x"]), expected)


if __name__ == "__main__": unittest.main()
