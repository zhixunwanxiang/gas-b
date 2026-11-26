"""Safety and continuation contracts for group/tail selection and resource changes."""
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
from gas_a.regression.assessment import Measurements, StageEvidence
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.evaluation import evaluate, promotion_gate
from gas_a.regression.grouping import fit_groups, attach_groups
from gas_a.regression.training import prepare_training
from gas_a.regression.training import continue_candidate
from gas_a.regression.model import component_digest
from gas_a.sensor_training.reference.model import Model as OriginalModel
from gas_a.sensor_training.io import digest
from tests.test_regression import make_fixture, RegressionHost
from tests.test_submission import ROOT, writes_only_in


class ScoreAlignedRegressionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="score-aligned-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))

    def fixture(self, dataset):
        root = self.root / dataset
        parent = make_fixture(root, dataset)
        train, val = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", dataset)
        return root, parent, train, val

    def run_host(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = host.root / f"work-{rnd.round_index}"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        return json.loads((host.saved_state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])

    def metric(self, errors, dataset="B"):
        errors = np.asarray(errors, dtype=float)
        data = {"x": np.zeros((len(errors), 1)), "y": np.zeros(len(errors)),
                "groups": np.asarray(["one"] * (len(errors) // 2) + ["two"] * (len(errors) - len(errors) // 2)),
                "grouping": {"source": "test_groups", "fit_sha256": "same"}}
        return evaluate(None, data, dataset, prediction=errors)

    def rules(self, dataset):
        return submission.dataset_config(self.config, dataset)["controller"]

    def test_local_rejection_has_no_upload_and_remains_a_counted_attempt(self):
        root, parent, _, _ = self.fixture("B")
        host = RegressionHost(root, parent, "B")
        with patch.object(submission, "promotion_gate", return_value={"eligible": False, "rmse_reduction": -.1}):
            record = self.run_host(host)
        self.assertIsNone(host.candidate)
        self.assertEqual(record["selected_artifact_id"], "parent-1")
        self.assertTrue(record["configuration"]["submission_skipped"])
        strategy = json.loads((host.saved_state / "regression_strategy.json").read_text(encoding="utf-8"))
        attempt = strategy["attempts"][-1]
        self.assertEqual(attempt["outcome"], "performance_rejected")
        self.assertTrue(attempt["counts_toward_rejection_limit"])
        self.assertEqual(attempt["status"], "local_measured_not_submitted")

    def test_unique_observe_reuses_real_plan_receipt(self):
        root, parent, _, _ = self.fixture("C")
        host = RegressionHost(root, parent, "C")
        host.action = "observe"
        record = self.run_host(host)
        self.assertEqual(len(host.calls), 1)
        self.assertTrue(record["selection_calls"]["selection_reused_plan"])
        self.assertEqual(record["model_call_ids"], host.calls)
        self.assertEqual(record["selection_calls"]["decision_source"], "model")

    def test_balancing_preserves_features_and_has_bounded_mean_one_weights(self):
        _, parent, train, _ = self.fixture("B")
        train["groups"] = np.asarray(["large"] * 76 + ["small"] * 4)
        train["grouping"] = {"source": "public_bearing_metadata"}
        model = load_model(parent, "B")
        original = deepcopy(model)
        x = train["x"].copy()
        prepared = prepare_training(model, train, {"weighting": "bearing_sqrt_inverse"})
        weights = prepared["weights"]
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertGreater(weights[-1], weights[0])
        self.assertGreaterEqual(weights.min(), .5 - 1e-12)
        self.assertLessEqual(weights.max(), 2 + 1e-12)
        np.testing.assert_array_equal(train["x"], x)
        np.testing.assert_array_equal(model.mean, original.mean)
        for key in model.params:
            np.testing.assert_array_equal(model.first[key], original.first[key])

    def test_tail_threshold_is_frozen_from_parent_training_residuals(self):
        _, parent, train, validation = self.fixture("C")
        model = load_model(parent, "C")
        residual = np.abs(model.predict(train["x"]) - train["y"])
        spec = {"weighting": "parent_tail", "tail_quantile": .9, "tail_weight": 2.}
        before = prepare_training(model, train, spec)
        validation["y"][:] = 1e8
        after = prepare_training(model, train, spec)
        self.assertAlmostEqual(before["evidence"]["threshold"], np.quantile(residual, .9))
        np.testing.assert_array_equal(before["weights"], after["weights"])
        self.assertAlmostEqual(before["weights"].mean(), 1.)

    def test_improved_overall_rmse_does_not_hide_bad_group(self):
        before = self.metric([10.] * 40)
        after = self.metric([1.] * 20 + [11.] * 20)
        self.assertLess(after["rmse"], before["rmse"])
        gate = promotion_gate(before, after, self.rules("B"), reference=before)
        self.assertFalse(gate["eligible"])
        self.assertTrue(any("group_regression" in r for r in gate["guard_failures"]))

    def test_stage_reference_blocks_accumulated_group_regression(self):
        before = self.metric([10.] * 40)
        after = self.metric([9.5] * 40)
        reference = self.metric([9.] * 20 + [12.] * 20)
        gate = promotion_gate(before, after, self.rules("B"), reference=reference)
        self.assertFalse(gate["eligible"])
        self.assertTrue(any("stage_reference:group_regression" in r for r in gate["guard_failures"]))

    def test_nasa_overflow_recovery_can_select_a_finite_model(self):
        before = self.metric([10000.] * 40, "C")
        after = self.metric([5.] * 40, "C")
        self.assertTrue(before["nasa_score_overflow"])
        self.assertTrue(promotion_gate(before, after, self.rules("C"), reference=before)["eligible"])
        bad = deepcopy(after)
        bad.update(nasa_score=None, nasa_score_overflow=True)
        self.assertFalse(promotion_gate(before, bad, self.rules("C"), reference=before)["eligible"])

    def test_missing_or_small_groups_are_not_reported_as_guard_passes(self):
        before = self.metric([10.] * 4)
        after = self.metric([5.] * 4)
        gate = promotion_gate(before, after, self.rules("B"), reference=before)
        self.assertFalse(gate["eligible"])
        self.assertTrue(any("insufficient_group" in r for r in gate["guard_failures"]))

    def test_metadata_does_not_enter_features_and_proxy_is_labeled(self):
        root, _, train, validation = self.fixture("B")
        original = train["x"].copy()
        self.assertEqual(validation["grouping"]["source"], "train_input_proxy")
        path = root / "data/train.jsonl"
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["meta"]["bearing"] = "trace-only"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        changed, _ = load_pair(path, root / "data/validation.jsonl", "B")
        np.testing.assert_array_equal(changed["x"], original)
        self.assertEqual(changed["grouping"]["source"], "public_bearing_metadata")

    def test_cache_reuse_requires_data_and_code_identity_and_keeps_origin(self):
        _, parent, _, val = self.fixture("C")
        model = load_model(parent, "C")
        cache = Measurements(self.root, "C", val, 1)
        with patch.object(model, "predict", wraps=model.predict) as predict:
            first, source = cache.measure(parent, model)
            second, reused = cache.measure(parent, model)
            self.assertEqual(predict.call_count, 1)
        self.assertEqual(first, second)
        self.assertTrue(source["measured_this_round"])
        self.assertFalse(reused["measured_this_round"])
        self.assertEqual(reused["origin_round"], 1)
        cache.save()
        next_cache = Measurements(self.root, "C", val, 2)
        self.assertFalse(next_cache.measure(parent, model)[1]["measured_this_round"])
        next_cache.data = {**val, "sha256": "different-data"}
        self.assertTrue(next_cache.measure(parent, model)[1]["measured_this_round"])
        next_cache.implementation = "changed-code"
        self.assertTrue(next_cache.measure(parent, model)[1]["measured_this_round"])

    def test_cache_corruption_is_rejected(self):
        _, parent, _, val = self.fixture("C")
        model = load_model(parent, "C")
        cache = Measurements(self.root, "C", val, 1)
        cache.measure(parent, model)
        next(iter(cache.entries.values()))["metrics"]["rmse"] = 0.
        with self.assertRaises(ValueError):
            cache.measure(parent, model)

    def test_projection_change_is_not_labeled_as_official_stage(self):
        _, parent, train, val = self.fixture("C")
        before = evaluate(load_model(parent, "C"), val, "C")
        rnd = SimpleNamespace(task={}, round_index=1,
                              parent=SimpleNamespace(artifact_id="parent", checkpoint_sha256="identity"))
        evidence = StageEvidence(self.root, rnd, train, val, before, "code")
        self.assertEqual(evidence.stage["scope_kind"], "data_projection_event")
        evidence.record(rnd, "parent", before, {}, "retain_parent")
        changed = StageEvidence(self.root, rnd, train, {**val, "sha256": "new"}, before, "code")
        self.assertTrue(changed.changed)
        self.assertEqual(changed.stage["scope_kind"], "data_projection_event")
        rnd.task = {"stage_id": "authorized-stage"}
        formal = StageEvidence(self.root, rnd, train, val, before, "code")
        self.assertEqual(formal.stage["scope_kind"], "host_stage")

    def test_unsupported_residual_is_rejected_instead_of_silently_dropped(self):
        _, parent, _, _ = self.fixture("C")
        model = OriginalModel.load(parent)
        model.meta["rul_residual_v1"] = {"weights": [1.]}
        path = self.root / "unsupported.npz"
        model.save(path)
        with self.assertRaisesRegex(ValueError, "complete compatible loader"):
            load_model(path, "C")

    def test_zero_component_and_export_preserve_complete_prediction(self):
        _, parent, train, val = self.fixture("C")
        model = load_model(parent, "C")
        expected = model.predict(val["x"])
        self.assertTrue(model.initialize_component(train))
        np.testing.assert_array_equal(model.predict(val["x"]), expected)
        model.update_component(train)
        path = self.root / "complete.npz"
        model.save(path)
        loaded = load_model(path, "C")
        with self.assertRaises(ValueError):
            OriginalModel.load(path)  # Older MLP-only code must fail instead of dropping the head.
        np.testing.assert_array_equal(model.predict(val["x"]), loaded.predict(val["x"]))
        np.testing.assert_allclose(loaded.predict(val["x"]), np.asarray([loaded.predict(x[None])[0] for x in val["x"]]), atol=1e-10)
        self.assertEqual(loaded.meta["regression_components"]["updates"], 1)

    def test_component_update_inherits_adam_without_inventing_neural_steps(self):
        _, parent, train, val = self.fixture("C")
        cfg = submission.dataset_config(self.config, "C")
        spec = next(s for s in cfg["experiments"] if s["name"] == "condition_ridge")
        original = load_model(parent, "C")
        before = evaluate(original, val, "C")
        source = parent
        for index in (1, 2):
            out = self.root / f"component-{index}"
            report = continue_candidate(source, train, val, out, spec, cfg["limits"], cfg["controller"], "C",
                digest(source), reference=before)
            loaded = load_model(out / "checkpoint.npz", "C")
            self.assertEqual(report["executed_epochs"], 0)
            self.assertEqual(report["final_steps"], original.meta["steps"])
            self.assertEqual(loaded.meta["regression_components"]["updates"], index)
            self.assertTrue(all(report["checks"].values()))
            for key in original.params:
                np.testing.assert_array_equal(loaded.params[key], original.params[key])
                np.testing.assert_array_equal(loaded.first[key], original.first[key])
                np.testing.assert_array_equal(loaded.second[key], original.second[key])
            source = out / "checkpoint.npz"
        inherited = component_digest(loaded.meta["regression_components"])
        neural = next(s for s in cfg["experiments"] if s["name"] == "head_short")
        report = continue_candidate(source, train, val, self.root / "neural", neural,
            cfg["limits"], cfg["controller"], "C", digest(source), reference=before)
        resumed = load_model(self.root / "neural/checkpoint.npz", "C")
        self.assertEqual(component_digest(resumed.meta["regression_components"]), inherited)
        self.assertGreater(resumed.meta["steps"], loaded.meta["steps"])

    def test_sdk_can_submit_and_predict_complete_component_checkpoint(self):
        root, parent, _, val = self.fixture("C")
        model = load_model(parent, "C")
        model.params["b2"] += 2
        parent.unlink()
        model.save(parent)
        host = RegressionHost(root, parent, "C")
        host.experiment = "condition_ridge"
        config_root = self.root / "configuration"
        config_root.mkdir()
        self.config["regression"]["enable_research_experiments"] = True
        (config_root / "submission_config.json").write_text(json.dumps(self.config), encoding="utf-8")
        with patch.object(submission, "ROOT", config_root):
            record = self.run_host(host)
        self.assertIsNotNone(host.candidate)
        self.assertEqual(record["selected_artifact_id"], "candidate-1")
        loaded = load_model(host.candidate / "checkpoint.npz", "C")
        self.assertEqual(loaded.meta["regression_components"]["updates"], 1)
        self.assertEqual(loaded.meta["steps"], model.meta["steps"])
        sample = SimpleNamespace(inputs=json.loads((root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()[0])["inputs"],
                                 dataset_key="C", task="regression")
        with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(host.candidate / "checkpoint.npz")}):
            value = contest_submission.predict(sample, host.candidate)["value"]
        self.assertAlmostEqual(value, loaded.predict(val["x"][:1])[0])

    def test_compiled_head_matches_clipped_basis_for_normal_and_extreme_inputs(self):
        _, parent, train, val = self.fixture("C")
        model = load_model(parent, "C")
        model.initialize_component(train)
        model.update_component(train)
        x = np.vstack((val["x"], np.full((1, 24), 1e6), np.full((1, 24), -1e6)))
        normalized = model.normalized(x)
        c = model.component
        groups = model.component_groups(normalized)
        from gas_a.regression.model import basis
        correction = basis(normalized[:, :21], c["means"][0], c["scales"][0]) @ c["weights"][0]
        correction += np.einsum("ij,ij->i", basis(normalized[:, :21], c["means"][groups], c["scales"][groups]), c["weights"][groups])
        raw = OriginalModel.forward(model, normalized)[1][:, 0]
        cap = model.meta["regression_components"]["cap"]
        expected = np.maximum(0, (raw + np.clip(correction, -cap, cap)) * model.meta["target_scale"] + model.meta["target_mean"])
        np.testing.assert_allclose(model.predict(x), expected, rtol=1e-12, atol=1e-10)

    def test_history_restores_complete_component_after_parent_fault(self):
        root, parent, train, _ = self.fixture("C")
        model = load_model(parent, "C")
        model.initialize_component(train)
        model.update_component(train)
        historical = root / "historical"
        historical.mkdir()
        model.save(historical / "checkpoint.npz")
        expected_component = component_digest(model.meta["regression_components"])
        model.params["b2"] += 100
        parent.unlink()
        model.save(parent)
        host = RegressionHost(root, parent, "C")
        host.add_artifact("history-head", historical)
        original_write = host.write
        def choose_history(text):
            result = original_write(text)
            frame = json.loads(text)
            if frame["type"] == "model_request":
                request = json.loads(frame["messages"][0]["content"])
                value = ({"action": "rollback", "experiment": "none", "target_artifact_id": "history-head",
                          "hypothesis": "Restore the measured complete historical predictor.", "reason": "The restored predictor passed current-scope guards."}
                         if request["phase"] == "plan" else {"selected_artifact_id": "history-head", "reason": "Select the authorized complete predictor."})
                host.pending[-1]["response"]["choices"][0]["message"]["content"] = json.dumps(value)
            return result
        host.write = choose_history
        record = self.run_host(host)
        self.assertEqual(record["selected_artifact_id"], "history-head")
        self.assertTrue(record["historical_validation"]["history-head"]["gate"]["eligible"])
        self.assertEqual(component_digest(load_model(historical / "checkpoint.npz", "C").meta["regression_components"]), expected_component)

    def test_missing_unknown_and_overflowed_component_metadata_are_rejected(self):
        _, parent, train, _ = self.fixture("C")
        model = load_model(parent, "C")
        model.initialize_component(train)
        complete = self.root / "complete.npz"
        model.save(complete)
        with np.load(complete, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored}
        original = json.loads(str(arrays["metadata"]))
        for index, kind in enumerate(("missing", "unknown", "overflow")):
            metadata = deepcopy(original)
            if kind == "missing":
                del metadata["regression_components"]
            elif kind == "unknown":
                metadata["regression_components"]["format"] = "unknown-format"
            else:
                metadata["regression_components"]["cap"] = 10 ** 400
            broken = self.root / f"broken-{index}.npz"
            np.savez(broken, **{**arrays, "metadata": np.asarray(json.dumps(metadata))})
            with self.assertRaises(ValueError):
                load_model(broken, "C")


if __name__ == "__main__":
    unittest.main()
