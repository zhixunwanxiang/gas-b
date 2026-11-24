"""Formal SDK closed loop, using an offline host and authorized test fixtures."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_submission import ROOT, OfflineHost, writes_only_in
from contest_harness.round import Round
from contest_harness.errors import HarnessError
from gas_a import submission
from gas_a.evolution import advance_policy, initial_policy, promotion_gate
from gas_a.monitoring import fit_reference, monitor, diagnose
from gas_a.sensor_training.data import evaluate, load_pair
from gas_a.sensor_training.reference.model import Model
from gas_a.sensor_training.io import CLASSES, digest, write_json


class EvolutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gas-evolution-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / "data/public", self.root / "data")
        (self.root / "previous").mkdir()
        self.parent = self.root / "previous/checkpoint.npz"
        shutil.copyfile(ROOT / "models/reference.npz", self.parent)
        self.train, self.validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl")
        self.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["controller"]

    def run_round(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = self.root / f"workspace-{rnd.round_index}"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store):
            submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertFalse(host.pending)
        return next(work.glob("round-*/state"))

    def test_monitor_reference_is_train_only_and_does_not_follow_validation_labels(self):
        reference = fit_reference(self.train)
        before = deepcopy(reference)
        same = monitor(reference, self.train, self.config["drift_thresholds"])
        self.assertFalse(same["drift"])
        self.assertTrue(all(f["psi"] == 0 for f in same["features"]))
        changed = dict(self.validation, x=self.validation["x"].copy(), y=np.roll(self.validation["y"], 17))
        changed["x"][:, 0:8] += 1000 * self.train["x"][:, 0:8].std(axis=0)
        measured = monitor(reference, changed, self.config["drift_thresholds"])
        self.assertTrue(measured["drift"])
        self.assertTrue(next(s for s in measured["sensors"] if s["sensor"] == 1)["drift"])
        self.assertEqual(measured, monitor(reference, {**changed, "y": self.validation["y"]}, self.config["drift_thresholds"]))
        self.assertEqual(reference, before)
        with self.assertRaises(ValueError):
            monitor({**reference, "origin": "validation"}, changed, self.config["drift_thresholds"])

    def test_diagnosis_uses_supported_errors_and_never_compares_different_scopes(self):
        metrics = evaluate(Model.load(self.parent), self.validation)
        drift = monitor(fit_reference(self.train), self.validation, self.config["drift_thresholds"])
        diagnosis = diagnose(metrics, drift, self.validation["sha256"], {"validation_sha256": "other", "selected_macro_f1": 1})
        self.assertIsNone(diagnosis["same_scope_previous_macro_f1_delta"])
        self.assertTrue(diagnosis["weak_classes"])
        self.assertTrue(diagnosis["top_confusions"])
        self.assertFalse(diagnosis["hidden_feedback_available"])

    def test_observe_finishes_without_training_or_candidate_and_records_model_identity(self):
        host = OfflineHost(self.root, self.parent)
        host.action = "observe"
        with patch("gas_a.submission.continue_candidate", side_effect=AssertionError("observation trained")):
            state = self.run_round(host)
        self.assertIsNone(host.candidate)
        self.assertEqual(host.record["selected_artifact_id"], "parent-1")
        self.assertEqual(host.record["configuration"]["controller_model_name"], "offline-host-stub")
        request = json.loads(host.messages[0][0]["content"])
        self.assertIn("monitoring", request)
        self.assertIn("diagnosis", request)
        self.assertTrue((state / "strategy.json").is_file())

    def test_completed_bad_json_is_corrected_once_and_all_real_calls_are_cited(self):
        host = OfflineHost(self.root, self.parent)
        host.action, host.malformed_responses = "observe", 1
        state = self.run_round(host)
        self.assertEqual(len(host.calls), 3)
        self.assertEqual(host.record["model_call_ids"], host.calls)
        record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(len(record["planning_calls"]["corrections"]), 1)
        self.assertIn("correction", json.loads(host.messages[1][0]["content"]))

    def test_invalid_json_exhaustion_records_program_guard_without_fake_model_decision(self):
        host = OfflineHost(self.root, self.parent)
        host.malformed_responses = 10
        state = self.run_round(host)
        self.assertEqual(len(host.calls), 4)
        self.assertIsNone(host.candidate)
        self.assertEqual(host.record["selected_artifact_id"], "parent-1")
        record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
        for phase in ("planning_calls", "selection_calls"):
            self.assertEqual(record[phase]["decision_source"], "program_guard")
            self.assertEqual(len(record[phase]["corrections"]), 2)
        evidence = [json.loads(line) for line in (state / "model-responses.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["call_id"] for r in evidence], host.calls)
        self.assertEqual(host.record["model_call_ids"], host.calls)

    def test_class_recall_guard_can_reject_higher_overall_f1(self):
        before = evaluate(Model.load(self.parent), self.validation)
        after = deepcopy(before)
        after["macro_f1"] += .01
        protected = next(k for k,v in before["per_class"].items() if v["support"] >= 20 and v["recall"] > .21)
        after["per_class"][protected]["recall"] -= .21
        gate = promotion_gate(before, after, self.config)
        self.assertFalse(gate["eligible"])
        self.assertIn(protected, gate["class_recall_losses"])
        self.assertTrue(promotion_gate(before, {**before, "macro_f1": before["macro_f1"] + .01}, self.config)["eligible"])

    def recovery_fixture(self):
        historical = self.root / "historical"
        historical.mkdir()
        shutil.copyfile(self.parent, historical / "checkpoint.npz")
        strong = Model.load(self.parent)
        # Synthetic fault fixture: labels are the historical model predictions.
        # These are test labels, never reported as an official validation score.
        rows = [json.loads(line) for line in (self.root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()]
        for row, target in zip(rows, strong.predict(self.validation["x"]).argmax(1)):
            row["answer"] = CLASSES[target]
        (self.root / "data/validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        damaged = Model.load(self.parent)
        damaged.params["w2"][:] = 0
        damaged.params["b2"][:] = 0
        self.parent.unlink()
        damaged.save(self.parent)
        return historical

    def test_actual_historical_rollback_then_next_round_uses_restored_parent(self):
        historical = self.recovery_fixture()
        host = OfflineHost(self.root, self.parent, history=[{"id": "past", "selected_artifact_id": "historical-good"}])
        host.action = "rollback"
        receipt = host.add_artifact("historical-good", historical)
        state = self.run_round(host)
        self.assertEqual(host.record["selected_artifact_id"], "historical-good")
        self.assertIn("Restored and selected historical", host.record["change_summary"])
        self.assertIsNone(host.candidate)
        self.assertTrue(any(f["type"] == "artifact_read_request" for f in host.frames))
        selection = json.loads(host.messages[-1][0]["content"])
        self.assertIsNone(selection["candidate_gate"])
        self.assertIsNone(selection["host_candidate_receipt"])
        self.assertFalse(selection["new_candidate_admission_applicable"])
        admission = selection["historical_admission"]["historical-good"]
        self.assertTrue(admission["eligible"])
        self.assertEqual(admission["validation_sha256"], digest(self.root / "data/validation.jsonl"))
        self.assertEqual(admission["measurement_source"], "restored_artifact_current_public_re_evaluation")
        self.assertAlmostEqual(admission["fresh_metrics"]["macro_f1"], 1.)
        next_host = OfflineHost(self.root, historical / "checkpoint.npz", 2, [{"id": "recovery", **host.record}], state,
                                experiment="conservative_head")
        next_host.frame["parent_artifact"] = receipt
        self.run_round(next_host)
        continuation = json.loads((next_host.candidate / "continuation.json").read_text(encoding="utf-8"))
        self.assertEqual(continuation["parent"]["checkpoint_sha256"], digest(historical / "checkpoint.npz"))
        self.assertNotEqual(continuation["parent"]["checkpoint_sha256"], digest(self.parent))

    def test_corrupt_historical_receipt_fails_before_reading_or_selecting(self):
        historical = self.recovery_fixture()
        host = OfflineHost(self.root, self.parent)
        host.add_artifact("historical-good", historical)["artifact_sha256"] = "sha256:" + "0" * 64
        with self.assertRaises(HarnessError) as caught:
            self.run_round(host)
        self.assertEqual(caught.exception.code, "ARTIFACT_RECEIPT_MISMATCH")
        self.assertIsNone(host.record)
        self.assertFalse(any(f["type"] == "artifact_read_request" for f in host.frames))

    def test_interrupted_historical_read_poisons_rpc_without_retry(self):
        historical = self.recovery_fixture()
        class InterruptedHost(OfflineHost):
            def readline(self):
                if self.frames and self.frames[-1]["type"] == "artifact_read_request":
                    raise TimeoutError("Interrupted historical read")
                return super().readline()
        host = InterruptedHost(self.root, self.parent)
        host.add_artifact("historical-good", historical)
        rnd = Round.start(host, host)
        rnd.parent.root = self.parent.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}):
            with self.assertRaises(TimeoutError):
                submission.train(rnd)
        self.assertEqual(sum(f["type"] == "artifact_read_request" for f in host.frames), 1)
        with self.assertRaises(HarnessError) as caught:
            rnd.history()
        self.assertEqual(caught.exception.code, "RPC_OUTCOME_UNKNOWN")
        self.assertEqual(host.calls, [])
        self.assertIsNone(host.record)

    def test_ten_rounds_stop_repeating_rejected_experiment_and_keep_reference_fixed(self):
        state, history, reference_bytes = None, [], None
        actions = []
        for index in range(1, 11):
            host = OfflineHost(self.root, self.parent, index, history, state, experiment="conservative_head")
            host.force_retain = True
            # The offline controller follows the menu; once this experiment is
            # blocked it chooses observation, as the real model is instructed to.
            if index > 2:
                host.action = "observe"
            state = self.run_round(host)
            payload = (state / "monitor_reference.json").read_bytes()
            reference_bytes = payload if reference_bytes is None else reference_bytes
            self.assertEqual(payload, reference_bytes)
            plan_request = json.loads(host.messages[0][0]["content"])
            if index > 2:
                self.assertIn("conservative_head", plan_request["strategy"]["blocked_repeated_experiments"])
                self.assertNotIn("conservative_head", plan_request["experiments"])
                self.assertIsNone(host.candidate)
            history.append({"id": f"ten-round-{index}", **host.record})
            actions.append(host.record["configuration"]["action"])
        self.assertEqual(actions, ["experiment"] * 2 + ["observe"] * 8)
        self.assertEqual(len((state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()), 10)
        self.assertEqual(json.loads((state / "strategy.json").read_text(encoding="utf-8"))["version"], 1)

    def test_strategy_promotes_only_adopted_experiments_and_rolls_back_with_artifact(self):
        base = initial_policy()
        kwargs = dict(parent_id="initial", candidate_id="new", plan={"experiment": "four_sensor"},
                      attempt_key="key", gain=.1, validation_sha256="val", selected_f1=.8)
        rejected = advance_policy(base, selected="initial", **kwargs)
        self.assertEqual(rejected["version"], 1)
        accepted = advance_policy(base, selected="new", **kwargs)
        self.assertEqual(accepted["version"], 2)
        self.assertEqual(accepted["preferred_experiment"], "four_sensor")
        reverted = advance_policy(accepted, selected="initial", parent_id="new", candidate_id=None,
                                  plan={"experiment": "none"}, attempt_key=None, gain=None,
                                  validation_sha256="val", selected_f1=.7)
        self.assertEqual(reverted["version"], 1)
        self.assertIsNone(reverted["preferred_experiment"])
        renewed = advance_policy(reverted, selected="another", **{**kwargs, "candidate_id": "another"})
        self.assertEqual(renewed["version"], 3)


if __name__ == "__main__":
    unittest.main()
