"""Controller evidence contracts and the rejected-highest-epoch regression."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost, writes_only_in
from contest_harness.round import Round
from gas_a import submission
from gas_a.controller_prompts import VERSION, admission, execution_policy, history_feedback, request_digest
from gas_a.evolution import available_experiments, fingerprint, initial_policy
from gas_a.sensor_training.data import load_pair, evaluate
from gas_a.sensor_training.reference.model import Model
from gas_a.sensor_training.io import digest


class ControllerPromptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gas-controller-v2-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / "data/public", self.root / "data")
        (self.root / "previous").mkdir()
        self.parent = self.root / "previous/checkpoint.npz"
        shutil.copyfile(ROOT / "models/reference.npz", self.parent)
        self.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
        self.train, self.validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl")
        self.before = evaluate(Model.load(self.parent), self.validation)
        self.execution = execution_policy(self.config["controller"], self.config["limits"])

    def run_round(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = self.parent.parent
        work = self.root / "workspace"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store):
            submission.train(rnd)
        self.assertTrue(rnd._finished)
        return next(work.glob("round-*/state"))

    def test_prompt_exposes_numeric_floors_and_actual_experiment_effect(self):
        host = OfflineHost(self.root, self.parent)
        host.action = "observe"
        self.run_round(host)
        plan = json.loads(host.messages[0][0]["content"])
        self.assertEqual(plan["controller_prompt"]["version"], VERSION)
        self.assertEqual(plan["admission"], admission(self.before, self.config["controller"]))
        self.assertAlmostEqual(plan["admission"]["protected_recall_floors"]["Ethanol"], 158/164 - .2)
        self.assertIn("Toluene", plan["admission"]["unprotected_low_support_classes"])
        self.assertEqual(plan["experiment_guide"]["four_sensor"]["weak_class_extra_multiplier_before_frequency_normalization"], 3)
        self.assertFalse(plan["data_scope"]["hidden_labels_available"])
        self.assertNotIn("OFFICIAL_PROMPT_SENTINEL", json.dumps(host.messages))
        select = json.loads(host.messages[1][0]["content"])
        self.assertEqual(select["controller_prompt"]["version"], VERSION)
        self.assertEqual(select["local_validation_sha256"], self.validation["sha256"])

    def test_history_distinguishes_scope_recipe_and_execution_changes(self):
        spec = next(s for s in self.config["experiments"] if s["name"] == "four_sensor")
        cfg = {"parent_checkpoint_sha256": digest(self.parent), "train_sha256": self.train["sha256"],
               "validation_sha256": "different-public-validation", "experiment": spec,
               "experiment_config_sha256": request_digest(spec), "local_validation_before": .7, "local_validation_after": .75,
               "candidate_gate": {"eligible": False, "macro_f1_gain": .05, "class_recall_losses": {"Ethanol": .21}}}
        history = [{"id": "measured-old", "configuration": cfg}]
        result = history_feedback(history, digest(self.parent), self.train["sha256"], self.validation["sha256"], self.execution, {"four_sensor": spec})[0]
        self.assertTrue(result["same_parent"])
        self.assertFalse(result["same_validation_data"])
        self.assertFalse(result["same_execution_policy"])
        self.assertTrue(result["same_experiment_configuration"])
        self.assertEqual(result["result"], "class_recall_regression")
        changed = {**spec, "learning_rate": .001}
        result = history_feedback(history, digest(self.parent), self.train["sha256"], self.validation["sha256"], self.execution, {"four_sensor": changed})[0]
        self.assertFalse(result["same_experiment_configuration"])

    def test_execution_change_does_not_inherit_old_rejection_block(self):
        menu = {s["name"]: s for s in self.config["experiments"]}
        spec = menu["four_sensor"]
        policy = initial_policy()
        old = fingerprint(digest(self.parent), self.train["sha256"], self.validation["sha256"], spec)
        policy["attempts"] = [{"fingerprint": old, "adopted": False}] * 2
        rnd = SimpleNamespace(parent=SimpleNamespace(checkpoint_sha256=digest(self.parent)))
        allowed, _ = available_experiments(policy, menu, rnd, self.train, self.validation, self.config["controller"], self.execution)
        self.assertIn("four_sensor", allowed)
        current = fingerprint(digest(self.parent), self.train["sha256"], self.validation["sha256"], spec, self.execution)
        policy["attempts"] = [{"fingerprint": current, "adopted": False, "outcome": "performance_rejected"}] * 2
        allowed, blocked = available_experiments(policy, menu, rnd, self.train, self.validation, self.config["controller"], self.execution)
        self.assertNotIn("four_sensor", allowed)
        self.assertIn("four_sensor", blocked)

    def test_history_exposes_host_rejection_and_local_failure_instead_of_claiming_eligible(self):
        spec = next(s for s in self.config["experiments"] if s["name"] == "four_sensor")
        config = {"experiment": spec, "candidate_gate": {"eligible": True, "macro_f1_gain": .05},
                  "host_candidate_gate": {"eligible": False, "reason": "host_gain_below_promotion_threshold"}}
        def feedback():
            return history_feedback([{"id": "failed", "configuration": config}], "parent", "train", "validation",
                                    self.execution, {"four_sensor": spec})[0]
        result = feedback()
        self.assertEqual(result["result"], "host_measurement_rejected")
        self.assertEqual(result["host_admission"], config["host_candidate_gate"])
        config["experiment_failure"] = {"error_type": "TimeoutError", "candidate_submitted": False}
        result = feedback()
        self.assertEqual(result["result"], "local_training_failure")
        self.assertEqual(result["experiment_failure"], config["experiment_failure"])

    def test_completed_response_correction_has_original_text_and_request_hashes(self):
        host = OfflineHost(self.root, self.parent)
        host.action, host.malformed_responses = "observe", 1
        state = self.run_round(host)
        request = json.loads(host.messages[1][0]["content"])
        self.assertEqual(request["correction"]["previous_completed_response"], "{invalid JSON")
        record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
        hashes = record["planning_calls"]["request_sha256"]
        self.assertEqual(len(hashes), 2)
        self.assertEqual(hashes, [request_digest(json.loads(msg[0]["content"])) for msg in host.messages[:2]])
        requests = [frame for frame in host.frames if frame["type"] == "model_request"]
        self.assertTrue(requests)
        self.assertTrue(all(frame["enable_thinking"] is False for frame in requests))

    def test_actual_sdk_exports_and_adopts_feasible_epoch_instead_of_rejecting_best_raw_f1(self):
        host = OfflineHost(self.root, self.parent, experiment="four_sensor")
        self.run_round(host)
        report = json.loads((host.candidate / "training.json").read_text(encoding="utf-8"))
        epochs = {item["epoch"]: item for item in report["epoch_history"]}
        self.assertFalse(epochs[43]["checkpoint_eligible"])
        self.assertTrue(epochs[44]["checkpoint_eligible"])
        self.assertGreater(epochs[43]["validation"]["macro_f1"], epochs[44]["validation"]["macro_f1"])
        self.assertEqual(report["final_epochs"], 44)
        self.assertTrue(report["checkpoint_selection"]["selected_epoch_eligible"])
        self.assertEqual(report["checkpoint_selection"]["mode"], "best_controller_eligible_epoch")
        self.assertAlmostEqual(report["validation_after"]["macro_f1"], .7443301294233811)
        self.assertEqual(host.record["selected_artifact_id"], "candidate-1")
        self.assertTrue(host.record["configuration"]["candidate_gate"]["eligible"])
        selection = json.loads(host.messages[-1][0]["content"])
        self.assertTrue(selection["candidate_gate"]["eligible"])
        self.assertEqual(selection["checkpoint_selection"]["selected_epoch"], 44)
