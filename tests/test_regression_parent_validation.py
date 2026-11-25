"""Reproduce C's ten-round compact-parent admission failure with the real SDK."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_regression import make_fixture, RegressionHost
from tests import test_regression as regression_tests
from contest_harness.round import Round
from gas_a.regression import submission
from gas_a.regression.data import load_pair, load_model
from gas_a.regression.evaluation import evaluate, current_parent_reference, parent_reference


class RegressionParentValidationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="c-parent-validation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = make_fixture(self.root, "C")
        self.train, self.validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", "C")
        self.before = evaluate(load_model(self.checkpoint, "C"), self.validation, "C")

    def reference(self, value):
        host = RegressionHost(self.root, self.checkpoint, "C")
        host.frame["current_validation"] = value
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        return current_parent_reference(rnd, self.checkpoint, self.before, self.validation["sha256"])

    def test_compact_parent_uses_local_reevaluation_without_certifying_host_scope(self):
        score = self.before["rmse"]
        values = [{"rmse": score}, {"metrics": {"rmse": score}},
                  {"summary": {"metric": "rmse", "score": score}},
                  {"rmse": score, "scope": "public_validation"},
                  {"rmse": score, "validation_sha256": self.validation["sha256"]}]
        for index, value in enumerate(values):
            with self.subTest(value=value):
                # A fresh peer store per case; all cases measure the same actual parent.
                if index:
                    (self.root / "host-1").rename(self.root / f"used-host-{index}")
                self.assertFalse(parent_reference(value, self.before, self.validation["sha256"])["comparable"])
                original = deepcopy(value)
                ref = self.reference(value)
                self.assertTrue(ref["comparable"])
                self.assertEqual(ref["source"], "local_current_parent_public_reevaluation")
                self.assertFalse(ref["host_reference"]["comparable"])
                self.assertEqual(ref["rmse"], score)
                self.assertEqual(value, original)

    def test_explicit_conflicts_and_invalid_scores_remain_rejected(self):
        values = [{"rmse": self.before["rmse"], **fields} for fields in (
            {"scope": "hidden"}, {"scope": None}, {"validation_sha256": None},
            {"validation_sha256": "wrong"}, {"artifact_id": "other"}, {"dataset_key": "B"},
            {"round_index": 99},
            {"checkpoint_sha256": "other"}, {"summary": {"rmse": self.before["rmse"], "scope": "hidden"}})]
        values += [{"rmse": self.before["rmse"] + 1}, {"score": self.before["rmse"]},
                   {"rmse": True}, {"rmse": float("nan")},
                   {"metrics": {"rmse": self.before["rmse"]}, "rmse": self.before["rmse"] + 1}]
        for index, value in enumerate(values):
            with self.subTest(value=value):
                if index:
                    (self.root / "host-1").rename(self.root / f"used-host-{index}")
                self.assertFalse(self.reference(value)["comparable"])

    def test_validation_changed_since_reevaluation_is_rejected(self):
        with (self.root / "data/validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaises(ValueError):
            self.reference({"rmse": self.before["rmse"]})

    def test_undeclared_copy_cannot_be_used_as_actual_parent(self):
        copied = self.root / "copy.npz"
        copied.write_bytes(self.checkpoint.read_bytes())
        host = RegressionHost(self.root, self.checkpoint, "C")
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        with self.assertRaises(ValueError):
            current_parent_reference(rnd, copied, self.before, self.validation["sha256"])

    def test_two_rounds_adopt_compact_receipts_and_continue_selected_parent(self):
        parent, state, history = self.checkpoint, None, []
        for index in (1, 2):
            host = RegressionHost(self.root, parent, "C", index, history=history, state=state)
            before = evaluate(load_model(parent, "C"), self.validation, "C")
            host.frame["current_validation"] = {"rmse": before["rmse"]}
            original_write = host.write
            def compact(text):
                result = original_write(text)
                if json.loads(text)["type"] == "candidate_request":
                    host.pending[-1]["summary"] = {"rmse": host.candidate_metrics["rmse"]}
                return result
            host.write = compact
            state = regression_tests.RegressionTest.run_host(self, host)
            self.assertEqual(host.record["selected_artifact_id"], f"candidate-{index}")
            cfg = host.record["configuration"]
            self.assertTrue(cfg["candidate_gate"]["eligible"])
            self.assertTrue(cfg["host_candidate_gate"]["eligible"])
            self.assertEqual(cfg["parent_host_reference"]["source"], "local_current_parent_public_reevaluation")
            report = json.loads((host.candidate / "training.json").read_text(encoding="utf-8"))
            self.assertEqual(report["initial_steps"], load_model(parent, "C").meta["steps"])
            self.assertTrue(all(report["checks"].values()))
            history.append({"id": f"history-{index}", **host.record})
            parent = host.candidate / "checkpoint.npz"
        self.assertLess(evaluate(load_model(parent, "C"), self.validation, "C")["rmse"], self.before["rmse"])

    def test_unverifiable_parent_observes_without_training(self):
        host = RegressionHost(self.root, self.checkpoint, "C")
        host.frame["current_validation"] = {"rmse": self.before["rmse"], "scope": "hidden"}
        host.action = "observe"
        with patch.object(submission, "continue_candidate", side_effect=AssertionError("must not train")):
            regression_tests.RegressionTest.run_host(self, host)
        self.assertIsNone(host.candidate)
        self.assertEqual(host.record["selected_artifact_id"], "parent-1")
        self.assertTrue(host.record["configuration"]["admission_blocker"])
        request = json.loads(host.messages[0][0]["content"])
        self.assertEqual(request["response_schema"]["action"], ["observe"])


if __name__ == "__main__":
    unittest.main()
