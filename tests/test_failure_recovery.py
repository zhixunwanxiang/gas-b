"""Actual SDK rounds with offline peers and injected local/host failures."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost, writes_only_in
from contest_harness.round import Round
from contest_harness.errors import HarnessError
from gas_a import submission


class FailureRecoveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="failure-recovery-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(ROOT/"data/public",self.root/"data")
        (self.root/"previous").mkdir()
        self.parent = self.root/"previous/checkpoint.npz"
        shutil.copyfile(ROOT/"models/reference.npz",self.parent)

    def run_host(self,host):
        rnd = Round.start(host,host)
        rnd.parent.root = self.parent.parent
        work = self.root/f"work-{rnd.round_index}"
        with patch.dict(os.environ,{"CONTEST_WORK":str(work)}),writes_only_in(work,host.store):
            submission.train(rnd)
        self.assertTrue(rnd._finished)
        return host.saved_state

    def test_host_regression_and_unknown_scope_keep_parent(self):
        for index,mode in enumerate(("regression","missing_digest"),1):
            host = OfflineHost(self.root,self.parent,index)
            original = host.write
            def inject(text):
                result = original(text)
                if json.loads(text)["type"] == "candidate_request":
                    if mode == "regression":
                        host.pending[-1]["summary"]["metrics"]["macro_f1"] = 0.
                    else:
                        host.pending[-1]["summary"].pop("validation_sha256")
                return result
            host.write = inject
            self.run_host(host)
            self.assertTrue(host.record["configuration"]["candidate_gate"]["eligible"])
            self.assertFalse(host.record["configuration"]["host_candidate_gate"]["eligible"])
            self.assertEqual(host.record["selected_artifact_id"],host.frame["parent_artifact"]["artifact_id"])

    def test_local_timeout_finishes_and_records_failure_with_model_decision(self):
        host = OfflineHost(self.root,self.parent)
        with patch.object(submission,"continue_candidate",side_effect=TimeoutError("Injected local deadline")):
            state = self.run_host(host)
        self.assertIsNone(host.candidate)
        self.assertEqual(len(host.calls),2)
        self.assertEqual(host.record["configuration"]["experiment_failure"]["error_type"],"TimeoutError")
        self.assertEqual(host.record["selected_artifact_id"],"parent-1")
        records = [json.loads(line) for line in (state/"experiment_attempts.jsonl").read_text().splitlines()]
        self.assertEqual([r["status"] for r in records],["started","failed"])
        self.assertTrue((state/"decisions.jsonl").is_file())
        strategy = json.loads((state/"strategy.json").read_text())
        self.assertEqual(strategy["attempts"][-1]["status"],"local_training_failed")

    def test_repeated_local_failures_are_blocked_without_repeating_training(self):
        state = None
        for index in (1,2):
            host = OfflineHost(self.root,self.parent,index,state=state)
            with patch.object(submission,"continue_candidate",side_effect=TimeoutError("local failure")):
                state = self.run_host(host)
        host = OfflineHost(self.root,self.parent,3,state=state)
        host.action = "observe"
        with patch.object(submission,"continue_candidate",side_effect=AssertionError("Blocked experiment ran")):
            self.run_host(host)
        request = json.loads(host.messages[0][0]["content"])
        self.assertNotIn("four_sensor",request["experiments"])
        self.assertIn("four_sensor",request["strategy"]["blocked_repeated_experiments"])

    def test_rpc_errors_are_never_recovered_as_local_compute_failure(self):
        host = OfflineHost(self.root,self.parent)
        with patch.object(submission,"continue_candidate",side_effect=HarnessError("RPC_OUTCOME_UNKNOWN")), \
                self.assertRaises(HarnessError):
            self.run_host(host)
        self.assertEqual(len(host.calls),1)
        self.assertIsNone(host.record)


if __name__ == "__main__":
    unittest.main()
