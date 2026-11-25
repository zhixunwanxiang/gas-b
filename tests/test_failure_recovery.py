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
from gas_a.host_validation import bind_parent_validation
from gas_a.sensor_training.data import evaluate, load_pair, load_model
from gas_a.sensor_training.io import digest


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

    def test_host_regression_and_conflicting_digest_keep_parent(self):
        for index,mode in enumerate(("regression","conflicting_digest"),1):
            host = OfflineHost(self.root,self.parent,index)
            original = host.write
            def inject(text):
                result = original(text)
                if json.loads(text)["type"] == "candidate_request":
                    if mode == "regression":
                        host.pending[-1]["summary"]["metrics"]["macro_f1"] = 0.
                    else:
                        host.pending[-1]["summary"]["validation_sha256"] = "sha256:" + "0"*64
                return result
            host.write = inject
            self.run_host(host)
            self.assertTrue(host.record["configuration"]["candidate_gate"]["eligible"])
            self.assertFalse(host.record["configuration"]["host_candidate_gate"]["eligible"])
            self.assertEqual(host.record["selected_artifact_id"],host.frame["parent_artifact"]["artifact_id"])

    def test_compact_sdk_receipt_keeps_host_score_and_binds_current_projection(self):
        host = OfflineHost(self.root,self.parent)
        original = host.write
        def compact(text):
            result = original(text)
            if json.loads(text)["type"] == "candidate_request":
                score = host.pending[-1]["summary"]["metrics"]["macro_f1"]
                host.pending[-1]["summary"] = {"macro_f1": score}
            return result
        host.write = compact
        state = self.run_host(host)
        gate = host.record["configuration"]["host_candidate_gate"]
        self.assertTrue(gate["eligible"])
        self.assertEqual(host.record["selected_artifact_id"],"candidate-1")
        self.assertEqual(gate["candidate"]["scope_source"],"sdk_current_candidate_public_validation")
        decision=json.loads((state/"decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertNotIn("scope",decision["host_candidate_receipt"]["summary"])
        self.assertNotIn("validation_sha256",decision["host_candidate_receipt"]["summary"])

    def test_compact_start_parent_measurement_allows_real_sdk_promotion(self):
        host = OfflineHost(self.root, self.parent, experiment="frozen_router")
        _, validation = load_pair(self.root/"data/train.jsonl", self.root/"data/validation.jsonl")
        before = evaluate(load_model(self.parent), validation)
        host.frame["current_validation"] = {"macro_f1": before["macro_f1"], "accuracy": before["accuracy"]}
        original = json.loads(json.dumps(host.frame["current_validation"]))
        self.run_host(host)
        gate = host.record["configuration"]["host_candidate_gate"]
        self.assertTrue(gate["eligible"])
        self.assertEqual(gate["parent"]["source"], "host")
        self.assertEqual(gate["parent"]["scope_source"], "sdk_current_parent_public_validation")
        self.assertEqual(host.record["selected_artifact_id"], "candidate-1")
        self.assertEqual(host.frame["current_validation"], original)
        calibration = json.loads((host.candidate/"routing.json").read_text())["calibration"]
        self.assertTrue(calibration["selected_gate"]["eligible"])

    def test_parent_binding_checks_projection_and_actual_primary_checkpoint(self):
        host = OfflineHost(self.root, self.parent)
        host.frame["current_validation"] = {"macro_f1": .7}
        # 所有声明的检查点逐文件核验；平台父身份可以是与任何文件不同的聚合摘要。
        auxiliary = self.parent.parent/"auxiliary.npz"
        auxiliary.write_bytes(b"unrelated auxiliary checkpoint")
        host.frame["parent_artifact"]["artifact"]["checkpoint_paths"].insert(0, auxiliary.name)
        host.frame["parent_artifact"]["files"].append({"path": auxiliary.name, "sha256": digest(auxiliary),
                                                     "size_bytes": auxiliary.stat().st_size})
        rnd = Round.start(host, host)
        rnd.parent.root = self.parent.parent
        sha = digest(rnd.data.validation_path)
        self.assertEqual(bind_parent_validation(rnd, sha).artifact_id, "parent-1")
        with self.assertRaisesRegex(ValueError, "projection changed"):
            bind_parent_validation(rnd, "sha256:" + "0"*64)
        rnd.parent.checkpoint_sha256 = "sha256:" + "0"*64
        self.assertEqual(bind_parent_validation(rnd, sha).checkpoint_sha256, rnd.parent.checkpoint_sha256)
        auxiliary.write_bytes(b"tampered")
        with self.assertRaises(HarnessError) as caught:
            bind_parent_validation(rnd, sha)
        self.assertEqual(caught.exception.code, "PARENT_DIGEST_MISMATCH")

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
