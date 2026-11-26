"""Strict completed-reply correction, preservation and original SDK round ACK."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import contest_submission
from contest_harness.errors import HarnessError
from contest_harness.round import Round
from gas_a.controller_prompts import request_digest
from gas_a.sensor_training.io import digest, json_text
from tests.controller_reply_faults import CASES, inject_reply
from tests.test_submission import ROOT, OfflineHost, writes_only_in
from tests.test_regression import RegressionHost, make_fixture
from scripts.validate_multitask import complete_experiment


class CompletedReplyRecoveryTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="completed-reply-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def peer(self, dataset, case):
        root = self.root / (dataset + "-" + case)
        if dataset == "A":
            shutil.copytree(ROOT / "data/public", root / "data")
            (root / "previous").mkdir()
            parent = root / "previous/checkpoint.npz"
            shutil.copyfile(ROOT / "models/reference.npz", parent)
            host = OfflineHost(root, parent)
        else:
            parent = make_fixture(root, dataset)
            host = RegressionHost(root, parent, dataset)
        host.frame["parent_artifact"]["artifact_id"] = "actual-parent-" + dataset
        return host

    def run_peer(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = host.root / "work"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertFalse(host.pending)  # The actual SDK consumed round_ack.
        return host.saved_state

    def test_completed_reply_faults_finish_all_datasets_with_full_evidence(self):
        for dataset in "ABC":
            for case in CASES:
                with self.subTest(dataset=dataset, case=case):
                    host = self.peer(dataset, case)
                    host.action = "experiment" if case == "wrong_selection" else "observe"
                    host.history = [{"id": "prior-round", "configuration": {}}]
                    parent_sha = digest(host.parent)
                    attempts, expected_responses = {}, []
                    original = host.write
                    def inject(text):
                        result = original(text)
                        frame = json.loads(text)
                        if frame["type"] == "model_request":
                            request = json.loads(frame["messages"][0]["content"])
                            phase = request["phase"]
                            attempts[phase] = attempts.get(phase, 0) + 1
                            inject_reply(host.pending[-1]["response"], request, case, attempts[phase])
                            expected_responses.append(host.pending[-1]["response"])
                        return result
                    host.write = inject
                    state = self.run_peer(host)
                    decisions = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
                    evidence = [json.loads(line) for line in (state / "model-responses.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual([r["response"] for r in evidence], expected_responses)
                    self.assertEqual([r["call_id"] for r in evidence], host.calls)
                    self.assertEqual(host.record["model_call_ids"], host.calls)
                    self.assertEqual(digest(host.parent), parent_sha)
                    self.assertEqual(host.record["selected_artifact_id"], "actual-parent-" + dataset)
                    self.assertLessEqual(max(attempts.values()), 2)
                    self.assertFalse(complete_experiment({"history_ids": host.record["history_ids"],
                        "candidate_submitted": host.candidate is not None, "record": host.record}))
                    for row in evidence:
                        self.assertEqual(row["request_sha256"], request_digest(row["request"]))
                    sources = [decisions[key]["decision_source"] for key in ("planning_calls", "selection_calls")]
                    if case in ("plain", "fence_crlf", "metadata_surrogate"):
                        self.assertEqual(sources, ["model", "model"])
                        self.assertEqual(len(host.calls), 2 if dataset == "A" else 1)
                    elif case == "repair_once":
                        self.assertEqual(sources, ["model", "model"])
                        self.assertEqual(len(host.calls), 3 if dataset == "A" else 2)
                        self.assertEqual(evidence[1]["request"]["correction"]["previous_completed_response"],
                                         evidence[0]["response"]["choices"][0]["message"]["content"])
                    elif case in ("wrong_parent", "unknown_experiment"):
                        self.assertEqual(sources, ["program_guard", "model"])
                        self.assertIsNone(host.candidate)
                    elif case == "wrong_selection":
                        self.assertEqual(sources, ["model", "program_guard"])
                        self.assertIsNotNone(host.candidate)
                    else:
                        self.assertEqual(sources, ["program_guard", "program_guard"])
                        self.assertIsNone(host.candidate)
                    frames = [f["type"] for f in host.frames]
                    self.assertLess(frames.index("history_request"), frames.index("model_request"))
                    self.assertLess(frames.index("state_request"), frames.index("round_complete"))

    def test_only_measured_model_loop_counts_towards_acceptance(self):
        host = self.peer("B", "experiment-success")
        host.history = [{"id": "prior-round", "configuration": {}}]
        self.run_peer(host)
        self.assertTrue(complete_experiment({"history_ids": host.record["history_ids"],
            "candidate_submitted": host.candidate is not None, "record": host.record}))

    def test_long_failed_reply_is_complete_in_correction_and_state(self):
        host = self.peer("B", "full-correction")
        host.action = "observe"
        original = host.write
        failed_text = "x" * 6000
        def inject(text):
            result = original(text)
            if json.loads(text)["type"] == "model_request" and len(host.calls) == 1:
                host.pending[-1]["response"]["choices"][0]["message"]["content"] = failed_text
            return result
        host.write = inject
        state = self.run_peer(host)
        responses = [json.loads(line) for line in (state / "model-responses.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(responses[1]["request"]["correction"]["previous_completed_response"], failed_text)
        self.assertEqual(responses[0]["response"]["choices"][0]["message"]["content"], failed_text)

    def test_serialization_keeps_unicode_and_nonfinite_checks_strict(self):
        original = {"reason": "中😀" + chr(0xD800)}
        encoded = json_text(original)
        self.assertEqual(json.loads(encoded), original)
        self.assertIn("中😀", encoded)
        encoded.encode("utf-8")
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                json_text({"live_measurement": value})

    def test_unknown_model_rpc_is_not_corrected_or_finished(self):
        host = self.peer("C", "rpc-unknown")
        original = host.write
        def interrupt(text):
            if json.loads(text)["type"] == "model_request":
                raise HarnessError("RPC_OUTCOME_UNKNOWN")
            return original(text)
        host.write = interrupt
        with self.assertRaises(HarnessError):
            self.run_peer(host)
        self.assertIsNone(host.record)
        self.assertFalse(any(f["type"] in ("candidate_request", "state_request", "round_complete") for f in host.frames))


if __name__ == "__main__":
    unittest.main()
