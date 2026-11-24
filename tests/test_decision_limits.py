"""Completed-output recovery and exact SDK metadata byte boundaries."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost, writes_only_in
from tests.test_regression import make_fixture, RegressionHost
import contest_submission
from contest_harness.errors import HarnessError
from contest_harness.round import ModelReceipt, Round, _declaration
from gas_a import submission
from gas_a.sdk_records import bounded_configuration, configuration_bytes


class DecisionLimitsTest(unittest.TestCase):
    def test_json_fences_keep_strict_schema_and_utf8_limits(self):
        for opening in ("```json\n", "```JSON\r\n", "```\n"):
            text = opening + json.dumps({"reason": "中" * 1333 + "a"}, ensure_ascii=False) + "\n```"
            rnd = SimpleNamespace(max_output_tokens=2048, model=lambda *a, **k: ModelReceipt("real-completed-1", {
                "choices": [{"message": {"content": text}}]}))
            value, calls = submission._decision(rnd, {"phase": "select"}, ("reason",), 2048, lambda v: None, 0)
            self.assertEqual(len(value["reason"].encode("utf-8")), 4000)
            self.assertEqual(calls["decision_source"], "model")
        for invalid in ('{"reason":"' + "中" * 1334 + '"}', '{"reason":"x","extra":"x"}',
                        '{"reason":"x","reason":"y"}', '{"reason":NaN}', 'prefix {"reason":"x"}'):
            rnd.model = lambda *a, **k: ModelReceipt("real-completed-2", {"choices": [{"message": {"content": invalid}}]})
            with self.assertRaises(submission.DecisionValidationError):
                submission._decision(rnd, {"phase": "select"}, ("reason",), 2048, lambda v: None, 0)

    def test_configuration_sdk_serialization_boundary_and_lossless_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            exact = {"x": "a" * (65536 - len(configuration_bytes({"x": ""})))}
            self.assertEqual(len(configuration_bytes(exact)), 65536)
            self.assertIs(bounded_configuration(exact, state, 1, "finish"), exact)
            _declaration("h", "c", exact, 42)
            oversized = {"x": exact["x"] + "中", "action": "observe", "nested": {"reason": "中" * 20000}}
            compact = bounded_configuration(oversized, state, 1, "finish")
            self.assertLessEqual(len(configuration_bytes(compact)), 65536)
            _declaration("中" * 1333 + "a", "c", compact, 42)
            with self.assertRaises(HarnessError):
                _declaration("中" * 1334, "c", compact, 42)
            saved = json.loads((state / "configuration-evidence.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["configuration"], oversized)
            self.assertEqual(saved["sha256"], compact["configuration_evidence"]["sha256"])

    def run_host(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = host.root / "work"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        return host.saved_state

    def test_both_regression_datasets_keep_parent_on_completed_invalid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            for dataset in ("B", "C"):
                root = Path(tmp) / dataset
                parent = make_fixture(root, dataset)
                host = RegressionHost(root, parent, dataset)
                original = host.write
                def invalid_selection(text):
                    result = original(text)
                    frame = json.loads(text)
                    if frame["type"] == "model_request" and json.loads(frame["messages"][0]["content"])["phase"] == "select":
                        host.pending[-1]["response"]["choices"][0]["message"]["content"] = json.dumps({
                            "selected_artifact_id": "not-authorized", "reason": "invalid target"})
                    return result
                host.write = invalid_selection
                state = self.run_host(host)
                self.assertIsNotNone(host.candidate)
                self.assertEqual(host.record["selected_artifact_id"], "parent-1")
                self.assertEqual(host.record["model_call_ids"], host.calls)
                record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(record["selection_calls"]["decision_source"], "program_guard")
                self.assertEqual(len(record["selection_calls"]["corrections"]), 2)
                self.assertEqual(len(host.calls), 3)

    def test_model_rpc_unknown_outcome_is_never_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = make_fixture(root, "B")
            host = RegressionHost(root, parent, "B")
            original = host.write
            def interrupt(text):
                if json.loads(text)["type"] == "model_request":
                    raise HarnessError("RPC_OUTCOME_UNKNOWN")
                return original(text)
            host.write = interrupt
            with self.assertRaises(HarnessError):
                self.run_host(host)
            self.assertIsNone(host.record)
            self.assertFalse(any(f["type"] in ("state_request", "candidate_request", "round_complete") for f in host.frames))
