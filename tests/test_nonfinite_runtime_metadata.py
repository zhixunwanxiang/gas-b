"""Finite data, optional nonfinite host metadata, and strict numeric boundaries."""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost, writes_only_in
from tests.test_regression import RegressionHost, make_fixture
import contest_submission
from contest_harness.round import Round
from contest_harness.errors import HarnessError
from gas_a.sdk_records import configuration_bytes, portable_record
from gas_a.sensor_training.reference.data import flatten, read_dataset
from gas_a.regression.data import load_pair, load_model
from gas_a.regression.evaluation import evaluate
from gas_a.controller_prompts import history_feedback
from gas_a import submission


class NonfiniteRuntimeMetadataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="runtime-metadata-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def peer(self, dataset, name):
        root = self.root / name
        if dataset == "A":
            shutil.copytree(ROOT / "data/public", root / "data")
            (root / "previous").mkdir()
            parent = root / "previous/checkpoint.npz"
            shutil.copyfile(ROOT / "models/reference.npz", parent)
            host = OfflineHost(root, parent)
        else:
            parent = make_fixture(root, dataset)
            host = RegressionHost(root, parent, dataset)
        return host

    def run_peer(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = host.root / "work"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        return host.saved_state

    def test_evidence_preserves_nonfinite_spelling_and_original_measurement(self):
        value = {"metrics": {"optional": float("nan"), "positive": float("inf"), "negative": -float("inf")}}
        with contextlib.redirect_stderr(io.StringIO()):
            clean = portable_record(value)
        self.assertIsNot(clean, value)
        self.assertEqual(clean["metrics"], dict.fromkeys(("optional", "positive", "negative")))
        self.assertEqual([w["original"] for w in clean["recording_warnings"]["nonfinite_fields"]], ["NaN", "Infinity", "-Infinity"])
        configuration_bytes(clean)
        with self.assertRaises(ValueError):
            configuration_bytes(value)

    def test_completed_model_response_metadata_does_not_abort_any_dataset(self):
        for dataset in "ABC":
            host = self.peer(dataset, "response-" + dataset)
            host.action = "observe"
            original = host.write
            def inject(text):
                result = original(text)
                if json.loads(text)["type"] == "model_request":
                    host.pending[-1]["response"]["optional_diagnostic"] = float("nan")
                return result
            host.write = inject
            state = self.run_peer(host)
            evidence = [json.loads(line) for line in (state / "model-responses.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(evidence), 2)
            self.assertEqual(host.record["model_call_ids"], host.calls)
            self.assertTrue(all(e["recording_warnings"]["nonfinite_fields"] for e in evidence))
            self.assertTrue(all(e["response"]["optional_diagnostic"] is None for e in evidence))

    def test_optional_candidate_nan_is_recorded_and_valid_candidate_can_be_adopted(self):
        for dataset in "ABC":
            host = self.peer(dataset, "candidate-" + dataset)
            original = host.write
            def inject(text):
                result = original(text)
                if json.loads(text)["type"] == "candidate_request":
                    host.pending[-1]["summary"]["metrics"]["optional_r2"] = float("nan")
                return result
            host.write = inject
            state = self.run_peer(host)
            self.assertEqual(host.record["selected_artifact_id"], "candidate-1")
            record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8"))
            self.assertIsNone(record["host_candidate_receipt"]["summary"]["metrics"]["optional_r2"])
            self.assertTrue(record["recording_warnings"]["nonfinite_fields"])

    def test_nonfinite_primary_candidate_score_still_rejects_promotion(self):
        for dataset in "ABC":
            host = self.peer(dataset, "invalid-score-" + dataset)
            original = host.write
            def inject(text):
                result = original(text)
                if json.loads(text)["type"] == "candidate_request":
                    host.pending[-1]["summary"]["metrics"]["macro_f1" if dataset == "A" else "rmse"] = float("nan")
                return result
            host.write = inject
            state = self.run_peer(host)
            self.assertEqual(host.record["selected_artifact_id"], "parent-1")
            self.assertFalse(host.record["configuration"]["host_candidate_gate"]["eligible"])

    def test_metric_case_is_consistent_through_parent_and_candidate_bindings(self):
        for dataset in "ABC":
            host = self.peer(dataset, "metric-" + dataset)
            host.frame["task_contract"]["metric"] = "MACRO_F1" if dataset == "A" else "RMSE"
            self.run_peer(host)
            self.assertEqual(host.record["selected_artifact_id"], "candidate-1")

    def test_input_nonfinite_strings_bools_and_overflowing_integers_remain_invalid(self):
        for bad in (float("nan"), float("inf"), -float("inf"), "1.5", "NaN", True, None, 10 ** 1000):
            with self.subTest(value_type=type(bad).__name__), self.assertRaises(ValueError):
                flatten({"features": [bad]})
        path = self.root / "public.jsonl"
        path.write_text('{"id":"s","inputs":{"features":["1.5"]},"answer":1}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "public.jsonl: line 1"):
            read_dataset(path, "regression", [])

    def test_sdk_unknown_outcome_remains_fatal(self):
        host = self.peer("C", "unknown-rpc")
        original = host.write
        def interrupt(text):
            if json.loads(text)["type"] == "model_request":
                raise HarnessError("RPC_OUTCOME_UNKNOWN")
            return original(text)
        host.write = interrupt
        with self.assertRaises(HarnessError):
            self.run_peer(host)
        self.assertIsNone(host.record)

    def test_unavailable_historical_numbers_do_not_invent_a_gain_or_abort(self):
        history = [{"id": "old", "configuration": {"experiment": {"name": "four_sensor", "learning_rate": float("nan")},
            "candidate_gate": None, "local_validation_before": float("nan"), "local_validation_after": .8}}]
        with contextlib.redirect_stderr(io.StringIO()):
            compact = submission._history_view(history)
        feedback = history_feedback(compact, "p", "t", "v", {}, {})
        self.assertIsNone(feedback[0]["measured_macro_f1_gain"])
        configuration_bytes({"history": compact, "feedback": feedback})
