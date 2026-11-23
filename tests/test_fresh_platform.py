"""全流程总控与结果归档的离线测试：不访问网络，不调用真实模型。"""
from datetime import datetime, timezone, timedelta
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts import test_fresh_platform as fresh
from scripts.results_only_journal import ResultsOnlyJournal
from gas_a.sensor_training.io import digest


class FreshPlatformTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def journal(self):
        journal = ResultsOnlyJournal(SimpleNamespace(Text=lambda text: text), {}, self.root,
            {"dataset_key": "A", "source": "SOURCE_SENTINEL", "model": "authorized"}, "unit")
        journal.run = SimpleNamespace(log=Mock())
        return journal

    def test_archive_only_reads_sanitized_events(self):
        journal = self.journal()
        (self.root / "private.py").write_text("PRIVATE_SOURCE_SENTINEL")
        (self.root / "checkpoint.npz").write_bytes(b"PRIVATE_WEIGHT_SENTINEL")
        (self.root / "train.jsonl").write_text("PRIVATE_SAMPLE_SENTINEL")
        journal.emit({"normal/selected_macro_f1": .8}, {"audit/result": {"passed": True}})
        journal.archive()
        with zipfile.ZipFile(io.BytesIO((self.root / "evidence-bundle.zip").read_bytes())) as archive:
            self.assertEqual(archive.namelist(), ["results-only-events.jsonl"])
            body = archive.read(archive.namelist()[0])
        self.assertNotIn(b"PRIVATE_", body)
        self.assertNotIn("source", journal.experiment_config)
        self.assertFalse(journal.receipt["source_uploaded"])
        self.assertTrue(journal.bundle["results_only"])

    def test_model_request_is_digest_only_and_unknown_cloud_keys_rejected(self):
        journal = self.journal()
        journal.before_call({"messages": [{"content": "PRIVATE_PROMPT_SENTINEL"}], "max_tokens": 2048},
                            {"dataset_key": "A", "round_index": 1})
        journal.after_call("actual-id", {"model": "authorized", "choices": [{"message": {"content": "decision"}}]},
                           [{"content": "PRIVATE_PROMPT_SENTINEL"}])
        body = (self.root / "cloud-events.jsonl").read_text()
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", body)
        self.assertIn("actual-id", body)
        with self.assertRaises(ValueError):
            journal.emit(texts={"source/code": "private"})

    def fixture(self):
        training, recovery = self.root / "training", self.root / "recovery"
        for folder, phase in ((training, "normal"), (recovery, "recovery")):
            folder.mkdir()
            (folder / "suite.json").write_text('{"image_id":"immutable"}')
            calls = folder / phase / "provider-calls/call-001"
            calls.mkdir(parents=True)
            (calls / "receipt.json").write_text(json.dumps({"call_id": phase,
                "started_at": datetime.now(timezone.utc).isoformat()}))
        baseline = training / "official-baseline/checkpoint.npz"
        baseline.parent.mkdir()
        baseline.write_bytes(b"official-model")
        host = training / "normal/host-1"
        host.mkdir()
        self.protocol = host / "protocol.jsonl"
        self.frames = [{"frame": {"state_path": None, "parent_artifact": {"artifact_id": "initial"},
                       "parent_checkpoint_sha256": digest(baseline)}},
                       {"frame": {"type": "history_response", "records": []}}]
        self.save_frames()
        return training, recovery

    def save_frames(self):
        self.protocol.write_text("\n".join(json.dumps(row) for row in self.frames))

    def test_freshness_requires_same_image_and_empty_initial_history(self):
        training, recovery = self.fixture()
        started = datetime.now(timezone.utc) - timedelta(seconds=5)
        result = fresh.fresh_case_checks(training, recovery, "A", "immutable", started)
        self.assertTrue(result["fresh_initial_model"])
        self.frames[1]["frame"]["records"] = [{"id": "old"}]
        self.save_frames()
        with self.assertRaises(ValueError):
            fresh.fresh_case_checks(training, recovery, "A", "immutable", started)
        with self.assertRaises(ValueError):
            fresh.fresh_case_checks(training, recovery, "A", "different-image", started)

    def test_receipt_from_before_run_rejected(self):
        training, recovery = self.fixture()
        with self.assertRaises(ValueError):
            fresh.fresh_case_checks(training, recovery, "A", "immutable", datetime.now(timezone.utc) + timedelta(seconds=5))

    def test_execution_changes_and_failed_phase_cannot_count_as_pass(self):
        with patch.object(fresh, "frozen_source", return_value={"a": "changed"}):
            with self.assertRaises(RuntimeError):
                fresh.assert_frozen({"a": "original"})
        args = SimpleNamespace(expected_image_id="immutable")
        with patch.object(fresh, "frozen_source", return_value={}), \
                patch.object(fresh, "execute_phases", side_effect=TimeoutError("unknown")):
            with self.assertRaises(TimeoutError):
                fresh.execute(args, {}, self.root)
        summary = fresh.read(self.root / "summary.json")
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["requirements_passed"])
        self.assertEqual(summary["error_type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
