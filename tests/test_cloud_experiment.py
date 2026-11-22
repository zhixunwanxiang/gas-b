"""Cloud audit checks without networking, model calls, or real credentials."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.cloud_experiment import CloudJournal, media_items
from scripts.download_cloud_evidence import download_bundle


class CloudExperimentTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="cloud-journal-test-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.enterContext(patch.dict(os.environ, {"SWANLAB_API_KEY": "private-test-key-unique-123",
                                                 "PYTRIO_API_KEY": "private-pytrio-key-unique-456"}))
        self.sdk = SimpleNamespace(Text=lambda x: x, has_run=Mock(return_value=False), get_run=Mock())
        self.journal = CloudJournal(self.sdk, {"project": "gas-b", "workspace": "unit"}, self.root,
                                    {"dataset_key": "A"}, "unit-experiment")
        self.journal.run = SimpleNamespace(id=self.journal.run_id, log=Mock(), finish=Mock())

    def test_secrets_are_rejected_before_any_cloud_log(self):
        with self.assertRaises(ValueError):
            self.journal.emit(texts={"unsafe": os.environ["PYTRIO_API_KEY"]})
        self.journal.run.log.assert_not_called()

    def test_log_failure_preserves_audit_and_does_not_interrupt_real_response(self):
        self.journal.run.log.side_effect = RuntimeError("cloud unavailable")
        self.journal.after_call("genuine-test-id", {"model": "Qwen/Qwen3.8-27B"}, [])
        self.assertEqual(self.journal.errors[0]["error_type"], "RuntimeError")
        self.assertIn("genuine-test-id", (self.root / "cloud-events.jsonl").read_text(encoding="utf-8"))
        self.assertNotIn(os.environ["PYTRIO_API_KEY"], (self.root / "cloud-receipt.json").read_text())

    def prepare_cloud(self):
        (self.root / "normal").mkdir()
        (self.root / "normal/config.json").write_text('{"seed":42}', encoding="utf-8")
        self.journal.emit({"normal/selected_macro_f1": .8, "normal/complete_history_cycle": 1})
        self.journal.emit({"recovery/selected_macro_f1": .8})
        self.journal.archive()
        bodies, series = {}, {}
        for line in (self.root / "cloud-events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            for key, body in event["texts"].items():
                url = f"https://unit.invalid/{len(bodies)}"
                bodies[url] = body.encode()
                series.setdefault(key, []).append({"index": event["step"], "items": [{"url": url}]})
        media = {"list": [{"key": k, "metrics": v} for k, v in series.items()]}
        scalar = {"list": [{"key": k, "metrics": [{"step": s, "value": v} for s, v in self.journal.scalars_for(k).items()]}
                          for k in ("normal/selected_macro_f1", "normal/complete_history_cycle", "recovery/selected_macro_f1")]}
        remote = SimpleNamespace(url="https://unit.invalid/run", state="FINISHED",
                                 medias=Mock(return_value=media), metrics=Mock(return_value=scalar))
        self.journal.api = SimpleNamespace(run=Mock(return_value=remote))
        self.journal.receipt.update(cloud_finish_verified=True, run_path="unit/gas-b/own-id")
        return remote, bodies

    def test_cloud_readback_reconstructs_identical_complete_evidence_zip(self):
        remote, bodies = self.prepare_cloud()
        with patch("scripts.cloud_experiment.urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url])):
            result = self.journal.verify_cloud()
        self.assertTrue(result["verified"])
        self.assertEqual((self.root / "cloud-downloaded-evidence.zip").read_bytes(), (self.root / "evidence-bundle.zip").read_bytes())
        remote.metrics.assert_called_once()

    def test_cloud_only_download_restores_archive_and_never_overwrites_it(self):
        remote, bodies = self.prepare_cloud()
        target = self.root / "downloaded.zip"
        with patch("scripts.download_cloud_evidence.urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url])):
            manifest = download_bundle(remote, target)
            self.assertEqual(manifest["sha256"], self.journal.bundle["sha256"])
            self.assertEqual(target.read_bytes(), (self.root / "evidence-bundle.zip").read_bytes())
            with self.assertRaises(FileExistsError):
                download_bundle(remote, target)

    def test_cloud_only_download_rejects_corrupted_chunk_without_creating_output(self):
        remote, bodies = self.prepare_cloud()
        key, step, item = next(row for row in media_items(remote.medias.return_value)
                               if row[0] == "evidence/bundle_chunk")
        value = json.loads(bodies[item["url"]])
        value["chunk_sha256"] = "0" * 64
        bodies[item["url"]] = json.dumps(value).encode()
        target = self.root / "must-not-exist.zip"
        with patch("scripts.download_cloud_evidence.urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url])), \
                self.assertRaises(ValueError):
            download_bundle(remote, target)
        self.assertFalse(target.exists())

    def test_cloud_only_download_rejects_duplicate_chunk_identity(self):
        remote, bodies = self.prepare_cloud()
        key, step, item = next(row for row in media_items(remote.medias.return_value)
                               if row[0] == "evidence/bundle_chunk")
        remote.medias.return_value["list"].append({"key": key, "metrics": [{"index": step + 1, "items": [item]}]})
        with patch("scripts.download_cloud_evidence.urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url])), \
                self.assertRaises(ValueError):
            download_bundle(remote, self.root / "invalid.zip")

    def test_corrupt_cloud_text_cannot_be_reported_as_verified(self):
        _, bodies = self.prepare_cloud()
        bodies[next(iter(bodies))] = b"changed cloud content"
        with patch("scripts.cloud_experiment.urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url])), \
                self.assertRaises(ValueError):
            self.journal.verify_cloud()
        self.assertNotIn("content_verified", self.journal.receipt)

    def test_eventual_media_index_is_rechecked_without_model_calls_or_cloud_writes(self):
        remote, bodies = self.prepare_cloud()
        media = remote.medias.return_value
        remote.medias.side_effect = [{"list": []}, media]
        with patch("scripts.cloud_experiment.urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url])), \
                patch("scripts.cloud_experiment.time.sleep"):
            result = self.journal.verify_cloud()
        self.assertTrue(result["verified"])
        self.assertEqual(remote.medias.call_count, 2)

    def test_readback_resume_does_not_reinitialize_or_finish_a_cloud_run(self):
        self.prepare_cloud()
        sdk = SimpleNamespace(Api=Mock(return_value=self.journal.api), init=Mock(), finish=Mock())
        resumed = CloudJournal.resume_verification(sdk, self.root)
        self.assertEqual(resumed.texts, self.journal.texts)
        self.assertEqual(resumed.bundle, self.journal.bundle)
        sdk.init.assert_not_called()
        sdk.finish.assert_not_called()

    def test_cloud_scalar_mismatch_cannot_be_reported_as_verified(self):
        remote, bodies = self.prepare_cloud()
        remote.metrics.return_value["list"][0]["metrics"][0]["value"] = .1
        with patch("scripts.cloud_experiment.urllib.request.urlopen", side_effect=lambda url, timeout: io.BytesIO(bodies[url])), \
                self.assertRaises(ValueError):
            self.journal.verify_cloud()

    def test_finish_only_affects_owned_run_and_verifies_cloud_terminal_state(self):
        self.journal.api = Mock()
        self.journal.receipt["run_path"] = "unit/gas-b/own-id"
        with patch("scripts.cloud_experiment.verify_finish", return_value={"status": "verified", "state": "FINISHED"}):
            self.journal.finish("success")
        self.journal.run.finish.assert_called_once()
        self.assertTrue(self.journal.receipt["cloud_finish_verified"])
        self.journal.run.id = "someone-else"
        self.journal.run.finish.reset_mock()
        self.journal.finish("success")
        self.journal.run.finish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
