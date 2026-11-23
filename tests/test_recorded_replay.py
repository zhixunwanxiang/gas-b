"""重放必须只消费已保存回复，且数值不一致时不能宣称复现通过。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contest_submission
from scripts.replay_multitask_cloud import RecordedReplies, compare


class RecordedReplayTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="recorded-only-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_exhausted_recordings_never_fall_back_to_network(self):
        folder = self.root / "normal/provider-calls/call-001"
        folder.mkdir(parents=True)
        response = {"id": "original-provider-id", "choices": [{"message": {"content": "{}"}}]}
        (folder / "response.json").write_text(json.dumps(response), encoding="utf-8")
        recorded = RecordedReplies(self.root)
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")) as request:
            self.assertEqual(recorded.next(), ("original-provider-id", response))
            with self.assertRaisesRegex(ValueError, "no model fallback"):
                recorded.next()
        request.assert_not_called()
        self.assertEqual(recorded.verified, [])
        self.assertEqual(recorded.attempts, [])

    def test_changed_predictions_or_checkpoint_fail_reproduction(self):
        source, target = self.root / "source", self.root / "target"
        old = source / "normal/host-1/candidate"
        new = target / "host-1/candidate"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        for folder in (old, new):
            (folder / "checkpoint.npz").write_bytes(b"same-model-bytes")
        (source / "normal/host-1/protocol.jsonl").write_text(json.dumps({"frame": {
            "type": "candidate_request", "artifact": {"checkpoint_paths": ["checkpoint.npz"]}}})+'\n', encoding="utf-8")
        for folder in (source / "normal/measurement-1", target / "measurement-1"):
            folder.mkdir()
            (folder / "predictions.jsonl").write_text('{"outputs":{"value":1}}\n', encoding="utf-8")
        report = {"rounds": [{"round": 1, "selected_artifact_id": "candidate-1", "candidate_submitted": True}]}
        self.assertTrue(compare(source, target, report, report)["selected_artifacts_equal"])
        (new / "checkpoint.npz").write_bytes(b"different-model")
        with self.assertRaisesRegex(ValueError, "checkpoint/route bytes differ"):
            compare(source, target, report, report)
        (new / "checkpoint.npz").write_bytes(b"same-model-bytes")
        (target / "measurement-1/predictions.jsonl").write_text('{"outputs":{"value":2}}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "predictions differ"):
            compare(source, target, report, report)


if __name__ == "__main__":
    unittest.main()
