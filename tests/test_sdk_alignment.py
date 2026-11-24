"""SDK 1.0.4 request/response ordering and cumulative host-test time budgets.

Only offline peers are used. The SDK implementation is never modified.
"""
from collections import deque
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost
from tests.test_regression import RegressionHost, make_fixture
import contest_submission
from contest_harness.round import Round
from scripts.docker_rehearsal import child_environment
from scripts.validate_platform import exchange
from gas_a.sensor_training.io import digest


class TraceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def write(self, text):
        # A second request before consuming the previous reply is illegal.
        if self.pending:
            raise AssertionError("Participant sent another request before reading its SDK reply")
        self.trace.append(("participant", json.loads(text)))
        return super().write(text)

    def readline(self):
        text = super().readline()
        self.trace.append(("host", json.loads(text)))
        return text


class ClassificationPeer(TraceMixin, OfflineHost):
    pass


class RegressionPeer(TraceMixin, RegressionHost):
    pass


class SdkAlignmentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sdk-alignment-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_peer(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent  # Map only the unavailable /previous mount on Windows.
        with patch.dict(os.environ, {"CONTEST_WORK": str(host.root / "work")}), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertFalse(host.pending)
        return rnd

    def assert_order(self, host):
        trace = host.trace
        self.assertEqual(trace[0][0], "host")
        self.assertEqual(trace[0][1]["protocol"], "contest.harness.workspace.v2")
        expected = {"history_request": "history_response", "model_request": "model_response",
                    "upload_begin_request": "upload_begin_response", "upload_chunk_request": "upload_chunk_response",
                    "upload_finish_request": "upload_finish_response", "candidate_request": "candidate_response",
                    "state_request": "state_response", "round_complete": "round_ack"}
        for index in range(1, len(trace), 2):
            direction, request = trace[index]
            response_direction, response = trace[index + 1]
            self.assertEqual((direction, response_direction), ("participant", "host"))
            self.assertEqual(response["type"], expected[request["type"]])
            if request["type"] == "round_complete":
                self.assertTrue(response["accepted"])
            else:
                self.assertEqual(response["id"], request["id"])
        requests = [frame for direction, frame in trace[1:] if direction == "participant"]
        kinds = [frame["type"] for frame in requests]
        self.assertEqual(kinds[0], "history_request")
        self.assertEqual(kinds[-1], "round_complete")
        self.assertEqual([f["id"] for f in requests[:-1]], list(range(len(requests) - 1)))
        self.assertEqual(kinds.count("model_request"), 2)
        self.assertLess(kinds.index("model_request"), kinds.index("candidate_request"))
        self.assertLess(kinds.index("candidate_request"), len(kinds) - 1 - kinds[::-1].index("model_request"))
        finished_uploads = set()
        for direction, frame in trace:
            if direction == "host" and frame.get("type") == "upload_finish_response":
                finished_uploads.add(frame["upload_id"])
            if direction == "participant" and frame.get("type") in ("candidate_request", "state_request"):
                self.assertIn(frame["upload_id"], finished_uploads)
        self.assertEqual(kinds.count("state_request"), 1)
        self.assertEqual(kinds.count("round_complete"), 1)

    def test_actual_abc_rounds_follow_sdk_order_and_share_one_source_snapshot(self):
        for dataset in ("A", "B", "C"):
            with self.subTest(dataset=dataset):
                root = self.root / dataset
                if dataset == "A":
                    root.mkdir()
                    shutil.copytree(ROOT / "data/public", root / "data")
                    (root / "previous").mkdir()
                    parent = root / "previous/checkpoint.npz"
                    shutil.copyfile(ROOT / "models/reference.npz", parent)
                    host = ClassificationPeer(root, parent)
                else:
                    parent = make_fixture(root, dataset)
                    host = RegressionPeer(root, parent, dataset)
                self.run_peer(host)
                self.assert_order(host)
                candidate_frame = next(f for f in host.frames if f["type"] == "candidate_request")
                self.assertEqual(candidate_frame["artifact"]["predict_argv"], ["python3", "/app/harness.py", "predict"])
                self.assertFalse((host.candidate / "source").exists())
                if dataset == "A":
                    training = json.loads((host.candidate / "training.json").read_text(encoding="utf-8"))
                    snapshot = host.candidate / training["source_snapshot"]
                    for name, sha in training["source_sha256"].items():
                        self.assertEqual(digest(snapshot / name), sha)
                    self.assertEqual(len(list(host.candidate.rglob("*.py"))),
                                     len(list((ROOT / "src").rglob("*.py"))) + 2)

    def test_interrupted_upload_never_starts_another_rpc_or_sends_finish(self):
        parent = make_fixture(self.root, "C")
        host = RegressionPeer(self.root, parent, "C")
        original = host.readline

        def interrupted():
            if host.pending and host.pending[0].get("type") == "upload_chunk_response":
                raise TimeoutError("SDK response outcome is unknown")
            return original()

        with patch.object(host, "readline", side_effect=interrupted), self.assertRaises(TimeoutError):
            self.run_peer(host)
        self.assertEqual(host.frames[-1]["type"], "upload_chunk_request")
        self.assertIsNone(host.record)
        self.assertFalse(any(f["type"] == "state_request" for f in host.frames))

    def test_host_rejects_cumulative_overrun_even_when_each_dispatch_is_short(self):
        parent = make_fixture(self.root, "C")
        source = RegressionHost(self.root, parent, "C")
        frame = source.frame
        frame["wall_seconds"] = 10
        clock = [0.0]

        class TimedPeer:
            def __init__(self):
                self.frame, self.pending, self.record = frame, deque([frame]), None

            def readline(self):
                return json.dumps(self.pending.popleft()) + "\n"

            def write(self, text):
                request = json.loads(text)
                clock[0] += 6  # Each dispatch is below 10 seconds; their sum is not.
                self.pending.append({"type": "history_response", "id": request["id"], "records": []})

        code = "from contest_harness.round import Round; r=Round.start(); r.history(); r.history()"
        peer = TimedPeer()
        with patch("scripts.validate_platform.time.monotonic", side_effect=lambda: clock[0]), \
                self.assertRaisesRegex(TimeoutError, "host_dispatch.*history_request"):
            exchange(peer, [sys.executable, "-B", "-X", "utf8", "-c", code], child_environment(), source.store)
        transcript = [json.loads(line) for line in (source.store / "protocol.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["frame"].get("type") for e in transcript if e["direction"] == "harness"],
                         ["history_request", "history_request"])
        self.assertEqual(sum(e["direction"] == "host" for e in transcript), 2)  # start plus the first reply only.

    def test_host_sends_ack_and_accepts_a_completed_round_within_budget(self):
        parent = make_fixture(self.root, "C")
        source = RegressionHost(self.root, parent, "C")
        frame, clock = source.frame, [0.0]
        frame["wall_seconds"] = 10

        class Peer:
            def __init__(self):
                self.frame, self.pending, self.record = frame, deque([frame]), None

            def readline(self):
                return json.dumps(self.pending.popleft()) + "\n"

            def write(self, text):
                request = json.loads(text)
                clock[0] += 1
                if request["type"] == "history_request":
                    reply = {"type": "history_response", "id": request["id"], "records": []}
                elif request["type"] == "model_request":
                    reply = {"type": "model_response", "id": request["id"], "call_id": "offline-call",
                             "response": {"choices": [{"message": {"content": "retain parent"}}]}}
                else:
                    self.record = request["record"]
                    reply = {"type": "round_ack", "accepted": True}
                self.pending.append(reply)

        code = """
from contest_harness.round import Round
r = Round.start()
r.history()
r.model([{'role': 'user', 'content': 'Offline protocol check'}], max_tokens=32)
r.finish(selected_artifact_id=r.parent.artifact_id, hypothesis='Retain parent',
         change_summary='Observed public inputs', reason='Offline model selected parent',
         configuration={}, seed=1)
"""
        peer = Peer()
        with patch("scripts.validate_platform.time.monotonic", side_effect=lambda: clock[0]):
            exchange(peer, [sys.executable, "-B", "-X", "utf8", "-c", code], child_environment(), source.store)
        self.assertEqual(peer.record["model_call_ids"], ["offline-call"])
        self.assertFalse(peer.pending)
        transcript = [json.loads(line) for line in (source.store / "protocol.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(transcript[-1]["frame"], {"type": "round_ack", "accepted": True})


if __name__ == "__main__":
    unittest.main()
