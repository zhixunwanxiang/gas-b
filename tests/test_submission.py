"""Exercise real SDK frames and numerical training with an offline host.

No external model, account, or contest evaluation is called by these tests.
"""
import base64
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from contest_harness.round import Round
from contest_harness.errors import HarnessError
import numpy as np
from gas_a import submission
from gas_a.sensor_training.data import evaluate, load_model, load_pair
from gas_a.sensor_training.io import CLASSES, digest
from gas_a.sensor_training.reference.model import Model
from gas_a.frozen_router import load_predictor, checkpoint_paths as artifact_checkpoint_paths


_WRITE_ROOTS = None


def _write_audit(event, args):
    if _WRITE_ROOTS is None:
        return
    paths = []
    if event == "open":
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            paths = [path]
    elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"):
        paths = [args[0]]
    elif event == "os.rename":
        paths = [args[0], args[1]]
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            target = Path(os.fsdecode(path)).resolve()
            if not any(target.is_relative_to(root) for root in _WRITE_ROOTS):
                raise PermissionError(f"Test denied a write outside writable mounts: {target}")


sys.addaudithook(_write_audit)


@contextmanager
def writes_only_in(*roots):
    global _WRITE_ROOTS
    _WRITE_ROOTS = [Path(root).resolve() for root in roots]
    try:
        yield
    finally:
        _WRITE_ROOTS = None


class OfflineHost:
    """Synchronous host peer for the unmodified SDK's JSONL RPC implementation."""
    def __init__(self, root, parent, round_index=1, history=(), state=None, experiment="four_sensor"):
        self.root = root
        self.parent = parent
        self.history = list(history)
        self.experiment = experiment
        self.messages = []
        self.frames = []
        self.pending = deque()
        self.record = None
        self.candidate = None
        self.candidate_metrics = None
        self.calls = []
        self.uploads = {}
        self.reserved_upload_bytes = {"candidate": 0, "state": 0}
        self.next_sequence = 0
        self.bad_plan = False
        self.force_retain = False
        self.action = "experiment"
        self.malformed_responses = 0
        self.artifact_dirs = {}
        self.store = root / f"host-{round_index}"
        self.store.mkdir()
        self.frame = {
            "protocol": "contest.harness.workspace.v2", "dataset_key": "A", "round_index": round_index,
            "prompt": "OFFICIAL_PROMPT_SENTINEL", "prompt_version": "industrial-workspace-flexible-v3",
            "task_contract": {"task": "classification", "metric": "macro_f1", "metric_direction": "higher",
                              "public_projection": {"train_path": str(root / "data/train.jsonl"),
                                                    "validation_path": str(root / "data/validation.jsonl")}},
            "model_name": "offline-host-stub", "max_output_tokens": 2048,
            "parent_checkpoint_sha256": digest(parent),
            "parent_artifact": {"artifact_id": f"parent-{round_index}",
                                "artifact": {"checkpoint_paths": [parent.name], "code_paths": [],
                                             "predict_argv": submission.PREDICT_ARGV},
                                "files": [{"path": parent.name, "sha256": digest(parent),
                                           "size_bytes": parent.stat().st_size}]},
            "artifacts": [], "state_path": str(state) if state else None,
            "state_sha256": "", "current_validation": {},
            "resource_limits": {"artifact_upload_protocol": "chunked-v1", "artifact_bytes": 512 * 1024 * 1024,
                                "state_bytes": 512 * 1024 * 1024, "candidate_upload_bytes_per_round": 1024 * 1024 * 1024,
                                "state_upload_bytes_per_round": 512 * 1024 * 1024, "upload_chunk_bytes": 64 * 1024},
        }
        self.pending.append(self.frame)

    def checkpoint_paths(self, directory, checkpoint="checkpoint.npz"):
        return artifact_checkpoint_paths(Path(directory) / checkpoint)

    def measure_candidate(self):
        _, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl")
        return evaluate(load_predictor(self.candidate / "checkpoint.npz"), validation)

    def add_artifact(self, identifier, directory, checkpoint="checkpoint.npz"):
        directory = Path(directory)
        manifest = [{"path": p.relative_to(directory).as_posix(), "sha256": digest(p), "size_bytes": p.stat().st_size}
                    for p in sorted(directory.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
        checkpoints = self.checkpoint_paths(directory, checkpoint)
        descriptor = {"checkpoint_paths": checkpoints,
                      "code_paths": ["routing.json"] if (directory / "routing.json").is_file() else [],
                      "predict_argv": submission.PREDICT_ARGV}
        receipt = {"artifact_id": identifier, "files": manifest, "artifact": descriptor}
        encoded = json.dumps({"files": manifest, "artifact": descriptor}, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        receipt["artifact_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
        self.frame["artifacts"].append(receipt)
        self.artifact_dirs[identifier] = directory
        return receipt

    def write(self, text):
        frame = json.loads(text)
        self.frames.append(frame)
        kind = frame["type"]
        if kind != "round_complete":
            assert frame["id"] == self.next_sequence
            self.next_sequence += 1
        reply = {"id": frame.get("id")}
        if kind == "history_request":
            reply.update(type="history_response", records=self.history)
        elif kind == "model_request":
            self.messages.append(frame["messages"])
            request = json.loads(frame["messages"][0]["content"])
            call_id = f"call-{self.frame['round_index']}-{len(self.calls) + 1}"
            self.calls.append(call_id)
            if request["phase"] == "plan":
                actions = request["response_schema"]["action"]
                action = self.action if self.action in actions else "observe"
                decision = {"action": action,
                            "experiment": ("unknown" if self.bad_plan else self.experiment) if action == "experiment" else "none",
                            "target_artifact_id": (request["response_schema"]["target_artifact_id"][-1]
                                                   if action == "rollback" else request["parent_artifact_id"]),
                            "hypothesis": "A bounded continuation can improve the measured weak class.",
                            "reason": "Read the public validation confusion matrix and available history."}
            else:
                scores = request["same_local_validation"]
                eligible = request["response_schema"]["selected_artifact_id"]
                adopt = (not self.force_retain
                         and request["candidate_artifact_id"] in eligible
                         and scores["candidate"] is not None
                         and (scores["candidate"]["macro_f1"] > scores["parent"]["macro_f1"] + 1e-12
                              if self.frame["task_contract"]["task"] == "classification"
                              else scores["candidate"]["rmse"] < scores["parent"]["rmse"] - 1e-12))
                decision = {"selected_artifact_id": request["candidate_artifact_id"] if adopt else request["parent_artifact_id"],
                            "reason": "Selected using the same public validation Macro-F1; retain the parent on a tie."}
                if request["action"] == "rollback" and not self.force_retain:
                    decision["selected_artifact_id"] = eligible[-1]
            content = json.dumps(decision)
            if self.malformed_responses:
                self.malformed_responses -= 1
                content = "{invalid JSON"
            reply.update(type="model_response", call_id=call_id,
                         response={"choices": [{"message": {"content": content}}]})
        elif kind == "artifact_read_request":
            path = self.artifact_dirs[frame["artifact_id"]] / frame["path"]
            raw = path.read_bytes()
            chunk = raw[frame["offset"]:frame["offset"] + frame["length"]]
            reply.update(type="artifact_read_response", data_base64=base64.b64encode(chunk).decode(),
                         eof=frame["offset"] + len(chunk) == len(raw))
        elif kind == "upload_begin_request":
            upload_kind = frame["kind"]
            assert upload_kind in ("candidate", "state")
            manifest = frame["files"]
            assert len(manifest) <= 64
            assert manifest or upload_kind == "state"
            assert len({f["path"] for f in manifest}) == len(manifest)
            assert all(set(f) == {"path", "size_bytes", "sha256"} for f in manifest)
            total = sum(f["size_bytes"] for f in manifest)
            limits = self.frame["resource_limits"]
            assert total <= limits["artifact_bytes" if upload_kind == "candidate" else "state_bytes"]
            self.reserved_upload_bytes[upload_kind] += total
            assert self.reserved_upload_bytes[upload_kind] <= limits[upload_kind + "_upload_bytes_per_round"]
            upload_id = f"upload-{len(self.uploads) + 1}"
            self.uploads[upload_id] = {"kind": upload_kind, "manifest": manifest,
                                       "contents": {f["path"]: bytearray() for f in manifest}, "finished": False}
            reply.update(type="upload_begin_response", upload_id=upload_id)
        elif kind == "upload_chunk_request":
            upload = self.uploads[frame["upload_id"]]
            assert not upload["finished"]
            data = base64.b64decode(frame["data_base64"], validate=True)
            assert len(data) <= self.frame["resource_limits"]["upload_chunk_bytes"]
            content = upload["contents"][frame["path"]]
            assert frame["offset"] == len(content)
            content.extend(data)
            item = next(f for f in upload["manifest"] if f["path"] == frame["path"])
            assert len(content) <= item["size_bytes"]
            reply.update(type="upload_chunk_response", offset=len(content))
        elif kind == "upload_finish_request":
            upload = self.uploads[frame["upload_id"]]
            for item in upload["manifest"]:
                raw = upload["contents"][item["path"]]
                assert len(raw) == item["size_bytes"]
                assert "sha256:" + hashlib.sha256(raw).hexdigest() == item["sha256"]
            upload["finished"] = True
            reply.update(type="upload_finish_response", upload_id=frame["upload_id"])
        elif kind == "candidate_request":
            assert frame["parent_checkpoint_sha256"] == digest(self.parent)
            assert "files" not in frame
            upload = self.uploads[frame["upload_id"]]
            assert upload["kind"] == "candidate" and upload["finished"]
            self.candidate = self.store / "candidate"
            self.candidate.mkdir()
            for name, content in upload["contents"].items():
                target = self.candidate / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            assert frame["artifact"]["checkpoint_paths"] == self.checkpoint_paths(self.candidate)
            assert frame["artifact"]["predict_argv"] == submission.PREDICT_ARGV
            for name in frame["artifact"]["code_paths"]:
                assert (self.candidate / name).is_file()
            self.candidate_metrics = self.measure_candidate()
            reply.update(type="candidate_response", artifact_id=f"candidate-{self.frame['round_index']}",
                         summary={"metrics": {k: v for k, v in self.candidate_metrics.items() if k in ("macro_f1", "rmse", "mae")},
                                  "validation_sha256": digest(self.root / "data/validation.jsonl"),
                                  "scope": "authorized_public_validation"})
        elif kind == "state_request":
            assert "files" not in frame
            upload = self.uploads[frame["upload_id"]]
            assert upload["kind"] == "state" and upload["finished"]
            self.saved_state = self.store / "state"
            self.saved_state.mkdir()
            for name, content in upload["contents"].items():
                target = self.saved_state / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            reply.update(type="state_response", state_sha256="sha256:" + hashlib.sha256(
                json.dumps(upload["manifest"], sort_keys=True).encode()).hexdigest())
        elif kind == "round_complete":
            self.record = frame["record"]
            reply = {"type": "round_ack", "accepted": True}
        else:
            raise AssertionError(f"Unexpected RPC {kind}")
        self.pending.append(reply)
        return len(text)

    def flush(self):
        pass

    def readline(self):
        return json.dumps(self.pending.popleft()) + "\n"


class SubmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="gas-a-sdk-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "previous").mkdir()
        (self.root / "data").mkdir()
        self.checkpoint = self.root / "previous/checkpoint.npz"
        shutil.copyfile(ROOT / "models/reference.npz", self.checkpoint)
        for name in ("train.jsonl", "validation.jsonl"):
            shutil.copyfile(ROOT / "data/public" / name, self.root / "data" / name)
        submission._prediction_model.cache_clear()

    def env(self, checkpoint=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CONTEST_")}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                   CONTEST_ARTIFACT_PATH=str(checkpoint or self.checkpoint))
        return env

    def sample(self, **updates):
        values = dict(task="classification", dataset_key="A", inputs={"features": [0.0] * 128})
        values.update(updates)
        return SimpleNamespace(**values)

    def test_describe_advertises_native_classification_and_regression(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "harness.py"), "describe"],
                                env=self.env(), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        description = json.loads(result.stdout)
        self.assertEqual(description["protocol"], "contest.harness.workspace.v2")
        self.assertEqual(description["model_access"], "host_gateway_rpc")
        self.assertEqual(description["sdk_version"], "1.0.4")
        self.assertEqual(description["tasks"], ["classification", "regression"])

    def test_legacy_host_is_rejected_before_model_calls_or_training(self):
        host = OfflineHost(self.root, self.checkpoint)
        host.frame["resource_limits"].pop("artifact_upload_protocol")
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        with self.assertRaises(HarnessError) as caught:
            submission.train(rnd)
        self.assertEqual(caught.exception.code, "CHUNKED_UPLOAD_UNSUPPORTED")
        self.assertEqual(host.frames, [])
        self.assertEqual(host.calls, [])
        self.assertIsNone(host.candidate)

    def test_candidate_and_state_use_chunked_uploads_without_inline_files(self):
        host = OfflineHost(self.root, self.checkpoint)
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}):
            submission.train(rnd)
        self.assertTrue(rnd._finished)
        for kind in ("candidate_request", "state_request"):
            frame = next(f for f in host.frames if f["type"] == kind)
            self.assertNotIn("files", frame)
            self.assertTrue(host.uploads[frame["upload_id"]]["finished"])
        chunks = [f for f in host.frames if f["type"] == "upload_chunk_request"]
        self.assertTrue(any(f["offset"] > 0 for f in chunks))
        self.assertTrue(all(len(base64.b64decode(f["data_base64"])) <= 64 * 1024 for f in chunks))

    def test_host_advertised_artifact_limit_is_honored_before_upload(self):
        host = OfflineHost(self.root, self.checkpoint)
        host.frame["resource_limits"]["artifact_bytes"] = 128
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}):
            with self.assertRaises(HarnessError) as caught:
                submission.train(rnd)
        self.assertEqual(caught.exception.code, "ARTIFACT_LIMIT")
        self.assertEqual(host.uploads, {})
        self.assertFalse(any(f["type"] == "candidate_request" for f in host.frames))

    def test_interrupted_model_request_stops_application_without_retry(self):
        host = OfflineHost(self.root, self.checkpoint)
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        original = host.readline
        def interrupted():
            if host.pending and host.pending[0].get("type") == "model_response":
                raise TimeoutError("The host may already have processed this model request")
            return original()
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}), \
                patch.object(host, "readline", side_effect=interrupted):
            with self.assertRaises(TimeoutError):
                submission.train(rnd)
            sent, late = len(host.frames), len(host.pending)
            with self.assertRaises(HarnessError) as caught:
                rnd.history()
        self.assertEqual(caught.exception.code, "RPC_OUTCOME_UNKNOWN")
        self.assertEqual(len(host.frames), sent)
        self.assertEqual(len(host.pending), late)
        self.assertEqual(len(host.calls), 1)
        self.assertIsNone(host.candidate)

    def test_interrupted_upload_does_not_retry_or_submit_incomplete_candidate(self):
        host = OfflineHost(self.root, self.checkpoint)
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        original = host.write
        def interrupted(text):
            result = original(text)
            if json.loads(text)["type"] == "upload_chunk_request":
                raise TimeoutError("The chunk may already have reached the host")
            return result
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}), \
                patch.object(host, "write", side_effect=interrupted):
            with self.assertRaises(TimeoutError):
                submission.train(rnd)
        self.assertEqual(sum(f["type"] == "upload_chunk_request" for f in host.frames), 1)
        self.assertGreater(host.reserved_upload_bytes["candidate"], 0)
        self.assertFalse(any(f["type"] in ("candidate_request", "state_request", "round_complete") for f in host.frames))
        with self.assertRaises(HarnessError) as caught:
            rnd.history()
        self.assertEqual(caught.exception.code, "RPC_OUTCOME_UNKNOWN")

    def test_predict_flushes_each_row_before_next_input_or_eof(self):
        env = self.env()
        env["CONTEST_PREDICTION_CONTEXT"] = json.dumps({"protocol": "contest.predict-context.v1",
            "dataset_key": "A", "task": "classification", "artifact": {}, "artifact_id": "chosen-model"})
        process = subprocess.Popen([sys.executable, "-B", str(ROOT / "harness.py"), "predict"],
                                   cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8")
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        lines = queue.Queue()
        def read_lines():
            for line in process.stdout:
                lines.put(line)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        for _ in range(2):
            process.stdin.write(json.dumps({"task": "classification", "inputs": {"features": [0.0] * 128}}) + "\n")
            process.stdin.flush()
            result = json.loads(lines.get(timeout=10))
            outputs = result["outputs"]
            self.assertEqual([x["label"] for x in outputs], CLASSES)
            scores = np.array([x["score"] for x in outputs])
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(((0 <= scores) & (scores <= 1)).all())
            self.assertAlmostEqual(float(scores.sum()), 1.0)
            self.assertEqual(int(np.count_nonzero(scores == scores.max())), 1)
            self.assertIsNone(process.poll())
        process.stdin.close()
        self.assertEqual(process.wait(timeout=10), 0, process.stderr.read())
        reader.join(timeout=2)
        process.stdout.close()
        process.stderr.close()

    def test_predict_uses_only_selected_checkpoint_and_does_not_write(self):
        model = Model.load(self.checkpoint)
        for value in model.params.values():
            value[:] = 0
        model.params["b2"][4] = 9
        selected = self.root / "selected.npz"
        model.save(selected)
        with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(selected)}), writes_only_in():
            result = submission.predict(self.sample(), self.root / "missing-model-dir")
        self.assertEqual(max(result, key=lambda x: x["score"])["label"], "Acetone")

    def test_predict_resolves_checkpoint_from_model_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            result = submission.predict(self.sample(), self.checkpoint.parent)
        self.assertEqual(len(result), 6)

    def test_predict_breaks_exact_score_ties(self):
        model = Model.load(self.checkpoint)
        for value in model.params.values():
            value[:] = 0
        tied = self.root / "tied.npz"
        model.save(tied)
        with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(tied)}):
            result = submission.predict(self.sample(), self.root)
        scores = [x["score"] for x in result]
        self.assertEqual(scores.count(max(scores)), 1)
        self.assertAlmostEqual(sum(scores), 1)

    def test_predict_rejects_invalid_feature_contracts(self):
        for inputs in [{"features": [0] * 127}, {"features": [True] * 128},
                       {"features": [float("nan")] * 128}, {"features": [[0]] * 128},
                       {"features": [0] * 128, "answer": "Ethanol"}]:
            with self.subTest(inputs=str(inputs)[:60]), self.assertRaises(ValueError):
                submission.predict(self.sample(inputs=inputs), self.root)

    def test_rejects_other_datasets_and_tasks(self):
        for task, dataset in [("regression", "A"), ("classification", "B")]:
            with self.assertRaises(ValueError):
                submission.predict(self.sample(task=task, dataset_key=dataset), self.root)

    def test_three_rounds_preserve_actual_parent_and_only_write_workspace(self):
        parent = self.checkpoint
        histories, state = [], None
        for index in range(1, 4):
            host = OfflineHost(self.root, parent, index, histories, state,
                               experiment="four_sensor" if index == 1 else "conservative_head")
            rnd = Round.start(host, host)
            rnd.parent.root = parent.parent
            work = self.root / f"workspace-{index}"
            with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store):
                contest_submission.train(rnd)
            self.assertTrue(rnd._finished)
            self.assertFalse(host.pending)
            record = host.record
            self.assertEqual(record["history_ids"], [r["id"] for r in histories])
            self.assertEqual(record["model_call_ids"], host.calls)
            self.assertEqual(len(host.calls), 2)
            self.assertEqual([json.loads(m[0]["content"])["phase"] for m in host.messages], ["plan", "select"])
            self.assertNotIn("OFFICIAL_PROMPT_SENTINEL", json.dumps(host.messages))
            before, after = Model.load(parent), Model.load(host.candidate / "checkpoint.npz")
            self.assertGreater(after.meta["steps"], before.meta["steps"])
            self.assertTrue(np.array_equal(before.mean, after.mean))
            self.assertTrue(np.array_equal(before.scale, after.scale))
            self.assertEqual(after.meta["seed"], before.meta["seed"])
            self.assertEqual(record["configuration"]["parent_checkpoint_sha256"], digest(parent))
            continuation = json.loads((host.candidate / "continuation.json").read_text(encoding="utf-8"))
            self.assertEqual(continuation["parent"]["checkpoint_sha256"], digest(parent))
            self.assertEqual(continuation["parent"]["artifact_id"], rnd.parent.artifact_id)
            self.assertTrue(all(continuation["checks"].values()))
            state = next(work.glob("round-*/state"))
            self.assertEqual(len((state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()), index)
            self.assertFalse(any(p.suffix == ".dat" for p in host.candidate.rglob("*")))
            histories.append({"id": f"history-{index}", **record})
            if record["selected_artifact_id"].startswith("candidate-"):
                parent = host.candidate / "checkpoint.npz"

    def test_parent_digest_mismatch_is_rejected(self):
        host = OfflineHost(self.root, self.checkpoint)
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        self.checkpoint.write_bytes(b"tampered")
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}):
            with self.assertRaises(HarnessError) as caught:
                submission.train(rnd)
        self.assertEqual(caught.exception.code, "PARENT_DIGEST_MISMATCH")
        self.assertIsNone(host.candidate)

    def test_train_cli_exchanges_protocol_frames_and_waits_for_ack(self):
        host = OfflineHost(self.root, self.checkpoint)
        host.frame["prompt"] = "官方任务提示：仅使用授权的公开训练集续训。"
        env = self.env()
        env.update(CONTEST_WORK=str(self.root / "workspace"),
                   GAS_A_TEST_ROOT=str(ROOT), GAS_A_TEST_PARENT=str(self.checkpoint.parent))
        # Replace only the unavailable /previous mount location on Windows;
        # execute the real harness entry, real SDK, and participant module.
        bootstrap = """
import os, runpy, sys
from pathlib import Path
from contest_harness.round import ParentView
original = ParentView.__init__
def init(self, frame):
    original(self, frame)
    self.root = Path(os.environ['GAS_A_TEST_PARENT'])
ParentView.__init__ = init
root = Path(os.environ['GAS_A_TEST_ROOT'])
sys.path.insert(0, str(root))
sys.argv = [str(root / 'harness.py'), 'train']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        process = subprocess.Popen([sys.executable, "-B", "-c", bootstrap], cwd=self.root,
                                   env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8")
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        lines = queue.Queue()
        def read_lines():
            for line in process.stdout:
                lines.put(line)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        process.stdin.write(host.readline())
        process.stdin.flush()
        while host.record is None:
            line = lines.get(timeout=15)
            self.assertNotEqual(json.loads(line).get("type"), "round_failed", line)
            host.write(line)
            if host.record is not None:
                # finish() must remain blocked until the host confirms.
                self.assertIsNone(process.poll())
            process.stdin.write(host.readline())
            process.stdin.flush()
        process.stdin.close()
        self.assertEqual(process.wait(timeout=10), 0, process.stderr.read())
        reader.join(timeout=2)
        self.assertTrue(lines.empty())
        self.assertEqual(len(host.record["model_call_ids"]), 2)
        process.stdout.close()
        process.stderr.close()

    def test_model_cannot_inject_unconfigured_experiment(self):
        host = OfflineHost(self.root, self.checkpoint)
        host.bad_plan = True
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}):
            with self.assertRaisesRegex(ValueError, "outside the configured menu"):
                submission.train(rnd)
        self.assertIsNone(host.candidate)

    def test_work_directory_cannot_be_in_read_only_code_or_data(self):
        for work in (ROOT / "runtime", self.root / "data/work", self.checkpoint.parent / "work"):
            host = OfflineHost(self.root, self.checkpoint, round_index=len(list(self.root.glob("host-*"))) + 1)
            rnd = Round.start(host, host)
            rnd.parent.root = self.checkpoint.parent
            with patch.dict(os.environ, {"CONTEST_WORK": str(work)}):
                with self.assertRaisesRegex(ValueError, "read-only"):
                    submission.train(rnd)

    def test_parent_checkpoint_can_have_any_declared_name_and_location(self):
        nested = self.root / "previous/weights/selected-stage.bin"
        nested.parent.mkdir()
        shutil.copyfile(self.checkpoint, nested)
        auxiliary = Model.load(self.checkpoint)
        auxiliary.params["b2"][0] += 1
        auxiliary_path = self.root / "previous/auxiliary.npz"
        auxiliary.save(auxiliary_path)
        host = OfflineHost(self.root, nested)
        host.frame["parent_artifact"]["artifact"]["checkpoint_paths"] = ["auxiliary.npz", "weights/selected-stage.bin"]
        host.frame["parent_artifact"]["files"] = [
            {"path": str(p.relative_to(nested.parent.parent)).replace("\\", "/"),
             "sha256": digest(p), "size_bytes": p.stat().st_size} for p in (auxiliary_path, nested)]
        rnd = Round.start(host, host)
        rnd.parent.root = nested.parent.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}):
            submission.train(rnd)
        record = json.loads((host.candidate / "continuation.json").read_text(encoding="utf-8"))
        self.assertEqual(record["parent"]["checkpoint_sha256"], digest(nested))
        self.assertNotEqual(record["parent"]["checkpoint_sha256"], digest(auxiliary_path))

    def test_training_uses_changed_authorized_projection_without_archive_hashes(self):
        projection_dir = self.root / "next-stage-public"
        projection_dir.mkdir()
        sources = {}
        for name, start, count in [("train.jsonl", 25, 100), ("validation.jsonl", 15, 180)]:
            rows = (self.root / "data" / name).read_text(encoding="utf-8").splitlines()[start:start+count]
            target = projection_dir / name
            target.write_text("\n".join(rows) + "\n", encoding="utf-8")
            sources[name] = target
        host = OfflineHost(self.root, self.checkpoint, experiment="conservative_head")
        host.frame["task_contract"]["public_projection"].update(
            train_path=str(sources["train.jsonl"]), validation_path=str(sources["validation.jsonl"]))
        rnd = Round.start(host, host)
        rnd.parent.root = self.checkpoint.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace")}):
            submission.train(rnd)
        record = json.loads((host.candidate / "continuation.json").read_text(encoding="utf-8"))
        self.assertEqual(record["public_projection"]["train_sha256"], digest(sources["train.jsonl"]))
        self.assertEqual(record["public_projection"]["validation_rows"], 180)
        self.assertEqual(record["public_projection"]["train_rows"], 100)
        self.assertTrue(rnd._finished)

    def test_rejected_candidate_is_not_used_as_the_next_parent(self):
        first = OfflineHost(self.root, self.checkpoint)
        first.force_retain = True
        rnd = Round.start(first, first)
        rnd.parent.root = self.checkpoint.parent
        work = self.root / "workspace-first"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}):
            submission.train(rnd)
        self.assertEqual(first.record["selected_artifact_id"], rnd.parent.artifact_id)
        state = next(work.glob("round-*/state"))
        second = OfflineHost(self.root, self.checkpoint, 2,
                             [{"id": "retained-round", **first.record}], state, "conservative_head")
        rnd = Round.start(second, second)
        rnd.parent.root = self.checkpoint.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / "workspace-second")}):
            submission.train(rnd)
        record = json.loads((second.candidate / "continuation.json").read_text(encoding="utf-8"))
        self.assertEqual(record["parent"]["state"]["steps"], Model.load(self.checkpoint).meta["steps"])
        self.assertEqual(record["parent"]["checkpoint_sha256"], digest(self.checkpoint))
        self.assertNotEqual(record["parent"]["checkpoint_sha256"], digest(first.candidate / "checkpoint.npz"))

    def test_prediction_context_resolves_nested_checkpoint_and_rejects_ambiguity(self):
        directory = self.root / "artifact"
        nested = directory / "weights/chosen.bin"
        nested.parent.mkdir(parents=True)
        shutil.copyfile(self.checkpoint, nested)
        context = {"artifact": {"checkpoint_paths": ["weights/chosen.bin"]}}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(len(submission.predict(self.sample(context=context), directory)), 6)
            for paths in [["a.npz", "b.npz"], ["../checkpoint.npz"]]:
                with self.assertRaises(ValueError):
                    submission.predict(self.sample(context={"artifact": {"checkpoint_paths": paths}}), directory)

    def test_prediction_cache_observes_new_weights_at_the_same_path(self):
        selected = self.root / "latest.bin"
        shutil.copyfile(self.checkpoint, selected)
        with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(selected)}):
            before = submission.predict(self.sample(), self.root)
            model = Model.load(selected)
            for value in model.params.values():
                value[:] = 0
            model.params["b2"][5] = 12
            replacement = self.root / "replacement.npz"
            model.save(replacement)
            os.replace(replacement, selected)
            stamp = selected.stat()
            os.utime(selected, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
            after = submission.predict(self.sample(), self.root)
        self.assertNotEqual(before, after)
        self.assertEqual(max(after, key=lambda p: p["score"])["label"], "Toluene")

    def test_optimizer_input_state_is_loaded_exactly_from_parent(self):
        import gas_a.sensor_training.optimize as optimize
        from gas_a.continuation import continue_candidate
        original = optimize.fit_weighted
        expected = Model.load(self.checkpoint)
        self.assertTrue(any(np.any(value != 0) for value in expected.first.values()))
        observed = []
        def fit(model, *args, **kwargs):
            for group in ("params", "first", "second"):
                for name, value in getattr(expected, group).items():
                    self.assertTrue(np.array_equal(value, getattr(model, group)[name]))
            self.assertEqual(model.meta["steps"], expected.meta["steps"])
            observed.append(True)
            return original(model, *args, **kwargs)
        train, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl")
        config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
        spec = dict(next(s for s in config["experiments"] if s["name"] == "conservative_head"), epochs=1)
        with patch.object(optimize, "fit_weighted", side_effect=fit):
            continue_candidate(self.checkpoint, train, validation, self.root / "continued", spec,
                               {"weakest_class": "Acetaldehyde"}, config["limits"],
                               expected_parent_sha256=digest(self.checkpoint))
        self.assertEqual(observed, [True])

    def test_default_entry_describes_instead_of_loading_archived_replay(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "run.py")], env=self.env(),
                                cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["tasks"], ["classification", "regression"])
        self.assertFalse((self.root / "runtime").exists())

    def test_local_cli_baseline_then_resume_uses_explicit_new_checkpoint(self):
        baseline = self.root / "official-fixture"
        baseline.mkdir()
        shutil.copyfile(self.checkpoint, baseline / "checkpoint.npz")
        shutil.copytree(self.root / "data", baseline / "public")
        (baseline / "baseline-info.json").write_text(json.dumps({"dataset_key": "A", "task": "classification",
            "checkpoint_sha256": digest(baseline / "checkpoint.npz")}), encoding="utf-8")
        (baseline / "training.json").write_text('{"fixture_history":true}\n', encoding="utf-8")
        first, second = self.root / "round-one", self.root / "round-two"
        first_command = [sys.executable, "-B", str(ROOT / "run.py"), "train-local", "--baseline-dir", str(baseline),
                         "--output", str(first), "--epochs", "1", "--experiment", "conservative_head"]
        result = subprocess.run(first_command, cwd=self.root, env=self.env(), capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((first / "parent_evidence/training.json").read_bytes(), (baseline / "training.json").read_bytes())
        result = subprocess.run([sys.executable, "-B", str(ROOT / "rebuild_models.py"),
            "--resume", str(first / "checkpoint.npz"), "--train", str(baseline / "public/train.jsonl"),
            "--validation", str(baseline / "public/validation.jsonl"), "--output", str(second),
            "--epochs", "1", "--experiment", "conservative_head"], cwd=self.root, env=self.env(),
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((second / "continuation.json").read_text(encoding="utf-8"))
        local_run = json.loads((second / "local_run.json").read_text(encoding="utf-8"))
        self.assertEqual(local_run["selection_status"], "candidate_only")
        self.assertFalse(local_run["platform_scored"])
        self.assertEqual(record["parent"]["checkpoint_sha256"], digest(first / "checkpoint.npz"))
        self.assertGreater(record["child"]["state"]["steps"], record["parent"]["state"]["steps"])
        requests = self.root / "new-input.jsonl"
        requests.write_text(json.dumps({"task": "classification", "inputs": {"features": [0.0] * 128}}) + "\n",
                            encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "run.py"), "predict-local",
            "--checkpoint", str(second / "checkpoint.npz"), "--input", str(requests)],
            cwd=self.root, env=self.env(), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([p["label"] for p in json.loads(result.stdout)["outputs"]], CLASSES)


if __name__ == "__main__":
    unittest.main()
