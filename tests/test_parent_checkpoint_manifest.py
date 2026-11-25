"""复现平台整组父摘要：真实 SDK 清单校验、多文件续训、损坏必须拒绝。"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.test_submission import ROOT, OfflineHost, simulated_checkpoint_identity
from tests.test_regression import RegressionHost, make_fixture
from contest_harness.round import Round
from contest_harness.errors import HarnessError
from gas_a import submission
from gas_a.frozen_router import load_predictor
from gas_a.host_validation import bind_parent_validation, parent_reference
from gas_a.parent_checkpoint import primary_checkpoint, verified_checkpoints
from gas_a.sensor_training.data import load_pair, evaluate
from gas_a.sensor_training.io import digest
from gas_a.regression import submission as regression_submission


class ParentCheckpointManifestTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="parent-manifest-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def a_parent(self):
        shutil.copytree(ROOT / "data/public", self.root / "data")
        (self.root / "initial").mkdir()
        parent = self.root / "initial/checkpoint.npz"
        shutil.copyfile(ROOT / "models/reference.npz", parent)
        return parent

    def run_host(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        with patch.dict(os.environ, {"CONTEST_WORK": str(self.root / f"work-{rnd.round_index}")}):
            (submission.train if rnd.dataset_key == "A" else regression_submission.train)(rnd)
        self.assertTrue(rnd._finished)
        self.assertEqual(host.frames[-1]["type"], "round_complete")
        for frame in host.frames:
            if frame["type"] == "candidate_request":
                self.assertEqual(frame["parent_checkpoint_sha256"], host.frame["parent_checkpoint_sha256"])
        return host.saved_state

    def test_a_four_checkpoints_next_round_uses_aggregate_and_continues_every_component(self):
        parent = self.a_parent()
        _, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl")
        first = OfflineHost(self.root, parent, experiment="frozen_router")
        state = self.run_host(first)
        self.assertEqual(first.record["selected_artifact_id"], "candidate-1")
        second = OfflineHost(self.root, first.candidate / "checkpoint.npz", 2,
                             history=[{"id": "history-1", **first.record}], state=state, experiment="frozen_router")
        receipt = second.add_artifact("candidate-1", first.candidate)
        # 顺序也反转：主文件依据包标记确定，不依赖第一个文件恰好正确。
        receipt["artifact"]["checkpoint_paths"].reverse()
        self.assertEqual(len(receipt["artifact"]["checkpoint_paths"]), 4)
        second.frame["parent_artifact"] = receipt
        aggregate = simulated_checkpoint_identity(receipt)
        second.frame["parent_checkpoint_sha256"] = aggregate
        before = load_predictor(second.parent)
        second.frame["current_validation"] = {"metrics": evaluate(before, validation),
                                               "parent_checkpoint_sha256": aggregate}
        for item in receipt["files"]:
            self.assertNotEqual(item["sha256"], aggregate)
        self.run_host(second)
        self.assertIsNotNone(second.candidate)
        cfg = second.record["configuration"]
        self.assertNotIn("experiment_failure", cfg)
        self.assertTrue(cfg["host_parent_reference"]["comparable"])
        self.assertEqual(cfg["parent_checkpoint_sha256"], aggregate)
        self.assertEqual(cfg["parent_primary_file_sha256"], digest(second.parent))
        after = load_predictor(second.candidate / "checkpoint.npz")
        for role, original in before.models.items():
            with self.subTest(role=role):
                self.assertGreater(after.models[role].meta["steps"], original.meta["steps"])
                self.assertGreater(after.models[role].meta["epochs"], original.meta["epochs"])
        continuation = json.loads((second.candidate / "continuation.json").read_text(encoding="utf-8"))
        self.assertEqual(continuation["parent"]["checkpoint_sha256"], digest(second.parent))

    def test_a_single_file_opaque_identity_and_host_identity_conflict(self):
        parent = self.a_parent()
        host = OfflineHost(self.root, parent)
        aggregate = simulated_checkpoint_identity(host.frame["parent_artifact"])
        host.frame["parent_checkpoint_sha256"] = aggregate
        train, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl")
        score = evaluate(load_predictor(parent, train["paths"]), validation)
        host.frame["current_validation"] = {"macro_f1": score["macro_f1"], "checkpoint_sha256": aggregate}
        self.run_host(host)
        self.assertIsNotNone(host.candidate)
        self.assertNotIn("experiment_failure", host.record["configuration"])
        peer = OfflineHost(self.root, parent, 2)
        rnd = Round.start(peer, peer)
        # 使用独立 Round 验证显式错误的宿主父身份仍不能被忽略。
        rnd.parent.root = parent.parent
        rnd.parent.checkpoint_sha256 = aggregate
        raw = {"macro_f1": score["macro_f1"], "checkpoint_sha256": digest(parent)}
        rnd.frame["current_validation"] = raw
        binding = bind_parent_validation(rnd, validation["sha256"])
        self.assertFalse(parent_reference(raw, score, validation["sha256"], binding=binding)["comparable"])

    def test_bc_two_rounds_opaque_identity_reaches_training_and_preserves_protocol(self):
        for dataset in ("B", "C"):
            with self.subTest(dataset=dataset):
                root = self.root / dataset
                root.mkdir()
                parent = make_fixture(root, dataset)
                state, history = None, []
                for index in (1, 2):
                    host = RegressionHost(root, parent, dataset, index, history=history, state=state)
                    host.frame["parent_checkpoint_sha256"] = simulated_checkpoint_identity(host.frame["parent_artifact"])
                    rnd = Round.start(host, host)
                    rnd.parent.root = parent.parent
                    with patch.dict(os.environ, {"CONTEST_WORK": str(root / f"work-{index}")}):
                        regression_submission.train(rnd)
                    self.assertTrue(rnd._finished)
                    self.assertIsNotNone(host.candidate)
                    self.assertNotIn("experiment_failure", host.record["configuration"])
                    self.assertEqual(host.record["configuration"]["parent_primary_file_sha256"], digest(parent))
                    report = json.loads((host.candidate / "training.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["parent_checkpoint_sha256"], digest(parent))
                    self.assertTrue(report["checks"]["actual_continuation"])
                    sent = next(frame for frame in host.frames if frame["type"] == "candidate_request")
                    self.assertEqual(sent["parent_checkpoint_sha256"], host.frame["parent_checkpoint_sha256"])
                    parent, state = host.candidate / "checkpoint.npz", host.saved_state
                    history.append({"id": f"history-{index}", **host.record})

    def test_aggregate_does_not_allow_tampered_or_missing_nonprimary_files(self):
        parent = self.a_parent()
        auxiliary = parent.parent / "auxiliary.npz"
        auxiliary.write_bytes(parent.read_bytes())
        host = OfflineHost(self.root, parent)
        receipt = host.frame["parent_artifact"]
        receipt["artifact"]["checkpoint_paths"].append(auxiliary.name)
        receipt["files"].append({"path": auxiliary.name, "sha256": digest(auxiliary), "size_bytes": auxiliary.stat().st_size})
        host.frame["parent_checkpoint_sha256"] = simulated_checkpoint_identity(receipt)
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        auxiliary.write_bytes(b"tampered")
        with self.assertRaises(HarnessError) as caught:
            verified_checkpoints(rnd)
        self.assertEqual(caught.exception.code, "PARENT_DIGEST_MISMATCH")
        auxiliary.unlink()
        with self.assertRaises(HarnessError):
            verified_checkpoints(rnd)

    def test_ambiguous_plain_multifile_parent_is_rejected(self):
        parent = self.a_parent()
        extra = parent.parent / "another.npz"
        extra.write_bytes(parent.read_bytes())
        host = OfflineHost(self.root, parent)
        receipt = host.frame["parent_artifact"]
        receipt["artifact"]["checkpoint_paths"].append(extra.name)
        receipt["files"].append({"path": extra.name, "sha256": digest(extra), "size_bytes": extra.stat().st_size})
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        with self.assertRaisesRegex(ValueError, "exactly one"):
            primary_checkpoint(rnd)


if __name__ == "__main__":
    unittest.main()
