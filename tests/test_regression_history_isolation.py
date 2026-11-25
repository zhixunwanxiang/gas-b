"""坏历史只失去资格；有效历史仍可回滚，当前父模型和未知 RPC 结果不能被吞掉。"""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import contest_submission
from contest_harness.errors import HarnessError
from contest_harness.round import Round
from gas_a.regression import submission
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.evaluation import evaluate
from gas_a.sensor_training.io import digest
from tests.test_regression import make_fixture, RegressionHost
from tests.test_submission import writes_only_in


class RegressionHistoryIsolationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="history-isolation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def corrupt(self, parent, directory, kind):
        directory.mkdir()
        target = directory / "checkpoint.npz"
        if kind == "bad_zip":
            target.write_bytes(b"not an NPZ archive")
        else:
            with zipfile.ZipFile(parent) as source, zipfile.ZipFile(target, "w") as out:
                for name in source.namelist():
                    body = source.read(name)
                    if (kind == "raw_metadata" and name == "metadata.npy"
                            or kind == "raw_tensor" and name == "w1.npy"):
                        body = b"not an NPY array"
                    elif kind not in ("raw_metadata", "raw_tensor") and name == "metadata.npy":
                        buffer = io.BytesIO()
                        np.lib.format.write_array(buffer, np.asarray(kind), allow_pickle=False)
                        body = buffer.getvalue()
                    out.writestr(name, body)
        return target

    def run_host(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        self.rnd = rnd
        work = host.root / "work"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertFalse(host.pending)  # 原 SDK 已实际消费 accepted round_ack。
        return json.loads((host.saved_state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])

    def test_bad_archives_and_nonobject_metadata_are_skipped_for_both_datasets(self):
        for dataset in "BC":
            for kind in ("bad_zip", "[]", "null", '"text"', "1", "raw_metadata", "raw_tensor"):
                with self.subTest(dataset=dataset, kind=kind):
                    root = self.root / (dataset + "-" + str(len(list(self.root.iterdir()))))
                    parent = make_fixture(root, dataset)
                    self.corrupt(parent, root / "old", kind)
                    before_sha = digest(parent)
                    host = RegressionHost(root, parent, dataset)
                    host.action = "observe"
                    host.history = [{"id": "past", "selected_artifact_id": "parent-1"}]
                    host.add_artifact("bad-old", root / "old")
                    record = self.run_host(host)
                    old = record["historical_validation"]["bad-old"]
                    self.assertFalse(old["gate"]["eligible"])
                    self.assertEqual(old["error_type"], "ValueError")
                    self.assertEqual(old["stage"], "historical_model_evaluation")
                    self.assertEqual(host.record["selected_artifact_id"], "parent-1")
                    self.assertEqual(host.record["history_ids"], ["past"])
                    self.assertEqual(host.record["model_call_ids"], host.calls)
                    self.assertEqual(len(host.calls), 2)
                    self.assertIsNone(host.candidate)
                    self.assertEqual(digest(parent), before_sha)
                    self.assertTrue(any(f["type"] == "artifact_read_request" for f in host.frames))

    def test_unreadable_history_does_not_hide_the_next_valid_rollback_target(self):
        for dataset in "BC":
            root = self.root / dataset
            parent = make_fixture(root, dataset)
            good = root / "good"
            good.mkdir()
            shutil.copyfile(parent, good / "checkpoint.npz")
            fault = load_model(parent, dataset)
            fault.params["b2"] += 100
            fault_path = root / "previous/fault.npz"
            fault.save(fault_path)
            self.corrupt(good / "checkpoint.npz", root / "bad", "bad_zip")
            host = RegressionHost(root, fault_path, dataset, history=[{"id": "past", "selected_artifact_id": "parent-1"}])
            host.action = "rollback"
            host.add_artifact("good", good)
            host.add_artifact("bad", root / "bad")  # 恢复顺序先坏、后好，覆盖错误隔离后的继续遍历。
            record = self.run_host(host)
            self.assertFalse(record["historical_validation"]["bad"]["gate"]["eligible"])
            self.assertTrue(record["historical_validation"]["good"]["gate"]["eligible"])
            self.assertEqual(host.record["selected_artifact_id"], "good")
            self.assertEqual(record["plan"]["action"], "rollback")
            self.assertIsNone(host.candidate)

    def test_invalid_actual_parent_still_stops_before_model_or_state_submission(self):
        for dataset in "BC":
            for kind in ("bad_zip", "[]"):
                root = self.root / (dataset + "-" + str(len(list(self.root.iterdir()))))
                good = make_fixture(root, dataset)
                parent = self.corrupt(good, root / "damaged-parent", kind)
                host = RegressionHost(root, parent, dataset)
                host.add_artifact("good", good.parent)
                with self.assertRaises(ValueError):
                    self.run_host(host)
                self.assertIsNone(host.record)
                self.assertEqual(host.calls, [])
                self.assertFalse(any(f["type"] in ("candidate_request", "state_request", "round_complete") for f in host.frames))

    def test_unknown_outcome_during_actual_restore_rpc_is_not_recovered(self):
        for dataset in "BC":
            root = self.root / dataset
            parent = make_fixture(root, dataset)
            old = root / "old"
            old.mkdir()
            shutil.copyfile(parent, old / "checkpoint.npz")
            host = RegressionHost(root, parent, dataset)
            host.add_artifact("old", old)
            original = host.write
            def interrupt(text):
                if json.loads(text)["type"] == "artifact_read_request":
                    raise HarnessError("RPC_OUTCOME_UNKNOWN")
                return original(text)
            host.write = interrupt
            with self.assertRaises(HarnessError) as error:
                self.run_host(host)
            self.assertEqual(error.exception.code, "RPC_OUTCOME_UNKNOWN")
            self.assertTrue(self.rnd._outcome_unknown)
            count = len(host.frames)
            with self.assertRaises(HarnessError):
                self.rnd.history()
            self.assertEqual(len(host.frames), count)
            self.assertIsNone(host.record)
            self.assertEqual(host.calls, [])

    def test_unexpected_implementation_error_is_not_hidden_as_bad_history(self):
        root = self.root / "unexpected"
        parent = make_fixture(root, "B")
        host = RegressionHost(root, parent, "B")
        old = root / "old"
        old.mkdir()
        shutil.copyfile(parent, old / "checkpoint.npz")
        host.add_artifact("old", old)
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        _, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl", "B")
        before = evaluate(load_model(parent, "B"), validation, "B")
        work = root / "work"
        work.mkdir()
        with patch.object(submission, "load_model", side_effect=RuntimeError("unexpected implementation failure")), \
                writes_only_in(work), self.assertRaisesRegex(RuntimeError, "unexpected implementation"):
            submission._restore_history(rnd, [], work, validation, before, {"max_history_restores": 2}, "B")


if __name__ == "__main__":
    unittest.main()
