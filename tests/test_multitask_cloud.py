"""真实多任务宿主桥接的回归测试；使用本地假回执，不联网、不计费。"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from tests.test_regression import make_fixture
from tests.test_submission import OfflineHost
from scripts.test_multitask_cloud import RealPeerFactory, MODEL
from scripts.cloud_experiment import CloudJournal


class MultitaskCloudTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="real-multitask-bridge-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_real_bridge_never_enters_offline_model_branch(self):
        for dataset in ("A", "B", "C"):
            root = self.root / dataset
            parent = make_fixture(root, "C" if dataset == "A" else dataset)
            response = {"model": MODEL, "choices": [{"message": {"content": "raw provider text"}}]}
            gateway = SimpleNamespace(invoke=Mock(return_value=("actual-provider-id", response, [])))
            journal = SimpleNamespace(before_call=Mock(), after_call=Mock())
            with patch("scripts.test_multitask_cloud.RealGateway", return_value=gateway):
                host = RealPeerFactory({"model": MODEL}, "synthetic-not-a-key", journal, 16)(root, parent, dataset, 1, [], None)
            frame = {"type": "model_request", "id": 0, "messages": [{"role": "user", "content": "{}"}], "max_tokens": 512}
            with patch.object(OfflineHost, "write", side_effect=AssertionError("Offline branch was entered")):
                host.write(json.dumps(frame))
            self.assertIs(host.pending[-1]["response"], response)
            self.assertEqual(host.pending[-1]["call_id"], "actual-provider-id")
            self.assertEqual(host.calls, ["actual-provider-id"])
            self.assertEqual(host.frame["model_name"], MODEL)
            gateway.invoke.assert_called_once_with(frame, host.frame)

    def test_gateway_failure_does_not_fall_back_or_retry(self):
        parent = make_fixture(self.root, "B")
        gateway = SimpleNamespace(invoke=Mock(side_effect=TimeoutError("unknown outcome")))
        journal = SimpleNamespace(before_call=Mock(), after_call=Mock())
        with patch("scripts.test_multitask_cloud.RealGateway", return_value=gateway):
            host = RealPeerFactory({"model": MODEL}, "synthetic", journal, 16)(self.root, parent, "B", 1, [], None)
        frame = {"type": "model_request", "id": 0, "messages": [{"role": "user", "content": "{}"}]}
        with self.assertRaises(TimeoutError), patch.object(OfflineHost, "write", side_effect=AssertionError("fallback")):
            host.write(json.dumps(frame))
        gateway.invoke.assert_called_once()
        journal.after_call.assert_not_called()
        self.assertEqual(host.calls, [])

    def test_regression_cloud_round_uses_rmse_and_auxiliary_metrics(self):
        state = self.root / "state"
        state.mkdir()
        cfg = {"selected_local_rmse": 50., "validation_before": {"rmse": 53.}, "strategy_version_after": 2}
        (state / "decisions.jsonl").write_text(json.dumps({"configuration": cfg})+'\n', encoding="utf-8")
        (state / "monitoring.json").write_text('{"drift":true}', encoding="utf-8")
        (state / "diagnosis.json").write_text('{}', encoding="utf-8")
        host = SimpleNamespace(saved_state=state, candidate=None, candidate_metrics=None,
            frame={"round_index": 1, "task_contract": {"task": "regression"}},
            record={"history_ids": [], "model_call_ids": ["actual"], "configuration": cfg},
            selected_metrics={"rmse": 50., "mae": 38., "nasa_score": 123456789.123456})
        sdk = SimpleNamespace(Text=lambda x: x)
        with patch.dict(os.environ, {}, clear=True):
            journal = CloudJournal(sdk, {}, self.root, {"dataset_key": "C"}, "unit")
        journal.run = SimpleNamespace(log=Mock())
        journal.after_round(host)
        self.assertEqual(list(journal.scalars_for("normal/selected_rmse").values()), [50.])
        self.assertEqual(list(journal.scalars_for("normal/selected_nasa_score").values()), [123456789.123456])
        self.assertFalse(journal.scalars_for("normal/selected_macro_f1"))


if __name__ == "__main__":
    unittest.main()
