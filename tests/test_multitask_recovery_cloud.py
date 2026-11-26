"""故障回滚验收器：真实SDK和数值模型，离线决策替身；不联网、不计费。"""
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from tests.test_submission import ROOT, OfflineHost
from tests.test_regression import make_fixture, RegressionHost
import contest_submission
from contest_harness.round import Round
from scripts import test_multitask_recovery_cloud as recovery
from scripts.validate_platform import receipt
from gas_a.sensor_training.io import digest, write_json
from gas_a.sensor_training.reference.model import Model
from gas_a.regression.data import load_model
from gas_a.regression.evaluation import evaluate
from gas_a.frozen_router import load_predictor
from gas_a.sensor_training.data import evaluate as classify


class RecoveryCloudTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="native-recovery-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def source(self, dataset):
        source = self.root / f"source-{dataset}"
        normal = source / "normal"
        if dataset == "A":
            (normal / "previous").mkdir(parents=True)
            shutil.copytree(ROOT / "data/public", normal / "data")
            checkpoint = normal / "previous/checkpoint.npz"
            shutil.copyfile(ROOT / "models/reference.npz", checkpoint)
        else:
            checkpoint = make_fixture(normal, dataset)
        (source / "official-baseline").mkdir()
        shutil.copyfile(checkpoint, source / "official-baseline/checkpoint.npz")
        (normal / "initial").mkdir()
        shutil.copyfile(checkpoint, normal / "initial/checkpoint.npz")
        host = normal / "host-1"
        (host / "candidate").mkdir(parents=True)
        (host / "state").mkdir()
        shutil.copyfile(checkpoint, host / "candidate/checkpoint.npz")
        descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                      "predict_argv": ["python3", "/app/harness.py", "predict"]}
        frames = [{"direction": "harness", "frame": {"type": "candidate_request", "artifact": descriptor}},
                  {"direction": "host", "frame": {"type": "candidate_response", "artifact_id": "candidate-1"}}]
        (host / "protocol.jsonl").write_text("".join(json.dumps(f)+'\n' for f in frames), encoding="utf-8")
        _, validation = (recovery.classification_pair(normal / "data/train.jsonl", normal / "data/validation.jsonl") if dataset == "A" else
                         recovery.regression_pair(normal / "data/train.jsonl", normal / "data/validation.jsonl", dataset))
        metric = classify(load_predictor(checkpoint), validation) if dataset == "A" else evaluate(load_model(checkpoint, dataset), validation, dataset)
        write_json(normal / "report.json", {"baseline": metric, "final": metric,
            "rounds": [{"round": 1, "record": {"selected_artifact_id": "candidate-1", "configuration": {}}}]})
        return source

    def test_fault_changes_only_output_params_and_preserves_originals(self):
        for dataset in ("A", "B", "C"):
            source = self.source(dataset)
            before = recovery.model_manifest(source)
            target = self.root / f"prepared-{dataset}"
            registry, history, state, _ = recovery.prepare(source, target, dataset)
            self.assertEqual(before, recovery.model_manifest(source))
            original = Model.load(source / "official-baseline/checkpoint.npz")
            fault = Model.load(target / "fault-parent/checkpoint.npz")
            self.assertEqual(original.meta, fault.meta)
            for group in ("first", "second"):
                for name in original.params:
                    np.testing.assert_array_equal(getattr(original, group)[name], getattr(fault, group)[name])
            np.testing.assert_array_equal(original.params["w1"], fault.params["w1"])
            np.testing.assert_array_equal(original.params["b1"], fault.params["b1"])
            np.testing.assert_array_equal(original.mean, fault.mean)
            np.testing.assert_array_equal(original.scale, fault.scale)
            self.assertNotEqual(digest(source / "official-baseline/checkpoint.npz"), digest(target / "fault-parent/checkpoint.npz"))
            for name in ("train.jsonl", "validation.jsonl"):
                self.assertEqual(digest(source / "normal/data" / name), digest(target / "data" / name))
            self.assertEqual(history[0]["id"], f"real-local-{dataset}-history-1")
            self.assertTrue(state.is_dir())
            self.assertEqual(set(registry), {"initial", "candidate-1", "fault-parent"})

    def run_recovery(self, dataset, retain=False):
        source = self.source(dataset)
        target = self.root / f"recovery-{dataset}"
        gateway = SimpleNamespace(verified=[], attempts=[])
        hosts = []
        class Factory:
            def __init__(self):
                self.gateway = gateway
            def __call__(self, root, parent, key, index, history, state):
                host = (OfflineHost(root, parent, index, history=history, state=state) if key == "A" else
                        RegressionHost(root, parent, key, index, history=history, state=state))
                host.action = "rollback" if not hosts else "observe"
                host.force_retain = retain
                hosts.append(host)
                return host
        runtime = SimpleNamespace(cli=["unit-docker"], train=Mock(return_value=([], "owned-test")))
        journal = SimpleNamespace(after_round=Mock())
        def measurement(runtime, directory, descriptor, path, validation, key, out):
            out.mkdir()
            checkpoint = directory / descriptor["checkpoint_paths"][0]
            return classify(load_predictor(checkpoint), validation) if key == "A" else evaluate(load_model(checkpoint, key), validation, key)
        def exchange(host, argv, env, out, name, **kwargs):
            host.frame["task_contract"]["public_projection"].update(
                train_path=str(host.root / "data/train.jsonl"), validation_path=str(host.root / "data/validation.jsonl"))
            call_state = target / "prior-state" if len(hosts) == 1 else hosts[-2].saved_state
            host.frame["state_path"] = str(call_state)
            rnd = Round.start(host, host)
            rnd.parent.root = host.parent.parent
            with tempfile.TemporaryDirectory(prefix="recovery-work-") as work, patch.dict(os.environ, {"CONTEST_WORK": work}):
                contest_submission.train(rnd)
            self.assertTrue(rnd._finished)
            gateway.verified.extend({"usage": {"total_tokens": 1}} for _ in host.calls)
        with patch.object(recovery, "measure", side_effect=measurement), patch.object(recovery, "exchange", side_effect=exchange):
            return recovery.recovery_phase(source, target, dataset, runtime, Factory(), journal)

    def test_actual_historical_selection_and_complete_next_parent_required(self):
        for dataset in ("A", "B", "C"):
            result = self.run_recovery(dataset)
            self.assertTrue(result["requirements_passed"])
            self.assertTrue(result["checks"]["actual_historical_rollback"])
            self.assertTrue(result["checks"]["next_round_uses_complete_restored_artifact"])
            self.assertEqual(result["real_model_calls"], 4)  # Synthetic receipts, never an actual provider call in these unit tests.

    def test_retaining_fault_parent_cannot_be_reported_as_recovery(self):
        result = self.run_recovery("B", retain=True)
        self.assertFalse(result["requirements_passed"])
        self.assertFalse(result["checks"]["actual_historical_rollback"])
        self.assertFalse(result["checks"]["next_round_uses_complete_restored_artifact"])
        self.assertFalse(result["checks"]["recovered_not_worse_than_public_baseline"])


if __name__ == "__main__":
    unittest.main()
