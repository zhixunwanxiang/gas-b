"""Routed artifacts use validation labels during selection and features during inference."""
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
import numpy as np
from contest_harness.round import Round
from contest_harness.errors import HarnessError
from gas_a.continuation import continue_candidate, state_identity
from gas_a.frozen_router import FrozenRouter, calibrate_policy, load_predictor, checkpoint_paths
from gas_a.sensor_training.data import evaluate, load_pair
from gas_a.sensor_training.io import digest
from gas_a.evolution import promotion_gate
from tests.test_submission import OfflineHost, writes_only_in


class FrozenRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="gas-frozen-tests-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
        cls.spec = next(s for s in cls.config["experiments"] if s["name"] == "frozen_router")
        cls.train, cls.validation = load_pair(ROOT / "data/public/train.jsonl", ROOT / "data/public/validation.jsonl")
        cls.bundle = cls.root / "routed"
        cls.report = continue_candidate(ROOT / "models/reference.npz", cls.train, cls.validation, cls.bundle,
                                        cls.spec, {"weakest_class": "Acetaldehyde"}, cls.config["limits"])

    def test_exported_policy_matches_public_validation_and_preserves_expert_weights(self):
        model = load_predictor(self.bundle / "checkpoint.npz")
        self.assertIsInstance(model, FrozenRouter)
        metric = evaluate(model, self.validation)
        self.assertAlmostEqual(metric["macro_f1"], self.report["validation_after"]["macro_f1"])
        calibration = model.manifest["calibration"]
        self.assertEqual(calibration["scope"], "authorized_public_validation")
        self.assertEqual(calibration["validation_sha256"], self.validation["sha256"])
        self.assertGreaterEqual(metric["macro_f1"] + 1e-12, calibration["four_sensor_metrics"]["macro_f1"])
        state = {k: digest(p) for k, p in model.component_paths.items()}
        parameter_states = {k: state_identity(m) for k,m in model.models.items()}
        changed = dict(self.validation, y=np.roll(self.validation["y"], 251))
        policy_before = deepcopy(model.policy)
        calibrate_policy(model.models, changed, model.policy["cutoffs"], self.spec["routing"]["selection"])
        self.assertEqual(policy_before, model.policy)
        self.assertEqual(state, {k: digest(p) for k,p in model.component_paths.items()})
        self.assertEqual(parameter_states, {k: state_identity(m) for k,m in model.models.items()})
        memory = model.models["home_router"]
        z = memory.normalized(self.train["x"])[:, memory.feature_mask]
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        for row, target in zip(memory.memory_x, memory.memory_y):
            self.assertTrue(np.any(np.all(np.isclose(z, row, rtol=0, atol=1e-14), axis=1)
                                   & (self.train["y"] == target)))

    def test_feature_only_stream_equals_batch_even_with_causal_correction(self):
        loaded = load_predictor(self.bundle / "checkpoint.npz")
        manifest = deepcopy(loaded.manifest)
        manifest["policy"]["feature_correction"] = True
        model = FrozenRouter(loaded.models, manifest, loaded.component_paths, loaded.manifest_path)
        x = self.validation["x"][:200]
        expected = model.predict(x)
        frozen_policy = deepcopy(model.policy)
        self.assertEqual(model._stream["drift"][2], 0)
        with patch("gas_a.router.FourSensorRouter.release_labels", side_effect=AssertionError("No label callback allowed")), \
                patch("gas_a.signals.reveal_labels", side_effect=AssertionError("No label reads allowed")), writes_only_in():
            streamed = np.vstack([model.predict_one(row[None, :]) for row in x])
        np.testing.assert_allclose(streamed, expected, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(streamed.argmax(1), expected.argmax(1))
        self.assertEqual(frozen_policy, model.policy)
        self.assertEqual(model._stream["drift"][2], len(x))
        self.assertFalse(hasattr(model, "waiting"))
        self.assertFalse(hasattr(model, "release_labels"))

    def test_sdk_predict_flushes_routed_rows_without_labels_or_data_files(self):
        env = {k: v for k,v in os.environ.items() if not k.startswith("CONTEST_")}
        env.update(CONTEST_ARTIFACT_PATH=str(self.bundle / "checkpoint.npz"), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8", GAS_TEST_CODE=str(ROOT))
        bootstrap = """
import os, runpy, sys
from pathlib import Path
root = Path(os.environ['GAS_TEST_CODE'])
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(root / 'data') or path.is_relative_to(root / 'models'):
            raise PermissionError('Prediction cannot read local datasets or packaged weights')
        if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError('Prediction cannot write files')
sys.addaudithook(audit)
sys.path.insert(0, str(root))
sys.argv = [str(root / 'harness.py'), 'predict']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        p = subprocess.Popen([sys.executable, "-B", "-c", bootstrap], cwd=self.root, env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8")
        self.addCleanup(lambda: p.kill() if p.poll() is None else None)
        received = queue.Queue()
        def read():
            for line in p.stdout:
                received.put(line)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        x = self.validation["x"][:10]
        expected = load_predictor(self.bundle / "checkpoint.npz").predict(x)
        for i, row in enumerate(x):
            p.stdin.write(json.dumps({"task": "classification", "inputs": {"features": row.tolist()}}) + "\n")
            p.stdin.flush()
            output = json.loads(received.get(timeout=10))["outputs"]
            self.assertEqual(len(output), 6)
            np.testing.assert_allclose([v["score"] for v in output], expected[i], atol=1e-10, rtol=0)
            self.assertIsNone(p.poll())
        p.stdin.close()
        self.assertEqual(p.wait(timeout=10), 0, p.stderr.read())
        reader.join(timeout=2)
        p.stdout.close()
        p.stderr.close()

    def test_missing_or_changed_bundle_files_fail_without_plain_model_fallback(self):
        folder = self.root / "broken"
        shutil.copytree(self.bundle, folder)
        manifest = folder / "routing.json"
        original = manifest.read_bytes()
        manifest.unlink()
        with self.assertRaises(FileNotFoundError):
            load_predictor(folder / "checkpoint.npz")
        manifest.write_bytes(original)
        five = folder / "routing/five_sensor.npz"
        five.write_bytes(five.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "component digest mismatch"):
            load_predictor(folder / "checkpoint.npz")

    def test_two_sdk_rounds_continue_all_declared_experts_and_record_decisions(self):
        root = self.root / "sdk"
        root.mkdir()
        (root / "previous").mkdir()
        shutil.copytree(ROOT / "data/public", root / "data")
        parent = root / "previous/checkpoint.npz"
        shutil.copyfile(ROOT / "models/reference.npz", parent)
        history, state = [], None
        parent_receipt = None
        first_candidate = None
        for index in (1, 2):
            host = OfflineHost(root, parent, index, history, state, experiment="frozen_router")
            if parent_receipt is not None:
                host.frame["parent_artifact"] = parent_receipt
            rnd = Round.start(host, host)
            rnd.parent.root = parent.parent
            work = root / f"workspace-{index}"
            # This test covers accepted expert continuity. Use an explicitly
            # relaxed class floor here; the default rejection gate is tested separately.
            relaxed_gate = lambda before, after, cfg: promotion_gate(before, after, {**cfg, "maximum_class_recall_drop": 1.0})
            with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                    patch("gas_a.submission.promotion_gate", side_effect=relaxed_gate):
                contest_submission.train(rnd)
            self.assertTrue(rnd._finished)
            descriptor = next(f["artifact"] for f in host.frames if f["type"] == "candidate_request")
            self.assertEqual(len(descriptor["checkpoint_paths"]), 4)
            self.assertIn("routing.json", descriptor["code_paths"])
            loaded = load_predictor(host.candidate / "checkpoint.npz")
            self.assertIsInstance(loaded, FrozenRouter)
            if index == 2:
                info = json.loads((host.candidate / "routing_training.json").read_text(encoding="utf-8"))
                self.assertEqual(info["five_sensor"]["parent_checkpoint_sha256"], digest(first_candidate / "routing/five_sensor.npz"))
                self.assertEqual(info["home"]["parent_checkpoint_sha256"], digest(first_candidate / "routing/home_trunk.npz"))
                self.assertEqual(info["home"]["router_parent_checkpoint_sha256"], digest(first_candidate / "routing/home_router.npz"))
            state = next(work.glob("round-*/state"))
            history.append({"id": f"route-history-{index}", **host.record})
            if index == 1:
                self.assertEqual(host.record["selected_artifact_id"], "candidate-1")
                first_candidate = host.candidate
                parent = host.candidate / "checkpoint.npz"
                parent_receipt = {"artifact_id": "candidate-1", "artifact": descriptor,
                    "files": [{"path": p.relative_to(host.candidate).as_posix(), "sha256": digest(p), "size_bytes": p.stat().st_size}
                              for p in host.candidate.rglob("*") if p.is_file()]}

    def test_parent_router_cannot_use_undeclared_auxiliary_files(self):
        root = self.root / "undeclared"
        root.mkdir()
        shutil.copytree(ROOT / "data/public", root / "data")
        host = OfflineHost(root, self.bundle / "checkpoint.npz", experiment="frozen_router")
        rnd = Round.start(host, host)
        rnd.parent.root = self.bundle
        with patch.dict(os.environ, {"CONTEST_WORK": str(root / "workspace")}):
            with self.assertRaises(HarnessError) as caught:
                contest_submission.train(rnd)
        self.assertEqual(caught.exception.code, "PARENT_FILE_NOT_DECLARED")
        self.assertIsNone(host.candidate)

    def test_historical_rollback_restores_complete_frozen_router_via_sdk(self):
        root = self.root / "router-recovery"
        root.mkdir()
        shutil.copytree(ROOT / "data/public", root / "data")
        (root / "previous").mkdir()
        parent = root / "previous/checkpoint.npz"
        from gas_a.sensor_training.reference.model import Model
        from gas_a.sensor_training.io import CLASSES
        damaged = Model.load(ROOT / "models/reference.npz")
        damaged.params["w2"][:] = 0
        damaged.params["b2"][:] = 0
        damaged.save(parent)
        # A synthetic fault drill makes recovery deterministic; its score is
        # explicitly unrelated to official validation/hidden performance.
        predicted = load_predictor(self.bundle / "checkpoint.npz").predict(self.validation["x"]).argmax(1)
        rows = [json.loads(line) for line in (root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()]
        for row, target in zip(rows, predicted):
            row["answer"] = CLASSES[target]
        (root / "data/validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        host = OfflineHost(root, parent, history=[{"id": "deployed-router", "selected_artifact_id": "router-good"}])
        host.action = "rollback"
        host.add_artifact("router-good", self.bundle)
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        work = root / "workspace"
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store):
            contest_submission.train(rnd)
        self.assertEqual(host.record["selected_artifact_id"], "router-good")
        restored = next(work.glob("round-*/restored-1"))
        for path in checkpoint_paths(self.bundle / "checkpoint.npz") + ["routing.json"]:
            self.assertEqual(digest(restored / path), digest(self.bundle / path))
        self.assertIsInstance(load_predictor(restored / "checkpoint.npz"), FrozenRouter)
        self.assertIsNone(host.candidate)

    def test_a_plain_continuation_of_a_routed_parent_has_no_stale_bundle_marker(self):
        spec = dict(next(s for s in self.config["experiments"] if s["name"] == "conservative_head"), epochs=1)
        report = continue_candidate(self.bundle / "checkpoint.npz", self.train, self.validation,
                                    self.root / "plain-child", spec, {"weakest_class": "Acetaldehyde"}, self.config["limits"])
        model = load_predictor(self.root / "plain-child/checkpoint.npz")
        self.assertNotIsInstance(model, FrozenRouter)
        self.assertNotIn("prediction_bundle", model.meta)
        self.assertAlmostEqual(report["validation_before"]["macro_f1"], self.report["validation_after"]["macro_f1"])


if __name__ == "__main__":
    unittest.main()
