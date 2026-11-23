"""No Docker server or model API is called by these tests."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.docker_rehearsal import DockerRuntime, child_environment
from scripts.test_real_sdk import RealModelHost, finalize_acceptance


class DockerRehearsalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="docker-rehearsal-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runtime = DockerRuntime("mutable-tag", executable="docker-client", context="test-context",
                                     expected_id="sha256:fixed")

    def test_service_credentials_and_parent_environment_are_not_inherited(self):
        with patch.dict(os.environ, {"PYTRIO_API_KEY": "synthetic-test-secret", "SWANLAB_API_KEY": "synthetic",
                                    "OPENAI_API_KEY": "synthetic", "CONTEST_DATA": "private", "NORMAL_SETTING": "ok"}):
            env = child_environment()
        self.assertNotIn("PYTRIO_API_KEY", env)
        self.assertNotIn("SWANLAB_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CONTEST_DATA", env)
        self.assertEqual(env["NORMAL_SETTING"], "ok")

    def test_wrong_image_stops_before_any_container_or_api_call(self):
        info = [{"Id": "sha256:different", "Os": "linux", "Architecture": "amd64"}]
        with patch("scripts.docker_rehearsal.subprocess.run", return_value=subprocess.CompletedProcess([], 0,
                   stdout=json.dumps(info))) as execute, self.assertRaises(ValueError):
            self.runtime.preflight()
        self.assertEqual(execute.call_count, 1)
        self.assertIsNone(self.runtime.image)

    def test_preflight_probes_use_sanitized_environment(self):
        info = [{"Id": "sha256:fixed", "Os": "linux", "Architecture": "amd64", "Config": {"User": "65534:65534"}}]
        replies = [info, {"verified": True}, {"sdk_version": "1.0.4", "tasks": ["classification", "regression"]}]
        with patch.dict(os.environ, {"PYTRIO_API_KEY": "synthetic-test-secret"}), \
                patch("scripts.docker_rehearsal.subprocess.run", side_effect=[
                    subprocess.CompletedProcess([], 0, stdout=json.dumps(r)) for r in replies]) as execute:
            result = self.runtime.preflight()
        self.assertEqual(result["image_id"], "sha256:fixed")
        for call in execute.call_args_list:
            self.assertNotIn("PYTRIO_API_KEY", call.kwargs["env"])

    def test_train_and_predict_use_immutable_image_and_only_declared_mounts(self):
        self.runtime.image, self.runtime.probe = "sha256:fixed", {"verified": True}
        for name in ("parent", "data", "state", "predict-log"):
            (self.root / name).mkdir()
        command, name = self.runtime.train(self.root / "parent", self.root / "data", self.root / "state", self.root / "train.jsonl")
        self.assertTrue(name.startswith("gas-real-train-"))
        self.assertEqual(command[-2:], ["sha256:fixed", "train"])
        prediction = self.runtime.predict(self.root / "parent", self.root / "parent/nested/checkpoint.npz", self.root / "predict-log")
        self.assertIn("CONTEST_ARTIFACT_PATH=/model/nested/checkpoint.npz", prediction)
        self.assertEqual(set(self.runtime.launches[0]["mounts"]), {"/previous", "/data", "/state"})
        self.assertEqual(set(self.runtime.launches[1]["mounts"]), {"/model"})
        self.assertNotIn("mutable-tag", command)
        self.assertTrue(self.runtime.audit()["container_policy_verified"])

    def test_multitask_preflight_rejects_old_classification_only_image(self):
        self.runtime.required_tasks = {"classification", "regression"}
        replies = [[{"Id": "sha256:fixed", "Os": "linux", "Architecture": "amd64", "Config": {"User": "65534:65534"}}],
                   {"verified": True}, {"sdk_version": "1.0.4", "tasks": ["classification"]}]
        with patch("scripts.docker_rehearsal.subprocess.run", side_effect=[
                subprocess.CompletedProcess([], 0, stdout=json.dumps(r)) for r in replies]), self.assertRaises(ValueError):
            self.runtime.preflight()

    def test_cleanup_refuses_containers_not_owned_by_the_test(self):
        with patch("scripts.docker_rehearsal.subprocess.run") as execute, self.assertRaises(ValueError):
            self.runtime.cleanup("someone-elses-container")
        execute.assert_not_called()

    def test_real_host_maps_container_paths_without_a_model_call(self):
        parent = self.root / "checkpoint.npz"
        parent.write_bytes(b"identity-only-fixture")
        state = self.root / "state"
        state.mkdir()
        host = RealModelHost(self.root, parent, state=state, measurements=object(), gateway=object(), runtime=self.runtime)
        self.assertEqual(host.frame["state_path"], "/state")
        projection = host.frame["task_contract"]["public_projection"]
        self.assertEqual(projection["validation_path"], "/data/validation.jsonl")
        self.assertEqual(host.calls, [])

    def test_docker_acceptance_requires_enforced_policy_and_actual_train_and_predict(self):
        report = {"rounds": [{"finish_acknowledged": True, "model_call_ids": ["call"]}] * 4,
                  "baseline_macro_f1": .7, "final_macro_f1": .8,
                  "complete_history_experiment_decision_records": 3,
                  "audit": {"mode": "docker_enforced_mounts_and_network", "container_policy_verified": True,
                            "train_containers": 4, "prediction_containers": 5}}
        self.assertEqual(finalize_acceptance(report), 0)
        report["audit"]["container_policy_verified"] = False
        self.assertEqual(finalize_acceptance(report), 1)


if __name__ == "__main__":
    unittest.main()
