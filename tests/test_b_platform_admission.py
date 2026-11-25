"""B 平台精简父分数的兼容回归；使用真实 SDK、数值训练和离线模型回复。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from tests import test_regression as fixtures
from gas_a.regression.data import load_model, load_pair
from contest_harness.round import Round
from gas_a.regression.evaluation import evaluate, current_parent_reference
from gas_a.regression import submission as regression


# 使用辅助方法而不继承整套回归测试，避免 discover 重复执行已有测试。
class BPlatformAdmissionTest(unittest.TestCase):
    setUp = fixtures.RegressionTest.setUp
    run_host = fixtures.RegressionTest.run_host

    def parent_score(self, checkpoint, dataset="B"):
        _, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", dataset)
        return evaluate(load_model(checkpoint, dataset), validation, dataset)["rmse"]

    def test_compact_parent_and_candidate_enable_real_multiround_continuation(self):
        parent = fixtures.make_fixture(self.root, "B")
        history, state = [], None
        for index in range(1, 4):
            host = fixtures.RegressionHost(self.root, parent, "B", index, history=history, state=state)
            score = self.parent_score(parent)
            raw_parent = ({"metrics": {"rmse": round(score, 6)}} if index == 1 else
                          {"summary": {"metric": "rmse", "value": score}} if index == 2 else
                          {"rmse": score, "scope": "public_validation"})
            host.frame["current_validation"] = deepcopy(raw_parent)
            original = host.write

            def compact(text):
                result = original(text)
                if json.loads(text)["type"] == "candidate_request":
                    host.pending[-1]["summary"] = {"rmse": host.candidate_metrics["rmse"]}
                return result

            host.write = compact
            state = self.run_host(host)
            self.assertEqual(host.record["selected_artifact_id"], f"candidate-{index}")
            cfg = host.record["configuration"]
            self.assertTrue(cfg["candidate_gate"]["eligible"])
            self.assertTrue(cfg["host_candidate_gate"]["eligible"])
            self.assertEqual(cfg["parent_host_reference"]["source"], "local_current_parent_public_reevaluation")
            self.assertEqual(host.frame["current_validation"], raw_parent)
            self.assertEqual(host.record["history_ids"], [row["id"] for row in history])
            next_parent = host.candidate / "checkpoint.npz"
            self.assertGreater(load_model(next_parent, "B").meta["steps"], load_model(parent, "B").meta["steps"])
            history.append({"id": f"history-{index}", **host.record})
            parent = next_parent

    def test_conflicting_parent_disables_unusable_training_and_restoration(self):
        parent = fixtures.make_fixture(self.root, "B")
        score = self.parent_score(parent)
        cases = [
            {"rmse": score, "scope": "hidden"},
            {"rmse": score, "scope": None},
            {"rmse": score, "validation_sha256": "wrong"},
            {"rmse": score, "validation_sha256": None},
            {"rmse": score + 1},
            {"rmse": score, "artifact_id": "other"},
            {"rmse": score, "dataset_key": "C"},
            {"rmse": score, "checkpoint_sha256": "wrong"},
            {"score": score},
            {"metric": "mae", "value": score},
            {"rmse": True},
            {"summary": {"rmse": score}, "scope": "hidden"},
        ]
        for index, raw in enumerate(cases, 1):
            with self.subTest(raw=raw):
                host = fixtures.RegressionHost(self.root, parent, "B", index)
                host.frame["current_validation"] = deepcopy(raw)
                host.action = "observe"
                with patch.object(regression, "continue_candidate") as training, \
                        patch.object(regression, "_restore_history") as restore:
                    state = self.run_host(host)
                training.assert_not_called()
                restore.assert_not_called()
                self.assertIsNone(host.candidate)
                self.assertEqual(host.record["selected_artifact_id"], f"parent-{index}")
                self.assertTrue(host.record["configuration"]["admission_blocker"])
                request = json.loads(host.messages[0][0]["content"])
                self.assertEqual(request["response_schema"]["action"], ["observe"])
                self.assertEqual(request["experiments"], {})
                self.assertFalse(request["parent_host_reference"]["comparable"])
                self.assertEqual(host.frame["current_validation"], raw)
                strategy = json.loads((state / "regression_strategy.json").read_text(encoding="utf-8"))
                self.assertEqual(strategy["attempts"], [])

    def test_host_candidate_disagreement_still_rejects_local_improvement(self):
        parent = fixtures.make_fixture(self.root, "B")
        host = fixtures.RegressionHost(self.root, parent, "B")
        host.frame["current_validation"] = {"rmse": self.parent_score(parent)}
        original = host.write

        def disagree(text):
            result = original(text)
            if json.loads(text)["type"] == "candidate_request":
                host.pending[-1]["summary"] = {"rmse": host.candidate_metrics["rmse"] + .1}
            return result

        host.write = disagree
        self.run_host(host)
        cfg = host.record["configuration"]
        self.assertTrue(cfg["candidate_gate"]["eligible"])
        self.assertFalse(cfg["host_candidate_gate"]["eligible"])
        self.assertEqual(host.record["selected_artifact_id"], "parent-1")

    def test_parent_reference_rejects_undeclared_checkpoint_and_changed_projection(self):
        parent = fixtures.make_fixture(self.root, "B")
        _, validation = load_pair(self.root / "data/train.jsonl", self.root / "data/validation.jsonl", "B")
        before = evaluate(load_model(parent, "B"), validation, "B")
        host = fixtures.RegressionHost(self.root, parent, "B")
        host.frame["current_validation"] = {"rmse": before["rmse"]}
        rnd = Round.start(host, host)
        rnd.parent.root = parent.parent
        copied = self.root / "undeclared.npz"
        copied.write_bytes(parent.read_bytes())
        with self.assertRaises(ValueError):
            current_parent_reference(rnd, copied, before, validation["sha256"])
        with (self.root / "data/validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            current_parent_reference(rnd, parent, before, validation["sha256"])


if __name__ == "__main__":
    unittest.main()
