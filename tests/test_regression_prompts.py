"""父模型身份、历史回滚与动作组合的提示合同回归测试。"""
import unittest
import contest_submission
from gas_a.regression.prompts import plan_contract, PLAN_INSTRUCTION, SELECT_INSTRUCTION


class RegressionPromptTest(unittest.TestCase):
    def test_history_best_is_not_declared_as_actual_parent(self):
        history = {"history-best": {"metrics": {"rmse": 50., "mae": 38.}},
                   "history-other": {"metrics": {"rmse": 60., "mae": 42.}}}
        contract = plan_contract("actual-fault", {"rmse": 800., "mae": 790.}, history,
                                 ["history-other", "history-best"], {"low_rate": {}})
        self.assertEqual(contract["actual_current_parent"]["artifact_id"], "actual-fault")
        self.assertEqual(contract["current_round_eligible_history"][0]["artifact_id"], "history-best")
        self.assertEqual(contract["evidence_based_preference"], "rollback")
        for row in contract["valid_plan_combinations"]:
            if row["action"] in ("observe", "experiment"):
                self.assertEqual(row["target_artifact_id"], "actual-fault")
            else:
                self.assertEqual(row["experiment"], "none")
                self.assertIn(row["target_artifact_id"], history)

    def test_unknown_or_unqualified_history_does_not_create_rollback_choices(self):
        contract = plan_contract("parent", {"rmse": 50., "mae": 30.}, {}, [], {"head_only": {}})
        self.assertEqual(contract["evidence_based_preference"], "observe_or_experiment")
        self.assertEqual({row["target_artifact_id"] for row in contract["valid_plan_combinations"]}, {"parent"})
        self.assertNotIn("rollback", {row["action"] for row in contract["valid_plan_combinations"]})

    def test_prompt_explains_current_identity_and_plan_selection_consistency(self):
        self.assertIn("历史最佳版本不是当前父模型", PLAN_INSTRUCTION)
        self.assertIn("valid_plan_combinations", PLAN_INSTRUCTION)
        self.assertIn("计划action=rollback", SELECT_INSTRUCTION)
        self.assertIn("不能在选版时临时改成", SELECT_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
