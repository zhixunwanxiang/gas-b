"""显式区分当前父模型与历史版本；动作、目标和训练起点必须保持一致。"""
VERSION = "rul-bc-controller-zh-v2"
# 提示词版本也进入实验身份，防止已修复的动作说明与旧失败实验被当作完全相同的上下文。


def plan_contract(parent_id, before, restored, rollback_ids, experiments):
    # 只整理已测量证据，不替换模型回复、不自动执行建议，也不放宽任何准入门槛。
    combinations = [{"action": "observe", "experiment": "none", "target_artifact_id": parent_id}]
    # 把允许的组合显式展开，减少模型把“历史最优”误认作“实际父版本”的歧义。
    # 排名只展示本轮已恢复、复评且合格的历史，不从历史记录里直接摘一个旧分数当排名。
    combinations += [{"action": "experiment", "experiment": name, "target_artifact_id": parent_id} for name in experiments]
    combinations += [{"action": "rollback", "experiment": "none", "target_artifact_id": key} for key in rollback_ids]
    ranking = sorted(({"artifact_id": key, "rmse": restored[key]["metrics"]["rmse"],
                       "mae": restored[key]["metrics"]["mae"], "current_round_gate_passed": True}
                      for key in rollback_ids), key=lambda row: row["rmse"])
    return {"actual_current_parent": {"artifact_id": parent_id, "metrics": before,
                                     "identity_source": "this_start_frame_parent_artifact"},
            "current_round_eligible_history": ranking,
            "valid_plan_combinations": combinations,
            "evidence_based_preference": "rollback" if ranking and ranking[0]["rmse"] < before["rmse"] else "observe_or_experiment",
            "action_meanings": {
                "observe": "只保留actual_current_parent，不训练，也不会恢复任何历史版本。",
                "experiment": "只能从actual_current_parent续训；不能直接以历史版本为训练起点。",
                "rollback": "恢复已在本轮复评达标的历史版本；experiment必须为none，本轮不训练，下轮才从恢复版本续训。"}}


PLAN_INSTRUCTION = (
    "你是工业剩余寿命回归实验规划者。先读当前身份表和实测分数，再读历史成败、漂移与误差证据。"
    "actual_current_parent/parent_artifact_id是本轮唯一正在使用的父模型，历史最佳版本不是当前父模型。"
    "必须逐项选择plan_contract.valid_plan_combinations中的一个完整组合，不能把不同组合的字段混搭。"
    "observe只能保留当前父模型；experiment只能从当前父模型续训。要使用其他历史模型，必须先rollback且experiment=none，下一轮才能继续训练。"
    "历史项已在同一当前公开验证集重新测量且gate合格时，应直接比较这些实测RMSE；若历史明显优于当前父模型，优先考虑恢复历史版本。"
    "不能把observe称为保留历史最佳，也不能凭空声称当前父模型就是某个历史ID、把已核验的模型退化解释成未经证实的数据异常。"
    "RMSE越低越好，MAE是辅助保护，C另记录NASA分数。菜单定义决定是否冻结特征，不能因父版本曾冻结就说它以后不能解冻。"
    "只用公开train更新梯度，validation只评估选版；不使用id/meta/隐藏标签，不重置父权重、Adam或缩放。"
    "reason说明当前父ID/实测RMSE、拟选版本/实测依据及风险，hypothesis写可验证假设，只返回规定JSON字段。"
)

SELECT_INSTRUCTION = (
    "按当前同一公开验证集的RMSE（越低越好）选择response_schema允许列表中的版本，同分保留实际父模型。"
    "父模型身份由parent_artifact_id决定，不由历史最优分数决定；新候选必须同时通过本地和宿主门槛。"
    "只有计划action=rollback时，本轮才可选择已计划并复评达标的历史目标；不能在选版时临时改成另一个未获准的历史ID。"
    "rollback没有新候选回执是正常情况，不代表历史版本不合格；应看本轮historical复评及gate。"
    "reason必须与最终ID及实测分数一致，不把保留故障父模型称为恢复历史模型。只返回规定JSON。"
)
