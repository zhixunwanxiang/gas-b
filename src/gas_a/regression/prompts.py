"""显式区分当前父模型与历史版本；动作、目标和训练起点必须保持一致。"""
VERSION = "rul-bc-controller-zh-v7-xiaoyang-port1"
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
                "observe": "本轮不训练，规划目标仍为actual_current_parent；最终可选本轮复评合格的历史版本。",
                "experiment": "只能从actual_current_parent续训；不能直接以历史版本为训练起点。",
                "rollback": "恢复已在本轮复评达标的历史版本；experiment必须为none，本轮不训练，下轮才从恢复版本续训。"}}


PLAN_INSTRUCTION = (
    "历史最佳版本不是当前父模型；只能选择plan_contract.valid_plan_combinations中的完整组合。"
    "根据validation_before、寿命段误差、工况、历史拒绝原因与算力预算，选择plan_contract允许的完整组合。"
    "observe与experiment指向当前父；rollback仅用本轮复评合格的历史，下轮才从所选模型训练。"
    "同条件下被明确拒绝的方案应尝试不同机制；不可比或另选合格产物不能作为性能失败。"
    "B若误差随振动特征或寿命段显著变化，可优先试bearing_tree_mix的原尺度和对数树融合；检查每根轴承相对父与起点的误差。"
    "C若多工况偏差明显，可试engine_cumulative_tree的有界残差；若正向修正使高估P99恶化，可试engine_tail_tree保留向下修正、减半向上修正。"
    "根据测量给假设和风险，不保证收益。bias为预测减真实。admission_blocker存在时观察。"
    "仅用公开train拟合，禁止使用隐藏答案、id或meta作特征。保留父权重、Adam和缩放。只返回规定JSON。"
)

SELECT_INSTRUCTION = (
    "合格历史版本无需规划预先声明rollback也可最终选择，但不能选择未经本轮核验的历史ID。"
    "从response_schema的允许ID中，比较同一公开验证上的当前父、候选和重新评估的历史，选择合格且RMSE较低者；同分保留父。"
    "候选须通过本地与宿主门槛。历史资格独立，见historical_admission。参考MAE、分组与尾部门槛，在reason中写实际差值与选择理由。只返回规定JSON。"
)
