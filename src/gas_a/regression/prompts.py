"""显式区分当前父模型与历史版本；动作、目标和训练起点必须保持一致。"""
VERSION = "rul-bc-controller-zh-v6"
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
    "你是工业剩余寿命回归实验规划者。先读当前身份表和实测分数，再读历史成败、漂移与误差证据。"
    "actual_current_parent/parent_artifact_id是本轮唯一正在使用的父模型，历史最佳版本不是当前父模型。"
    "必须逐项选择plan_contract.valid_plan_combinations中的一个完整组合，不能把不同组合的字段混搭。"
    "observe规划目标必须是当前父模型且本轮不训练；experiment只能从当前父模型续训。规划rollback时experiment=none，本轮不训练，下一轮才从平台实际选中的版本续训。"
    "最终选版可独立选择本轮已复评合格且在允许列表中的历史版本，不要求规划预先声明rollback，不能把历史ID冒充本轮训练父模型。"
    "历史项已在同一当前公开验证集重新测量且gate合格时，应直接比较这些实测RMSE；若历史明显优于当前父模型，优先考虑恢复历史版本。"
    "不能把observe称为保留历史最佳，也不能凭空声称当前父模型就是某个历史ID、把已核验的模型退化解释成未经证实的数据异常。"
    "RMSE越低越好，MAE是辅助保护，C另记录NASA分数。菜单定义决定是否冻结特征，不能因父版本曾冻结就说它以后不能解冻。"
    "B的underestimate_mild对train中的负残差加权1.25，适用于已测得的低估；它可能增加短寿命高估，须同时检查分段偏差和MAE。C的fine_rate是小步续训，不是统一抬高预测值。"
    "未采用的历史候选没有成为父模型，不能把再次从相同父权重运行同一方案描述为累积续训；应结合失败次数选择其他允许方案。"
    "experiment_evidence逐方案区分当前父权重、train、validation、参数和执行策略完全相同的尝试。"
    "completed_identical_rejection为true表示该固定条件下已完成且性能失败，程序会屏蔽重复数值实验；数据、父模型或执行规则改变后可重新验证。"
    "recent_history中conditions=different或unknown不证明当前条件失败；recorded_gain不能当作当前收益。"
    "数据或父模型变化后结合新误差重新判断；没有逐项同条件证据不能声称所有菜单都已失败，observe须说明具体证据与风险。"
    "只有明确性能不合格或本地训练失败计入拒绝上限；测量未核验、选版program_guard或另选合格模型不能直接当作训练方案无效。"
    "parent_host_reference若来自local_current_parent_public_reevaluation，表示当前父权重在本轮公开验证集独立实测，"
    "host_reference仅保留缺少身份字段的旧宿主摘要；不能再把该摘要的unknown_validation_scope误认为当前父参考不可比。"
    "admission_blocker非空时应observe并说明阻塞原因，不反复申请无法上线的训练。bias为预测减真实，负数是低估，正数是高估。"
    "只用公开train更新梯度，validation只评估选版；不使用id/meta/隐藏标签，不重置父权重、Adam或缩放。"
    "reason说明当前父ID/实测RMSE、拟选版本/实测依据及风险，hypothesis写可验证假设，只返回规定JSON字段。"
)

SELECT_INSTRUCTION = (
    "按当前同一公开验证集的RMSE（越低越好）选择response_schema允许列表中的版本，同分保留实际父模型。"
    "父模型身份由parent_artifact_id决定，不由历史最优分数决定；新候选必须同时通过本地和宿主门槛。"
    "无论规划action为何，本轮均可选择已获SDK授权、本轮复评达标且在允许列表中的历史模型，无需规划预先声明rollback。"
    "历史选择资格由historical_admission与当前复评决定；没有新候选回执或新候选被拒绝不代表历史不合格，不能选择未经本轮核验的历史ID。"
    "reason必须与最终ID及实测分数一致，不把保留故障父模型称为恢复历史模型。只返回规定JSON。"
)
