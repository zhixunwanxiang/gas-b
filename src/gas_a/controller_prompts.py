"""Versioned, evidence-bound prompts for the platform's 27B controller.

The host attaches the official prompt. These are participant instructions only.
"""
# 27B 决策证据组装：PLAN 用于实验规划，SELECT 用于已有测量后的产物选择。
# 官方任务提示由宿主注入，这里仅维护选手侧提示和结构化证据，不能重复附加 rnd.prompt。
# 下面的提示词字符串会实际发送给模型；本次中文说明只使用注释，不修改提示内容。
import hashlib
from .host_validation import HOST_ADMISSION_VERSION
from .sensor_training.io import json_text
from .evolution import EXPERIMENT_POLICY_VERSION, REPEAT_POLICY
from .sensor_training.reference.data import finite_number

VERSION = "gas-a-controller-zh-v9"
EPOCH_POLICY_VERSION = "retained-complete-bundle-per-epoch-v2"
ROUTING_POLICY_VERSION = "protected-routing-retained-auxiliaries-v3"

PLAN = """你是任务 A 的受控工业模型实验负责人。目标是在同一份授权公开验证集上，找到能真正通过上线门槛的 Macro-F1 改进，而非只生成更高但不可上线的候选。
先比较当前逐类错误、混淆矩阵、传感器分布变化和 history_feedback，再在 response_schema 允许值内选择一个动作。admission 明确给出了最低 F1、受保护类别召回底线和未受保护的小样本类别；底线是约束，不是应主动牺牲到的目标。
experiment_evidence 按每个菜单项给出当前父模型、train、validation、方案和执行策略完全相同的尝试与拒绝次数。recent_history 的 conditions=different 或 unknown 不证明当前条件已失败，recorded_gain 只属于旧条件。数据摘要变化后应依据新错误重新判断；没有逐项同条件证据时，不得声称所有方案均已失败或模型达到上限。observe 应引用具体方案、当前证据和风险。
completed_identical_rejection=true 表示固定条件下已完成的确定性训练已被性能门槛拒绝，本条件不再重复；换数据或父模型后重新判断。超时、测量核验和控制器错误不等于已完成的性能失败。
实验选择：优先检验有数据依据、尚未在相同父摘要+数据摘要+执行策略下验证的方案。根据 experiment_guide 判断该方案实际能改变什么，比较其潜在收益与类别代价。弱类增强与固定类别增强可能相乘，不能把增强弱类等同于整体收益。漂移大小不是特征重要性，更不证明传感器物理故障。
历史学习：区分无增益、类别召回超限、合格但未采用和不同数据/父模型的结果。同条件已失败且执行策略未改变时，不机械重复；执行策略已修复时允许明确说明变化后重新检验。不要把不同验证集或不同父模型的绝对分数直接作为本轮收益。
宿主核验：host_parent_reference 是本轮实际父模型的核验结果。历史 parent_measurement_unverified 属于测量核验拒绝，不能直接当作训练方案无效；当前父测量已核验且执行策略改变时，可重新检验有本地收益的允许方案。frozen_router 会在有界融合权重菜单中优先选择通过相同类别保护的路由。
有尚未验证且有可检验假设的允许实验时，不因某个旧方案失败而过早 observe。只有菜单已无新的合理检验、没有新证据或预算不允许时观察。不要为凑实验次数重复无效训练。只有本轮重新评估通过门槛的历史 ID 才可提出 rollback。未采用不等于训练无效，策略只对明确性能失败或本地训练失败计入拒绝上限。规划动作决定本轮是否训练；最终选版仍可选择本轮复评合格且在允许列表的历史模型，无需规划预先声明 rollback。
hypothesis 用简短文字说明：哪类错误、拟检验机制、预期改善的指标及可能代价。reason 引用实际指标和可用历史 id，解释为何选择它及失败时如何拒绝；预测收益必须标为假设，不能写成实测。只返回 response_schema 的五个字符串字段，不返回思考过程、Markdown、代码、额外超参数或额外字段。
训练只使用公开 train；公开 validation 可评估和选版。隐藏标签/遥测不可用，不能由 id、文件名或匿名标识推断。不可放宽门槛、重置父权重/Adam/scaler、伪造调用号或改写测量。"""

SELECT = """你是任务 A 的上线审核者。本轮评估已经完成，只根据本轮同范围测量和硬门槛选择产物，不再提出一个未经执行的新实验。
selected_artifact_id 必须来自 response_schema 的允许列表。先识别 action，再使用对应证据：
experiment：只有本轮新候选同时通过 candidate_gate 和 host_admission 才可上线；host_candidate_receipt、candidate_gate、host_admission 仅描述这个新候选。其公开范围、验证摘要或测量未知/冲突时不得用本地高分绕过限制。experiment_failure 表示本地训练失败，部分文件不是可选新模型。
历史选版：历史产物已通过 SDK 恢复并在本轮公开验证集重新运行。historical_admission 列出当前验证摘要、fresh_metrics、当前类别保护 gate 和最终允许资格；这是本轮复评，不能误称旧历史分数。无论规划是 experiment、observe 或 rollback，都可选择本轮复评合格且在允许列表的历史 ID，无需规划预先声明 rollback。新候选相关字段为 null 或新候选未通过宿主门槛，不表示独立复评的历史不合格，也不需要为历史制造新的 candidate_response。
observe 表示本轮未训练；最终仍可选择上述合格历史，若没有更好的合格版本则保留父产物。所有动作均不得放宽 admission 的类别召回和最小收益约束。比较完整预测器的同范围 Macro-F1，选择有实测支持的更优合格版本，同分保留父产物。不得用旧历史记录中的分数替换本轮复评结果，也不得将任何公开验证分数称为隐藏集成绩。
reason 简短列出：同口径父/候选/历史分数、关键类别召回或拒绝门槛、最终 ID 与选择理由。若看到主组件 epoch 指标和完整路由指标，必须区分范围；有可行 epoch 的诊断不等于该 epoch 已导出或已上线。
召回门槛相对当轮父模型，不保证多轮累计下降相对初始基线仍在同一范围；整体 F1 提高也不证明每个气体类别都改善。
只返回 response_schema 的两个字符串字段，不输出思考过程、Markdown 或额外字段。保留父模型是允许的真实结果，不能伪称已提升。"""


def execution_policy(controller, limits):
    # 把逐 epoch 选择、最终晋升和训练预算绑定为可序列化证据，供历史可比性和实验指纹使用。
    return {"epoch_policy_version": EPOCH_POLICY_VERSION, "repeat_policy": REPEAT_POLICY,
            "experiment_policy_version": EXPERIMENT_POLICY_VERSION,
            "routing_policy_version": ROUTING_POLICY_VERSION,
            "host_admission_policy_version": HOST_ADMISSION_VERSION,
            "promotion_gate": {key: controller[key] for key in (
                "minimum_macro_f1_gain", "minimum_class_support", "maximum_class_recall_drop")},
            "training_limits": dict(limits)}


def admission(before, controller):
    # 把程序门槛转换成模型可直接比较的最低 F1 和逐类召回底线，并明确小样本未保护类别。
    support = controller["minimum_class_support"]
    return {"scope": "same_authorized_public_validation",
            "parent_macro_f1": before["macro_f1"],
            "minimum_candidate_macro_f1": before["macro_f1"] + controller["minimum_macro_f1_gain"],
            "minimum_absolute_macro_f1_gain": controller["minimum_macro_f1_gain"],
            "strict_improvement_required": True,
            "maximum_class_recall_drop_percentage_points": 100 * controller["maximum_class_recall_drop"],
            "minimum_class_support": support,
            "protected_recall_floors": {name: max(0., item["recall"] - controller["maximum_class_recall_drop"])
                                        for name,item in before["per_class"].items() if item["support"] >= support},
            "unprotected_low_support_classes": [name for name,item in before["per_class"].items() if item["support"] < support]}


def history_feedback(history, parent_sha, train_sha, validation_sha, policy, menu):
    # 解释历史失败属于召回超限、宿主拒绝还是无提升，并分别标注父模型/数据/配置是否相同。
    rows = []
    for item in history:
        cfg = item.get("configuration", {})
        if not isinstance(cfg, dict):
            continue
        spec = cfg.get("experiment")
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            continue
        gate = cfg.get("candidate_gate", {})
        if not isinstance(gate, dict):
            gate = {}
        gain = gate.get("macro_f1_gain")
        if not finite_number(gain):
            before, after = cfg.get("local_validation_before"), cfg.get("local_validation_after")
            gain = after - before if finite_number(before) and finite_number(after) else None
        host_gate = cfg.get("host_candidate_gate")
        failure = ("local_training_failure" if cfg.get("experiment_failure") else
                   "class_recall_regression" if gate.get("class_recall_losses") else
                   "host_measurement_rejected" if isinstance(host_gate, dict) and not host_gate.get("eligible") else
                   "eligible" if gate.get("eligible") else "insufficient_gain_or_unrecorded_gate")
        rows.append({"history_id": item["id"], "experiment": spec["name"],
                     "same_parent": cfg.get("parent_checkpoint_sha256") == parent_sha,
                     "same_training_data": cfg.get("train_sha256") == train_sha,
                     "same_validation_data": cfg.get("validation_sha256") == validation_sha,
                     "same_execution_policy": cfg.get("execution_policy") == policy,
                     "same_experiment_configuration": (cfg.get("experiment_config_sha256") == request_digest(menu[spec["name"]])
                                                       if spec["name"] in menu else None),
                     "measured_macro_f1_gain": gain, "result": failure,
                     "host_admission": host_gate, "experiment_failure": cfg.get("experiment_failure"),
                     "class_recall_losses": gate.get("class_recall_losses", {}),
                     "epoch_selection": cfg.get("checkpoint_selection"),
                     "selected_artifact_id": item.get("selected_artifact_id")})
    return rows


def enrich_plan(request, controller, train, validation, policy):
    # 给规划请求补充实验实际能力、类别增强乘数、有限预算和同条件历史反馈，避免 AI 凭名称猜方案。
    weak = request["diagnosis"]["weak_classes"][0]["label"]
    guide = {}
    for name,spec in request["experiments"].items():
        if spec.get("pair_review"):
            guide[name] = {"actual_change": "fit_targeted_review_on_public_train_with_frozen_parent",
                           "review_targets": ["Ethanol/Ethylene", "Toluene disagreements"],
                           "admission": "both Accuracy and Macro-F1 must improve; existing recall and host checks apply",
                           "risk": "inspect Toluene support before generalizing any Macro-F1 gain"}
            continue
        kept = spec.get("kept_sensors")
        guide[name] = {"actual_change": "rebuild_frozen_expert_bundle" if "routing" in spec else
                      "output_head_only" if spec.get("freeze") == "features" else "neural_continuation_with_sensor_projection",
                      "kept_sensors": kept, "learning_rate": spec["learning_rate"], "epochs": spec["epochs"],
                      "routing_selection": spec.get("routing", {}).get("selection"),
                      "preserves_parent_auxiliaries": request.get("parent_predictor_scope") == "complete_frozen_bundle" and "routing" not in spec,
                      "weak_class": weak,
                      "weak_class_extra_multiplier_before_frequency_normalization": spec["weak_boost"] * spec.get("class_boosts", {}).get(weak, 1),
                      "risk": "sensor exclusion and class reweighting may trade recall between classes; head-only may provide little gain"}
    return {**request, "controller_prompt": {"version": VERSION, "instruction": PLAN},
            "admission": admission(request["validation_before"], controller),
            "data_scope": {"train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
                           "train_rows": len(train["x"]), "validation_rows": len(validation["x"]),
                           "hidden_labels_available": False},
            "execution_policy": policy, "experiment_guide": guide,
            "history_feedback": history_feedback(request["history"], request["parent_checkpoint_sha256"],
                                                 train["sha256"], validation["sha256"], policy, request["experiments"])}


def enrich_selection(request, controller, validation_sha, candidate_gate, checkpoint_selection):
    # 新候选准入与历史回滚准入分开提供。历史项必须标为本轮复评且验证摘要相同。
    # 没有新候选时 candidate_gate 可以为 None；不能把它误读为历史产物也不合格。
    allowed = request["response_schema"]["selected_artifact_id"]
    historical = {}
    for identifier, item in request["same_local_validation"].get("historical", {}).items():
        fresh = (item.get("reevaluated_this_round") is True
                 and item.get("validation_sha256") == validation_sha)
        historical[identifier] = {
            "eligible": bool(fresh and item["gate"].get("eligible") and identifier in allowed),
            "measurement_source": "restored_artifact_current_public_re_evaluation" if fresh else "unverified",
            "validation_sha256": item.get("validation_sha256"),
            "checkpoint_sha256": item.get("checkpoint_sha256"),
            "fresh_metrics": item.get("metrics") if fresh else None, "gate": item["gate"]}
    return {**request, "controller_prompt": {"version": VERSION, "instruction": SELECT},
            "admission": admission(request["same_local_validation"]["parent"], controller),
            "local_validation_sha256": validation_sha, "candidate_gate": candidate_gate,
            "new_candidate_admission_applicable": request.get("candidate_artifact_id") is not None,
            "host_admission_scope": "new_candidate_only", "historical_admission": historical,
            "checkpoint_selection": checkpoint_selection}


def request_digest(request):
    # 固定 JSON 键序和分隔符计算请求摘要，用于追踪真实发送的证据版本，不是模型调用回执。
    raw = json_text(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()
