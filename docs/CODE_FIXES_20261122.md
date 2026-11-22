# 审查问题修复（2026-11-22）

修复对象为 [前次代码审核](POST_REAL_CODE_REVIEW_20261122.md) 的五项问题。保留原审核、真实 27B 回执和已测成绩；本轮只使用公开数据及离线替身，不新增真实模型调用或云端写入。配置中的 20 个百分点保护等数值没有调整。

| 问题 | 修改后的行为 | 主要代码与回归测试 |
|---|---|---|
| 宿主测量未形成硬门槛 | 候选需同时通过本地类别保护和宿主测量校验；分数退化、摘要不一致、口径未知或明显数值冲突均不可选 | `src/gas_a/host_validation.py`、`submission.py`；`tests/test_host_validation.py`、`test_failure_recovery.py` |
| 本地训练失败中断闭环 | 先记录实验身份；仅捕获本地训练的已知超时、数值或参数异常，将失败证据交给模型后保留父产物、保存状态并完成回合。相同失败计入重复预算 | `submission.py`、`evolution.py`；`tests/test_failure_recovery.py` |
| 验收工具完成即报成功 | 分离执行完成与要求通过；未达三次完整历史实验、低于基线或回滚未发生等情形报告未达标并非零退出 | `scripts/test_real_sdk.py`；`tests/test_sdk_acceptance.py` |
| home 训练轮数不生效 | 按 `routing.home.epochs` 实际循环输出头训练并检查轮数预算和截止时间；记录执行次数，保留隐藏层及其 Adam 矩、标准化状态 | `home_training.py`、`frozen_router.py`；`tests/test_home_epochs.py` |
| 原生审计放行代码目录凭证 | 子进程剥离已知服务凭证变量；原生审计显式拒绝已知凭证文件路径，允许读目录不能覆盖该拒绝 | `scripts/simulation_audit.py`、`simulate_competition.py`；`tests/test_simulation_audit.py` |

## 宿主摘要适配边界

当前显式支持的候选回执为：

```json
{
  "artifact_id": "实际返回的产物ID",
  "summary": {
    "scope": "authorized_public_validation",
    "validation_sha256": "sha256:实际公开验证文件摘要",
    "metrics": {"macro_f1": 0.8}
  }
}
```

父测量从 `rnd.frame.current_validation` 读取同样的摘要对象，可直接是 `scope/validation_sha256/metrics`，也可包在 `summary` 中。`scope` 支持 `authorized_public_validation`、`public_validation`、本地替身的 `offline_public_validation`。分数必须为非布尔有限数且在 `[0,1]`；验证集摘要必须一致；本地与宿主 Macro-F1 允许 `1e-6` 的普通舍入差异。宿主候选收益仍须达到配置门槛且严格大于零。

宿主完全不提供父测量（空对象或空值）时，使用已计算的同一公开验证父分数，并记录 `source=local_parent_host_measurement_absent`。已提供但无法验证或与本地冲突的父摘要不能被本地评估覆盖；此时新候选及历史回滚均不进入允许列表。候选宿主测量缺失或未知时没有本地分数放行分支。

SDK 1.0.4 没有冻结上述 `summary` 的内部字段；这些是本项目明示支持的适配格式，不是声称官方必定提供的字段。正式宿主若使用其他格式，应依据实际合同修改 `host_validation.py` 并增加对应测试。在此之前保持父模型，避免猜测字段或把不同范围的分数混比。

## 失败与记录

`experiment_attempts.jsonl` 保存 `started`、`trained` 或 `failed` 事件及父/训练/验证摘要、实验配置和计划调用号。失败类型与原因也进入最终 `configuration.experiment_failure`、选版请求及 `decisions.jsonl`；策略状态保存失败次数。失败的部分候选不会提交，也不计作一次成功提交的完整实验。

恢复范围只包围 `continue_candidate()` 本地计算。`rnd.model()`、恢复、候选上传、状态上传和结束 RPC 不在捕获范围内；SDK 的 `RPC_OUTCOME_UNKNOWN` 等协议异常继续终止会话，不重试或发送后续 RPC。强制杀进程、宿主资源终止和本地 I/O 故障仍可能让回合无法完成。

当前提示词版本为 `gas-a-controller-zh-v3`。执行策略指纹新增宿主准入版本，避免把修复前的拒绝历史机械套用到新执行逻辑；仅修改提示词措辞仍不会清空失败预算。

## 验收及原生审计

`test_real_sdk.py`：退出码 `0` 表示本地要求通过，`1` 表示执行完成但要求未满足，`2` 表示运行期失败。报告逐项写入 `requirements`、`requirements_passed` 和 `status`。正常路径要求至少三次非空历史驱动的实际候选实验；回滚演练采用独立的历史回滚、下一轮父模型及公开基线检查。工具不会强改模型决定来凑实验次数。

原生审计拒绝 `.env`、`.env.*`（含模板文件）、`.netrc`、`_netrc`、`.aws/`、`.ssh/`、`.swanlab/`，同时检查原路径与解析后的路径。凭证测试只读取临时合成哨兵，未读取真实密钥。它仍是 Python 应用审计，不能替代操作系统隔离，也不能保证发现任意文件名的凭证。Docker 镜像继续通过构建白名单排除宿主凭证。

## 验证

项目全量测试及 SDK 三个模块的日志位于 [本轮验证目录](../runtime/code-fixes-20261122/)。其中四轮 observe-only 测试实际经过 SDK 和 Harness 子进程，验证最终返回 `1`、完整实验数为 `0`；合成模型原始回复在临时目录结束时清理，不发布为真实 27B 证据。新增历史反馈测试还核对：宿主否决必须明确标为 `host_measurement_rejected` 并传递原始原因，本地失败传递实际异常证据，不会因本地门槛通过就误称可上线。

最终验证结果：

| 检查 | 结果 |
|---|---|
| 全量项目 / SDK 离线测试 | 105 / 18 项全部通过 |
| 两组独立十轮模拟 | 权重、逐条预测、选择与分数一致；每组 9 次非空历史实验、5 次正常晋升 |
| 同一公开验证集的静态基线 / 最终 Macro-F1 | `0.714029722 → 0.800045300`，提升约 8.60 个百分点 |
| 独立故障回滚 | 选回历史产物，下一轮使用恢复的父模型 |
| 宿主冲突故障探针 | 注入候选宿主 F1=0 后保留父模型，回合收到结束确认 |
| 独立核算 | 重算 34 组、42,296 条保存预测；36 个模拟源码文件与工作区一致 |

完整机器核验见 [verification.json](../runtime/code-fixes-20261122/verification.json)。复核命令为 `python runtime/code-fixes-20261122/verify_results.py`，只重算保存的公开预测，不调用模型。`git diff --check` 通过。

最终代码的十轮公开数据模拟、独立重复和故障回滚结果保存在同目录的 `simulation-final/`；`simulation/` 是补充历史反馈前的首轮冒烟验证，保留原结果。该结果使用离线决策替身，不能替代正式宿主、真实 27B 或隐藏集评测。此前真实 27B 的 `0.714030 → 0.798702` 属于前次保存源码，本轮修复没有重新进行付费模型测试。
