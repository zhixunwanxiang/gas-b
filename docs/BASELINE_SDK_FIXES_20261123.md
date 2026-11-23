# 官方基线续训与 SDK 1.0.4 审查修复

本次针对三份官方 `TRAINING_GUIDE.md` 和 SDK 1.0.4 审查发现的问题修复代码。未新增真实模型调用或云端实验，下面的通过结果都是本地离线回归；此前固定 v6 镜像的公开数据成绩继续作为原始历史证据保留。

## 1. B/C 允许授权公开训练投影跨轮更新

原问题：首轮保存监测参考后，下一轮把参考的 `train_sha256` 与当轮 train 文件做相等比较；即使只新增一条合法训练样本，也会在模型调用前抛异常。

当前处理：

- 首轮仍只用公开 train 拟合监测参考，后续固定保留它，避免参考随漂移移动而掩盖输入变化。
- 校验原参考的数据集、特征路径、来源摘要格式、样本数及分箱统计。数据集或状态损坏仍明确拒绝，不静默重新初始化。
- 当前 train 摘要可以变化，监测报告分别记录 `reference_train_sha256`、`current_train_sha256`、`training_projection_changed` 和 `reference_policy`。
- 诊断寿命分段沿用首轮 train 的分位点，以保持跨轮口径；它们与模型的目标 scaler 是不同对象。
- 父权重、Adam、seed、输入和目标 scaler 继续从实际父检查点继承，绝不因投影更新重拟合。
- 云端结果归档白名单增加上述摘要与标志，不上传样本。

实现：[monitoring.py](../src/gas_a/regression/monitoring.py)、[回归主控](../src/gas_a/regression/submission.py)。

新增 SDK 回归案例分别对 B/C 执行首轮、增加一条授权训练样本、读取首轮历史与状态并执行第二轮，核对完成确认、累计步骤增加、scaler/seed 未变、参考未重拟合及两份摘要准确记录。

## 2. 精简候选回执按 SDK 当前事务绑定

SDK 说明将 `submit_candidate()` 的返回摘要定义为公开验证测量，但没有强制其必须包含冗余的 `scope`、`validation_sha256` 字段。原实现无条件要求这两个键，因此可拒绝确有改善的候选。

当前处理：

1. 只在真实 `rnd.submit_candidate()` 返回后，为该 `candidate_response` 创建不可变的当前候选绑定。
2. 校验 SDK 协议、数据集与任务、实际返回的产物 ID，并重新计算当前授权 validation 文件摘要，确认提交期间文件未改变。
3. 支持 `summary.metrics.<指标>`、扁平 `summary.<指标>`，以及明确指定 `metric` 的 `value`/`score`。不推测没有指标名的通用 `score`。
4. 宿主未提供范围/摘要键时，仅当前 SDK 候选绑定可补充调用来源和本地文件身份。规范化记录明确区分 `scope_source`、`validation_sha256_source`；原始回执不修改，也不把本地摘要说成宿主声明。
5. 回执显式提供隐藏范围、空/冲突摘要、错误产物 ID 或数据集时仍拒绝。宿主不给有效数字、不提升、与本地复核不一致、重复位置指标互相冲突，也仍拒绝。
6. 指标数值始终取宿主实际返回值；本地测量只用于复核和原有安全门槛。

父模型 `current_validation` 不属于当前候选事务，不能借用候选绑定。宿主没有提供父测量时沿用原有本地同范围父评估；已提供但来源不明或冲突的父测量仍保守拒绝。未知宿主摘要结构同样拒绝，需依据正式合同再增加显式字段适配。

实现：[共享回执适配](../src/gas_a/host_validation.py)、[B/C 评估](../src/gas_a/regression/evaluation.py)。A/B/C 主控均调用同一绑定逻辑；执行策略记录 `sdk-bound-public-validation-v2`，防止旧拒绝指纹阻止修复后的合理实验。

新增测试覆盖 A/B/C 的真实 SDK JSONL 候选交互：删去候选摘要的冗余身份字段并改成具名扁平分数后，合格候选可被采用，原始回执仍不含补造字段。另检查未绑定来源、显式冲突、错误产物、验证文件改变、无指标名分数、非有限数及不一致宿主分数都不能绕过门槛。

## 3. B/C 补齐指南约定的 training.json

每个成功完成的 B/C 候选现在同时输出：

| 文件 | 内容 |
|---|---|
| `checkpoint.npz` | 选中 epoch 的完整权重、Adam、scaler 和累计状态 |
| `history.jsonl` | 每个实际执行 epoch 的损失、公开验证指标和准入结果 |
| `training.json` | 指南约定路径的真实训练报告 |
| `continuation.json` | 原有父子状态及续训证据，继续保留 |

两份 JSON 报告由同一对象写出，数值一致；包含实际父摘要、导出检查点摘要、train/validation 摘要、参数、累计步骤、选中/执行轮数、前后指标及继承检查。输出仍使用全新目录，不覆盖父产物和原始基线训练记录。

实现：[training.py](../src/gas_a/regression/training.py)。原有 B/C 四轮 SDK 测试增加了两份报告相等、父摘要和导出检查点摘要正确的断言。

## 验证结果

- 项目完整离线测试：**159 项通过**，其中本次新增 8 项回归测试。
- SDK 官方 `test_sdk`、`test_contract_regressions`、`test_transport_regressions`：**18 项通过**。
- Windows SDK 测试显式设置 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`，保证其启动的子进程也使用 UTF-8。
- `git diff --check` 通过；本次没有修改官方 SDK、官方参考模型或原始公开基线数据。

日志：`runtime/baseline-sdk-fix-all-tests-20261123.log`、`runtime/baseline-sdk-fix-official-tests-20261123.log`。完整命令见 [README](../README.md)。

旧 v6 镜像不包含这些功能修复，已另行重建 v7，见[v7 交付说明](IMAGE_V7_20261123.md)。该镜像已核对当前应用源码和离线修复用例；正式宿主及隐藏集仍需另行验证，不能把 v6 的 36 次真实调用成绩或导出体积自动归给 v7。
