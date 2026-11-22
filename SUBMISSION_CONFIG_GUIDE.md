# submission_config.json 参数与实际代码逻辑

本文对应当前项目根目录的 `submission_config.json`。共核对 72 个参数路径；数组中的传感器号、权重候选等作为一个参数整体说明。重复的训练参数含义相同，取值和生效分支在后文列全。

这是项目的系统策略和实验菜单，不代表组委会规定了这些阈值。特别是召回下降 0.20、支持数 20、漂移阈值等，都是项目当前默认值。

## 1. 这份配置如何进入程序

| 配置区域 | 控制什么 | 实际入口 |
|---|---|---|
| `max_output_tokens` | 宿主大模型每次决策输出的 token 上限 | [submission.py:96](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:98) |
| `controller` | 回复纠错、历史恢复、实验重复限制、漂移判定和最终上线门槛 | [submission.py:195](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:218)、[evolution.py:8](D:/aiproject/gas-rsi-four-sensor/src/gas_a/evolution.py:8) |
| `limits` | 数值训练的轮数、耗时、最佳 epoch 导出和早停 | [optimize.py:322](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:322) |
| `experiments` | 大模型可选择的四个实验方案及各自训练参数 | [submission.py:214](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:237) |
| `experiments[0].routing` | 冻结路由方案的辅助专家训练和路由校准 | [frozen_router.py:195](D:/aiproject/gas-rsi-four-sensor/src/gas_a/frozen_router.py:195) |

正式 `train` 一轮的大致流程为：读取配置 → 核验并评估实际父产物 → 监测和诊断 → 恢复允许的历史产物并重新评估 → 模型选择观察、一个实验或回滚 → 如需实验则续训并提交 → 最终门槛检查及模型确认 → 保存状态、等待宿主确认。

`experiments` 是菜单，每轮不会固定执行全部四个方案。当前模型回复只能选择已有实验名和允许的动作，不能直接返回任意学习率或改写配置。调度多少个 SDK 回合由平台决定，配置中的 `epochs` 不表示 SDK 回合数。

## 2. max_output_tokens：模型决策输出

| 参数路径 | 当前值 | 含义与实际行为 |
|---|---:|---|
| `max_output_tokens` | `2048` | 每次 `rnd.model()` 的输出 token 预算。实际传入的是 `min(2048, rnd.max_output_tokens)`，因此不能突破宿主给的上限。它不控制训练样本数、输入维数或工业模型参数量。 |

代码：[submission.py:96](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:98)。普通回合有计划和选版两次模型调用，观察回合也有这两次调用。默认每个阶段最多增加一次回复纠错，所以正常返回但格式不合法的情况下，一轮最多四次决策调用；RPC 中断立即停止。

## 3. controller：控制器和最终上线门槛

| 完整参数路径 | 当前值 | 含义、边界和代码位置 |
|---|---:|---|
| `controller.json_corrections` | `1` | 每个决策阶段收到不合法 JSON 或不允许的选择后，最多补发一次纠错请求；首次调用不计在这一次里。`0` 表示不纠错。代码允许整数 `0～2`。只纠正已成功返回的回复，不重试结果未知的 RPC。[submission.py:96](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:98) |
| `controller.max_history_restores` | `2` | 每轮最多完整恢复并尝试评估两个宿主允许的历史产物，优先按近期历史选择记录排列，再补充其他允许产物。它不是保留历史记录的条数，也不保证找到了所有历史版本中的最佳模型。代码允许整数 `0～2`；`0` 关闭历史恢复。[submission.py:130](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:135) |
| `controller.max_repeat_rejections` | `2` | 相同父检查点摘要、训练集摘要、验证集摘要及完整实验配置，累计两次候选未被采用后，从当轮实验菜单移除该组合。改用其他父权重、其他数据或其他实验配置时是不同组合。按状态中最近 128 次实验记录计数；不是“总共只准训练两轮”，也不要求拒绝记录必须相邻。代码允许整数 `1～8`。[evolution.py:56](D:/aiproject/gas-rsi-four-sensor/src/gas_a/evolution.py:59) |
| `controller.minimum_macro_f1_gain` | `0.0005` | 最终候选或历史回滚版本相对当轮父模型，在同一份当前公开验证集上的 Macro-F1 至少提升 `0.0005`。例如 `0.7140 → 0.7145`，是 0.05 个百分点，不是相对提高 0.05%。代码允许 `0～1`；即使设为 `0`，仍要求提升严格大于 `1e-12`，同分不能晋升。[evolution.py:68](D:/aiproject/gas-rsi-four-sensor/src/gas_a/evolution.py:71) |
| `controller.minimum_class_support` | `20` | 只有父模型评估报告中真实样本数至少 20 的类别，才参加“类别召回下降保护”。这不是预测为该类别的样本数，也不限制是否计算该类别 F1。代码允许整数 `1～1000000`。[evolution.py:68](D:/aiproject/gas-rsi-four-sensor/src/gas_a/evolution.py:71) |
| `controller.maximum_class_recall_drop` | `0.20` | 对满足上述支持数的每个类别，计算 `父召回率 - 候选召回率`；超过 `0.20 + 1e-12` 就拒绝该候选。`0.20` 是 20 个百分点，例如 96% 降到 76%；不是“相对降低 20%”。一个受保护类别超限就足以拒绝。代码允许 `0～1`；`0` 表示基本不允许下降，`1` 数值上不再拦截合法召回范围内的降幅。[evolution.py:68](D:/aiproject/gas-rsi-four-sensor/src/gas_a/evolution.py:71) |

最终可晋升需同时满足：Macro-F1 严格改善、改善量达到配置值、所有受保护类别的召回降幅未超限。通过门槛仅获得被选择的资格，模型仍可保留父产物。未通过门槛的候选仍可能被上传用于记录，但不能被选择上线。

召回降幅的参考是**当轮实际父模型**，不是固定初始检查点。连续多轮分别通过 20 个百分点保护后，相对初始基线的累计下降仍可能超过 20 个百分点；当前没有额外施加初始基线的逐类固定底线。

当前公开验证集的甲苯只有 5 条，所以默认 `minimum_class_support=20` 下，甲苯不参加召回下降保护；它仍参与实际指标计算。降低该参数会让更多小样本类别进入保护范围。

**正式入口已统一逐 epoch 与最终门槛**：当前 `train` 将三个晋升参数派生为内部 `controller_checkpoint_gate`，在 `select_best_epoch=true` 时优先导出通过同一门槛的最佳 epoch；有达标 epoch 时，不会被更高 F1 但召回超限的 epoch 替代。若全部 epoch 不达标，仍导出最佳实测候选并标明不合格，最终决策不能选用。比较使用实际父预测器；路由父产物不会被偷换成较弱的主组件参考。JSON 没有新增或放宽阈值。

## 4. controller.drift_thresholds：漂移监测

参考分布只从首轮授权 train 的 128 维特征建立，保存在 SDK 状态；后续继续沿用。当前测量对象是当轮公开 validation 的特征，标签不参与漂移指标。它不表示获得了隐藏流遥测，也不会重新拟合模型的均值和标准差。

| 完整参数路径 | 当前值 | 实际比较的量及含义 |
|---|---:|---|
| `controller.drift_thresholds.minimum_rows` | `30` | 当前验证特征窗口至少有 30 条才允许报漂移。不足 30 条仍计算指标，但 `sufficient_rows=false`、所有传感器漂移标志为假。代码要求正整数。 |
| `controller.drift_thresholds.psi` | `0.2` | 每个特征先用训练参考的分位数分箱计算 PSI，再把同一传感器的 8 个特征 PSI 求平均；该平均值至少 0.2 可触发传感器漂移。不是单个特征 PSI 的最大值。代码要求有限正数。 |
| `controller.drift_thresholds.binned_ks` | `0.2` | 每个特征计算分箱累积分布的最大差，再在每个传感器内平均；平均值至少 0.2 可触发漂移。它是 KS 距离的分箱近似，0.2 不是显著性检验的 p 值。代码要求有限正数。 |
| `controller.drift_thresholds.absolute_mean_z` | `1.5` | 每个特征计算 `abs(当前均值 - 参考均值) / 参考标准差`，再在每个传感器内平均；至少 1.5 可触发漂移。量纲为参考标准差倍数。代码要求有限正数。 |

代码：[monitoring.py:11](D:/aiproject/gas-rsi-four-sensor/src/gas_a/monitoring.py:11)、[monitoring.py:23](D:/aiproject/gas-rsi-four-sensor/src/gas_a/monitoring.py:23)。PSI 的分箱比例使用 0.5 平滑，分箱边界来自训练特征的 10%～90% 分位数，并去重。

判定关系为：样本数达标，且某个传感器的三个平均指标任意一个达到阈值，就标记该传感器漂移；16 个传感器任意一个漂移，则整体 `drift=true`。阈值使用“或”，不是要求三个都超限。监测证据会进入模型计划提示，但没有“漂移即强制训练”的硬编码分支，也不会仅凭 PSI 宣称物理故障原因。

## 5. limits：训练预算、最佳 epoch 和早停

| 完整参数路径 | 当前值 | 含义与实际代码逻辑 |
|---|---:|---|
| `limits.max_epochs` | `32` | 主网络、五传感器辅助网络以及 home 输出头，每个组件单次允许的最大新增轮数。超出直接报错，不自动截成 32。不是累计 epoch 上限，也不是多专家合计轮数上限。主/五网络由 `train_candidate()` 校验，home 在 `train_home()` 校验。[optimize.py:58](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:58) |
| `limits.max_seconds_per_candidate` | `30` | 候选训练的本地墙钟预算，单位秒。主/五传感器在 epoch 和 mini-batch 边界检查；home 在每个 epoch 前检查同一截止时间；路由构建在阶段间检查总剩余时间。home 内部没有逐 batch 检查，阈值拟合和路由搜索也可能在下一检查点才发现超时。它不是精确的强制中断，也不包含模型 RPC、候选上传、历史恢复等整轮耗时。本地训练超时记录失败后进入保留父产物的决策，RPC 中断不会按本地失败恢复。宿主资源限制独立生效。[optimize.py:234](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:234)、[frozen_router.py:198](D:/aiproject/gas-rsi-four-sensor/src/gas_a/frozen_router.py:198) |
| `limits.select_best_epoch` | `true` | 逐 epoch 保存完整权重、Adam 一二阶矩及计数器。正式入口先比较是否通过 controller 门槛，再比较验证 Macro-F1 和 Accuracy；同分保留较早版本。本地 `train-local` 未启用 controller 门槛，仍按自己的训练选择逻辑运行。设为 `false` 时导出最终执行状态且不启用这套早停；home 按其配置轮数执行，不走该选择器。[optimize.py:383](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:383) |
| `limits.early_stopping_patience` | `6` | 当 `select_best_epoch=true`、patience 非零且已执行不少于该实验 `minimum_epochs` 时，连续 6 个 epoch 没有刷新最佳选择键就停止。刷新最佳会把计数归零。`0` 关闭早停，代码要求非负整数。[optimize.py:388](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:388) |

例如父模型累计训练到第 20 个 epoch，当前 `epochs=24`，最多执行到第 44 个。若最佳是第 43 个，最终导出的累计 epoch 是 43，实际执行量仍单独记录为 24 轮。`minimum_epochs=16` 保证的是正常早停前至少执行 16 轮，不保证最终导出的最佳 epoch 一定来自第 16 轮以后；异常、超时也不受这一最低执行数保证。

路由候选的 epoch 选择先针对主神经组件和五传感器组件分别执行；最终整体路由效果由路由校准和外层晋升门槛另行评估。

## 6. experiments：四个实验及其所有取值

数组索引与实验名的对应关系：`experiments[0]=frozen_router`、`experiments[1]=four_sensor`、`experiments[2]=five_sensor`、`experiments[3]=conservative_head`。

| 参数名 | frozen_router 主专家 `[0]` | four_sensor `[1]` | five_sensor `[2]` | conservative_head `[3]` |
|---|---|---|---|---|
| `name` | `frozen_router` | `four_sensor` | `five_sensor` | `conservative_head` |
| `learning_rate` | `0.003` | `0.003` | `0.003` | `0.00065` |
| `epochs` | `24` | `24` | `24` | `5` |
| `minimum_epochs` | `16` | `16` | `16` | 未配置，默认 `1` |
| `balance` | `0.5` | `0.5` | `0.5` | `0.35` |
| `weak_boost` | `1.5` | `1.5` | `1.5` | `1.0` |
| `class_boosts.Acetaldehyde` | `2` | `2` | `2` | 未配置，不额外增强 |
| `l2` | `0.001` | `0.001` | `0.001` | `0.0002` |
| `batch_size` | `128` | `128` | `128` | `128` |
| `kept_sensors` | `[3,14,15,16]` | `[3,14,15,16]` | `[3,13,14,15,16]` | 未配置，不新增传感器投影约束 |
| `freeze` | 未配置，默认 `none` | 未配置，默认 `none` | 未配置，默认 `none` | `features` |
| `routing` | 有：另训练五传感器和 home 专家，再校准组合策略 | 无 | 无 | 无 |

`name` 是模型选择菜单中的标识和报告名，不是自动决定算法行为的魔法开关。是否构建路由取决于有没有 `routing` 配置；保留哪些传感器由 `kept_sensors` 决定；冻结行为由 `freeze` 决定。只改名字不会自动改变网络结构。主/五传感器实验名需由字母、数字、下划线组成；顶层菜单名应保持唯一，否则构建字典时同名项会覆盖。

### 6.1 所有实验通用的训练参数

以下解释覆盖上表中所有已配置的同名路径，也覆盖 `experiments[0].routing.five_sensor` 内的同名参数。

| 参数 | 含义、调整方向及当前代码边界 | 使用位置 |
|---|---|---|
| `name` | 方案标识，用于选菜单、实验报告和状态记录，不是权重文件的来源。正式入口始终续训宿主给定的真实父产物。 | [submission.py:214](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:237)、[optimize.py:58](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:58) |
| `learning_rate` | Adam 的更新步长，主/五传感器范围 `0.000001～0.01`。当前没有配置学习率调度，因此按固定学习率训练。增大会放大每次更新，减小则更保守，但不能据此保证验证效果改善。沿用父模型的 Adam 状态与累计 steps。 | [optimize.py:232](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:232)、[optimize.py:301](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:301) |
| `epochs` | 本轮最多新增多少遍完整训练集，整数 `1～limits.max_epochs`。可能受早停影响提前结束；超时则报错，不能当正常早停成功。 | [optimize.py:78](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:78)、[optimize.py:234](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:234) |
| `minimum_epochs` | 早停前的最低新增执行轮数；默认 `1`，范围 `1～本实验epochs`。它不强制继续执行超时任务，也不限定最终最佳检查点必须来自哪些 epoch。 | [optimize.py:416](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:416) |
| `balance` | 类别逆频率权重的指数，范围 `0～1`。`0` 不做频率补偿；`0.5` 是平方根程度补偿；`1` 是完整逆频率补偿。统计来自 train，不来自验证集。 | [optimize.py:223](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:223) |
| `weak_boost` | 对当前父模型公开验证 F1 最低、且真实支持数非零的类别，再乘一个训练损失权重。范围 `0.1～5`。`1.5` 增强弱类，`1.0` 不额外增强；它不直接保证该类召回。弱类由诊断选择，不一定每轮都是同一类别。 | [monitoring.py:58](D:/aiproject/gas-rsi-four-sensor/src/gas_a/monitoring.py:58)、[submission.py:275](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:302)、[optimize.py:226](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:226) |
| `class_boosts.Acetaldehyde` | 额外对英文标签 `Acetaldehyde`（乙醛）乘 `2`。字典键必须是合法类别，数值范围 `0.1～5`。若本轮弱类也是乙醛，则先乘 `weak_boost=1.5`，再乘这里的 `2`，归一化前增强因子合计为 `3`。 | [optimize.py:227](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:227) |
| `l2` | 权重 L2 正则系数，范围 `0～0.01`，代码给权重梯度加 `l2 * W`，不对偏置使用这项权重正则。用于约束权重幅度；它不是“离父模型参数距离”的约束。冻结参数不会更新。 | [optimize.py:278](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:278) |
| `batch_size` | 每次梯度更新使用的训练样本数，主/五传感器若显式配置，要求整数 `16～256`。`128` 时，445 条训练样本每 epoch 分四批，最后一批 61 条。改变 batch 大小也改变每 epoch 的 Adam 更新次数。 | [optimize.py:231](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:231)、[optimize.py:247](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:247) |
| `kept_sensors` | 使用从 1 开始的传感器编号，每个对应连续 8 个特征；必须是 `1～16` 内非重复整数的非空列表。把未保留传感器对应的第一层权重行置零，并在更新后重新投影为零。四传感器保留 32 个输入维度的连接，五传感器保留 40 个；接口仍接收完整 128 维。不会重新计算输入标准化。 | [optimize.py:213](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:213) |
| `freeze` | `features` 表示只更新输出头 `w2/b2`，冻结其余特征层的权重及 Adam 一二阶矩；冻结参数会从梯度集合删除，避免旧动量导致变化。当前 `conservative_head` 使用此值。它沿用父模型已有特征；若父主网络已投影掉部分传感器，冻结不会恢复这些连接。不能同时配置“冻结第一层”和 `kept_sensors` 投影。 | [optimize.py:34](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:34)、[optimize.py:293](D:/aiproject/gas-rsi-four-sensor/src/gas_a/sensor_training/optimize.py:293) |

类别训练权重的实际计算可写为：

```text
基础权重[c] = (训练样本总数 / (6 × max(该类训练样本数, 1))) ^ balance
弱类权重 *= weak_boost
指定类别权重 *= class_boosts[类别]
所有类别权重 /= 每条训练样本对应权重的平均值
```

所以 `weak_boost=1.5` 并不表示最终报告中的权重一定是 1.5；它还叠加类别频率、指定类别增强及最后的归一化。验证标签用于选择弱类和评估，不被并入梯度训练数据。

## 7. frozen_router.routing.five_sensor：五传感器辅助专家

只有选择顶层 `frozen_router` 时才执行该子方案；选择独立 `five_sensor` 实验只训练一个五传感器网络，不会顺带训练 home 或构建组合路由。

| 完整参数路径 | 当前值 | 本分支作用 |
|---|---|---|
| `experiments[0].routing.five_sensor.name` | `five_sensor_auxiliary` | 辅助训练报告的方案名，不是额外的顶层可选菜单项。 |
| `experiments[0].routing.five_sensor.learning_rate` | `0.003` | 辅助网络 Adam 学习率。 |
| `experiments[0].routing.five_sensor.epochs` | `24` | 辅助网络最多新增 24 个 epoch，独立受 `max_epochs=32` 校验。 |
| `experiments[0].routing.five_sensor.minimum_epochs` | `16` | 辅助网络早停的最低新增执行轮数。 |
| `experiments[0].routing.five_sensor.balance` | `0.5` | 辅助网络训练类别频率补偿指数。 |
| `experiments[0].routing.five_sensor.weak_boost` | `1.5` | 沿用本轮诊断弱类，对其训练权重增强。 |
| `experiments[0].routing.five_sensor.class_boosts.Acetaldehyde` | `2` | 乙醛训练权重再增强两倍。 |
| `experiments[0].routing.five_sensor.l2` | `0.001` | 辅助网络权重正则。 |
| `experiments[0].routing.five_sensor.batch_size` | `128` | 辅助网络 mini-batch 大小。 |
| `experiments[0].routing.five_sensor.kept_sensors` | `[3,13,14,15,16]` | 保留五个传感器对应连接，比主专家额外使用第 13 号传感器。 |

代码：[frozen_router.py:215](D:/aiproject/gas-rsi-four-sensor/src/gas_a/frozen_router.py:215)。父产物已是路由包时，从其中真实的 `five_sensor` 检查点继续；否则从宿主主检查点继续。最终保存为 `routing/five_sensor.npz`，同时保留主专家和 home 组件。

## 8. frozen_router.routing.home：全输入辅助专家

代码：[home_training.py:151](D:/aiproject/gas-rsi-four-sensor/src/gas_a/home_training.py:151)、[home_training.py:82](D:/aiproject/gas-rsi-four-sensor/src/gas_a/home_training.py:82)。生成 `home_trunk.npz` 和 `home_router.npz`：前者只续训输出头；后者复制该神经状态，再用授权 train 拟合特征选择、KNN 和成对逻辑回归辅助头。这里的参数不是只控制一个普通 MLP。

| 完整参数路径 | 当前值 | 实际含义 |
|---|---:|---|
| `experiments[0].routing.home.epochs` | `1` | 实际新增输出头训练轮数，必须为 `1～limits.max_epochs` 的整数；改成 5 就执行 5 个 epoch，累加实际父模型的 epoch/step 计数。每轮前检查候选截止时间；执行次数和逐轮计数写入 `routing_training.json` 的 `home.training`。默认仍为 1。 |
| `experiments[0].routing.home.learning_rate` | `0.00065` | 各轮输出头训练使用的 Adam 学习率。第一层及其 Adam 矩保持不变。 |
| `experiments[0].routing.home.balance` | `0.35` | home 内部拟合子集的类别频率补偿指数。这里只做频率补偿，不读取主专家的 `weak_boost` 或 `class_boosts`。 |
| `experiments[0].routing.home.l2` | `0.0002` | 输出层权重 `w2` 的 L2 正则，不调整下面硬编码的 KNN 或逻辑回归正则。 |
| `experiments[0].routing.home.label_smoothing` | `0.02` | 标签平滑。六分类中，目标类概率为 `1 - 0.02 + 0.02/6 ≈ 0.983333`，其他每类为 `0.02/6 ≈ 0.003333`；降低硬标签的绝对确定性。只用于训练目标，不改真实标签。 |

home 的神经训练使用 train 内部分层拆出的拟合子集；保留子集不参与神经梯度，随后 KNN/成对头使用授权 train 的全部样本。home 的 batch 大小从实际父检查点继承，不读取主实验的 `batch_size=128`。它没有套用主/五专家的 `bounded_experiment()` 完整数值范围校验，也没有独立使用 `select_best_epoch` 或 patience。

当前配置没有暴露 KNN 邻居数、Fisher 保留维数、成对逻辑回归正则及 CV 阈值候选等；这些在 `home_training.py` 内另有固定值。例如保留 120 维、3 个近邻、成对头取 8 个特征等。改变这里的 `l2` 不会同时改变这些固定配置。

## 9. frozen_router.routing.selection：路由校准

这一步不更新神经网络梯度。它在授权公开 validation 上比较有限组合，选定策略后写进 `routing.json`，预测时按该策略运行，不读取标签反馈。

| 完整参数路径 | 当前值 | 实际作用及边界 |
|---|---|---|
| `experiments[0].routing.selection.inner_weights` | `[0.4,0.75,0.9]` | 搜索“五传感器辅助专家”的内部混合权重。每个候选 `w` 对应 `challenger = w × five + (1-w) × home`。`w` 越大，五传感器分数占比越高。代码要求 1～5 个有限数值，各在 `0～1`。不是给四个专家直接分配的三个总权重。 |
| `experiments[0].routing.selection.correction_choices` | `[false,true]` | 分别评估关闭/打开特征偏移修正，最终只导出选中的布尔值。开启时，修正器用已经到达的特征维护历史中位数偏移；不使用标签，也不改检查点的标准化参数。代码要求 1～2 个布尔值。仅保留 `[false]` 就只搜索关闭修正的策略。 |
| `experiments[0].routing.selection.minimum_support` | `20` | 某个“主专家预测类别与 challenger 预测类别的无序组合”，在排除第 13 号传感器平线报警样本后，至少覆盖 20 条验证样本，才尝试启用该组合的路由。不是某个真实类别支持数，也不是 controller 的类别召回保护。代码要求正整数。 |
| `experiments[0].routing.selection.minimum_macro_f1_gain` | `0.0005` | 每贪心添加一个类别组合路由时，整份验证集 Macro-F1 必须相对当前路由策略严格提升超过 `max(1e-12, 0.0005)`，才能加入。这里比较“当前路由策略与增加一个组合后的策略”，不是“完整候选与父模型”；严格大于也不同于 controller 的至少提升。代码要求有限非负数。 |

代码：[frozen_router.py:137](D:/aiproject/gas-rsi-four-sensor/src/gas_a/frozen_router.py:137)、[frozen_router.py:53](D:/aiproject/gas-rsi-four-sensor/src/gas_a/frozen_router.py:53)、[signals.py:67](D:/aiproject/gas-rsi-four-sensor/src/gas_a/signals.py:67)。

当前还存在不在 JSON 中的固定路由值：外层混合 `outer_weight=0.6`。对已启用且未平线报警的类别组合，最终为 `0.4 × four + 0.6 × challenger`；其他样本继续使用 four 分数。即使开启修正，four/five 的分数也各自取原始输入预测和修正输入预测的平均，不能简单理解为完全替换输入。未发现优于基准主专家的组合时，策略保留主专家输出。

`correction_choices=true` 表示允许被搜索，不表示最终一定启用。`inner_weights` 也不是根据隐藏标签在线重选的列表，最终权重和类别组合已冻结在产物清单中。特征状态可随输入更新。

## 10. 四个容易混淆的比较

| 容易混淆的参数 | 区别 |
|---|---|
| `controller.minimum_macro_f1_gain` 与 `routing.selection.minimum_macro_f1_gain` | 前者守住整版候选/历史版本对当前父模型的晋升资格；后者决定某个类别组合的路由调整是否加入。默认同为 0.0005，但处于不同阶段，且 `>=` 与严格 `>` 的边界不同。 |
| `controller.minimum_class_support` 与 `routing.selection.minimum_support` | 前者看真实类别有多少验证样本，决定是否保护该类召回；后者看两个专家预测类别组合覆盖多少样本，决定是否尝试该路由。 |
| `controller.maximum_class_recall_drop` 与 `limits.select_best_epoch` | 正式入口已把前者接入后者，先选达标 epoch；最终完整候选仍须独立过门槛。直接 `train-local` 不自动使用 controller 保护。 |
| `controller.drift_thresholds` 与 `routing.selection.correction_choices` | 前者产生公开特征漂移证据供模型决策；后者控制路由校准是否搜索特征修正。监测报警不会自动开启预测修正。 |

## 11. 修改配置会影响哪个入口

| 入口 | 会用到的配置 |
|---|---|
| 平台 `harness.py train` / `run.py train` | 本文所有分组；具体训练和 routing 参数取决于本轮选中的实验。 |
| `run.py train-local` / `rebuild_models.py` | 读取实验菜单与 `limits`，命令行 `--experiment` 直接选方案，默认 `frozen_router`；不调用模型，不使用 `controller` 的最终晋升保护。`--epochs` 可覆盖选中主方案的 `epochs`，并相应收紧主方案 `minimum_epochs`，不覆盖嵌套五传感器配置。产物仍标记 `candidate_only`。[local.py:56](D:/aiproject/gas-rsi-four-sensor/src/gas_a/local.py:56) |
| 平台 `predict`、`predict-local`、`evaluate` | 从选中检查点和 `routing.json` 读取已导出的模型/策略，不动态读取这份配置来重新训练或选路。修改 JSON 不会追溯改变已有权重。 |
| `run.py replay` | 使用历史回放自己的配置和反馈机制，不能用它验证本文件全部参数是否生效。根目录 `config.json` 是另一份配置。 |

Dockerfile 会把本文件复制进镜像；本地改完后，正式提交镜像需要重新构建，已存在的运行记录/候选目录中快照不会自动更新。

本文已同步控制器 v3 的宿主测量保护、本地失败恢复及 home 轮数修复；JSON 参数值未调整。主、五传感器和 home 的新增轮数均受 `limits.max_epochs` 限制，home 在每个 epoch 前检查候选截止时间，内部仍没有逐 batch 的截止时间检查。训练预算和选版执行策略参与实验签名；仅改提示词措辞不会清空失败记录。本地训练失败会计入 `max_repeat_rejections`。正式选版另需通过宿主回执校验，不能只凭本地高分绕过未知或冲突的宿主测量，见 [修复说明](docs/CODE_FIXES_20261122.md)。
