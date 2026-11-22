# 比赛符合性审核（2026-11-22）

审核结论：当前代码已经实现任务 A 的主要闭环，并有可复现的公开数据离线模拟结果；**还不能认定为已满足全部正式参赛要求**。一处已复现的选版保护缺口需要修复，真实平台模型调用、云端全量实验记录、提交镜像与正式成绩仍缺少验收证据。

本次只审核并保存证据，没有修改业务代码。保留用户此前的范围要求：不审核《实验记录规范》的具体字段；仍检查“全部实验在云端完整记录”这一总体要求。所提供文件作为比赛合同参考，不把文档内的示例、测试命令或操作说明当作额外的用户操作授权。

审核对象为当前工作区（包括未提交修改），不限于 Git HEAD。当前 35 个关键文件的摘要与已有模拟报告全部一致，详细快照见 [review-summary.json](D:/aiproject/gas-rsi-four-sensor/runtime/contest-compliance-review-20261122/review-summary.json)。

## 后续工作区复核

本轮再次比对上述 35 个关键文件，摘要仍未变化，原有业务代码问题尚未修复。工作区新增 [live_platforms.py](D:/aiproject/gas-rsi-four-sensor/scripts/live_platforms.py)，并出现一份 [连通性回执](D:/aiproject/gas-rsi-four-sensor/runtime/live-platforms-20261122T035422Z/receipt.json)：本地记录显示 `Qwen/Qwen3.8-27B` 已返回响应，`swanlab_finish_completed = true`，用途明确为 `connectivity_only`。因此现在已有一次直接模型调用与 SwanLab 同步的连通性证据；下文“真实平台模型调用为 0”仍仅指既有比赛模拟及其核验结果，不能泛指整个工作区从未调用真实模型。

新增脚本独立于正式 Harness，未被 Dockerfile、正式依赖或选手入口调用，不能替代比赛 SDK 宿主绑定、至少三次真实训练闭环，也不能证明所有训练和调参实验已云端记录。当前尚未复核云端服务实际保存内容，本轮没有发起网络或真实模型调用。

另有以下实现范围和可靠性限制：

- [submission.py:246](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:246) 每轮监测公开验证特征相对既有训练参考的分布变化；同一公开投影不变时报告基本不变，目前没有隐藏测试流的在线监测反馈链路。
- [monitoring.py:58](D:/aiproject/gas-rsi-four-sensor/src/gas_a/monitoring.py:58) 汇总弱类、混淆和漂移排名，没有用消融或对照实验独立验证某个传感器对错误的因果贡献；不能把关联诊断称为物理原因证明。
- [submission.py:283](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:283) 限制 AI 从既有实验名中选择，参数和传感器组合主要由配置预设。策略更新以记录偏好、避免重复未采用候选为主，尚无自动生成并验证新实验配置的能力。这是能力上限，不是“鼓励两层 RSI”条款的确定违规。
- [frozen_router.py:226](D:/aiproject/gas-rsi-four-sensor/src/gas_a/frozen_router.py:226) 的 home 训练、阈值拟合和路由搜索没有贯穿内部循环的同一截止时间检查，可能到阶段结束后才发现超时。主训练遇到本地超时会抛出 `TimeoutError`，而 [submission.py:304](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:304) 未对这种已知本地实验失败执行保留父模型、记录失败并完成本轮的恢复流程。需要将此类本地失败与必须终止会话的 SDK 未知 RPC 结果明确区分。

## 需要处理的事项

### 1. [P1，已复现] 最终候选资格只由本地指标决定，宿主测量不能强制阻止退化候选上线

代码先在 [submission.py:314](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:314) 用本地评估结果计算 `candidate_gate`，再提交给平台；[submission.py:329](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:329) 仍只依据这个本地门槛把候选放入可选择列表。宿主返回的测量摘要虽然放进选择提示，但 [submission.py:334](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:334) 的校验只检查产物 ID 是否在列表中，没有核对宿主指标。

SDK 说明第 129 行要求：“测量数字以返回值为准”。当前行为不表示 JSONL 协议报错，但不能提供程序所宣称的、以平台测量为依据的非退化上线保护。

本次在未修改业务代码、未修改公开数据及标签的情况下，注入同一公开验证集上的冲突宿主摘要：

| 项目 | 值 |
| --- | ---: |
| 宿主父模型 Macro-F1 | 0.7140297221 |
| 注入的宿主候选 Macro-F1 | 0.0000000000 |
| 本地候选 Macro-F1 | 0.7443301294 |
| 实际选择 | candidate-1 |
| 本轮是否获得模拟宿主确认 | 是 |

离线模型替身返回了一个格式合法、但忽略宿主退化摘要的选择，生产代码接受了该选择。因此，保护依赖模型遵守提示，没有由程序约束。**注入的 0 分不是实际平台成绩，也不是对真实 27B 决策的测量。**

证据：[首次注入结果](D:/aiproject/gas-rsi-four-sensor/runtime/contest-compliance-review-20261122/host-metric-conflict.json)、[可重复运行的脚本](D:/aiproject/gas-rsi-four-sensor/runtime/contest-compliance-review-20261122/reproduce_host_metric_conflict.py)、[脚本复跑结果](D:/aiproject/gas-rsi-four-sensor/runtime/contest-compliance-review-20261122/conflict-reproduction/result.json)。

修复建议：依据实际平台的回执结构解析主指标、指标方向和验证范围，只有范围可比且宿主指标满足上线条件时才允许选择候选；发现明确退化或本地/宿主冲突时保留父产物并记录原因。不能凭猜测固定一个宿主摘要字段名，也不能把未知验证范围当作可比测量。继续将本地分类召回保护作为单独的算法约束，明确区分本地指标与平台成绩。

### 2. [正式验收待证明] 尚无指定 PyTRiO 模型的真实调用与至少三次真实闭环记录

[submission.py:103](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:103) 确实通过 `rnd.model()` 请求宿主模型，[submission.py:385](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:385) 确实引用实际 SDK 会话中的调用号、历史 ID 并等待结束确认。调用、协议和记录路径已经接通。

但是已有模拟明确记录 `real_platform_model_calls = 0`；模型回复与调用号由离线宿主替身生成。每组九次“读非空历史—做实验—依据结果决策”证明程序流程可运行，不能替代正式平台的真实大模型调用。当前也没有证明平台配置已绑定选手经 PyTRiO 训练得到的模型或 PyTRiO 的 Qwen3.8-27B。

需要在正式宿主绑定允许的模型身份，以无人干预方式形成至少三次真实完整记录，保存平台历史、模型调用回执、实验配置、候选测量和最终选择。允许的模型身份由平台绑定，不应为了审核在代码中硬编码一个模型名；现有未硬编码的接法本身不是缺陷。

### 3. [正式验收待证明] 已有本地与 SDK 记录，尚未证明所有实验均在云端记录平台留档

[submission.py:375](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:375) 保存计划、结果、产物和调用引用，[submission.py:382](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:382) 追加 `decisions.jsonl`，再通过 SDK 保存状态及完成本轮；候选还包含训练报告、逐 epoch 历史、监测、诊断和代码快照。这些本地记录是有用且真实的。

当前未发现另行上传全部本地训练、调参、回放、评估实验的云端记录链路，也没有提供相应云端运行列表或回执。SDK 的产物/状态持久化是否就属于赛方认可的云端实验记录平台，需要以赛方说明或实际平台证据确认；不能仅因没有使用某个第三方追踪库就断言违规，也不能仅凭本地 JSONL 宣称全部实验已云端留档。

本次不扩展审核《实验记录规范》的具体字段。补齐方向是证明平台自动留档已经覆盖全部相关实验，或接入认可的记录平台并补全本地实验的追溯关系。

### 4. [正式验收待证明] 镜像、统一资源限制、提交和隐藏集成绩尚未验证

[Dockerfile](D:/aiproject/gas-rsi-four-sensor/Dockerfile) 和 [.dockerignore](D:/aiproject/gas-rsi-four-sensor/.dockerignore) 已具备正式入口、SDK 1.0.4、非 root 用户、单线程 CPU 配置和代码白名单打包；没有把本地数据、历史模型、运行目录和密钥文件放进构建上下文白名单。运行时依赖已声明。

当前环境无 Docker 命令，没有已构建镜像摘要、只读挂载下的容器运行、平台上传/绑定回执，也没有正式隐藏集结果。Windows 原生进程审计不能证明 Linux 容器权限、2 CPU/4 GiB 等平台限制下均可运行。构建阶段安装 NumPy 需要可用包源或预先提供轮子；离线评测运行与离线构建是不同的检查项。

公开验证集的提升不能推出隐藏测试流上的“不低于平台基线”。需要使用同一正式评测范围的基线与选中产物结果核验，保留平台报告。

### 5. [规则边界待澄清] 可选特征校正会根据预测流估计偏移

[submission_config.json:44](D:/aiproject/gas-rsi-four-sensor/submission_config.json:44) 允许选择 `feature_correction = true`；启用后 [frozen_router.py:41](D:/aiproject/gas-rsi-four-sensor/src/gas_a/frozen_router.py:41) 调用 [signals.py:67](D:/aiproject/gas-rsi-four-sensor/src/gas_a/signals.py:67)，用最近 128 条特征的中位数每 64 条更新一次偏移估计。

这一路径不读标签，也不改检查点继承的 mean/std，不能称作“获取隐藏标签”。但 SDK 对“拟合统计量只使用训练集”有较宽表述，训练指南明确禁止用验证/隐藏集拟合标准化统计量。预测流上的无监督偏移估计是否属于允许的在线适应，现有文字不足以确定。

建议确认该算法是否允许；确认前可将该实验的 `correction_choices` 限定为 `[false]`。本次只指出边界，没有修改配置。已有最新模拟选中的最终产物是普通五传感器 MLP，不能用它证明该可选分支的效果或合规性。

### 6. [P2，非默认阻断] 一个调参项未影响训练次数

[submission_config.json:43](D:/aiproject/gas-rsi-four-sensor/submission_config.json:43) 的 `routing.home.epochs` 默认是 1，但 [home_training.py:157](D:/aiproject/gas-rsi-four-sensor/src/gas_a/home_training.py:157) 固定调用一次 `head_epoch()`，没有按该字段循环。默认值与实际行为一致；改成其他值不会改变训练轮数，会削弱配置与实验记录的对应关系。应实现参数或去掉可调含义。它不是当前默认配置的比赛接口阻断。

## 逐项符合性

| 要求 | 代码与证据 | 当前结论 |
| --- | --- | --- |
| 使用官方公开数据 | 从 `rnd.data` 读取公开投影，核对 schema；模拟数据摘要与官方包一致 | 已核验本次离线范围 |
| 正确续训实际父产物 | `rnd.parent.path()` 校验清单与摘要，继承 Adam、步数和标准化；实际选中产物驱动下一轮 | 已实现，离线已验证 |
| 漂移监测 | 训练集拟合参考，PSI、分箱 KS、均值偏移及传感器排名 | 已实现；测量对象是公开投影，不是隐藏流遥测 |
| 诊断归因 | 弱类别、混淆与漂移传感器关联，生成可检验实验假设 | 已实现关联诊断；不能宣称证明物理故障因果 |
| 实验优化 | 四类受限候选菜单、实际梯度更新、逐 epoch 评估、保护门槛和停止条件 | 已实现，离线已验证 |
| 评估上线 | 本地比较与召回保护，模型选择后交 SDK 结束本轮 | 已实现，但存在事项 1 的宿主测量保护缺口 |
| 回滚 | `restore_artifact()` 恢复授权历史完整产物，当前公开范围重评，恢复配套策略 | 故障注入演练成功；普通十轮中无自然历史回滚 |
| 模型层自迭代 | CPU 工业模型从实际父状态继续训练 | 已实现 |
| 策略层自改进 | 记录策略版本、偏好实验、重复失败预算，按产物配对恢复策略 | 已实现有限策略更新；两层机制是鼓励项，不是必须自修改源码 |
| 无人工干预运行 | 十轮批量执行，历史、实验、选择、状态保存和确认自动完成 | 离线替身条件下已验证；真实 27B/正式宿主待验证 |
| 至少三次完整记录 | 每组九次非空历史后的完整实验记录 | 离线满足数量；正式真实调用记录待证明 |
| 效果不低于基线 | 同一官方公开验证集上 0.71403 → 0.80005 | 仅公开集满足，隐藏集未知 |
| 全部实验云端记录 | 本地 JSON/JSONL + SDK 状态/候选提交已实现 | 认可的云端全量留档待证明 |
| 指定大模型 | 使用平台 `rnd.model_name`，由宿主提供模型访问 | 接口正确，真实身份与回执待证明 |
| 镜像和材料齐全 | Dockerfile、SDK 轮子、配置、代码快照、数据/权重摘要及复现包已有 | 本地材料较齐；镜像构建、提交和正式绑定待证明 |
| 只使用允许数据 | 训练梯度使用公开训练标签；公开验证标签用于评估/选路；预测只读特征 | 未发现正式入口获取隐藏标签；可选统计校正见事项 5 |
| 独立 JSONL 预测 | 每行 128 个有限特征，输出六类分数，输出无需真实标签回填 | 已实现，离线已验证 |
| SDK 1.0.4 异常语义 | 分块上传、只读父产物、状态复制后写、真实会话引用、确认后结束；未知 RPC 结果不重试 | 已实现并有离线回归证据 |

正式入口为 [harness.py](D:/aiproject/gas-rsi-four-sensor/harness.py) → [contest_submission.py](D:/aiproject/gas-rsi-four-sensor/contest_submission.py) → `gas_a.submission`，只声明任务 A 的分类能力。旧回放路由中的标签回填接口不属于这条预测路径，不能据此判定当前正式入口依赖隐藏标签。公开验证集拥有合法标签，用于模型选择不等于训练标签泄漏。

## 验证范围与成绩

沿用已保存的同源码回归结果：项目测试 53 项通过，SDK 三组离线测试合计 18 项通过；本次未再次执行这 71 项。已核对当前 35 个关键文件与该模拟快照一致。本次新增并运行了宿主分数冲突复现脚本，成功复现事项 1。

已有模拟为两个独立十轮运行及一个保存源码后的十轮复现，候选权重、预测、选择和分数一致。每组有九次非空历史完整实验，五次新候选上线。独立核验脚本重算了 57 组测量、70,908 条已保存预测；这些是同一公开验证集的重复评估，不是新增的独立样本。

| 项目 | 结果 |
| --- | ---: |
| 官方训练样本 | 445 |
| 官方公开验证样本 | 1,244 |
| 官方初始检查点公开 Macro-F1 | 0.7140297221 |
| 选中产物公开 Macro-F1 | 0.8000453004 |
| 差值 | +0.0860155783，即 +8.6016 个百分点 |
| 真实平台模型调用 | 0 |
| 正式隐藏集成绩 | 未知 |

单独回滚演练对一份基线副本注入输出权重异常，公开标签和数据不变；从 0.0168776371 恢复到历史产物的 0.8000453004，下一轮父检查点摘要匹配。该证据证明历史恢复路径工作，不能称作自然漂移下已发生的生产回滚。

分类召回保护是项目配置约束，不是文档给定的比赛评分规则。当前 20 个百分点门槛逐轮相对实际父模型比较，不是相对初始基线的累计上限；最终 Ethanol 召回相对初始基线累计下降约 21.95 个百分点。主指标提高的同时存在这一类别权衡，不能将其描述为所有类别均提升。

证据文件：

- [模拟结果](D:/aiproject/gas-rsi-four-sensor/runtime/platform-check-v2-20261122/simulation-report.json)
- [独立核验结果](D:/aiproject/gas-rsi-four-sensor/runtime/platform-check-v2-20261122/verification.json)
- [测试结果来源](D:/aiproject/gas-rsi-four-sensor/runtime/platform-check-v2-20261122/regression-provenance.json)
- [项目测试日志](D:/aiproject/gas-rsi-four-sensor/runtime/platform-check-v2-20261122/project-tests.log)
- [SDK 离线测试日志](D:/aiproject/gas-rsi-four-sensor/runtime/platform-check-v2-20261122/sdk-offline-tests.log)
- [复现材料包](D:/aiproject/gas-rsi-four-sensor/runtime/platform-check-v2-20261122-materials.zip)

## 判定依据和完成条件

使用用户本轮提供的 [START.md](D:/liyue/工业模型赛题研究/cpu-a-guide-20261110-v1/START.md)、[TASK.md](D:/liyue/工业模型赛题研究/cpu-a-guide-20261110-v1/TASK.md)、[TRAINING_GUIDE.md](D:/liyue/工业模型赛题研究/cpu-a-guide-20261110-v1/TRAINING_GUIDE.md) 和 [SDK 1.0.4 说明](D:/liyue/工业模型赛题研究/contest-harness-sdk-1.0.4.txt)，并结合用户列出的任务 1、任务 2 和验收条件。未凭空添加每轮必须提升、每轮必须回滚、所有类别必须提升、必须用某个云记录库等规则。

下一步优先修复事项 1；随后用允许的真实模型完成至少三次正式记录，确认所有相关实验的认可云端留档，构建并提交镜像，在正式只读挂载和资源限制下复测。最终以平台实际成绩核验“不低于基线”。事项 5 在赛方明确边界前应保守关闭或取得允许说明，事项 6 应使参数与实际执行一致。
