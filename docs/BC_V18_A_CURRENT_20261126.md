# B/C 恢复 V18，A 保留当前版本

日期：2026-11-26。依据用户要求：“将 B、C 代码的修改还回到 V18 的版本，A 项目保留当前版本”。

## 回退范围

B/C 使用已交付 V18 的冻结源码 `runtime/parent-manifest-v18-20261125/source/`。恢复前逐文件核对交付清单 SHA256，并确认与 Git 提交 `9ab6a59` 的源码字节一致。

- `src/gas_a/regression/` 下 V18 的 7 个核心文件全部精确恢复：`__init__.py`、`data.py`、`evaluation.py`、`monitoring.py`、`prompts.py`、`submission.py`、`training.py`。
- `submission_config.json` 只替换整个 `regression` 值，其他内容保持原样。B/C 恢复四项共享实验：`low_rate`、`standard_rate`、`head_only`、`robust_huber`；最多 8 epoch、30 秒、早停耐心 3。
- 恢复 V18 的原 MLP 加载和 Adam 续训、RMSE/MAE 门槛、两阶段真实回执、候选提交与历史恢复流程，保留 V18 已有的整组父检查点摘要兼容修复。
- 撤下 B/C 后续的设备/尾部加权、`fine_rate`、短菜单、分组/阶段保护、测量缓存、观察轮单调用、失败候选免上传、工况岭回归头和 RBF 研究头；相应新增模块、研究筛选脚本和专属测试已从当前工作树移除。
- 旧实验、文档和产物仍可通过 Git 提交 `b106b47` 及先前提交、本地 `runtime/` 记录查阅，没有重写历史或清除实验结果。

## A 保留范围

A 保留回退前提交 `b106b47` 的实现。包括完整路由专家包继承、主组件续训时保留辅助专家、完整预测器逐 epoch 选版、现有解析优化、控制器与资源记录。

回退前后核对：A 及共用源码的 33 个文件字节不变，A 的全部配置值不变，现有 6 个模型文件字节不变。共用 SDK、入口、父检查点校验、输入解析、资源统计和实验证据辅助代码保留当前版本，以兼容 A；B/C 的业务模块和回归配置使用 V18。共用辅助实现与 V18 的接口兼容性另由数值对照及 SDK 测试验证。

测试中保留 A 的全部现有用例，只撤下不再适用的 B/C v19/v20/v21 用例，并恢复 V18 的 B/C 协议预期。`validate_first_four.py` 保留 A 轨迹，将 B/C 序列改为 V18 菜单；历史 `validate_item_tuning.py` 检查当前菜单后明确拒绝用于本次组合，避免误用旧 v20 实验名。

## 验证

- 当前完整测试 246 项通过，含 A 完整专家包续训、父摘要兼容及 B/C 续训、历史与协议用例。
- 冻结 V18 与当前恢复版分别从相同官方 B/C 父模型出发，在相同公开 train/validation 上执行 `low_rate` 三轮续训。导出的 B/C 检查点 SHA256、逐条预测、选中 epoch、累计步数和指标完全一致。
- B 对照导出 SHA256：`ea4da13a8c80c8a2404ef6a5d8918368bbc49e1d6f35b49c884fb9f9c35f7e20`。
- C 对照导出 SHA256：`99bff9813430fc2d01fbe74ea28a4db8836c6baf2fcfc82b255a9b8d39d544fc`。
- 上述对照用于验证恢复正确性，不是新的提分实验，也不宣称隐藏集收益。
- 恢复后的 B/C 各运行三轮本机离线 SDK 闭环，共六轮完成；逐轮核对两次合成模型回执及宿主完成确认。该验证使用离线控制器，不是 Docker 隔离或真实 27B 验收。

原始记录：[回退前身份清单](D:/aiproject/gas-rsi-four-sensor/runtime/rollback-bc-v18-keep-a-20261126/preservation-before.json)、[文件/配置/数值核验](D:/aiproject/gas-rsi-four-sensor/runtime/rollback-bc-v18-keep-a-20261126/verification.json)、[完整测试日志](D:/aiproject/gas-rsi-four-sensor/runtime/rollback-bc-v18-keep-a-20261126/full-tests.log)、[B/C 六轮 SDK 摘要](D:/aiproject/gas-rsi-four-sensor/runtime/rollback-bc-v18-keep-a-20261126/sdk-bc/summary.json)。

本次修改源码和配置，没有替换已有检查点、运行状态、镜像或平台产物。V21 新格式的组件检查点需要当时配套的预测器，不能作为 V18 普通 MLP 权重加载；历史产物应与对应源码配套使用。
