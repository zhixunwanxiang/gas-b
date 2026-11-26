# Xiaoyang B/C 接入、验证与镜像

日期：2026-11-26。分支：`codex/xiaoyang-bc`，从 `main` 的 `0c0ba6f` 创建。`main` 保持原样。用户随后要求生成该组合的镜像。

来源为用户提供目录 `C:/Users/Administrator/Desktop/xiaoyang版本-代码与本机指标-20261126/xiaoyang版本-20261126/`。这份新包的 B/C 实现和旧桌面本地研究脚本不同，本次以新包为准；未改写来源文件。

## 实现范围

- B：继承当前 MLP，增加原尺度与对数寿命浅树融合；每层权重为原预测 0.7、原尺度树 0.1、对数树 0.2，最多 2 层。每个尺度拟合 8 棵树，不使用旧研究版的大规模森林。
- C：导入 train 工况残差、累计树和正向修正减半的尾部方案。每层 12 棵浅树，最多 5 层；保留首次累计锚点及 train IQR 确定的半径，后续层和神经续训不能绕过累计边界。
- 新组件只用 train 拟合。神经骨干继续原 MSE/Huber 训练，树组件拟合训练目标或完整父预测残差；完整预测器用于验证选版。原权重、Adam、输入/目标缩放、种子和累计计数继承，不虚增训练次数。
- 保留来源的 RMSE/MAE 准入；B 每设备默认 15% 研发退化上限同时参考父模型和起点。C 约束 NASA、高估 P99 和最差工况。15% 不是官方规则，也不是此前 v21 的 2% 研发门槛。
- 原有 SDK 1.0.4、多文件父摘要修复、父评分核验、分块上传、历史恢复及错误边界继续保留。

## 为当前项目做的兼容适配

A 与共用模型代码不修改。xiaoyang 的文件流/嵌入模型能力独立放在 `regression/reference_model.py`，回归数据读取适配当前公共读取接口。首次切换到该代码即建立实际父模型的 B 分组起点，支持从任务中间轮次接入。

新复合检查点外层格式为 `contest.cpu-rul-xiaoyang.v1`，树/锚点均在一个已声明的 `checkpoint.npz` 中，并核验内部 SHA256。可读取来源包的旧复合格式；新导出的复合检查点会使旧普通 MLP 加载器明确报错，避免静默丢失组件。旧层前缀、C 锚点与半径、原残差头在后续更新中进行继承检查。

逐文件核对 A 与共用源码 33 个文件、全部 A 配置、原有 6 个模型文件，均与 main 一致。模型始终从 SDK 提供的实际父产物续训，没有用随包终点权重替换官方起点。

## 用户关心的两项 V18 保护

1. 父检查点通过 SDK `rnd.parent.path()` 按清单逐文件校验，产物聚合摘要不与单文件摘要混用。训练另记录实际主文件摘要。A 多组件及 B/C 完整检查点均保留此路径。
2. 父分数缺少 `scope` 等字段时，对实际父模型和本轮验证文件重新测量，并交叉核对宿主数值；显式错误的产物身份、验证摘要或分数仍拒绝升级。不把本地复评伪装成宿主对缺失范围的认证。

相关测试覆盖多组件下一轮续训、篡改/缺失文件、精简评分回执、冲突与验证文件变化。

## 新测与来源权重复算

首先用本分支 SDK 独立预测入口复算来源 B/C 终点权重，分别得到 B RMSE 587.890563、C RMSE 50.739451，与包内记录一致。随后从官方初始检查点和空状态重新运行 A/B/C 各十轮，结果如下：

| 项目 | 官方初始主指标 | 第十轮实际选中 | 十轮平均 | 实际采用次数 |
|---|---:|---:|---:|---:|
| A Macro-F1 | 0.714030 | 0.895614 | 0.874105 | 6 |
| B RMSE/分钟 | 681.159833 | 587.890563 | 596.625367 | 4 |
| C RMSE/循环 | 52.868573 | 50.739451 | 50.849671 | 3 |

B/C 相对官方初始 RMSE 分别降低 13.6927%、4.0272%。最终 B MAE 为 477.105411；C MAE 为 36.388882、NASA 为 3,351,296,107.7273、高估 P99 为 92.793333。A 保持现有十轮结果。

本机各任务新测全流程约为 A 35.5 秒、B 29.7 秒、C 232.1 秒，可并行执行。该时间包含本次宿主与独立预测，不含真实 27B 网络等待，不能用作正式资源评分或跨环境加速证明。运行采用离线控制器与真实数值训练、SDK 独立预测，真实模型调用为 0；每条路线十轮不是十次独立随机实验。隐藏成绩及正式阶段恢复速度未验证。

## 工程验收

- Windows 项目测试 254 项通过；Linux 254 项完成、1 项 Windows 专用操作系统审计检查跳过、无失败。跳过的是 `test_blocked_windows_version_query_is_distinct_from_data_access`：检查被拦截的 `cmd /c "ver"` 被归类为系统版本查询，不误报数据访问；该项已在 Windows 执行通过，A/B/C 业务与 SDK 检查未因此跳过。
- 完整组件续训、原层保持、C 累计边界、验证答案不参与拟合、格式拒绝、超时、SDK 上传/独立预测、B 起点继承与 C 完整历史恢复均有用例。
- 新测 30 轮均完成；逐条重算候选及实际选中模型的预测，核对实际父链、原始指标、回执与完成确认。
- 开发镜像在断网、只读、2 CPU、4 GiB 条件下独立预测新测 B/C 最终产物，指标与本机一致。镜像内 43 个应用文件 SHA256 与当前分支一致。
- 测试首轮发现来源短提示缺少原父身份说明，已补回；新增测试首轮的宿主参数名也已修正。原失败日志保留，随后测试通过，没有放松准入规则来获取上述结果。

## 镜像与图表

镜像内容固定 ID：`sha256:b8fab39b5a134e84bed0eef323688058abb7f04f9dccecaace314dfc820a9c7d`。

本地导出标签：`gas-rsi-abc:wyx-sdk104-xiaoyang-bc-20261126`。交付目录为 `output/submission-wyx-sdk104-xiaoyang-bc-20261126/`，随包提供 TAR、SHA256、加载命令、源码审查及验证摘要。它与已验证的开发镜像使用相同内容 ID，不包含本地密钥、训练/验证数据或来源终点权重。

- [本次新测完整图表](D:/aiproject/gas-rsi-four-sensor/output/xiaoyang-bc-fresh-rounds-20261126/index.html)
- [图表 ZIP](D:/aiproject/gas-rsi-four-sensor/output/本次新测-ABC十轮图表-xiaoyang-20261126.zip)
- [来源与 A 保留清单](D:/aiproject/gas-rsi-four-sensor/runtime/xiaoyang-bc-integration-20261126/source-and-preservation.json)
- [A 保留及镜像源码核验](D:/aiproject/gas-rsi-four-sensor/runtime/xiaoyang-bc-integration-20261126/preservation-verification.json)
- [来源权重复算](D:/aiproject/gas-rsi-four-sensor/runtime/xiaoyang-bc-integration-20261126/package-verification.json)
- [原始新测登记](D:/aiproject/gas-rsi-four-sensor/runtime/xiaoyang-bc-integration-20261126/fresh-30/registration.json)
- [Windows 测试](D:/aiproject/gas-rsi-four-sensor/runtime/xiaoyang-bc-integration-20261126/final-full-tests.log)、[Linux 测试](D:/aiproject/gas-rsi-four-sensor/runtime/xiaoyang-bc-integration-20261126/linux-full-tests.log)、[容器独立预测](D:/aiproject/gas-rsi-four-sensor/runtime/xiaoyang-bc-integration-20261126/container-verification.json)

新测登记中的 Git 提交为分支基点；当时移植改动尚未提交，实际运行版本由登记中的完整源码 SHA256 固定，最终提交保存同一份已验应用代码。已有 main 的 V18 镜像和原图表包保持不变。没有推送镜像或提交比赛平台。
