# Gas RSI：工业模型自进化系统

更新日期：2026-11-25。

项目支持 A 气体识别、B 轴承剩余寿命、C 发动机剩余寿命，使用 contest-harness SDK 1.0.4 接入平台。

核心思路是：**程序固定实验菜单和上线门槛，27B 根据数据、错误和历史选择方案，程序实际训练并测量，合格才允许采用，不合格则保留父模型，必要时选回本轮验证合格的历史模型。** AI 可以选择方案，不能随意改写训练代码、扩大预算或放宽门槛。

## 1. 当前交付与工作区版本

平台最新诊断已确认旧版的多文件父模型加载错误：`parent_checkpoint_sha256` 可为整组检查点的聚合摘要，旧代码误将它与每个单文件摘要比较，A 在采用四组件路由包后的下一轮因此退出。此前离线宿主只模拟单文件摘要，未覆盖这一情况。

工作区已修复：按 `checkpoint_paths` 对每个文件调用 `rnd.parent.path()`，由 SDK 按产物清单核验各文件；四组件包依据主模型内的路由标记确定入口，不依赖文件名或清单顺序。平台父身份继续原样用于 SDK 协议、父评分绑定和实验指纹；续训文件校验独立使用实际主文件摘要，另记为 `configuration.parent_primary_file_sha256`。A、B、C 的父评分及续训路径均已调整。程序不自行推测官方聚合算法，文件损坏、缺失或声明冲突仍应拒绝。

新增回归测试为 `tests/test_parent_checkpoint_manifest.py`，离线容器宿主支持 `--aggregate-parent` 和 `--compact-validation` 联合测试；聚合算法只是测试夹具，程序不会自行重算或替换官方身份。

当前交付为 **v18 父产物清单校验版**，修复已在 v17 镜像中复现的聚合摘要误比较错误，并保留此前历史输入、数值溢出和路由清单结构保护。

| 项目 | 内容 |
|---|---|
| 镜像标签 | `gas-rsi-abc:wyx-sdk104-v18-parent-manifest-20261125` |
| 不可变镜像 ID | `sha256:3bf46eec43814d6fcaef9e885d7629d8b87507115d87e4c36fc5c6d72522c774` |
| 镜像包 | [v18 TAR](output/submission-wyx-sdk104-v18-parent-manifest-20261125/gas-rsi-abc-wyx-sdk104-v18-parent-manifest-20261125.tar) |
| 完整交付目录 | `output/submission-wyx-sdk104-v18-parent-manifest-20261125/` |
| 交付总量 | 镜像、源码审查包及说明合计小于 **60 MiB**；逐文件摘要见目录内 `delivery-manifest.json` |
| 已测源码快照 | `runtime/parent-manifest-v18-20261125/source/`，交付包另含更新后的 README；应用代码摘要与镜像逐文件匹配 |
| 运行依赖 | Python 3.12.13、NumPy 1.26.4、SDK 1.0.4 |
| 验证范围 | 离线宿主模拟、真实数值训练与容器预测；未确认正式平台通过 |

本 README 的更新不会改变已经交付的镜像、源码审查包及其摘要。

v18 已包含 v16/v17 的分类拒绝计数、最终历史选版和历史输入保护。后续再改本地代码时仍需另行构建，已有镜像不会自动更新。

| 行为 | 旧 v15 | v16/v17/v18 |
|---|---|---|
| 重复拒绝计数 | 未采用/本地失败可能计入重复限制 | 区分性能拒绝、测量未核验、控制器保护、合格但未选择等结果；只对明确性能拒绝或本地训练失败计数 |
| 最终选择历史模型 | 规划需要明确选择 rollback，才能选对应合格历史目标 | 规划有效时，observe/experiment 结束后也可选择本轮复评合格、SDK 授权且进入白名单的历史模型 |
| 规划进入 program_guard | 只能保留实际父模型 | 仍只能保留实际父模型，不开放历史选择 |

相关修改在 [策略模块](src/gas_a/evolution.py)、[A 主控](src/gas_a/submission.py)、[B/C 主控](src/gas_a/regression/submission.py)及对应提示词中。最终动作记录在 `configuration.selection_action`，便于区分“规划未训练”和“最终选择了历史模型”。

2026-11-25 已清理本项目 16 个旧 Docker 镜像及 21 个旧导出包，保留 v15 最终版、基础镜像、源码和实验记录。历史文档中的旧镜像包不再保证存在。`runtime/` 和 `output/` 不进入 Git，克隆仓库不会自动获得这些本地产物。

## 2. “漂移监测—诊断归因—实验优化—评估上线—回滚”如何运行

平台每次调用 `train(rnd)`，执行一个回合。下一回合由平台传入实际选中的父产物，不能由程序私自指定其他起点。

| 步骤 | 什么时候做、具体做什么 | 依据什么决定 | 留下什么结果 |
|---|---|---|---|
| 漂移监测 | 读取本轮公开投影后，把当前 validation 的输入与首轮 train 建立的固定参考比较 | 分箱占比变化、累计分布距离、均值偏移及样本数 | `monitoring.json`，是否告警及变化位置 |
| 诊断归因 | 在公开 validation 上测量实际父模型，分析错误集中在哪里 | A 看弱类别、混淆和召回；B/C 看误差方向与寿命分段 | `diagnosis.json`，提供待验证的原因线索 |
| 实验优化 | 先读历史和策略，再由 27B 从允许菜单中提出实验假设 | 当前错误、可比历史、重复失败限制、执行预算 | 完整模型回复及回执、候选权重、逐 epoch 训练记录 |
| 评估上线 | 本地评估候选，通过 SDK 提交给宿主测量，再让 27B 选择最终版本 | 本地门槛和宿主回执均合格，最终 ID 在白名单内 | 候选回执、采用或拒绝理由、选中产物 ID |
| 回滚 | 从本帧允许的历史产物中恢复模型，在本轮 validation 重新测量后选回 | 当前父模型与历史模型的同口径实测、完整文件摘要 | 历史复评、最终选择及配套策略，下轮继承该产物 |

漂移告警只说明“输入分布变了”，不直接证明模型变差，也不能仅凭统计指标断言某个传感器发生物理故障。是否值得优化，还要结合模型错误和实际测量。

模型层通过继续训练改变权重；策略层根据已发生的结果调整实验偏好和重复限制。两层都受固定规则约束。

## 3. program_guard：AI 方案不合法时怎么办

**通俗理解：AI 没按要求给出可执行方案，纠正后仍不合法，程序就保守地沿用平台本轮指定的模型，不擅自做实验或换模型。**

“实际父模型”指 `rnd.parent.artifact_id` 对应的产物，通常是平台上一轮实际选中的版本。它不一定是最近训练出的候选，也不一定是最初基线。

当前 `json_corrections=1`：每个决策阶段先请求一次；拿到完整回复后，如 JSON、字段、菜单或目标不合法，再允许一次纠正。因此：

- **规划校验耗尽**：进入 `program_guard`，规划设为 `observe / none`，目标锁定实际父模型，本轮不启动新训练，也不改选历史模型。
- **最终选版校验耗尽**：即使候选已经训练和测量，也不采用它，仍保留实际父模型。
- **SDK 请求中断、超时或结果未知**：终止会话，不进入上述纠正保护，不自动重试。

例如本轮父模型是 V3，AI 两次都选了不允许的方案，程序就继续使用 V3；不能自行改用 V1，也不能直接选历史上分数更高的 V5。

规划保护后，程序在会话仍有效时还会继续最终选版、保存状态和等待结束确认。规划与选版各最多两次模型调用，两个阶段都需要纠正时，一轮最多四次，不能理解为整轮固定两次。

程序会保存 `model-responses.jsonl`、`decisions.jsonl`，记录真实回执及 `decision_source=program_guard`。**程序兜底不等于 27B 给出了合法决定，也不计为一次成功的完整实验。**

实现见 [共享决策与保护逻辑](src/gas_a/submission.py)。

## 4. 固定菜单、上线门槛与数据边界

### 数据与原生输出

| 数据集 | 输入 | 预测结果 | 主要指标 |
|---|---|---|---|
| A：气体识别 | 128 维特征，16 个传感器各 8 维 | 六类气体分数 | Macro-F1 越高越好 |
| B：轴承剩余寿命 | 水平、垂直通道各 17 项振动特征 | 有限非负数，单位分钟 | RMSE 越低越好，辅助 MAE |
| C：发动机剩余寿命 | 21 个 sensors、3 个 settings | 有限非负数，单位循环 | RMSE 越低越好，辅助 MAE、NASA 评分 |

只使用平台授权的公开 train 和 validation。梯度更新及最初统计拟合使用 train；validation 可用于诊断、选 epoch、选路和选版，不并入梯度训练。

`id` 和 `meta` 不进入模型特征，不从文件名、样本号或匿名标识推断标签。隐藏标签不会挂入训练容器，也不获取隐藏标签。

续训恢复实际父检查点的权重、Adam 一二阶矩、输入/目标缩放、seed、epochs 和 steps。未采用候选不能自动成为下轮父模型；B/C 授权 train 更新后，监测仍保留首轮参考，模型 scaler 沿用父检查点。

### AI 可以选择什么

- A：`frozen_router`、`four_sensor`、`five_sensor`、`conservative_head`。
- B/C：`low_rate`、`standard_rate`、`head_only`、`robust_huber`。
- `observe` 规划不启动训练；`experiment` 只能从实际父模型续训一个允许方案；`rollback` 规划不训练，提出经过本轮复评的历史目标。
- 最终历史选择范围按第 1 节区分镜像与工作区。无论哪个版本，AI 都不能绕过白名单、改写实际父模型或放宽门槛。

### 默认门槛

| 范围 | 判断依据 |
|---|---|
| A 上线 | Macro-F1 严格提高，且至少增加 0.0005，即 **0.05 个百分点** |
| A 类别保护 | 父 validation 支持数至少 20 的类别，单步召回下降最多 0.20，即 **20 个百分点** |
| B/C 上线 | RMSE 至少减少 `max(0.000001, 父 RMSE × 0.0005)`，MAE 最多相对增加 2% |
| 宿主核验 | 公开验证身份可核验，宿主分数与本地复核一致；显式冲突或无效测量不能放行 |
| 漂移 | 当前样本至少 30 条；PSI ≥ 0.2、分箱距离 ≥ 0.2 或均值偏移 ≥ 1.5 个训练标准差 |
| 漂移汇总 | A 每个传感器的 8 维指标先取均值再判断；B/C 对 34/24 列逐项判断 |
| 本地训练 | A 最多 32 个 epoch、B/C 最多 8 个；单候选数值阶段最多 30 秒，并受菜单及早停限制 |
| 历史恢复 | 每轮最多恢复 2 个历史产物，失败历史也占一次恢复额度 |

保护门槛逐轮相对**实际父模型**比较，不是相对初始基线的永久累计保护。A 少于 20 个验证样本的类别不受上述召回硬保护，C 的 NASA 评分只作辅助记录。

参数详见 [submission_config.json](submission_config.json) 与 [参数逻辑说明](SUBMISSION_CONFIG_GUIDE.md)。根目录 `config.json` 属于历史回放。

## 5. 坏历史模型和其他异常如何处理

v17 继续保留 v15 的坏 NPZ 隔离，并修复后续实际镜像故障注入发现的三类缺口。

| 情况 | 处理方式 |
|---|---|
| B/C 历史 NPZ/ZIP 损坏、元数据不是 JSON 对象、内部条目不是 NumPy 数组 | 标记历史不可用，继续检查其他允许历史，或保留有效父模型 |
| A/B/C 历史记录的可选产物 ID 为列表、对象、空白等非法类型 | 记录 `history_selection_hint_rejected` 并跳过该提示，仍检查本帧其他授权产物 |
| A/B/C 历史检查点包含无法转为有限数的超大整数 | 识别为检查点格式错误，排除旧历史；实际父模型有同样错误时仍停止 |
| A 历史路由清单、policy、cutoffs 或组件字段类型错误 | 在加载时明确拒绝该产物，错误留在 `historical_validation`，不再泄漏 TypeError/AttributeError |
| 本地数值训练失败 | 记录失败，拒绝局部候选，进入后续合法选版 |
| 完整模型回复不合法 | 有限纠正；耗尽后进入 program_guard |
| 当前父模型、输入或旧状态损坏 | 明确失败，不能换一个模型冒充实际父模型 |
| SDK 恢复、模型、上传、状态或 finish 请求结果未知 | 终止会话，不捕获后继续业务 RPC，也不自动重试 |

坏历史的原因保存在 `decisions.jsonl` 的 `historical_validation[产物ID]`，包括 `error`、`error_type`、`stage` 和 `gate.eligible=false`。

已知文件格式错误在 [检查点加载器](src/gas_a/sensor_training/reference/model.py) 中规范为 ValueError；[B/C 历史复评](src/gas_a/regression/submission.py) 只隔离本地加载和测量错误。SDK 的恢复与摘要校验仍在这个捕获范围之外，没有用 `except Exception` 吞掉所有问题。

## 6. 加载现有 v18 镜像

在项目根目录使用 PowerShell。以下变量也供后面的命令使用。

```powershell
$projectRoot = (Get-Location).Path
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
$dockerExe = 'C:/Users/Administrator/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'
$baselineRoot = 'D:/liyue/工业模型赛题研究'
$sdkArchive = Join-Path $baselineRoot 'contest-harness-sdk-1.0.4.tar.gz'
$v18Tag = 'gas-rsi-abc:wyx-sdk104-v18-parent-manifest-20261125'
$v18Id = 'sha256:3bf46eec43814d6fcaef9e885d7629d8b87507115d87e4c36fc5c6d72522c774'
$v18Archive = Join-Path $projectRoot 'output/submission-wyx-sdk104-v18-parent-manifest-20261125/gas-rsi-abc-wyx-sdk104-v18-parent-manifest-20261125.tar'

& $dockerExe --context desktop-linux load --input $v18Archive
& $dockerExe --context desktop-linux image inspect $v18Tag --format '{{.Id}}'
& $dockerExe --context desktop-linux run --rm --pull=never --network none --read-only $v18Id describe
```

Docker 已在 PATH 时可将 `$dockerExe` 改为 `docker`。镜像已验证可加载；正式入口保持 SDK 原模板：

```text
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

训练需要宿主双向 JSONL 交互，不能用静态 start 文件代替整个会话。SDK 读取一帧 start 后才进入训练，不等待 stdin EOF。预测没有 start 帧，必须在读取下一条样本前输出并刷新本条结果。

预测时平台通过 `CONTEST_PREDICTION_CONTEXT` 和 `CONTEST_ARTIFACT_PATH` 指定任务与主检查点，`model_dir` 是 /model 根目录。A 输出六类分数数组，B/C 输出 `{"outputs":{"value":123.4}}`。stdout 只写协议 JSONL，日志写 stderr。

## 7. 构建和测试当前工作区源码

若在 v18 之后继续修改工作区代码，需要构建新镜像。已有镜像不会自动更新，测试工具的源码一致性检查也会拒绝混用版本。

```powershell
$localImageTag = 'gas-rsi-abc:local-review-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $dockerExe --context desktop-linux build --pull=false --network none -f Dockerfile.offline --build-arg "CONTEST_RUNTIME=$v18Tag" -t $localImageTag .
$localImageId = (& $dockerExe --context desktop-linux image inspect $localImageTag --format '{{.Id}}').Trim()
& $dockerExe --context desktop-linux run --rm --network none --read-only $localImageId describe
```

必须显式传入上述 CONTEST_RUNTIME：Dockerfile.offline 的旧默认值仍引用已清理的 v4。需要从基础环境重新安装依赖时，可使用根目录 Dockerfile 正常构建；它需要准备基础镜像和依赖。

### 本地单元测试与官方 SDK 离线测试

安装环境后运行：

```powershell
if (-not (Test-Path $pythonExe)) { py -3.12 -m venv .venv }
& $pythonExe -m pip install -r requirements.txt
& $pythonExe -B -X utf8 -m unittest discover -s tests -t . -v

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location vendor/contest-harness-sdk-1.0.4
& $pythonExe -B -X utf8 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
Pop-Location
```

SDK wheel 随源码保存在 vendor，来源见 [SDK 清单](vendor/sdk-provenance.json)，不要从 PyPI 搜索同名比赛 SDK。以上测试不调用真实模型；SDK 的 `tests/test_live.py` 是另行运行的真实模型测试，不包含在命令内。

### 当前源码的 A/B/C 并行镜像测试

三份官方包应保持原名称，放在 `$baselineRoot` 下。先完成当前源码构建，再运行：

```powershell
$offlineOut = Join-Path $projectRoot ('runtime/offline-current-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
& $pythonExe -B -X utf8 scripts/benchmark_sdk_agent.py --baseline-root $baselineRoot --sdk-archive $sdkArchive --output $offlineOut --image $localImageTag --expected-image-id $localImageId --rounds 4 --modes parallel --compact-validation --aggregate-parent --docker-executable $dockerExe --docker-context desktop-linux
```

命令显式指定镜像及 ID，不使用测试脚本遗留的 v10 默认值。`--compact-validation` 模拟精简测量摘要，`--aggregate-parent` 令父身份与任一单文件摘要不同，覆盖本次平台暴露的对接场景；去掉两个选项可测试原先的显式摘要、单文件父身份模式。

输出目录必须全新。测试实际执行训练、SDK 分块传输、历史恢复和容器预测，模型选择由明确标记的离线替身产生；不能当作真实 27B 的实验结果。

### 原样复现已经验证的 v15

使用与镜像一致的冻结源码，不能在存在新改动的项目根目录直接测试旧镜像：

```powershell
$v15Source = Join-Path $projectRoot 'runtime/history-guard-v15-20261125/source-final'
$v15Tag = 'gas-rsi-abc:wyx-sdk104-v15-history-guard-final-20261125'
$v15Id = 'sha256:bb7cd7e69b1a5632568db81dd828de9d942e5607d7311192ccdcc9a8a21432e0'
$v15Out = Join-Path $projectRoot ('runtime/v15-repeat-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
Push-Location $v15Source
& $pythonExe -B -X utf8 scripts/benchmark_sdk_agent.py --baseline-root $baselineRoot --sdk-archive $sdkArchive --output $v15Out --image $v15Tag --expected-image-id $v15Id --rounds 4 --modes parallel --compact-validation --rpc-delay-seconds 0.02 --model-delay-seconds 2 --docker-executable $dockerExe --docker-context desktop-linux
Pop-Location
```

源码审查 ZIP 中的 source 也可作为源码目录；官方大数据和基线权重需另行准备。运行单元测试还需要对应公开测试夹具，不能把精简审查包当作完整 runtime。

容器测试固定 Linux、非 root、2 CPU、4 GiB、只读根目录和禁网。测试的工作/临时卷采用 tmpfs，与平台说明中的磁盘卷不完全相同。

## 8. 最新验证结果与历史真实模型实验

### v18：整组父摘要与单文件校验分开

已用 v17 镜像复现：首轮采用四组件路由包，第二轮在首次模型调用前抛出 `No declared parent checkpoint matches parent_checkpoint_sha256`。v18 使用相同的四个检查点字节和测试聚合身份，第二轮已完成续训、提交和结束确认。Linux 项目测试 238 项中 237 项通过、1 项仅限 Windows 的检查跳过；SDK 原包 18 项通过，镜像内 9 个 SDK 模块和 Harness 原模板未改动；30 个逐行预测用例通过。

多轮聚合身份容器测试和独立复核结果见 `runtime/parent-manifest-v18-20261125/verification.json`，交付说明见 v18 目录。本次使用离线模型替身，无真实 27B 调用和云端写入，不能据此宣称正式平台或隐藏集已通过。

A/B/C 各完成 10 轮，30 轮均使用与单文件摘要不同的父身份并收到结束确认；各自完成 9/8/9 次非空历史实验。独立重算 62 组、543733 条预测，A Macro-F1 从 0.714030 到 0.805881（+9.19 个百分点），B RMSE 从 681.159833 到 651.626661（降低 4.34%），C RMSE 从 52.868573 到 51.614535（降低 2.37%）。单轮训练进程最大耗时分别约 12.80/12.84/31.44 秒；其中模型等待为模拟延迟，不代表真实 27B 延迟。

### v17：历史输入隔离修复

v16 实际镜像中复现的 9 个坏历史失败用例，在 v17 中均完成本轮并收到 accepted round_ack。坏历史被排除，有效父模型未被改写；实际父模型损坏、非法公开数据、状态损坏或 SDK 会话失败仍按边界停止。

修复仅改变历史 ID 检查、数值格式检查及路由清单结构检查，未修改官方 SDK、指标、类别保护或训练方案。详细用例、原始协议与完整验收数据位于 `runtime/history-input-v17-20261125/`，交付说明见对应 v17 目录。

### 历史 v15：2026-11-25 离线镜像验证

| 检查 | 结果 |
|---|---|
| 项目测试 | 221 项：220 通过、1 项 Windows 专用检查跳过 |
| 实际镜像故障用例 | 13 个全部完成，覆盖坏历史隔离及坏历史后的有效回滚 |
| 正常多轮运行 | A/B/C 各 4 轮，共 12 轮；各 3 次完整历史实验 |
| 数值一致性 | 检查点与逐条预测和 v14 对应轮次逐字节一致 |
| 路由记录 | 策略、组件摘要及其他字段一致，只有一次父模型预测耗时记录不同 |
| SDK 与归档 | 9 个 SDK 模块、官方入口模板、39 个应用文件及 OCI blob 摘要核验通过 |
| 新增外部调用 | 真实模型调用 0，云端写入 0 |

| 指标 | 同口径公开基线 | v15 四轮结果 |
|---|---:|---:|
| A Macro-F1 ↑ | 0.714029722 | 0.805881336 |
| B RMSE ↓，分钟 | 681.159832604 | 651.626661250 |
| C RMSE ↓，循环 | 52.868572566 | 51.756636508 |

v14 的十轮 C 结果 51.614535 来自另一轮数的实验，不能写成 v15 四轮结果。v15 的证据见 [修复说明](output/submission-wyx-sdk104-v15-history-guard-20261125/修复与使用说明.txt) 和 `runtime/history-guard-v15-20261125/verification-final.json`。

### 最近一次保留的全新真实 27B 实验：v7

2026-11-23 的固定 v7 镜像实验完成 12 轮正常训练及 6 轮故障回滚/继承检查，共 36 次真实 PyTRiO Qwen3.8-27B 调用，提供方报告 225,491 tokens。公开结果为：

- A Macro-F1：0.714029722 → 0.798701650，提高约 8.47 个百分点。
- B RMSE：681.159832604 → 651.885364673，降低约 4.30%。
- C RMSE：52.868572566 → 51.761745579，降低约 2.09%。

它与 v15 离线验证分开统计。详细证据及历史云端链接见 [v7 真实实验报告](docs/FRESH_V7_REAL_TEST_20261123.md)，完整本地记录位于 `runtime/fresh-v7-full-20261123/`。旧 v7 镜像和导出包已清理，重新做原镜像重放需要先恢复当时的固定镜像，不能直接用 v15 冒充。

上述成绩均是公开数据上的本地宿主模拟，不是组委会正式隐藏集成绩。

## 9. 真实 27B 调用与云端记录

真实模型测试会产生费用，并向云端写实验结果，需要单独执行。先安装 requirements-platform.txt，按 .env.example 在宿主配置 PYTRIO_API_KEY 和 SWANLAB_API_KEY。不要把真实密钥写入源码、Git、镜像、候选或实验记录。

需要测试当前工作区时，使用第 7 节新构建的匹配镜像：

```powershell
& $pythonExe -m pip install -r requirements-platform.txt
$liveOut = Join-Path $projectRoot ('runtime/fresh-abc-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
& $pythonExe -B -X utf8 scripts/test_fresh_platform.py --baseline-root $baselineRoot --output $liveOut --image $localImageTag --expected-image-id $localImageId --docker-executable $dockerExe --docker-context desktop-linux
```

该总控对 A/B/C 执行正常训练、故障回滚、继承核验、保存回复重放和云端结果汇总。正常阶段每组至少 4 轮、至多 6 轮，常见流程约 36 次真实调用，上限 66 次；纠正请求也算调用。失败会保留记录，不自动重跑付费请求或借用旧成绩补齐。

测试宿主把真实提供方响应返回 SDK，但不冒充正式比赛宿主或其计费回执。正式平台的密钥、端点和模型绑定由宿主持有，镜像只调用 `rnd.model()`。

当前总控使用 [ResultsOnlyJournal](scripts/results_only_journal.py)，云端保存指标、方案、选版理由、调用号和诊断摘要；原始请求、源码、样本、权重保留在本地。一个 Run 可以包含多轮记录，Run 数量不等于实验轮数；总览 Run 本身不算一次历史实验。

## 10. 记录、核心代码与当前限制

| 记录 | 保存的内容 |
|---|---|
| `protocol.jsonl`、`sdk-timing.json` | 实际 SDK 帧、上传确认、回执引用、结束确认与耗时 |
| `state/model-responses.jsonl` | 完整模型请求、回复、回执与纠正过程 |
| `state/decisions.jsonl` | 规划、选版、program_guard 来源及历史不可用原因 |
| `monitoring.json`、`diagnosis.json` | 漂移与错误诊断 |
| `strategy.json`、`regression_strategy.json` | 配套策略版本、偏好及实验结果 |
| `training.json`、`continuation.json`、`history.jsonl` | 实际父模型、继承检查、训练过程及测量 |
| `measurement-*/predictions.jsonl` | 通过原生预测入口得到的逐条结果 |
| `image-preflight.json` | 镜像 ID、源码/SDK 一致性及隔离检查 |
| `cloud-receipt.json`、`cloud-verification.json` | 云端 Run 与回读核验证据 |

核心阅读顺序：

| 文件 | 作用 |
|---|---|
| [选手模块](contest_submission.py)、[任务分发](src/gas_a/multitask.py) | SDK 回调、任务声明、CPU 线程和 A/B/C 原生分发 |
| [A 主控](src/gas_a/submission.py)、[B/C 主控](src/gas_a/regression/submission.py) | 父模型、历史、AI 规划、训练、选版、状态保存及结束 |
| [策略](src/gas_a/evolution.py)、[A 提示词](src/gas_a/controller_prompts.py)、[B/C 提示词](src/gas_a/regression/prompts.py) | 实验分类、重复限制、历史资格及结构化决定 |
| [A 监测](src/gas_a/monitoring.py)、[B/C 监测](src/gas_a/regression/monitoring.py) | 固定训练参考与错误诊断 |
| [A 续训](src/gas_a/continuation.py)、[B/C 续训](src/gas_a/regression/training.py) | 权重与优化器继承、真实训练及导出证明 |
| [冻结路由](src/gas_a/frozen_router.py)、[宿主核验](src/gas_a/host_validation.py) | 无标签预测、完整模型评估及宿主测量门槛 |
| [元数据限额](src/gas_a/sdk_records.py)、[检查点加载](src/gas_a/sensor_training/reference/model.py) | JSON 字节上限、完整证据和坏文件格式检查 |
| [历史隔离测试](tests/test_regression_history_isolation.py)、[决策保护测试](tests/test_completed_reply_recovery.py) | 正常恢复、应拒绝的情况和 SDK 中断边界 |

当前仍需注意：

1. **完整实验次数没有硬保证。** 只观察、只回滚或 program_guard 回合不能代替三次实际实验。工作区在 experiment 实际训练后选择合格历史，仍可按完整记录统计实验；它与没有训练的回滚回合不同。测试宿主会核验次数，生产主控仍可能因模型选择或条件不足而未达标。
2. **本地训练 30 秒不等于整轮 30 秒。** 模型等待、历史恢复、上传、宿主测量、保存状态都占时间。业务代码尚未统一按 rnd.wall_seconds 调度剩余预算。
3. **协议通过不代表平台一定成功。** SDK 1.0.4 结果未知必须终止；真实网关排队、资源争用及正式宿主行为未由离线替身覆盖。
4. **源码、镜像和实验记录必须配套。** 工具遗留默认镜像仍可能指向已清理的旧版本，务必显式指定标签和 ID。
5. **公开验证调优存在选择偏差。** 本地不低于公开基线、合成故障回滚成功，都不能保证隐藏流必然改善。

SDK 采用 `upload_begin → upload_chunk → upload_finish`，再提交候选或状态事务；finish 收到 `round_ack / accepted=true` 后才结束。当前测试宿主默认按整轮 180 秒及 start 中更小的 wall_seconds 核验，这不是 SDK 对所有平台调用的统一时限，也不保证能抢占已发出的同步宿主操作。

历史材料见 [v10 验收](docs/SDK104_V10_ACCEPTANCE_20261124.md)、[v8 超时排查](docs/IMAGE_V8_TIMEOUT_20261124.md)、[续训修复](docs/BASELINE_SDK_FIXES_20261123.md)、[历史回放边界](docs/LEGACY_REPLAY.md)及 [SDK 原说明](vendor/contest-harness-sdk-1.0.4/HARNESS_SDK.md)。这些报告描述各自当时版本，不代表当前镜像仍保留，也不代表所有旧问题已经由后续版本彻底消除。
