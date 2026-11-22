# Gas RSI · 工业气体分类自进化系统

本项目面向任务 A：对 **128 维特征（16 个传感器，每个 8 维）**进行六类气体分类。正式入口使用 **contest-harness SDK 1.0.4**，支持从平台实际父产物续训、公开特征漂移监测、诊断、实验选版和历史版本回滚。

目前已通过选手侧离线功能与协议测试，并完成本地 SDK 1.0.4 桥接真实 Qwen3.8-27B 的连续回合及单独回滚演练。Docker 实际沙箱、正式宿主模型绑定和回归、比赛隐藏集成绩仍待验证；本地测试不代表正式评测得分。

## 文档导航

| 文档 | 用途 |
|---|---|
| 本 README | 项目入口、环境、常用命令、当前验证结果与待补项 |
| [参数说明](SUBMISSION_CONFIG_GUIDE.md) | `submission_config.json` 全部 72 个参数路径的含义、代码位置和生效限制 |
| [27B 控制器提示词](docs/CONTROLLER_PROMPT_V2.md) | 当前 v3 文案、动态证据、硬门槛及真实模型 A/B 边界 |
| [审查问题修复说明](docs/CODE_FIXES_20261122.md) | 宿主分数保护、本地失败恢复、验收退出码、home 轮数与原生审计 |
| [平台验收说明](PLATFORM_VALIDATION.md) | 正式闭环、离线演练、Docker 验收和真实平台验证边界 |
| [云端连通性与异常收尾](docs/LIVE_PLATFORM_CHECK.md) | 真实连通性检查、SwanLab 结束状态核验、遗留运行的只读排查 |
| [真实 27B 的 SDK 测试](docs/REAL_SDK_REHEARSAL.md) | `rnd.model()` 的真实回复桥接、多轮 CPU 训练与回执，及正式宿主边界 |
| [历史回放说明](docs/LEGACY_REPLAY.md) | 固定 batch3–10 回放、旧配置、归档权重重建和历史指标口径 |
| [SDK 官方说明](vendor/contest-harness-sdk-1.0.4/HARNESS_SDK.md) | 宿主协议、模型访问、挂载目录、分块上传和 SDK 接口 |

## 1. 快速开始

以下本地命令在项目根目录使用 PowerShell 执行。项目环境基线为 Python **3.12**，当前 `.python-version` 指定 **3.12.13**；依赖为 NumPy **1.26.4** 和随包 SDK **1.0.4**。

```powershell
# 新工作区尚无虚拟环境时创建；已有环境可直接使用
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.12 -m venv .venv
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B run.py describe
```

SDK wheel 位于 `vendor/contest_harness-1.0.4-py3-none-any.whl`，由用户提供的官方源码归档离线构建，来源与摘要见 [sdk-provenance.json](vendor/sdk-provenance.json)。不需要从 PyPI 查找同名比赛 SDK；安装 NumPy 仍需可用包源或本地 wheel 缓存。

`run.py` 不带参数时执行 `describe`。也可通过 `start.ps1` 使用项目虚拟环境，例如 `.\start.ps1 describe`。当前工作区 `.venv` 依赖项目内 `.python` 的解释器，迁移环境时应重新创建虚拟环境。

## 2. 入口怎么选

| 入口 | 用途 | 输入和结果 |
|---|---|---|
| `harness.py describe` | 查看 SDK 协议和任务声明 | 输出描述 JSON，不执行训练 |
| `harness.py train` | 平台正式训练回合 | 宿主双向 JSONL 协议、授权公开数据、实际父产物、模型网关 |
| `harness.py predict` | 平台逐行预测 | 选中产物及逐行特征；每行立即返回六类分数 |
| `run.py train-local` | 本地数值续训、生成候选 | 显式基线或已选检查点及公开数据；不调用决策大模型 |
| `run.py predict-local` | 本地逐行预测 | 显式检查点及 JSONL 请求 |
| `run.py evaluate` | 本地公开验证评估 | 显式检查点及带标签的公开验证集 |
| `scripts/validate_platform.py` | 离线宿主替身演练 | 执行真实入口和 SDK，模拟模型决策与宿主，不调用真实模型 |
| `run.py replay` | 固定历史数据回放 | 归档权重、`config.json` 和 batch3–10，依赖批后标签反馈 |

`run.py describe/train/predict` 委托 SDK；`rebuild_models.py` 的命令行等同于 `train-local`。平台 `train` 需要宿主持续交互，不能用单个静态 start 文件或等待 EOF 代替。

## 3. 正式 Harness 的运行逻辑

容器约定入口：

```sh
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

协议为 `contest.harness.workspace.v2`，模型访问方式为 `host_gateway_rpc`，项目只声明 `classification`，实现任务 A。

```text
宿主实际父产物 + 当轮授权公开数据 + 历史/状态
    ↓
核验父文件摘要，评估当前父模型
    ↓
公开特征漂移监测、错误诊断、允许的历史版本恢复与复评
    ↓
宿主模型选择 observe / experiment / rollback
    ├─ observe：保留父产物，不训练
    ├─ experiment：从实际父产物续训一个受限方案，评估并提交候选
    └─ rollback：提出当前公开验证达标的历史产物
    ↓
按最终门槛限定可选版本，模型确认选版
    ↓
保存 SDK 状态，提交真实调用/历史引用，等待 round_ack
```

训练起点以宿主本轮父产物清单和 `parent_checkpoint_sha256` 为准，保留父权重、Adam 一二阶矩、seed、标准化参数及累计计数器。未被选中的候选不能自动成为下一轮父模型。路由父产物按各自组件继续训练，历史恢复通过 SDK 核验完整文件和摘要。

梯度、训练统计、Fisher 特征、KNN 记忆和成对分类头只使用授权 train。公开 validation 的标签可以用于评估、选 epoch、诊断和选路。样本 ID 不作为推理特征，也不用于推断隐藏标签。训练容器没有隐藏集标签或遥测回传通道，公开数据监测结果不能描述成隐藏流监测结果。

计划和选版均通过 `rnd.model()` 调用宿主模型，官方提示由宿主附加。默认成功返回但内容不合法的决策最多纠正一次；RPC 发出后若中断，立即结束本轮，不重试结果未知的调用、上传或恢复操作。

## 4. 本地训练、续训与评估

官方基线包示例为 `D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1`。以下生成一个独立候选目录，不代表自动上线：

```powershell
$baselineDir = 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1'
$runRoot = Join-Path 'runtime' ('local-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$candidateDir = Join-Path $runRoot 'candidate'

.\.venv\Scripts\python.exe -B run.py train-local --baseline-dir $baselineDir --experiment frozen_router --output $candidateDir
.\.venv\Scripts\python.exe -B run.py evaluate --checkpoint "$candidateDir/checkpoint.npz" --validation "$baselineDir/public/validation.jsonl"
```

`--baseline-dir` 会核验基线包 `baseline-info.json` 中的检查点摘要。也可使用 `--from-baseline 文件`，此时必须同时提供 `--train`、`--validation`。输出目录必须尚不存在。

继续训练时，先把 `$selectedCheckpoint` 明确设置为**实际选中的**检查点；如果上一轮候选被拒绝，应继续使用原先的已选版本：

```powershell
# 这里示例保留官方基线；根据真实选版结果设置路径
$selectedCheckpoint = Join-Path $baselineDir 'checkpoint.npz'
$resumeOutput = Join-Path 'runtime' ('resume-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))

.\.venv\Scripts\python.exe -B run.py train-local --resume $selectedCheckpoint --train "$baselineDir/public/train.jsonl" --validation "$baselineDir/public/validation.jsonl" --experiment conservative_head --output $resumeOutput
```

本地 `--epochs` 可在上限内覆盖所选主实验的新增轮数。默认本地方案为 `frozen_router`，其他菜单项为 `four_sensor`、`five_sensor`、`conservative_head`。

本地输出包括 `checkpoint.npz`、`training.json`、`history.jsonl`、`continuation.json`、`local_run.json`；基线证据保存到 `parent_evidence/`。`local_run.json` 标记 `selection_status=candidate_only`，其父模型 F1 下限检查不等于正式 controller 的完整晋升门槛。

## 5. 预测接口与模型产物

预测请求每行包含 `task=classification` 和 `inputs.features`，数组必须是 **128 个有限数值**。输出每行是 `outputs` 数组，包含六个英文类别及 `score`，类别顺序为：

| 标签 | 气体 |
|---|---|
| `Ethanol` | 乙醇 |
| `Ethylene` | 乙烯 |
| `Ammonia` | 氨气 |
| `Acetaldehyde` | 乙醛 |
| `Acetone` | 丙酮 |
| `Toluene` | 甲苯 |

使用一条授权公开样本构造不含答案的预测请求：

```powershell
$baselineDir = 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1'
$validationFile = Join-Path $baselineDir 'public/validation.jsonl'
$selectedCheckpoint = Join-Path $baselineDir 'checkpoint.npz'

Get-Content -LiteralPath $validationFile -Encoding utf8 -TotalCount 1 |
    ForEach-Object {
        $row = $_ | ConvertFrom-Json
        @{ task = 'classification'; inputs = $row.inputs } | ConvertTo-Json -Depth 10 -Compress
    } |
    .\.venv\Scripts\python.exe -B run.py predict-local --checkpoint $selectedCheckpoint
```

已有 JSONL 请求文件时，可执行 `run.py predict-local --checkpoint 检查点路径 --input 请求文件路径`。预测不读取标签、不等待批次反馈、不写文件；每条结果在下一条输入和 EOF 之前刷新。

正式预测优先使用 `CONTEST_ARTIFACT_PATH`。未提供时可由上下文选择唯一声明的检查点；没有上下文声明时才使用 `model_dir/checkpoint.npz`。多检查点产物需宿主明确指定主文件。

冻结路由产物必须整体保存和挂载，通常包含：

```text
checkpoint.npz              四传感器主专家
routing.json                冻结策略、组件摘要和公开验证校准记录
routing/
    five_sensor.npz         五传感器辅助专家
    home_trunk.npz          home 神经分支
    home_router.npz         home KNN/成对头分支
routing_training.json       各辅助分支的父产物与训练证据
```

推理加载时核验所用组件。仅复制主 `.npz` 会丢失路由依赖。已导出的融合权重、启用类别组合及阈值固定在产物中；平线检测和可选偏移修正只维护当前预测进程的特征缓冲，不接收真实标签。

## 6. 配置与当前实现限制

正式配置为 [submission_config.json](submission_config.json)，逐项说明见 [SUBMISSION_CONFIG_GUIDE.md](SUBMISSION_CONFIG_GUIDE.md)。根目录 `config.json` 属于历史回放。

| 默认配置 | 当前值 | 作用 |
|---|---:|---|
| `max_output_tokens` | 2048 | 每次决策输出预算，受宿主上限约束 |
| `controller.minimum_macro_f1_gain` | 0.0005 | 整版候选/历史版本相对当前父模型的最低公开验证 F1 增益 |
| `controller.minimum_class_support` | 20 | 参加类别召回保护的最低真实类别样本数 |
| `controller.maximum_class_recall_drop` | 0.20 | 受保护类别召回最多下降 20 个百分点 |
| `controller.max_history_restores` | 2 | 每轮最多恢复并尝试评估的历史产物数 |
| `controller.max_repeat_rejections` | 2 | 相同父模型、数据和配置组合被拒绝后的重复限制 |
| `limits.max_epochs` | 32 | 主/五传感器/home 各组件每次训练允许的最大新增 epoch 数 |
| `limits.max_seconds_per_candidate` | 30 秒 | 本地候选训练预算，按训练/阶段检查点检查 |
| `limits.select_best_epoch` / `early_stopping_patience` | true / 6 | 导出最佳 epoch，满足最低执行轮数后允许早停 |

这些是项目策略值，20 个百分点不是比赛明文门槛。正式 `train` 现已把同一门槛接入最佳 epoch 导出，优先选择合格 epoch，仍由最终完整候选评估决定是否上线；直接 `train-local` 保持独立行为。当前仍有以下实现边界：

- `routing.home.epochs` 已按配置执行新增输出头训练轮数，范围为 `1～limits.max_epochs`；保持隐藏层参数及其 Adam 矩、标准化状态不变，实际轮数写入 `routing_training.json`。
- 30 秒不是整轮 Harness 的硬中断时限，模型 RPC、历史恢复、上传等独立消耗时间，部分路由阶段到下一个检查点才发现超时。
- 当前允许搜索因果特征偏移修正，该分支会从已到达特征估计偏移；其是否满足官方“拟合统计量只使用 train”的边界，仍需主办确认。

修改配置后，已有权重和路由策略不会追溯改变；容器提交需重新构建镜像。

## 7. 离线测试与十轮演练

### 项目功能测试

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

测试覆盖实际父产物续训、完整路由恢复、公开特征监测、观察/实验/回滚、策略状态、JSON 纠错、RPC 中断不重试、逐行预测和写入路径限制。模型与宿主使用离线替身。2026-11-22 审查问题修复的测试与模拟记录在 `runtime/code-fixes-20261122/`，结果详见[修复说明](docs/CODE_FIXES_20261122.md)。

正式选版还须通过宿主测量校验：候选回执需含可识别的公开验证范围、相同验证集摘要和 `metrics.macro_f1`，与本地结果明显冲突、分数退化或口径不明时不能上线。宿主完全没有父测量时，明确记录使用本地父评估；宿主已给出但不能核验的父测量不能被本地分数覆盖。SDK 未冻结摘要字段结构，正式宿主若采用其他格式，需在 `host_validation.py` 显式适配；详见修复说明。

本地训练超时或已知数值错误会写入 `experiment_attempts.jsonl`，再交由模型依据失败证据选择父产物并完成本轮；相同失败计入重复限制。模型、上传和提交 RPC 的未知结果仍会立即终止会话。真实测试工具现在以 `requirements_passed` 判断是否达标，未满足三次完整实验等要求时返回退出码 1，不再仅因回合跑完就返回成功。

### 官方 SDK 离线测试

在项目根目录执行下列 PowerShell 命令，显式限定三个离线模块：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location vendor/contest-harness-sdk-1.0.4
try {
    ..\..\.venv\Scripts\python.exe -B -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
} finally {
    Pop-Location
}
```

Linux/macOS 则在 SDK 目录运行：

```sh
python3 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
```

这三个模块只使用 JSONL 宿主替身。官方宿主仓库中的 `runtime/tests/test_sdk_host_contract.py` 未包含在 SDK 下载包或本项目中，当前没有该项宿主验证证据。

`tests/test_live.py` 单独运行。设置 `PYTRIO_API_KEY` 后可能产生真实调用费用；`CONTEST_PI` 可指向精简 pi 的 `pi.mjs` 以测试 SDK 传输层握手。运行前应确认密钥、模型及环境，默认离线命令不会执行它。SDK `example/contest_submission.py` 只演示协议和模型调用，不能证明工业模型成绩。

### 十轮原生 SDK 演练

```powershell
$baselineDir = 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1'
$rehearsalOutput = Join-Path 'runtime' ('sdk-rehearsal-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))

.\.venv\Scripts\python.exe -B scripts/validate_platform.py --parent "$baselineDir/checkpoint.npz" --train "$baselineDir/public/train.jsonl" --validation "$baselineDir/public/validation.jsonl" --output $rehearsalOutput --rounds 10
```

该脚本使用真实 Harness 子进程、SDK 分块协议、数值训练及状态传递，宿主决策模型是离线替身。默认实验为 `four_sensor`，第 3 轮主动观察、第 4 轮注入一次已成功返回的错误 JSON；被重复限制的实验不再强行执行。它不会调用平台模型或消耗正式试运行额度。

输出含 `validation-summary.json`、`prediction-check.json`、各回合协议/日志、候选及 SDK 状态。末尾验证十条预测逐行刷新且与选中产物一致。合成故障下的实际历史回滚另由项目测试覆盖，合成分数不作为官方数据成绩。

### 按验收条件模拟：至少三次完整实验记录

另有专用模拟器，根据已读历史依次尝试尚未测过的菜单方案，通过真实逐行预测接口测量全部公开验证样本。默认执行两组独立的十轮运行，比较候选权重、预测摘要、选版和分数，并在独立目录注入权重故障验证历史回滚及下一轮父产物。

```powershell
$baselineDir = 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1'
$acceptanceOutput = Join-Path 'runtime' ('acceptance-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
.\.venv\Scripts\python.exe -B scripts/simulate_competition.py --baseline-dir $baselineDir --output $acceptanceOutput --rounds 10 --replicas 2
```

输出包含 `SIMULATION_REPORT.md`、`simulation-report.json`、逐轮完整协议/决策/文件访问记录、逐条请求和预测、官方输入副本及 `reproduction/` 源码与复现说明。首轮历史为空不计入“三次非空历史实验”统计。应用审计阻止项目归档数据、未授权数据目录和联网访问，但不等于 Docker 或官方沙箱隔离。

旧版对照见 [模拟报告](runtime/competition-acceptance-20261122/SIMULATION_REPORT.md)：当时最终公开验证分数与基线相同，正常候选晋升为零。该记录保留，不用新版本结果覆盖。离线调用号不能作为平台真实模型回执，故障演练结果与正常训练成绩分别记录。

27B 提示词 v2 与 epoch 门槛修复后的新结果见 [调优报告](runtime/controller-v2-validated-20261122/TUNING_REPORT.md)：两组十轮都从 **0.714029722** 提升至 **0.800045300**，每组 5 次正常候选晋升，权重、逐条预测及选版一致；项目 53 项测试和 SDK 离线 18 项通过。此收益包含训练选版代码修复，不是已验证的真实 27B 提示词独立收益。最终乙醇召回低于初始基线，逐轮召回保护也不代表相对初始基线的累计保护。

v2 代码复测见 [平台模拟复测报告](runtime/platform-check-v2-20261122/SIMULATION_REPORT.md)：两组十轮和源码快照的独立十轮一致，每组 9 次非空历史驱动实验、5 次上线；独立重算 57 组测量、70,908 条保存预测，并验证故障回滚后的父产物。完整材料、已选模型和复现源码位于同一运行目录。模型调用仍为离线替身，不能替代真实平台回执。

审查问题修复后的当前 v3 结果见 [修复说明](docs/CODE_FIXES_20261122.md)：105 项项目测试及 18 项 SDK 离线测试通过；两组独立十轮的权重、预测与选择一致，公开 Macro-F1 均为 **0.714030 → 0.800045**，独立故障回滚通过。宿主候选分数为 0 的冲突探针已正确保留父模型。独立重算 34 组、42,296 条保存预测，详见 [核验结果](runtime/code-fixes-20261122/verification.json)。本轮没有新增真实模型调用。

## 8. 已有验证结果与待完成事项

最新真实 27B 测试：四轮正常 SDK 回合使用 8 次真实回复，形成 3 次含非空历史的完整实验记录，公开 Macro-F1 为 **0.714030 → 0.798702**；另两轮真实回滚演练使用 4 次回复，主动恢复历史产物并确认下一轮父模型一致。加上修复前两次推理耗尽 Token 的失败回复，共 14 次实际调用。原始 API 响应与 SDK 回执逐条匹配，工作区和保存源码中的 85 项项目测试均通过。详见 [真实 SDK 测试说明](docs/REAL_SDK_REHEARSAL.md) 与 [本次核验](runtime/real-27b-sdk-20261122-results/verification.json)。

下面保留 **2026-11-22 较早的验收快照**，详见 [completion-audit.json](runtime/rsi-completion-20261122/completion-audit.json) 和 [sdk-offline-verification.json](runtime/rsi-completion-20261122/sdk-offline-verification.json)。`runtime/` 不纳入 Git，其他工作区需重新生成记录。

| 检查 | 已记录结果 | 可证明的范围 |
|---|---|---|
| 项目测试 | 44 项通过 | 已覆盖的本地功能及协议行为 |
| SDK 指定离线模块 | 18 项通过，未执行 live 测试 | SDK 与 JSONL 替身的回归行为 |
| 原生 SDK 演练 | 10 轮完成 | 连续回合、状态传递、观察及回复纠错等 |
| 逐行预测 | 10 条通过 | 在下一条输入/EOF 前响应，并与选中产物一致 |
| Docker 实际运行 | 尚未验证 | 当前记录中的机器未安装 Docker |
| 获准 PyTRiO 模型、官方宿主与正式隐藏集 | 该旧快照尚未验证 | 当时没有真实调用、宿主回归或正式成绩证据；最新本地真实调用见上文 |

这次演练中，四传感器候选的公开验证 Macro-F1 从 **0.714030** 提升至 **0.750708**，但乙醇召回从 **158/164（96.34%）**降至 **125/164（76.22%）**，下降 **20.12195 个百分点**，超过项目保护线。因此候选被拒绝，十轮最终保留父模型；这不是已证明性能提升的结果。

旧版漏选第 44 个合格 epoch 的衔接缺口已在 v2 修复，历史记录和旧版分数保留用于对照。上表为旧验收快照，最新结果见前述 v2 调优报告。按用户范围，《实验记录规范》的完整性不在本次审核结论内。

## 9. Docker 构建与沙箱验收

Dockerfile 使用 Python 3.12 slim、非 root 用户和单线程数值计算。`.dockerignore` 仅允许正式代码、平台配置、依赖和 SDK 1.0.4 wheel 进入构建上下文；`models/`、`data/`、`runtime/`、本地环境及账号文件不进入镜像。父产物、公开数据和模型网关由宿主提供。

在已安装并启动 Docker 的机器上执行：

```powershell
docker build --platform linux/amd64 -t gas-a-rsi:sdk104 .
docker run --rm --network none --read-only --tmpfs /tmp:rw,mode=1777 gas-a-rsi:sdk104 describe

$baselineDir = 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1'
$containerOutput = Join-Path 'runtime' ('container-rehearsal-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
.\.venv\Scripts\python.exe -B scripts/validate_platform.py --image gas-a-rsi:sdk104 --parent "$baselineDir/checkpoint.npz" --train "$baselineDir/public/train.jsonl" --validation "$baselineDir/public/validation.jsonl" --output $containerOutput --rounds 10
```

脚本以禁网、只读根文件系统、2 CPU、4 GiB 内存及独立可写 tmpfs 运行；train 只读挂载授权数据、父产物和状态，predict 只挂载选中模型。该检查仍使用离线模型替身。构建镜像需要预备基础镜像及依赖，正式运行不下载模型或安装依赖。

正式运行时 `/app`、`/previous`、`/data`、`/state`、`/model` 按各阶段只读挂载，训练写入 `/workspace`。本地替身可设置 `CONTEST_WORK`，其位置必须在代码、数据、模型及状态目录之外。

SDK 1.0.4 要求宿主声明 `artifact_upload_protocol=chunked-v1`，不支持时本项目提前报 `CHUNKED_UPLOAD_UNSUPPORTED`。候选与状态按摘要清单分块上传，上传期间不能改文件；具体容量以当轮 `rnd.limits` 为准。默认单次最多 64 个普通文件、512 MiB，每轮候选累计 1 GiB、状态累计 512 MiB。项目模型加载器另有单检查点 16 MiB、单 JSONL 128 MiB 的能力上限。

完成真实提交前，仍需在平台绑定获准的 PyTRiO 训练模型/Sampler 或 PyTRiO Qwen3.8-27B，核验真实模型回执，执行正式平台连续回合、资源与同口径基线评测。`configuration.controller_model_name` 只记录宿主声明的名字，不能单独证明来源合规。镜像导出及上传方式以平台要求为准。

## 10. 代码位置

| 文件/目录 | 当前职责 |
|---|---|
| [harness.py](harness.py)、[contest_submission.py](contest_submission.py) | SDK CLI 与任务模块入口 |
| [submission.py](src/gas_a/submission.py) | 正式训练回合、模型决策、历史恢复、选版与预测回调 |
| [continuation.py](src/gas_a/continuation.py) | 从实际父检查点续训，保存父子状态和摘要证据 |
| [monitoring.py](src/gas_a/monitoring.py) | train 来源的固定参考、公开特征漂移监测及诊断 |
| [evolution.py](src/gas_a/evolution.py) | 最终晋升门槛、实验重复限制和策略版本 |
| [controller_prompts.py](src/gas_a/controller_prompts.py) | 27B 计划/选版提示词、数值准入线、历史反馈与版本摘要 |
| [frozen_router.py](src/gas_a/frozen_router.py) | 路由包构建、公开验证校准、完整产物校验和无标签流式预测 |
| [home_training.py](src/gas_a/home_training.py) | home 输出头、训练内特征选择、KNN/成对头和阈值拟合 |
| [sensor_training/](src/gas_a/sensor_training) | 数据契约、模型加载、数值训练、Adam 与逐 epoch 选择 |
| [local.py](src/gas_a/local.py)、[run.py](run.py) | 本地命令和分发 |
| [validate_platform.py](scripts/validate_platform.py)、[tests/](tests) | 离线宿主演练及功能回归测试 |
| [simulate_competition.py](scripts/simulate_competition.py)、[simulation_audit.py](scripts/simulation_audit.py) | 验收条件模拟、独立复现、逐行测量、故障回滚及原生进程访问审计 |
| [live_platforms.py](scripts/live_platforms.py) | 本地 PyTRiO / SwanLab 连通性检查、异常收尾及云端状态查询，不属于正式镜像入口 |
| [replay.py](replay.py)、[router.py](src/gas_a/router.py)、[config.json](config.json) | 历史批次回放与标签反馈路由 |

## 11. 历史回放

归档回放命令为 `.\.venv\Scripts\python.exe run.py replay`，依赖本地归档模型与数据。它固定读取 batch3–10，在每批提交预测后揭晓标签并更新后续路由，不能作为 SDK 独立预测入口。

历史目标值 **0.649151222** 来自 12,221 条归档样本的合并混淆矩阵、固定六类 Macro-F1，与正式训练端“真实或预测中出现的类别”的平均口径有区别。该指标属于已使用历史样本的开发回放，不是当前平台隐藏集成绩。

详细回放命令、数据和模型关系、重建限制及旧闭环工具见 [历史回放说明](docs/LEGACY_REPLAY.md)。
