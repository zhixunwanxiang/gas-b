# Gas RSI · 四传感器路由

正式 Harness 已增加公开特征漂移监测、诊断、观察/实验/历史回滚动作、受限策略版本和安全 JSON 纠错。
默认选版加入类别召回保护，旧的“Macro-F1 上升即采用”不再足够。
当前行为、十轮演练命令与尚待平台核验的内容以 [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md) 为准。
`submission_config.json` 的全部参数含义、具体代码位置和当前生效限制见 [SUBMISSION_CONFIG_GUIDE.md](SUBMISSION_CONFIG_GUIDE.md)。

## 官方基线续训与动态入口

`run.py` 默认执行 `describe`；`train`、`predict` 委托比赛 SDK。
固定 batch3–10 的回放需要显式执行 `run.py replay`，实现位于 `replay.py`。
`rebuild_models.py` 现在等同于 `run.py train-local`，要求显式指定续训起点和数据，
不再默认复制项目 `models/start.npz` 或 `models/reference.npz`。

用户提供的 `D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1` 可作为首轮基线包。
本地首轮读取其中的 `checkpoint.npz`，核对该包 `baseline-info.json` 声明的摘要，
读取公开 train/validation，保留原权重、Adam 一阶与二阶矩、seed、标准化和累计步数。
下一轮使用实际选中的检查点；某轮候选未被采用时继续使用此前选中的版本。

```powershell
$baselineDir = 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1'
$roundRoot = Join-Path 'runtime' ('baseline-chain-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))

# 首轮：从指定官方基线生成候选，输出目录必须尚不存在
.\.venv\Scripts\python.exe run.py train-local --baseline-dir $baselineDir --experiment frozen_router --output "$roundRoot/round-1"

# 仅当前一轮候选实际被选中时，下一轮才使用这个检查点
.\.venv\Scripts\python.exe run.py train-local --resume "$roundRoot/round-1/checkpoint.npz" --train "$baselineDir/public/train.jsonl" --validation "$baselineDir/public/validation.jsonl" --experiment conservative_head --output "$roundRoot/round-2"

# 任意已选检查点和当前公开验证文件，均不绑定归档哈希或批次编号
.\.venv\Scripts\python.exe run.py evaluate --checkpoint "$roundRoot/round-2/checkpoint.npz" --validation "$baselineDir/public/validation.jsonl"
.\.venv\Scripts\python.exe run.py predict-local --checkpoint "$roundRoot/round-2/checkpoint.npz" --input requests.jsonl
```

`requests.jsonl` 每行格式为 `{"task":"classification","inputs":{"features":[128个数值]}}`，
其中数组须替换为实际特征。也可省略 `--input` 从 stdin 逐行读取。
`--from-baseline 路径` 可直接指定首轮检查点，但必须同时提供 `--train`、`--validation`。
`--resume` 同样要求当轮的两个授权数据路径；不会自动回退到项目自带数据。
`--epochs` 可在既定预算内覆盖本地训练轮数。

每个新训练目录保留 `training.json`、`history.jsonl`、`checkpoint.npz`，新增
`continuation.json` 记录实际父产物摘要、训练前后参数和 Adam 矩的摘要、累计步数、
标准化状态以及当轮数据摘要和样本数。首轮的 `parent_evidence/` 原样保存基线包的
历史说明和训练证据；不改写这些历史记录，也不要求新权重与旧归档逐字节一致。
本地命令执行数值训练、产生候选，不调用平台模型或宣称取得正式成绩。
`local_run.json` 标记 `selection_status=candidate_only`，并报告候选是否达到父模型的
公开验证 Macro-F1 下限；产出检查点本身不代表该候选已被选用。

## 比赛 Harness 接入

平台入口已独立于下文的固定批次回放入口：

```text
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

`describe` 声明协议 `contest.harness.workspace.v2`、模型访问方式
`host_gateway_rpc`，且只声明 `classification`。`contest_submission.py`
调用 `src/gas_a/submission.py` 实现任务 A；SDK 负责协议 JSONL 的读写和逐行刷新。

`predict` 从 `CONTEST_ARTIFACT_PATH` 读取被选中的检查点；未设置时使用预测上下文
清单声明的唯一检查点，缺少声明时才读取 `model_dir/checkpoint.npz`。多个检查点
必须由宿主明确指定。检查点可以使用不同名称或子目录。每行仅接收 `task=classification` 和
`inputs.features` 的 128 个有限数值，立即返回六个英文类别及概率分数。
预测不读取标签、不等待批次反馈、不写文件，支持官方 NumPy MLP、训练模块显式迁移的检查点，
以及新导出的冻结路由产物。冻结路由的主检查点是 `checkpoint.npz`；SDK 同时挂载完整产物目录，
其中包含 `routing.json` 和 `routing/` 下的辅助权重。仅复制主 `.npz` 不能代替完整的路由产物。
旧回放目录里的 KNN/多专家检查点需要通过新的训练流程生成合法产物，不能直接套用固定回放配置。

`train` 每次处理平台发来的一轮：通过 SDK 核验实际父检查点，读取当轮授权的
公开 train/validation 和历史记录，先监测漂移并诊断，再由宿主模型选择观察、受限实验或历史回滚，
依据同一当轮公开验证集的实测结果和保护门槛确认选版，保存原生状态并等待结束确认。
十轮调度由平台负责。梯度只使用 train，validation 只用于评估和选 epoch。
父产物以当轮清单和 `parent_checkpoint_sha256` 为准；每个声明文件通过 SDK 验证后，
按指定摘要选择真正的父检查点。此前保存的状态和未选中的候选不会覆盖这个起点。
当轮 train/validation 路径来自 SDK 的 public projection，允许样本和文件摘要逐轮变化。
菜单在 `submission_config.json` 中，包括冻结路由、四传感器、五传感器和保守输出层更新；
各实验沿用父权重、Adam、标准化参数及累计步数。候选的本地验证 Macro-F1
低于父模型时不能被选用；默认还要求至少提升 0.0005，并限制有足够样本的类别召回降幅。
成功收到但格式不合法的回复最多纠正一次；RPC 中断不重试。不会伪造模型回执或决策。

所有训练输出放在 `/workspace` 下的独立新目录；本地宿主替身可用
`CONTEST_WORK` 指定其他可写目录，但不能位于代码、数据、父产物或状态挂载中。
镜像只包含代码、独立的平台配置、NumPy 和 SDK，不包含 `models/`、`data/`、
`runtime/`、本地环境或账号文件。宿主提供父模型、数据和模型网关。

SDK 1.0.4 来源于用户提供的归档，原件和解包源码位于 `vendor/`，归档摘要记录于
`vendor/sdk-provenance.json`。随包 wheel 由这些源码离线构建；依赖不需要从
PyPI 寻找同名比赛 SDK。在项目根目录安装并检查：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe harness.py describe
.\.venv\Scripts\python.exe -m pip check

# 将一条公开样本转换为不含答案的预测请求
$env:CONTEST_ARTIFACT_PATH = (Resolve-Path models\reference.npz).Path
.\.venv\Scripts\python.exe -c "import json; r=json.loads(open('data/public/validation.jsonl',encoding='utf-8').readline()); print(json.dumps({'task':'classification','inputs':r['inputs']}))" | .\.venv\Scripts\python.exe harness.py predict

# 新接入和冻结路由的离线集成测试，不调用真实大模型
.\.venv\Scripts\python.exe -B -m unittest tests.test_submission tests.test_frozen_router tests.test_evolution -v

# 官方 SDK 离线测试，在 SDK 源码目录运行
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location vendor\contest-harness-sdk-1.0.4
..\..\.venv\Scripts\python.exe -B -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
Pop-Location
```

SDK 1.0.4 仍使用 `contest.harness.workspace.v2`，但上传采用 `chunked-v1`。
宿主必须在 `rnd.limits.artifact_upload_protocol` 声明此能力；旧宿主会在本项目训练开始时
收到 `CHUNKED_UPLOAD_UNSUPPORTED`，需要平台升级宿主，不会回退到旧版内联上传。
候选与状态由 SDK 先计算文件清单及摘要，再分块上传并等待宿主验证回执。
应用在 `submit_candidate()` 返回前不修改候选目录，保存状态同样使用分块上传。

新版上限为单次候选/状态各 64 个文件、512 MiB；每轮候选累计 1 GiB、状态累计
512 MiB，以当轮 `rnd.limits` 为准。工作环境为 2 核 CPU、4 GiB 内存，工作卷 4 GiB，
输出和临时卷各 1 GiB。本项目保持单线程；当前小型 A 题模型无需扩大检查点格式。
模型加载器自身仍限制单个检查点 16 MiB、单份 JSONL 128 MiB，这些属于本模型的能力边界，
不代表已经支持任意 512 MiB 模型。镜像和大文件在实际沙箱中的资源表现仍需平台测试。

RPC 发出后遇到信号、超时或异常，宿主处理结果可能未知，SDK 会封闭本轮会话。
本项目让该异常向外传播并结束运行，不会重试上传、提交候选或继续调用模型。
SDK 后续调用会报 `RPC_OUTCOME_UNKNOWN`。正常回执之后的模型文本若不符合决策 JSON
约定，当前仍明确失败，没有自动纠错重试；它属于模型回复稳健性问题，不能通过重试
未知结果的 RPC 处理。新测试覆盖分块偏移/摘要、宿主动态额度、旧宿主拒绝，以及中断后停止。

`vendor/` 中保留的 SDK 1.0.2 归档和旧运行快照仅用于历史追溯；当前环境、依赖、
镜像和测试均使用 1.0.4，Docker 构建上下文只允许新版 wheel。

新版说明明确要求拟合统计量只使用 train。当前标准化参数、Fisher 特征、KNN 记忆和
二分类头均遵守这一边界；可选因果漂移校正仍会从预测流估计中位偏移，是否属于该约束下的
统计量拟合需要主办明确。在确认前可把 `routing.selection.correction_choices` 限为 `[false]`。
本次 SDK 1.0.4 复核导出的候选实际选中 `feature_correction=false`。
文中旧 `runtime/` 路径属于历史运行记录，当前工作区不一定保留这些目录；新验证需重新生成产物。

安装 Docker 后，在项目根目录构建 Linux/amd64 镜像并导出：

```powershell
docker build --platform linux/amd64 -t gas-a-harness:1.0.0 .
docker run --rm --read-only --network none --tmpfs /tmp gas-a-harness:1.0.0 describe
docker save -o runtime/gas-a-harness.tar gas-a-harness:1.0.0
```

只读预测检查可在前面的样本转换命令后，将管道右侧替换为：

```powershell
docker run --rm -i --read-only --network none --tmpfs /tmp --mount "type=bind,source=$((Resolve-Path models).Path),target=/model,readonly" -e CONTEST_ARTIFACT_PATH=/model/reference.npz gas-a-harness:1.0.0 predict
```

`train` 需要持续的宿主双向协议交互，不能用一个静态 start 文件或等待 EOF
代替宿主。离线测试使用真实 SDK、实际数值续训及宿主替身，覆盖三轮历史、父模型链、
状态保存、逐行预测和写入路径限制；它不代表真实模型调用、Docker 构建或平台评分通过。
下文的 `0.649151222` 只属于原有带标签反馈的本地回放。

### 公开验证选路与独立预测

`frozen_router` 是本地训练的默认实验，也作为平台决策模型可以选择的菜单项。
四传感器主专家、五传感器辅助专家、完整输入输出层与训练集 KNN/二分类头保留原有组合思路。
首次从宿主实际父检查点产生各分支；后续若父产物已是冻结路由，则四/五传感器和 home
分支分别从该父产物对应的权重及 Adam 状态继续。所有梯度、KNN 记忆、Fisher 特征和
二分类头只使用当轮公开 train。辅助分支的 epoch 及最终路由通过公开 validation 选择。

训练阶段比较有限的融合权重、是否开启因果漂移校正，以及类别对路由。
类别对需达到最少样本数和 Macro-F1 增益门槛，完整路由配置写入 `routing.json`。
该文件记录公开验证数据摘要、对照指标、每次路由比较和最终选择，作为产物文件参与摘要校验。
`routing_training.json` 记录辅助分支父摘要与训练状态；`training.json` 中的 epoch 历史属于
主神经网络，另行记录完整预测器的验证指标和总参数量。

预测阶段由 `src/gas_a/frozen_router.py` 加载已选权重和已固定的配置，立即返回每一行的六类分数。
它没有 `waiting` 或 `release_labels` 接口。融合权重、启用的类别对和校准阈值都不再随预测改变。
平线检测及可选漂移校正仅维护前序输入特征的内存缓冲；缓冲在每个预测进程启动时清空，
公开验证过程的标签和特征缓冲不会带入预测。整段评估与同序逐行预测使用相同的因果计算。
因此本方式适用于 SDK 无标签回填的逐行预测接口；平台实际评测结果仍须独立核验。

## 本地回放

这是图中橙线“本次四传感器路由”的单版本开发项目，对应实验 `four_sensor_router_20261121a` 的自然顺序 `macro` 分支。基准为 batch3–10 全部 12,221 条样本合并混淆矩阵后的固定六类 Macro-F1，预期值为 **0.649151222**。

## 本地归档回放命令

当前工作区已准备 Python **3.12.13** + NumPy **1.26.4**，进入项目目录后直接运行，无需先激活环境：

```powershell
.\.venv\Scripts\python.exe run.py replay
# 或使用启动脚本，参数会传给 run.py
.\start.ps1 replay
```

显式回放命令直接加载随包权重。各批预测先写入新建的 `runtime/` 子目录，再揭晓该批标签。程序会逐批核验预测摘要和合并混淆矩阵，并输出各批与合并 F1。全程使用本地 CPU，无需账号。

这是命令行实验项目；运行完成后退出，结果保留在 `runtime/`，没有网页地址或常驻服务。

### 当前环境与重建环境的方法

| 项目 | 当前设置 |
|---|---|
| Python | 3.12.13，由 `.python-version` 标明 |
| 解释器 | `.venv\Scripts\python.exe` |
| 运行依赖 | `numpy==1.26.4` 和随包 `contest-harness==1.0.4`，见 `requirements.txt` |
| 底层 Python | `.python/cpython-3.12.13-windows-x86_64-none/`，由 uv 下载到项目内 |
| 原虚拟环境 | `.venv-py313-backup/`，保留原 Python 3.13 环境文件 |
| 计算资源 | 本地 CPU；两个入口均在导入 NumPy 前设置单线程 |

本次将原 Python 3.13 空环境替换为项目要求的 3.12 环境，没有修改系统 Python。
`.venv` 依赖 `.python` 中的底层解释器，不应只移动或删除其中一个目录。
PyCharm 中可选择本项目的 `.venv\Scripts\python.exe` 作为解释器，运行入口为 `run.py`，按用途设置 `train-local`、`predict-local` 或 `replay` 等参数。

在新目录重新准备环境时，若已安装 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

若只有 uv，没有 Python 3.12，可按当前工作区相同方式安装。以下命令用于尚未创建 `.venv` 的目录：

```powershell
uv python install 3.12.13 --install-dir .python --no-bin --no-registry --cache-dir .cache\uv
.\.python\cpython-3.12.13-windows-x86_64-none\python.exe -m venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt --cache-dir .cache\uv
```

不要使用系统 Python 3.13 直接安装这个锁定依赖；当前项目的环境基线是 Python 3.12。

## 看哪几个文件

| 位置 | 用途 |
|---|---|
| `src/gas_a/router.py` | 本版本的核心。预测时读取特征，批次结束后再用已揭晓标签更新最近三批的路由统计。|
| `src/gas_a/signals.py` | 数据读取、传感器平线检测、按时间顺序做偏移校正及辅助模型融合。|
| `config.json` | 同一版本用到的权重摘要、训练配方、校准阈值和回放验收值。验收值只用于检查复现。|
| `models/` | 起点模型、标准化参考、四传感器主专家、五传感器辅助专家、完整输入辅助分支的两个检查点，共六个组件。|
| `data/public/`、`data/replay/` | 前者含官方公开训练与验证数据；后者含本地私测用的 batch3–10。|
| `run.py`、`rebuild_models.py`、`src/gas_a/continuation.py` | 从显式基线或实际选中的检查点续训，并记录父子产物。|
| `rebuild_replay_models.py`、`src/gas_a/home_training.py`、`src/gas_a/sensor_training/` | 重建归档回放所用的组件。|

要复核组件如何训练，可另开一个输出目录：

```powershell
.\.venv\Scripts\python.exe rebuild_replay_models.py runtime/rebuilt
# 只有重建过程的全部 SHA-256 校验通过，才可直接使用该目录回放
.\.venv\Scripts\python.exe run.py replay --models runtime/rebuilt
```

重建使用 batch1 训练、batch2 公开验证。私测流从 batch3 的共同起点开始，batch4 起使用第 3、14、15、16 号传感器主专家及辅助分支，batch7 起可依据已结束批次的错分决定后续路由。八批回放期间权重固定，变化的是路由决策。

**当前 Windows 环境的重建限制：** 本次执行已产出全部六个组件，但四个训练所得组件未通过逐字节 SHA-256 校验，最先报错的是 `home_trunk`。数组比较发现优化器状态与权重存在约 `10^-18` 至 `10^-16` 的差异，`home_router` 的逻辑回归头系数也存在末位差异；两个直接复制的起始组件完全一致。这与浮点计算环境差异相符，但没有原始训练环境信息可用于确认根因。

单独诊断时，仅在内存中将模型摘要绑定到重建文件的实际摘要，八批预测摘要和合并混淆矩阵全部匹配。该诊断说明本次重建的预测行为一致，不代表文件字节一致；原 `config.json`、随包模型以及运行入口的严格校验均保留。正常运行请使用默认 `models/`。诊断记录见 `runtime/rebuilt-environment-check/prediction_diagnostic.json`。

## 代码梳理

### 输入、输出与主调用链

任务是六类气体分类，每条样本有 **128 个数值特征 = 16 个传感器 × 每个 8 个特征**。模型概率列与内部标签按以下顺序对应：

| 内部标签 | 气体 |
|---|---|
| 0 | Ethanol，乙醇 |
| 1 | Ethylene，乙烯 |
| 2 | Ammonia，氨气 |
| 3 | Acetaldehyde，乙醛 |
| 4 | Acetone，丙酮 |
| 5 | Toluene，甲苯 |

DAT 文件的标签为 1–6，揭晓时转换为 0–5；公开训练 JSONL 使用气体英文名称。

```text
run.py replay → replay.py
  ├─ config.json + models/*.npz → 校验六个模型摘要 → FourSensorRouter
  └─ 按 batch3 → batch10 顺序逐批执行
       ├─ signals.dat_inputs：跳过标签字段，只解析特征并计算文件摘要
       ├─ FourSensorRouter.predict_batch：预测本批类别
       ├─ runtime/<新目录>/batchXX.npy：独占创建，保存并重新读取预测
       ├─ 校验预测 SHA-256
       ├─ signals.reveal_labels：确认文件未变后解析标签
       ├─ confusion / macro：本批及累计评估
       └─ FourSensorRouter.release_labels：更新供后续批次使用的路由统计
  └─ 校验累计混淆矩阵，写入 results.json
```

`run.py replay` 提供 `--models`、`--output` 两个参数。输出目录必须尚不存在，避免覆盖已有预测。
默认输出使用 `runtime/<时间戳>/`；模型目录默认为项目中的 `models/`。
这条入口绑定了随包数据和预测的验收摘要，不是任意新数据的通用推理 CLI。

### 推理核心：`router.py` 与 `signals.py`

`FourSensorRouter` 是有状态对象，必须按时间顺序调用。第一批对应 batch3，直接使用 `start.npz`。
从第二批开始，同时计算以下分支：

1. **四传感器主专家**：第 3、14、15、16 号传感器，原始输入与偏移校正后输入的概率取平均。
2. **五传感器辅助专家**：增加第 13 号传感器，同样平均原始与校正后的预测。
3. **完整输入辅助分支**：融合 `home_trunk`、`home_router`，再按训练集校准阈值对乙醇/乙烯与乙醛之间的判断做修正。

四、五传感器模型仍接收 128 维输入，通过第一层权重的零行屏蔽不用的传感器。
六个随包模型均为 128 → 32 → 6 的 MLP，参数总数 4,326；`start`、`home_trunk`、`home_router` 还包含 KNN 记忆与融合配置。

`signals.Flatline` 观察第 13 号传感器的 8 个特征，以最近 8 条读数的归一化跨度检测平线。
报警样本直接采用四传感器分支。`signals.correct` 逐条处理输入，维护 128 条历史特征，每 64 条更新一次偏移估计；当前样本先使用已有估计校正，再进入历史缓冲，标签不参与偏移估计。

非报警样本的辅助分支先按类别对融合完整输入专家与五传感器专家，再由外层路由决定是否与四传感器结果融合。
路由统计仅保存最近三个已揭晓批次；batch4–6 的统计积累后，最早从 batch7 开始生效。
切换需满足样本数量、至少两批获益及累计 Macro-F1 提升等条件。`predict_batch` 后必须调用一次 `release_labels`，才能预测下一批。

### 模型和训练模块

| 文件 | 职责与使用位置 |
|---|---|
| `model.py` | 当前回放使用的模型实现：标准化、tanh MLP、Adam、KNN、辅助分类头、检查点读写和校验。|
| `decisions.py` | 按样本 ID 确定性划分训练/校准集，以及模型预测使用的类别校准逻辑。|
| `home_training.py` | 完整输入辅助分支训练：仅更新输出层、Fisher 特征排序、逻辑回归头和五折阈值校准。|
| `sensor_training/data.py` | 128 维 JSONL 数据契约、训练/验证 ID 去重、模型兼容检查和评估。|
| `sensor_training/optimize.py` | 四/五传感器候选训练：类别加权、权重投影、Adam、每 epoch 验证、质量门槛、最佳检查点与早停。|
| `sensor_training/reference/` | 训练端保留的基础模型与数据加载器；与回放端 `model.py` 是不同实现，应分别理解。|
| `sensor_training/architecture.py` | 训练端检查点加载，以及显式架构迁移；本次配置使用基础架构。|
| `sensor_training/adaptive.py` | 训练端质量门槛和历史候选选择工具；本次重建使用其中的 `fixed_gate`。|
| `sensor_training/controller_policy.py` | 训练决策证据、计划和选择契约等工具；当前两个入口没有外部决策模型调用。|
| `sensor_training/io.py` | 类别常量、JSON/JSONL、摘要、配置继承与文件锁工具。|

归档重建调用链为：`rebuild_replay_models.py` 加载公开数据 → 复制 `start`、`reference` → 从 `start` 训练两个完整输入辅助组件 → 校准二分类阈值 → 从 `reference` 分别训练四/五传感器组件 → 检查六个文件的 SHA-256。

`config.json` 集中了模型身份、训练配方、质量门槛、校准阈值、八批数据/预测摘要以及最终验收矩阵。
修改训练参数会改变权重；当前回放入口仍要求原始摘要，因此新实验还需独立设计配置和验收方式。

### 数据与评分口径

| 数据 | 样本数 | 用途 |
|---|---:|---|
| `data/public/train.jsonl` | 445 | batch1，训练与训练内校准 |
| `data/public/validation.jsonl` | 1,244 | batch2，公开验证、选择 epoch |
| `data/replay/batch3.dat` 至 `batch10.dat` | 12,221 | 按顺序回放、延迟揭晓标签、评估 |

回放的 `router.macro` 固定平均六个类别，某类没有真实或预测样本时该类 F1 为零。
训练端 `sensor_training.data.score_predictions` 的 Macro-F1 仅平均真实或预测中出现的类别。
两者口径不同，比较单批指标时要保留这个区别。
最终回放成绩是在合并混淆矩阵上计算，不能用八个批次 F1 的算术平均代替。

## 本次验证记录（2026-11-21）

依赖检查 `python -m pip check` 通过；18 个 Python 源文件均通过语法解析。
使用随包权重的完整回放成功，六个模型摘要、八批数据摘要、八批预测摘要及合并混淆矩阵全部匹配。

| 批次 | 样本数 | 固定六类 Macro-F1 |
|---|---:|---:|
| batch3 | 1,586 | 0.593269 |
| batch4 | 161 | 0.685127 |
| batch5 | 197 | 0.688513 |
| batch6 | 2,300 | 0.760428 |
| batch7 | 3,613 | 0.675819 |
| batch8 | 294 | 0.505773 |
| batch9 | 470 | 0.344042 |
| batch10 | 3,600 | 0.537445 |
| 合并 | **12,221** | **0.649151222** |

本次默认回放结果：`runtime/20261121-172609-875846/results.json`。
训练重建的逐字节验收未通过，预测诊断通过，详情见前文重建限制。
以上是原始回放验证记录；后续已新增比赛接入与动态续训测试，运行命令见本文前部。

## 完整闭环验证

新增入口 `scripts/closed_loop_validation.py` 在独立输出目录中执行监测、诊断、真实训练、门槛评估、本地部署切换和回滚验证，不改变原回放算法或 `config.json`。

本次正式记录位于 `runtime/closed_loop_four_sensor_20261121_02/`，PDF 报告位于 `output/pdf/closed-loop-four-sensor-validation.pdf`。报告沿用提供的参考 PDF 的横向版式，包含逐批成绩、阶段时间线、传感器漂移热图、类别召回、实际参数和拒绝/晋升依据。

结果：Macro-F1 为 **0.649151222**，相对参考上轮 **0.585773862** 提高 **6.3377 个百分点**，与当前项目原回放逐条一致。7组路由候选包含14次传感器训练，只有初始原配方重训方案上线，后续6组均被拒绝；自然性能回滚0次，两项隔离演练通过。乙醛召回由14.08%降至8.85%，不能将总体提升表述为业务指标全面改善。本次使用本地规则控制器，外部AI调用0次。

需要重新执行时，指定一个尚不存在的输出目录：

```powershell
.\.venv\Scripts\python.exe scripts\closed_loop_validation.py `
  --dataset 'D:\liyue\工业模型赛题研究\gas+sensor+array+drift+dataset\Dataset' `
  --previous 'D:\aiproject\gas1\runtime\temporal_guarded_20261121_01' `
  --output runtime\closed_loop_new
.\.venv\Scripts\python.exe scripts\verify_closed_loop.py runtime\closed_loop_new
.\.venv-report\Scripts\python.exe scripts\build_closed_loop_report.py runtime\closed_loop_new
```

`.venv-report` 是独立的绘图/PDF环境，依赖锁定于 `requirements-report.txt`，不影响原推理环境。新工作区可用 Python 3.12 创建该环境并安装此文件。本次100项核验覆盖所有8批、14次传感器训练、两个home组件、状态恢复、独立指标重算和两项回滚演练；结果见正式记录中的 `verification.json`。训练重放核验应在新运行目录中首次执行，已有核验产物保留用于审阅。

阶段时间为本次计算机执行时间，并非原数据采集时间。所有批次先保存预测再揭晓标签，候选只用batch1进行梯度训练、batch2选择epoch，已结束窗口用于诊断和选版；更换版本仅影响后续批次。当前版本结构经过历史数据研究，因此本次也不构成全新留出集验证；本地部署不等同于HTTP服务或比赛SDK上线。

## 平台边界

上述回放入口用于开发审阅、本地训练复核与回放，依赖批次结束后揭晓标签；比赛 SDK 1.0.4 没有约定隐藏批次标签回填。正式平台入口见本文开头的 Harness 接入部分。平台模型身份应在提交页选择获准的模型，密钥只填平台密钥栏；仍需实际完成镜像构建、平台试运行与成绩核验。请勿把含私测数据的项目 ZIP 直接上传为平台镜像。0.649151222 是本地已使用历史样本的回放成绩。
