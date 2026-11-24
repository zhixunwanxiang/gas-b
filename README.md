# Gas RSI：工业模型自进化系统

系统以平台提供的实际父模型为起点，先检查公开数据是否变化、模型主要错在哪里，再由 27B 根据历史和当前证据，从预先配置的方案菜单中选择实验。程序实际训练并测量候选，达到硬门槛才允许上线；收益不足就保留父模型，需要回滚时恢复本轮重新验证合格的历史产物，并让下一轮继承它的完整文件和配套策略。

**当前支持 A 分类、B/C 剩余寿命回归，原生接口使用 contest-harness SDK 1.0.4。** 方案范围和安全门槛由代码限定，27B 负责依据证据做选择；系统不会任意生成并执行新的训练代码。

**2026-11-24 补充：当前交付更新为 v9。** `/app/harness.py` 恢复为用户提供 SDK 的原模板，镜像内 9 个 SDK 模块与原始压缩包逐字节核对。容器训练/预测验收显式使用 SDK 标准命令，并传入完整预测上下文，不再只测依赖 ENTRYPOINT 的简写与缺省上下文兼容分支。旧 v8 源码重复快照也已合并。当前验证属于离线与保存回复重放，不计为新的真实 27B 调用；不能据此宣布组委会 180 秒预检故障已消除。交付时须使用新的 v9 tar，重试旧提交不会更新镜像。

**2026-11-24：针对平台“预检单步超过 180 秒”的提示，已构建 v8 启动与预测优化镜像。** 本地未复现该 180 秒超时；已去除 describe 的训练模块加载，并在 OS 确认 `/model` 只读后复用路径/文件身份，减少每条预测的文件系统查询。C 的 23,049 条公开预测由约 28.7 秒降至 2.0 秒，输出字节一致。v8 的验证为离线检查和既有真实回复重放，不计作新的真实 27B 实验；最近一次真实调用成绩仍是下述 v7。新镜像、日志与重新试运行说明见 [v8 超时排查报告](docs/IMAGE_V8_TIMEOUT_20261124.md)。

当前 **v7 镜像**已纳入续训审查的三项修复：B/C 允许跨轮更新授权训练投影而保留最初监测参考；输出指南约定的 `training.json`；精简候选回执通过本轮 SDK 提交与当前文件摘要绑定，仍拒绝显式冲突和无法核验的宿主分数。实现见[续训与 SDK 修复说明](docs/BASELINE_SDK_FIXES_20261123.md)，构建与离线镜像验证见[v7 交付说明](docs/IMAGE_V7_20261123.md)。下列成绩来自修复后固定 v7 镜像的全新真实 27B 测试，未沿用 v6 回执或结果。

## 当前任务与最近结果

| 数据集 | 输入 | 预测结果 | 选版主指标 |
|---|---|---|---|
| A：气体识别 | 128 维特征，16 个传感器 | 六类气体分数 | Macro-F1 越高越好 |
| B：轴承剩余寿命 | 水平/垂直各 17 个振动特征 | 非负寿命，单位分钟 | RMSE 越低越好，辅助 MAE |
| C：发动机剩余寿命 | 3 项设置、21 项传感器值 | 非负寿命，单位循环 | RMSE 越低越好，辅助 MAE、NASA 评分 |

2026-11-23 13:59:36 至 14:22:56，最新一次全新测试使用同一个固定 v7 镜像，从三个官方公开包的初始模型、空历史和空状态开始，完成 **12 轮正常训练 + 6 轮故障回滚/继承检查**，共 **36 次真实 PyTRiO Qwen3.8-27B 调用**，提供方报告 225,491 tokens。每个数据集有 3 次“读非空历史—实际实验—依据结果决策”完整记录；运行中没有人工指定方案或改写回复。

| 任务 | 公开包初始基线 | 最终成绩 | 改善 | 正常阶段升级 |
|---|---:|---:|---|---:|
| A：Macro-F1 ↑ | 0.714029722 | 0.798701650 | +8.4672 个百分点 | 3 次 |
| B：RMSE ↓ | 681.159832604 | 651.885364673 | 误差降低 4.2977% | 2 次 |
| C：RMSE ↓ | 52.868572566 | 51.761745579 | 误差降低 2.0935% | 2 次 |

三组实际历史回滚和下一轮完整产物继承均通过；v7 构建前 159 项项目测试通过，镜像内另有 17 项修复回归测试通过。使用本次保存的回复重放后，16 个检查点/路由文件、27 组逐条预测及所选版本一致，重放新增模型调用为 0。7 个新云端 Run 均已回读确认 `FINISHED`，本次没有触发响应纠正请求。

[最新 v7 全流程结果、七个云端链接与验证边界](docs/FRESH_V7_REAL_TEST_20261123.md) · [SwanLab 总览](https://swanlab.cn/@1218/gas-b/runs/27bd34cfe76742dcac41df4e9a3f93ee/chart)

本次镜像、精简源码/证据包和报告位于 `output/fresh-platform-test-v7-20261123/`，含清单共 **62,851,302 字节（59.94 MiB）**，低于 60 MiB。材料 ZIP 对一份相同的初始权重去重，解压后先运行 `restore_materials.py`；恢复后与原材料逐字节核验 169 个文件。完整逐条预测、公开大数据和所有轮次权重仍在本地 runtime，精简包不替代该目录。

这些成绩来自**公开数据上的本地 SDK 宿主模拟**，不等于组委会正式宿主、最终平台基线或隐藏测试集成绩。旧实验及失败记录另见文末历史资料，不能与本次成绩混合统计。

## 闭环如何运行

每个 `train(rnd)` 回合执行一次受控动作；平台随后传入被选中的父产物，开始下一回合。检查点序号不表示整个程序只能训练一次。

| 步骤 | 什么时候做、做什么 | 决策依据 | 留下的结果 |
|---|---|---|---|
| 漂移监测 | 加载实际父模型、读取公开投影后，将当前验证输入与 train 建立的固定参考比较 | PSI、分箱累计分布距离、均值偏移及最小样本数 | `monitoring.json`，告警及变化位置 |
| 诊断归因 | 测量父模型后，分析公开验证标签上的错误 | A 的弱类别与混淆；B/C 的误差方向和寿命分段误差 | `diagnosis.json`，待验证的实验假设 |
| 实验优化 | 读取历史、策略与诊断，27B 在允许动作和菜单内提出方案 | 真实成败、同口径分数、重复失败限制、资源预算 | 规划回复、真实调用号、候选检查点、逐 epoch 记录 |
| 评估上线 | 本地评估候选，再通过 SDK 提交给宿主测量，27B 最终选版 | 本地硬门槛与宿主回执都合格；最终 ID 必须在白名单内 | 候选回执、采用/拒绝理由、选中产物 ID |
| 回滚 | 本轮允许的历史版本经恢复和当前验证复评达标后，由 27B 决定恢复 | 当前实际父模型与历史模型的本轮实测比较、完整文件摘要 | 历史恢复记录、配套策略；下一轮继承该产物 |

“输入发生变化”不直接证明模型变差；误差和分布统计也是原因线索，不是物理故障的因果证明。本次 A/B 的四轮监测都告警，C 四轮都未告警，系统仍可依据误差诊断开展优化。

### 固定的规则与 AI 可以选择的范围

- **动作**：`observe` 保留实际父模型；`experiment` 从实际父模型续训一个菜单方案；`rollback` 恢复本轮合格历史，本轮不训练，下轮才从恢复版本续训。观察或训练不能指向另一个历史版本。
- **A 菜单**：`frozen_router`、`four_sensor`、`five_sensor`、`conservative_head`。内部组件和路由参数以配置为准；公开验证用于选路，导出后的预测不等待标签反馈。
- **B/C 菜单**：低学习率、标准学习率、只训练输出层、Huber 稳健损失，名称分别为 `low_rate`、`standard_rate`、`head_only`、`robust_huber`。27B 不能通过回复覆盖它们的学习率、轮数或损失。
- **策略更新**：记录采用、拒绝和本地失败；只有实际采用新候选才增加配套策略版本。同一父模型、数据、方案及执行规则下，累计两次未采用/失败会屏蔽该重复实验；换父状态后重新判断。

### 上线门槛与漂移阈值

| 范围 | 默认依据 |
|---|---|
| A 上线 | Macro-F1 严格提高且至少增加 `0.0005`，即 **0.05 个百分点**；父验证支持数至少 20 的类别，单步召回下降最多 `0.20`，即 **20 个百分点** |
| B/C 上线 | RMSE 至少减少 `max(0.000001, 父 RMSE × 0.0005)`，同时 MAE 最多相对增加 2% |
| 宿主回执 | 授权公开验证身份由显式字段或本轮 SDK 候选事务核验，分数始终使用宿主值并与本地复核一致；冲突、未知指标或未绑定的缺省身份仍拒绝 |
| 漂移 | 当前样本至少 30 条；PSI ≥ 0.2、分箱距离 ≥ 0.2 或均值偏移 ≥ 1.5 个训练标准差 |
| 漂移汇总 | A 先对每个传感器的 8 维指标取均值再判断；B/C 按 34/24 列逐项判断 |
| 数值训练预算 | A 最多 32 个 epoch，B/C 最多 8 个；每个候选数值阶段最多 30 秒，还受各菜单轮数和早停限制 |

门槛逐轮相对**实际父模型**比较，不是对初始基线做永久累计保护。A 少于 20 条验证样本的类别不受上述召回硬保护，C 的 NASA 评分只作辅助记录，不保证每轮都改善。

全部参数及实际调用位置见 [SUBMISSION_CONFIG_GUIDE.md](SUBMISSION_CONFIG_GUIDE.md)。正式配置是 [submission_config.json](submission_config.json)；根目录 `config.json` 属于历史回放。

## 数据、父产物和状态边界

- 只使用平台授权的公开 `train` / `validation`。train 用于更新梯度和拟合训练统计，validation 可用于诊断、选 epoch、选路与选版。隐藏数据不挂进训练容器，也不获取隐藏标签。
- 续训以本轮 `rnd.parent` 声明的检查点和摘要为准，继承权重、Adam 一二阶矩、标准化、seed 和累计训练量。未采用候选不会自动变成下轮父模型。
- 样本 `id` 只用于数据一致性检查，不进入模型；不从文件名、样本号或匿名标识反推标签。B/C 特征结构、顺序、profile 与分钟/循环单位均严格校验。
- B/C 跨轮 train 更新时仍使用首轮监测参考，分别记录参考与当前 train 的摘要；父检查点的输入/目标 scaler 不重拟合。`training.json` 与 `continuation.json` 保存同一份真实续训报告。
- `/app`、`/data`、`/previous`、`/state`、`/model` 按阶段只读使用。候选与 `state_workspace()` 返回的状态副本写在 `/workspace`，通过 SDK 保存。
- SDK 1.0.4 使用 `chunked-v1` 上传。上传期间不修改文件，容量以 `rnd.limits` 为准；不支持新协议时明确报错。
- 完整但非法的模型回复最多纠正一次，并保留每次真实回执。RPC 发出后中断会立即结束会话，不捕获后继续发请求，也不自动重试结果未知的操作。

## SDK 1.0.4 顺序与超时边界

正式入口逐字节使用随附 SDK 的 `template/harness.py`，不在入口脚本中追加自定义启动逻辑，也不自行实现或修改 JSONL 协议。训练先由 `Round.start()` 读取 `start`；每个请求都必须收到对应响应类型、同 ID 的宿主回复后才能继续。上传按 `upload_begin → upload_chunk（逐块确认）→ upload_finish` 完成，再发 `candidate_request` 或 `state_request`。最后发送 `round_complete`，收到 `round_ack/accepted=true` 后退出。预测没有训练 start 帧，每条输入必须在下一条输入和 EOF 之前得到已刷新的输出。

SDK 没有规定 `save_state` 与 `submit_candidate` 必须按示例中的相对顺序执行。项目在候选实测、模型选版和配套策略写完后保存状态，随后结束本轮；不能为了机械模仿示例，把未完成的决策状态提前保存。`tests/test_sdk_alignment.py` 用实际 SDK 校验 A/B/C 完整回合的请求/回复顺序，以及上传中断后不会再发 RPC。

协议兼容不代表满足平台时限。SDK 1.0.4 的文件上传和历史恢复都是串行 RPC：小文件也要单独请求/确认，恢复历史还会传回整份产物的源码和报告。本项目原 A 候选把训练源码同时放在 `source/` 与 `src/gas_a/sensor_training/`，现改为共用后一处，减少上传和以后恢复时的重复往返。此改动改变产物源码布局，已纳入 v9；既有固定 v8 镜像和旧证据不包含它。

`scripts/validate_platform.py::exchange()` 现在以整轮累计时间验收，默认 180 秒，并受本地宿主 start 帧中更小的 `wall_seconds` 约束；可通过函数参数 `round_timeout` 调整。180 秒是本地测试预算，不是 SDK 对所有平台/RPC 的通用规定。同步宿主处理（模型请求或候选测量）返回后也复查预算，超限不再发送回复或继续会话。协议记录增加累计秒数并逐帧刷新，可区分宿主等待和选手计算。该检查不抢占已经发出的同步宿主操作，宿主操作本身仍需自己的超时与收尾；不能把它解释为每个测试进程都保证在 180 秒内结束。

定位超时须区分阶段：`describe` 不会进入历史恢复、模型调用或候选上传；`train` 的数值训练上限也不涵盖模型、上传和宿主评测的全部耗时。保存回复重放省去了真实模型等待，不能用其耗时推断正式平台一定通过。

### 测试 agent 计时与并发对照

`scripts/benchmark_sdk_agent.py` 使用用户提供的 SDK 1.0.4 源码核对安装包、原模板和固定镜像，再对 A/B/C 各执行 10 轮串行与三个独立进程并发的对照，共 60 轮。每个数据集内部仍严格按轮继承父模型、状态和历史，单个 SDK 会话只允许一个未完成请求。测试执行真实数值训练、分块传输和容器预测，模型决策使用离线替身，不读取密钥、不请求真实模型、不写云端。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/benchmark_sdk_agent.py `
  --baseline-root 'D:\liyue\工业模型赛题研究' `
  --sdk-archive 'D:\liyue\工业模型赛题研究\contest-harness-sdk-1.0.4.tar.gz' `
  --output runtime/my-sdk-agent-timing --rounds 10 `
  --docker-executable 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe' `
  --docker-context desktop-linux
```

输出目录必须全新。`round-timings.csv` 汇总逐轮耗时；每轮 `sdk-timing.json` 保存请求/回复顺序、上传偏移及摘要核验、宿主等待、选手输出间隔和应用阶段。`summary.json` 与 `REPORT.txt` 比较串行/并发耗时和模型输出一致性。Docker `start/die` 事件用于核验训练容器的真实时间交集，不能把同一数据集候选评测中嵌套启动的预测容器误算成跨数据集并发。候选评测时间已包含在候选 RPC 等待中，不能重复相加。

SDK 的 `Round._rpc()` 用锁串行完成单会话请求，SDK 不包含平台跨数据集调度器。仓库原 `test_fresh_platform.py` 按 A→B→C 串行执行；平台界面同时显示 A7/B8/C5 只能支持进度独立，不能凭该截图排除分时交错或界面异步更新，也不能证明同时执行。无法取得平台任务起止时间时，平台并行性应标为未确认；本地并发实验也不代表平台资源池或模型网关排队情况。

## 安装与入口

以下命令在项目根目录用 PowerShell 运行。数值环境为 Python **3.12.13**、NumPy **1.26.4**、SDK **1.0.4**。

```powershell
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.12 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -X utf8 harness.py describe
```

SDK wheel 已随源码放在 `vendor/contest_harness-1.0.4-py3-none-any.whl`，来源见 [sdk-provenance.json](vendor/sdk-provenance.json)。不要从 PyPI 查找同名比赛 SDK。

| 入口 | 用途 |
|---|---|
| `harness.py describe` | 查看协议和 `classification`、`regression` 任务声明 |
| `harness.py train` | 平台训练回合；需要宿主双向 JSONL 交互，不能靠静态 start 文件和等待 EOF 替代 |
| `harness.py predict` | 选中产物的逐行独立预测，读取一行即输出并刷新一行 |
| `scripts/validate_multitask.py` | A/B/C 离线宿主演练，实际训练/预测，模型决定为离线替身 |
| `scripts/test_fresh_platform.py` | 从初始基线开始的 A/B/C 真实 27B、回滚、云端记录和重放总控 |
| `scripts/replay_multitask_cloud.py` | 使用保存回复做零费用数值重放 |
| `scripts/validate_image_latency.py` | 保持 stdin 打开检查镜像首帧响应，并可对比已有公开预测输出，不调用模型 |
| `run.py train-local / predict-local / evaluate` | A 的本地单候选开发工具，不代替 B/C 原生入口 |
| `run.py replay` | 历史带标签反馈回放，仅供研究，不作为平台预测入口 |

容器正式命令为 `python3 /app/harness.py describe|train|predict`。stdout 只写协议 JSONL，普通日志写 stderr。

预测阶段没有训练 start 帧。SDK 根据 `CONTEST_PREDICTION_CONTEXT` 提供 `dataset_key` / `task`，通过 `CONTEST_ARTIFACT_PATH` 指定主检查点；回调的 `model_dir` 是 `/model` 根目录。stdin 每行包含原生 `task` 和 `inputs`，不带 `answer`；SDK 输出 A 的六类 `outputs` 分数数组，或 B/C 的 `{"outputs":{"value":123.4}}`。完整定义见 [SDK 预测说明](vendor/contest-harness-sdk-1.0.4/HARNESS_SDK.md)。

## 构建当前源码的镜像

```powershell
$dockerExe = 'docker'  # 不在 PATH 时改成 Docker Desktop 的 docker.exe 绝对路径
$imageTag = 'gas-rsi-abc:local-review'
& $dockerExe build -t $imageTag .
$imageId = (& $dockerExe image inspect $imageTag --format '{{.Id}}').Trim()
& $dockerExe run --rm --network none --read-only $imageTag describe
```

构建阶段需要准备基础镜像和依赖；正式容器不安装依赖、不持有模型/云端密钥、不访问外网。运行用户为 `65534:65534`；镜像测试工具额外固定只读根目录、2 CPU、4 GiB 和允许挂载。

已有核验过的 v6 运行镜像时，也可离线重建当前应用层：

```powershell
& $dockerExe build -f Dockerfile.offline `
  --build-arg CONTEST_RUNTIME=gas-rsi-abc:wyx-sdk104-v7-20261123 `
  -t $imageTag .
$imageId = (& $dockerExe image inspect $imageTag --format '{{.Id}}').Trim()
```

当前镜像为 `gas-rsi-abc:wyx-sdk104-v9-20261124`，不可变 ID 为 `sha256:be6ddcc96cd4b20be480cad0067abbdd0102bc03c2891d7acf5ac7b38003f0d4`，交付目录为 `output/submission-wyx-sdk104-v9-20261124/`。默认真实测试工具已指向 v9；本次未执行新的付费测试。v8 镜像与当时的验证保留，见 [v8 超时排查报告](docs/IMAGE_V8_TIMEOUT_20261124.md)。v7 镜像及其真实实验原样保留，见[v7 交付说明](docs/IMAGE_V7_20261123.md)。

v6 原始镜像与真实模型证据原样保留在 `output/fresh-platform-test-v6-20261123/` 和 `runtime/fresh-v6-full-20261123/`。v7 新增真实 27B 的完整证据位于 `runtime/fresh-v7-full-20261123/`，与构建时的离线替身测试分开保存。后续若再改动应用源码，仍须重建并显式传入对应 `$imageTag` / `$imageId`，严格预检会拒绝源码不匹配的旧镜像。

## 测试与复现

### 1. 离线单元测试，不产生模型费用

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
```

固定 v6 验收时为 151 项测试通过，修复后的完整测试结果见[修复说明](docs/BASELINE_SDK_FIXES_20261123.md)。官方 SDK 自身的离线协议测试可另在 `vendor/contest-harness-sdk-1.0.4/` 目录运行：

```powershell
$env:PYTHONUTF8 = '1'  # 同时作用于 SDK 测试启动的子进程
$env:PYTHONIOENCODING = 'utf-8'
Push-Location vendor/contest-harness-sdk-1.0.4
& ..\..\.venv\Scripts\python.exe -B -X utf8 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
Pop-Location
```

这些测试使用 JSONL 宿主替身，不调用真实模型。SDK 的 `tests/test_live.py` 是另行计费的真实模型检查，不包含在上述命令中。

### 2. A/B/C 离线多轮演练

先完成上面的镜像构建。官方三份包放在同一父目录，保持原名称和公开数据结构：

```text
工业模型赛题研究/
  cpu-a-guide-20261110-v1/
  cpu-b-rul-20261110-v1/
  cpu-c-guide-20261110-v1/
```

```powershell
$baselineRoot = 'D:\liyue\工业模型赛题研究'
$offlineOut = 'runtime/offline-abc-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
.\.venv\Scripts\python.exe -B -X utf8 scripts/validate_multitask.py `
  --baseline-root $baselineRoot --output $offlineOut --rounds 4 `
  --image $imageTag --docker-executable $dockerExe --docker-context desktop-linux
```

真实执行数值训练和原生预测，但方案由离线替身选择；通过不表示真实 27B 已验证，也不表示平台分数已通过。每次输出目录必须不存在，旧失败和成绩不覆盖。

### 3. 真实 27B + 云端完整流程，会产生费用

宿主先安装云端依赖并准备本地 `.env`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-platform.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在 `.env` 填写 `PYTRIO_API_KEY`、`SWANLAB_API_KEY`；模型固定为 `Qwen/Qwen3.8-27B`，端点 `https://pytrio.com`。`GAS_A_SWANLAB_PROJECT` 设置为目标项目（例如 `gas-b`）；`GAS_A_SWANLAB_WORKSPACE` 留空时使用当前授权个人空间，也可填写有权限的组织名。凭证只留在宿主，不进 Git、镜像或候选。

```powershell
$liveOut = 'runtime/fresh-abc-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
.\.venv\Scripts\python.exe -B -X utf8 scripts/test_fresh_platform.py `
  --baseline-root $baselineRoot --output $liveOut `
  --image $imageTag --expected-image-id $imageId `
  --docker-executable $dockerExe --docker-context desktop-linux
```

该总控对 A/B/C 各执行正常训练、两轮故障回滚/继承、独立核验，最后用本次原始回复重放数值训练并写云端总览。正常阶段每组至少 4 轮、至多 6 轮；预计约 36 次真实调用，预算上限为 66 次。任一阶段失败就停止并标记失败，不自动重跑付费请求，也不拿历史成绩补齐。

测试使用本地宿主实现 SDK 协议，真实模型回复原样返回 `rnd.model()`；它不冒充组委会真实宿主或正式计费回执。正式平台由宿主持有模型密钥并绑定获准模型，镜像内不放外部采样客户端。

### 4. 已有记录的零费用核验和重放

下例需要本机保留最新完整 `runtime` 目录，以及当时的固定 v7 镜像。`runtime/` 和 `output/` 未进入 Git，克隆仓库不会自动获得实验大文件。

```powershell
$recordedRun = 'runtime/fresh-v7-full-20261123'
$recordedImage = 'gas-rsi-abc:wyx-sdk104-v7-20261123'
$recordedImageId = 'sha256:259ad61698efd54431e15b5f7da917a3388261a4287a594bf0668695502caa37'

# 只读核验原始证据，不调用模型或云端
.\.venv\Scripts\python.exe -B -X utf8 scripts/verify_closed_loop.py `
  --normal "$recordedRun/training" --recovery "$recordedRun/recovery" `
  --output runtime/my-fresh-evidence-verification.json

# 在原固定镜像内重放，新增模型调用和云端写入均为 0
$replayOut = 'runtime/replay-abc-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
.\.venv\Scripts\python.exe -B -X utf8 scripts/replay_multitask_cloud.py `
  --source "$recordedRun/training" --output $replayOut `
  --image $recordedImage --expected-image-id $recordedImageId `
  --docker-executable $dockerExe --docker-context desktop-linux
```

重放证明保存回复条件下的数值可复现性，不保证重新请求大模型时得到完全相同的方案。

## 运行记录保存在哪里

| 位置 | 内容 |
|---|---|
| `summary.json` | 全流程状态、各组独立核验、真实调用数、总览链接 |
| `training/{A,B,C}/normal/host-*/protocol.jsonl` | 完整 SDK 请求/响应，包括读历史、上传和完成确认 |
| `host-*/state/` | 监测、诊断、决策及策略状态；失败尝试也保留 |
| `normal/provider-calls/`、`recovery/provider-calls/` | 本地原始请求、提供方响应与回执，未保存授权头 |
| `measurement-*/predictions.jsonl` | 原生预测入口逐条输出，以及对应实测指标 |
| `reproduction/`、`source-sha256.json` | 完整源码快照及字节摘要 |
| `image-preflight.json` | 镜像身份、应用文件一致性、资源与隔离检查 |
| `cloud-receipt.json`、`cloud-verification.json` | Run 身份、结束状态、云端内容回读和摘要校验 |
| `replay/verification.json` | 模型文件、预测和最终选版一致性 |

一个正常 Run 可以包含多轮完整记录；不能把网页上只有一个 Run 理解成只做了一轮。最新总控创建六个阶段 Run 和一个汇总 Run，汇总本身不算历史实验。

当前总控使用 [ResultsOnlyJournal](scripts/results_only_journal.py)：云端保存指标、方案/选版理由、调用号和诊断摘要；原始请求、完整源码、样本和权重保留在本地。对每条云端 Text 和关键指标回读，再重组 ZIP 核对摘要。旧 `CloudJournal` 的完整归档器仍供历史工具使用，不要把它与当前结果归档模式混淆。

## 核心代码阅读顺序

核心自有模块已补中文注释，按“为什么这样做—依据什么—输出什么”解释关键步骤；官方 SDK 和原始参考模型保留原样。

| 文件 | 重点 |
|---|---|
| [contest_submission.py](contest_submission.py)、[multitask.py](src/gas_a/multitask.py) | 任务声明、CPU 线程、A/B/C 原生分发 |
| [A 主控](src/gas_a/submission.py) | A 的模型规划、真实续训、硬门槛、历史恢复及共享 SDK 决策函数 |
| [B/C 主控](src/gas_a/regression/submission.py) | 从合同检查到保存/结束的 11 个阶段，动作目标不能混搭 |
| [回归数据](src/gas_a/regression/data.py) | B/C 特征顺序、单位、检查点与投影检查 |
| [回归监测诊断](src/gas_a/regression/monitoring.py) | 固定训练参考、三种分布距离、寿命分段误差 |
| [回归训练](src/gas_a/regression/training.py) | 继承 Adam、只训练输出层的方案、梯度裁剪、早停与最佳 epoch |
| [回归评估](src/gas_a/regression/evaluation.py) | 原单位误差、NASA、本地/宿主准入 |
| [回归提示词](src/gas_a/regression/prompts.py)、[A 提示词](src/gas_a/controller_prompts.py) | 当前父身份、允许组合与可验证证据 |
| [A 续训证明](src/gas_a/continuation.py)、[策略](src/gas_a/evolution.py)、[冻结路由](src/gas_a/frozen_router.py) | 完整父子状态、受控策略改进、无标签独立预测 |
| [全流程总控](scripts/test_fresh_platform.py)、[实际测量](scripts/validate_multitask.py) | 新起点、固定源码/镜像、真实/替身区分、失败终态 |
| [回滚测试](scripts/test_multitask_recovery_cloud.py)、[回复重放](scripts/replay_multitask_cloud.py) | 预设故障、完整继承、逐文件与逐条预测核对 |
| [结果归档](scripts/results_only_journal.py)、[云端生命周期](scripts/cloud_experiment.py) | 上传边界、结束收尾、回读校验 |

## 历史资料与验证边界

- [最新全新 v7 测试](docs/FRESH_V7_REAL_TEST_20261123.md)：2026-11-23 13:59 起，36 次新的真实调用，A/B/C 正常与回滚均为固定 v7 镜像。
- [此前全新 v6 测试](docs/FRESH_PLATFORM_TEST_20261123.md)：2026-11-23 12:22 起的历史验收，单独保留，不与 v7 混算。
- [此前 v6 修复与混合阶段验收](docs/CLOSED_LOOP_ACCEPTANCE_V6_20261123.md)：保留 v5 首次回滚失败及修复过程，正常阶段采用此前 v5 记录；不与最新一次混算。
- [v5 多任务适配](docs/MULTITASK_V5_20261123.md)、[v5 正常实验](docs/REAL_MULTITASK_RESULTS_20261123.md)：补齐 B/C 原生接口的开发记录。
- [旧 v4 A 云端实验](docs/CLOUD_TEST_20261122.md)、[历史回放](docs/LEGACY_REPLAY.md)：v4 仅支持 A；带批后标签反馈的归档回放不能作为平台预测入口。
- [参数说明](SUBMISSION_CONFIG_GUIDE.md)、[平台验收](PLATFORM_VALIDATION.md)、[SDK 官方说明](vendor/contest-harness-sdk-1.0.4/HARNESS_SDK.md)。

公开验证被用于调优和选择，因此收益仍可能有选择偏差；合成故障恢复也不等于已经覆盖真实隐藏流的所有退化。当前实现的受控菜单、少样本类别保护边界、跨轮累计退化风险与正式平台适配，都应结合真实平台连续回合继续验证。
