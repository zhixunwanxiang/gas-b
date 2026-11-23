# 正式闭环与平台验收

最新一次从初始基线开始的 A/B/C 全流程模拟已完成：同一 v6 镜像、18 轮真实决策、36 次真实 27B 调用、每组 3 次完整历史实验、三组回滚/继承及数值重放通过。7 个云端 Run 均为 FINISHED，详见[最新全新测试报告](docs/FRESH_PLATFORM_TEST_20261123.md)。正式平台基线和隐藏评分仍需实际提交确认；后续中文注释整理不改变逻辑，当前源码的严格镜像测试应先重建再核对摘要。

此前 v5 正常实验、v5 首次回滚失败和 v6 修复复测保留在[历史闭环验收](docs/CLOSED_LOOP_ACCEPTANCE_V6_20261123.md)，不与最新一次混算。

当前v5已补齐B/C原生寿命回归，同时声明classification与regression。旧v4只支持A，不能用于要求A/B/C全部可运行的提交。实现与验证范围见[多任务适配说明](docs/MULTITASK_V5_20261123.md)；下面的v4真实云端结果仅作为A任务历史证据。

最新 v4 云端实测：实际 Docker 镜像、12 次真实 27B 调用、四轮正常训练、三次非空历史实验、两轮独立故障回滚通过；公开 Macro-F1 `0.714030 → 0.798702`。[云端实验](https://swanlab.cn/@1218/gas-b/runs/928d73e54bbc4c46bfd28e67487d571e/chart)状态 **FINISHED**，121 条文本、10 个关键指标点及完整归档已下载核验。见[完整报告](docs/CLOUD_TEST_20261122.md)及[复现/下载命令](docs/CLOUD_EXPERIMENTS.md)。这仍是公开数据上的本地宿主模拟，正式隐藏集成绩未验证。

此前未接云端的独立 v4 测试为 `0.714030 → 0.838108`，真实模型选择与本次不同，见[历史报告](runtime/image-real-27b-v4-20261122/REAL_IMAGE_TEST_REPORT.md)。旧 v3 的真实回滚失败记录继续保留。

正式入口为 `harness.py train/predict`，协议为 SDK 1.0.4 的 `contest.harness.workspace.v2`。
`scripts/closed_loop_validation.py` 仍是历史回放工具；它的批次标签反馈机制不进入正式入口。
按用户要求，本次不审核《实验记录规范》。云端完整运行证据已完成留档及回读，不代替该规范的字段验收或正式平台提交。

2026-11-22 已新增本地真实 27B 的 SDK 测试：正常四轮形成三次完整历史实验记录，公开 Macro-F1 0.714030 → 0.798702；真实模型另行选择回滚并验证下一轮父产物。共 14 次模型响应（包含修复前两次空 JSON 回复），原始 API 响应与 SDK 回执已逐条核对。见 [真实测试说明](docs/REAL_SDK_REHEARSAL.md) 和 [核验结果](runtime/real-27b-sdk-20261122-results/verification.json)。这些是本地模拟宿主和 PyTRiO 提供方请求 ID，仍不是正式宿主模型绑定、计量或隐藏集成绩证据。

## 已补齐的行为

1. **监测与诊断**：首轮仅从授权 train 的特征建立固定分位数参考，保存到 SDK 状态；后续轮次沿用参考，不重新拟合模型标准化参数。用当轮公开 validation 的特征测量 PSI、分箱 KS 近似及均值偏移，汇总 16 个传感器。公开标签仅用于混淆矩阵、类别错误、评估和选版。诊断明确区分相关性与物理因果。
2. **自主决策**：每轮模型读取监测、诊断、历史、策略版本和当前实测指标，选择 `observe`、`experiment` 或 `rollback`。观察不训练、不提交伪候选；实验始终从宿主指定的实际父产物续训。
3. **上线门槛**：候选和历史版本在同一份当轮公开 validation 上重新评估。默认 Macro-F1 至少提升 0.0005；支持数至少 20 的类别召回最多下降 0.20（20 个百分点）。候选还需通过宿主公开范围、验证摘要和测量一致性校验，未知口径或冲突不能绕过。项目保护策略不代表官方另有该阈值，实际宿主摘要若不同需显式适配。
4. **真实回滚**：每轮最多恢复两个宿主允许的历史产物。使用 SDK 分块读取并核验完整文件清单、摘要和路由组件，重新评估后才允许选中历史 ID。回滚与保留父模型分别记录；下一轮起点仍以宿主父产物为准。历史源代码不会在训练进程中执行。
5. **受控策略版本**：记录实验结果；只有被采用的实验才能提升策略版本并成为优先实验。相同父摘要、数据摘要和实验配置累计拒绝两次后，该组合从菜单移除。回滚时恢复已知的配套策略；外部未知策略采用初始策略。这里只调整受限实验偏好，不开放任意改代码或扩大预算。
6. **模型回复纠错**：成功收到但不合法的 JSON 最多纠正一次；全部真实调用 ID 写入状态和最终记录。模型、上传、恢复等 RPC 中断时立即停止，不重试结果未知的请求。
7. **27B 提示词与训练选版衔接**：当前控制器 v3 明确数值准入线、实验能力与风险、同父/同数据/同执行策略的历史反馈；选版提示区分宿主测量、主组件与整体路由，并接收程序校验和本地失败证据。正式入口在逐 epoch 导出时也使用同一晋升门槛。此前真实模型证据对应 v2 保存源码；v3 本轮仅做离线测试，不能作为真实提示词 A/B 结论。
8. **本地失败恢复**：训练前写入实验身份，已知 CPU 训练失败写入失败记录，模型再决定保留父模型并结束本轮；失败计入重复限制。只有本地计算在恢复范围内，不捕获未知结果的 RPC 后继续发送请求。

`submission_config.json` 保存固定门槛和实验菜单。模型只能在已验证的动作、配置和产物 ID 内决策。
`monitor_reference.json`、`monitoring.json`、`diagnosis.json`、`strategy.json`、`decisions.jsonl` 随 SDK 状态跨轮传递；读写发生于可写工作目录。

训练容器只能监测其授权公开数据。SDK 没有提供隐藏流标签/遥测回传通道，本实现不会把公开验证变化描述为隐藏流变化。
预测继续只接收 128 维特征，逐行输出六类概率，不依赖标签回填。

## 本地验收

在项目根目录执行，输出目录必须尚不存在：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

$baselineDir = 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1'
.\.venv\Scripts\python.exe -B scripts/validate_platform.py --parent "$baselineDir/checkpoint.npz" --train "$baselineDir/public/train.jsonl" --validation "$baselineDir/public/validation.jsonl" --output runtime/rehearsal-new --rounds 10
```

演练调用真实入口和 SDK，宿主与决策模型使用离线替身，不访问云服务、不消耗正式试运行额度。
第 3 轮主动观察，第 4 轮注入一次已完成的错误 JSON；其余轮次按受限菜单决策。
每轮保留协议、候选、SDK 状态、模型调用和选版结果；结束后核验十条预测均在下一条输入和 EOF 前返回，并与选中产物一致。
演练只测兼容性与控制行为，不能证明正式成绩提升。合成故障回滚由 `tests/test_evolution.py` 和 `tests/test_frozen_router.py` 单独验证，其分数不作为官方数据成绩。

官方 SDK 的离线测试应在 SDK 源码目录显式运行以下三个模块，不使用会自动发现 `test_live.py` 的全目录扫描：

```sh
python3 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
```

本项目 Windows 环境可在项目根目录执行：

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

这三个模块使用 JSONL 宿主替身，不调用真实模型。官方宿主仓库的协议回归测试位于其 `runtime/tests/test_sdk_host_contract.py`；SDK 下载包不包含宿主，当前选手项目也未包含这份测试，选手侧测试不构成宿主回归验证。

`tests/test_live.py` 须另行运行。设置 `PYTRIO_API_KEY` 后，真实模型调用可能产生费用；设置 `CONTEST_PI` 指向精简 pi 的 `pi.mjs` 后，还可测试 SDK 传输层握手。运行前应确认密钥、模型和环境，离线验收不会自动运行它。此前记录的“两项跳过”表示没有执行真实模型验证。

SDK 的 `example/contest_submission.py` 仅展示协议和模型调用，不提供可得分的训练方案。本项目的数值训练、离线单元测试和宿主替身演练也都不能证明正式工业模型成绩；实际分数须以比赛平台评测为准。

## Docker 验收入口

在安装并启动 Docker 的机器执行：

```powershell
docker build -t gas-a-rsi:sdk104 .
docker run --rm --network none --read-only gas-a-rsi:sdk104 describe
.\.venv\Scripts\python.exe -B scripts/validate_platform.py --image gas-a-rsi:sdk104 --parent "$baselineDir/checkpoint.npz" --train "$baselineDir/public/train.jsonl" --validation "$baselineDir/public/validation.jsonl" --output runtime/container-rehearsal-new --rounds 10
```

脚本以非 root 镜像用户运行，禁网、只读根文件系统、2 CPU、4 GiB 内存；train 仅只读挂载授权数据、父产物与上轮状态，工作目录使用 4 GiB tmpfs。predict 仅挂载选中产物，不挂载训练数据，并测试逐行刷新。容器检查仍使用离线模型替身。
镜像构建需要基础镜像和 NumPy wheel；正式运行不下载依赖。2026-11-22 已使用本机 Docker Desktop 29.9.3 构建并完成实际容器验收：非 root、只读根目录、禁网、2 CPU/4 GiB、十轮离线实验及独立故障回滚均通过；镜像及材料合计 59.44 MiB。详见 [镜像交付说明](docs/IMAGE_RELEASE_20261122.md)。

## 仍需在比赛平台完成

在平台提交页绑定获准的 PyTRiO 训练模型/Sampler 或 PyTRiO Qwen3.8-27B，并以本届平台实际提供的标识为准。镜像通过 SDK 的宿主网关访问该模型，不在代码中写死身份或密钥。
每轮 `configuration.controller_model_name` 记录宿主声明的模型名；它本身不能证明模型来源合规，应与平台绑定页面、真实调用回执相互核验。

使用构建完成的镜像执行正式平台试运行，保存镜像摘要、实际模型绑定、完整十轮回执、资源报告和平台允许导出的成绩，确认无人工干预及同口径静态基线对比。当前已有旧版源码的本地真实 27B 调用证据；正式宿主模型绑定、容器运行和隐藏集成绩仍未验证，不能宣布全部参赛要求已通过。
