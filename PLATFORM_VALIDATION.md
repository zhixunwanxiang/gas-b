# 正式闭环与平台验收

正式入口为 `harness.py train/predict`，协议为 SDK 1.0.4 的 `contest.harness.workspace.v2`。
`scripts/closed_loop_validation.py` 仍是历史回放工具；它的批次标签反馈机制不进入正式入口。
按用户要求，本次不审核《实验记录规范》。以下文件是系统运行证据，不代表完成该规范或云端记录验收。

## 已补齐的行为

1. **监测与诊断**：首轮仅从授权 train 的特征建立固定分位数参考，保存到 SDK 状态；后续轮次沿用参考，不重新拟合模型标准化参数。用当轮公开 validation 的特征测量 PSI、分箱 KS 近似及均值偏移，汇总 16 个传感器。公开标签仅用于混淆矩阵、类别错误、评估和选版。诊断明确区分相关性与物理因果。
2. **自主决策**：每轮模型读取监测、诊断、历史、策略版本和当前实测指标，选择 `observe`、`experiment` 或 `rollback`。观察不训练、不提交伪候选；实验始终从宿主指定的实际父产物续训。
3. **上线门槛**：候选和历史版本在同一份当轮公开 validation 上重新评估。默认 Macro-F1 至少提升 0.0005；支持数至少 20 的类别召回最多下降 0.20（20 个百分点）。不达标只能保留父模型。这是项目保护策略，不是声称官方另有该阈值。
4. **真实回滚**：每轮最多恢复两个宿主允许的历史产物。使用 SDK 分块读取并核验完整文件清单、摘要和路由组件，重新评估后才允许选中历史 ID。回滚与保留父模型分别记录；下一轮起点仍以宿主父产物为准。历史源代码不会在训练进程中执行。
5. **受控策略版本**：记录实验结果；只有被采用的实验才能提升策略版本并成为优先实验。相同父摘要、数据摘要和实验配置累计拒绝两次后，该组合从菜单移除。回滚时恢复已知的配套策略；外部未知策略采用初始策略。这里只调整受限实验偏好，不开放任意改代码或扩大预算。
6. **模型回复纠错**：成功收到但不合法的 JSON 最多纠正一次；全部真实调用 ID 写入状态和最终记录。模型、上传、恢复等 RPC 中断时立即停止，不重试结果未知的请求。

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
镜像构建需要基础镜像和 NumPy wheel；正式运行不下载依赖。当前机器未安装 Docker，本地单元测试不能替代该项。

## 仍需在比赛平台完成

在平台提交页绑定获准的 PyTRiO 训练模型/Sampler 或 PyTRiO Qwen3.8-27B，并以本届平台实际提供的标识为准。镜像通过 SDK 的宿主网关访问该模型，不在代码中写死身份或密钥。
每轮 `configuration.controller_model_name` 记录宿主声明的模型名；它本身不能证明模型来源合规，应与平台绑定页面、真实调用回执相互核验。

使用构建完成的镜像执行正式平台试运行，保存镜像摘要、实际模型绑定、完整十轮回执、资源报告和平台允许导出的成绩，确认无人工干预及同口径静态基线对比。当前没有真实模型调用、容器运行或隐藏集成绩证据，不能据此宣布全部参赛要求已通过。
