# 使用真实 27B 回复测试 SDK 1.0.4

最新完整云端测试使用 `scripts/test_cloud_suite.py`：四轮正常训练、三次非空历史实验、两轮独立回滚，公开 Macro-F1 `0.714030 → 0.798702`；12 次真实调用、完整训练与决策证据均已留档，云端状态 FINISHED 且归档回读一致。见[云端运行/下载说明](CLOUD_EXPERIMENTS.md)和[本次报告](CLOUD_TEST_20261122.md)。

下文的 `scripts/test_real_sdk.py` 是独立宿主组件，本身不创建云端实验；外层 `test_cloud_suite.py` 负责完整云端留档。此前独立 v4 运行的公开分数为 `0.838108`，见[历史报告](../runtime/image-real-27b-v4-20261122/REAL_IMAGE_TEST_REPORT.md)。使用 `--image` 将 train 和 predict 都放进镜像，`--expected-image-id` 固定交付摘要，密钥只由宿主读取。

```powershell
.\.venv\Scripts\python.exe -B scripts/test_real_sdk.py --baseline-dir "D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1" --image gas-rsi-a:wyx-sdk104-v4-20261122 --expected-image-id sha256:388fb0d9721fa11d0cb9befcb6dee6dbadb23597dfdb59415913d276b4ec25fe --docker-executable "C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe" --docker-context desktop-linux --output runtime/image-real-new --rounds 4 --max-rounds 8 --max-model-calls 16
```

这条命令会产生真实费用。未指定 `--image` 时沿用本机进程测试。Docker 模式验证实际资源与挂载限制，不将未插桩的逐次文件/网络访问表述为已全面观测。`recovered_checkpoint_sha256` 只有真正选回历史产物时才设置，保留故障父模型不能记作回滚成功。

`scripts/test_real_sdk.py` 在本地模拟宿主中调用真实 `Qwen/Qwen3.8-27B`，选手代码仍通过安装的 SDK 1.0.4 执行 `rnd.model()`，实际训练并提交 CPU 工业模型候选。

请求链为：选手 `rnd.model()` → SDK JSONL `model_request` → 本地宿主代发 PyTRiO 的 OpenAI 兼容接口 → 原始响应与真实请求 ID → SDK `ModelReceipt` → 选手解析、训练、选版。模型请求分支不会执行离线规则引擎，不替换或补写真实回复。

这不是组委会正式宿主：固定系统提示明确标注为本地模拟，`call_id` 使用 PyTRiO 实际响应 ID，不冒充正式平台计量回执。公开验证集的评测由本地主机独立运行逐行预测完成，不代表隐藏集成绩。

## 运行

先安装根目录 `requirements.txt`，配置本机 `.env` 中的 `PYTRIO_API_KEY`、`PYTRIO_BASE_URL=https://pytrio.com`、`PYTRIO_MODEL=Qwen/Qwen3.8-27B`。不会回退到其他模型。密钥只由宿主读取，不复制进候选或传给训练进程；不需要 SwanLab 密钥，不创建云端实验记录。

以下命令会产生真实模型费用：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/test_real_sdk.py --baseline-dir "D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1" --output runtime/real-sdk-new --rounds 4 --max-rounds 6 --max-model-calls 16
```

输出目录必须尚不存在。至少跑四个检查点；如果仍不足三次“读非空历史—实际提交实验—依据结果决策”，允许继续到 `--max-rounds`。AI 可以选择观察，因此不保证数量必定达标；数量和未达标原因均如实写入报告，不强制改写为实验。

修复后的退出码：`0` 表示本地公开验收要求全部通过，`1` 表示回合跑完但要求未满足（`status=requirements_unmet`、`requirements_passed=false`），`2` 表示运行期异常。正常测试同时要求至少三次完整历史实验、各轮结束确认与调用回执、公开成绩不低于基线及无观测到的数据/网络违规；独立回滚演练另要求实际选回历史版本并在下一轮使用恢复的父模型。预检参数或环境错误也会非零退出。

每轮通常包含计划和选版两次模型请求；已返回但 JSON 不合法时，选手当前允许一次格式修正请求，也计入总请求上限。网络异常、超时或未知结果不会自动重发同一请求；没有真实回复时终止本次测试，不伪造成功。

需要单独验证真实模型是否主动回滚时，可对一份已完成会话运行两轮权重故障演练：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/test_real_sdk.py --baseline-dir "D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1" --output runtime/real-sdk-recovery-new --recovery-from runtime/real-sdk-new --max-model-calls 6
```

只对基线副本置换输出权重和偏置，公开数据与标签不变。两轮动作均由真实模型决定，不预设必须回滚或观察；报告如实记录是否恢复以及下一轮是否使用恢复的父模型。`--max-model-calls` 是单次进程的上限，多次运行需要累计已用请求数，不能把重新运行当成费用预算归零。

## 真实模型发现并修复的问题

初次实际调用中，27B 两次返回 `finish_reason=length`，2048 个输出 Token 全部被计入 reasoning，最终 `content` 为空。选手现在在 `rnd.model()` 中显式传 `enable_thinking=False`；保持原来的输出上限和温度，不从宿主侧悄悄修改采样参数。这一参数由 SDK 1.0.4 原样传给模型请求。

修复前后测试使用不同目录，保留所有真实回执与失败记录，不把初次失败计为完整实验。

## 保存的证据

- `provider-calls/call-*/request.json`：实际发给服务的请求，不含授权头。
- `response.json`、`receipt.json`：完整原始模型响应、实际请求 ID、用量、结束原因与请求状态。
- `host-*/protocol.jsonl`：实际 SDK 请求/响应、候选与状态上传、结束确认。
- `host-*/model-calls.jsonl`：模型原始消息、一次附加的本地固定任务提示和真实模型回复。
- `host-*/candidate/`、`host-*/state/`：实际数值训练、监测、关联诊断、计划、选版和状态证据。
- `measurements/`：逐行仅特征输入、六类预测及重新计算的公开指标。
- `rounds.jsonl`、`live-report.json`：历史引用、候选、选择、真实调用数量和最终成绩。
- `selected-model/`、`selected-state/`：选中模型和原生状态。

训练/预测使用原生进程访问审计，禁止其联网；原生子进程剥离 PyTRiO、SwanLab 等凭证环境变量，并拒绝 `.env`、`.env.*`、`.netrc`、`_netrc`、`.aws/`、`.ssh/`、`.swanlab/` 路径的文件访问。这是 Python 层的应用审计，不等同于 Docker 或正式平台操作系统隔离，也不保证识别任意文件名的凭证。真实模型回复本身不保证确定性或优于基线，最终以保存的测量和决策为准。

桥接层离线测试，不联网或计费：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest tests.test_real_sdk_bridge tests.test_controller_prompts -v
```
