# SDK 1.0.4：v10 修复与并行十轮验收

v10 已修复完整模型回复反复校验失败后直接退出的路径。A/B/C 三路并行、每组无人干预 10 轮的本地离线验收通过。按用户选择，新增付费调用为 0；保留已有 36 份 `Qwen/Qwen3.8-27B` 回执。平台截图只有 `ValueError` 分类，没有堆栈，不能确认其唯一根因，也不能把本地通过写成正式平台通过。

## 改动及原因

- 原先模型两次完整回复仍有 JSON、菜单、父模型组合或选版错误时，第二次校验异常直接结束训练。现在只对这类已完整返回的回复执行有限纠正；仍失败则保留实际父模型。规划阶段不训练，选版阶段不采用未经批准的候选。
- 保护决定标记为 `program_guard`，不伪装成模型选择；完整请求、原始回复、SDK 回执号、请求摘要和校验原因均留档。程序保护不计入成功模型实验闭环。
- 支持完整 JSON 围栏的 LF、CRLF、大小写 JSON 标记及无语言标记；仍拒绝额外字段、重复键、非有限数值、超长文本和白名单外选择，不从任意说明文字中猜测 JSON。
- SDK 请求异常不在恢复范围内。结果未知、上传中断、错误数据、检查点或状态损坏仍报错；不会在结果未知后继续请求或伪造确认。
- 原版 `harness.py` 和 SDK 源码保持不变。核验 start、单个未完成请求、连续 ID、对应回复、分块偏移和摘要、上传结束后事务、实际历史与调用引用、最终接受确认及进程退出。

## JSON 字节限制

以 SDK `contest_harness/round.py` 的 `_text`、`_declaration`、`finish` 为准：

| 字段 | 上限 | 30 轮实测最大值 |
| --- | ---: | ---: |
| hypothesis | 4000 UTF-8 字节 | 59 |
| change_summary | 4000 UTF-8 字节 | 119 |
| reason | 4000 UTF-8 字节 | 79 |
| configuration | 65536 序列化字节 | 5151 |

配置严格按 SDK 的 `sort_keys=True, allow_nan=False, separators=(",", ":")` 和默认 `ensure_ascii=True` 计算，中文转义后也计入长度。不是按字符数计算，也没有把 64 KiB 错套为所有协议帧总长度。

文本超限触发纠正和父模型保护，不截断成无效 JSON。配置超限时，完整对象及摘要写入 SDK 上传状态的 `configuration-evidence.jsonl`，协议只提交小型摘要与证据引用。已用原 SDK 验证 4000/4002 字节中文边界、65536 字节配置边界和超限完整留档。

## 本地验收

镜像：`gas-rsi-abc:wyx-sdk104-v10-20261124`。

不可变 ID：`sha256:de5a58d5c3fbeafa1399b531dadff78fd24067a818b094a16747a90cfba45b71`。

| 数据集 | 轮数 | 非空历史完整实验闭环 | 公开基线 | 最终公开指标 | 最长单轮 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A，Macro-F1 越高越好 | 10 | 9 | 0.7140297221 | 0.8058813363 | 6.797 秒 |
| B，RMSE 越低越好 | 10 | 8 | 681.1598326 | 651.6266612 | 6.797 秒 |
| C，RMSE 越低越好 | 10 | 9 | 52.8685726 | 51.6145352 | 28.438 秒 |

总耗时 242.604 秒，这是三组各十轮的总时间，不是单轮超时。数据集内部逐轮继承实际选中模型、状态与历史。三组最终指标均不低于同口径公开基线；离线模型替身决定实验，数值训练和预测真实执行。

Docker 捕获 92/92 个本次容器的完整 start/die 区间，最多 3 个训练容器同时运行。嵌套预测不算第四个训练任务。每个容器为 2 CPU、4 GiB、非 root、根目录只读、禁网。训练只挂载官方公开 train/validation、实际父模型及旧状态；预测只挂载模型并输入特征，不挂载带标签数据。未获取隐藏测试标签。

项目 187 项测试通过；源码 ZIP 解压至全新目录后，187 项测试再次全部通过。包内含 A 的公开单元测试输入与基线检查点。原版 SDK 的 18 项离线测试通过。Windows 默认编码下原 SDK 首次有 1 项失败，显式设置与镜像一致的 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8` 后通过；初次日志保留，未修改 SDK 测试或源码。

固定镜像另外通过 15 项故障注入：A/B/C 分别注入错误 JSON、菜单外方案、错误父 ID、非法选版及 4002 字节中文假设。全部保留父模型、持久化证据、引用实际回执并收到结束确认。这些保护回合不冒充成功实验。

既有真实回复在新镜像中各四轮重放通过：24 份原回复、16 个检查点/路由文件、27 组逐条预测及每轮选版一致。重放不是新真实调用，旧回执也不等于新镜像的真实网关验收。

## 材料和复现

交付目录：`output/submission-wyx-sdk104-v10-20261124/`。完整过程在 `runtime/sdk104-v10-20261124/`，不随 Git 提交。

- `acceptance.json`：独立逐轮检查，排除程序保护后统计完整闭环。
- `parallel-10/parallel/{A,B,C}/run/host-N/`：完整双向协议、stderr、SDK 计时、候选及上传状态。
- `state/model-responses.jsonl`：完整模型请求和回复；`state/decisions.jsonl`：计划、实测和最终决策；`experiment_attempts.jsonl`：已开始、训练成功或失败的尝试。
- `controller-recovery/verification.json`：15 项保护验证。
- `recorded-replay/verification.json`：既有真实回复重放验证。
- 原真实证据仍在 `runtime/fresh-v7-full-20261123/`，本次附摘要索引，不覆盖旧文件。

精简源码包不重复分发全部官方公开数据、逐条预测和轮次权重。复现需要三个官方公开包，各包含 `checkpoint.npz`、`baseline-info.json`、`public/train.jsonl`、`public/validation.jsonl`。摘要保存在 start 帧和材料清单中，完整 runtime 目录保留全部测试产物，精简 ZIP 不替代它。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
python -m pip install -r requirements.txt
docker load -i output/submission-wyx-sdk104-v10-20261124/gas-rsi-abc-wyx-sdk104-v10-20261124.tar
python scripts/benchmark_sdk_agent.py --output runtime/v10-parallel-reproduction `
  --baseline-root 'D:/liyue/工业模型赛题研究' `
  --sdk-archive 'D:/liyue/工业模型赛题研究/contest-harness-sdk-1.0.4.tar.gz' `
  --image gas-rsi-abc:wyx-sdk104-v10-20261124 `
  --expected-image-id sha256:de5a58d5c3fbeafa1399b531dadff78fd24067a818b094a16747a90cfba45b71 `
  --rounds 10 --modes parallel
python scripts/verify_sdk_agent_acceptance.py --run runtime/v10-parallel-reproduction `
  --output runtime/v10-parallel-reproduction/acceptance.json
```

Docker 不在 PATH 时加 `--docker-executable` 指定绝对路径，输出目录必须全新。`tested-source` 保存验收时的源码及摘要。验收后测试工具的默认镜像由 v9 改为已测 v10；显式运行参数不变，应用源码无变化。

正式平台基线、隐藏成绩及真实网关延迟仍须正式运行确认。本地不覆盖平台排队和资源争用。需上传新的 v10 TAR 创建新提交，重试旧 v3 提交不会替换镜像。
