# 真实镜像测试与云端实验留档

`scripts/test_cloud_suite.py` 将真实 27B、SDK 1.0.4、实际 Docker train/predict 和 SwanLab 留档串起来。云端记录发生在本地宿主，镜像继续禁网；PyTRiO、SwanLab 密钥不进入容器、候选或证据归档。仓库和云项目叫 `gas-b`，本次实际执行的任务合同是 **A：六类气体分类**。

## 本次结果

2026-11-22 的[云端实验](https://swanlab.cn/@1218/gas-b/runs/928d73e54bbc4c46bfd28e67487d571e/chart)状态为 **FINISHED**。四轮正常运行、三次非空历史驱动实验和两轮独立回滚共使用 12 次真实 Qwen/Qwen3.8-27B 回复；公开 Macro-F1 从 0.714029722 提升至 0.798701650。121 条云端文本和 10 个关键指标点已逐条下载核验，575 个文件的完整证据 ZIP 已从云端恢复且 SHA-256 一致。

详见可在 GitHub 查看及下载的[本次报告](CLOUD_TEST_20261122.md)。原始独立核验和云端回读文件位于本地 `runtime/cloud-rsi-v4-20261122/`，不随 Git 上传。本次将六个 SDK 回合记录在一个 SwanLab Run 内，三次完整记录不等于三组独立会话。这是官方公开数据上的本地宿主模拟验收，正式隐藏集成绩仍以比赛平台为准。

## 启动一份新实验

宿主需要 Python 3.12、Docker Desktop、官方基线项目，以及自己的 `.env`：

```dotenv
PYTRIO_API_KEY=填写自己的密钥
PYTRIO_BASE_URL=https://pytrio.com
PYTRIO_MODEL=Qwen/Qwen3.8-27B
SWANLAB_API_KEY=填写自己的密钥
GAS_A_SWANLAB_PROJECT=gas-b
# 可选：GAS_A_SWANLAB_WORKSPACE=自己的空间标识；留空时使用当前 API 账号
```

以下命令在项目根目录 PowerShell 执行，会产生真实模型费用。`--output` 必须是新目录；默认限制正常阶段 16 次、回滚阶段 6 次模型调用，并非保证全部用完。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-platform.txt
$out = Join-Path runtime ('cloud-rsi-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
.\.venv\Scripts\python.exe -B -X utf8 scripts/test_cloud_suite.py `
  --baseline-dir 'D:\liyue\工业模型赛题研究\cpu-a-guide-20261110-v1' `
  --output $out `
  --docker-executable 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe' `
  --docker-context desktop-linux
```

默认固定镜像 `gas-rsi-a:wyx-sdk104-v4-20261122` 及 ID `sha256:388fb0d9721fa11d0cb9befcb6dee6dbadb23597dfdb59415913d276b4ec25fe`。迁移机器先加载交付 tar；Docker 已在 PATH 时可改为 `--docker-executable docker`。若另行构建镜像，需显式给出匹配的 `--image` 和 `--expected-image-id`，宿主仍会核对 `/app` 中的实际源码摘要。

程序自动完成：预检模型/镜像/凭证隔离 → 新建私有云端实验 → 正常四至八轮 → 非空历史实验计数 → 独立权重故障回滚两轮 → 上传完整证据 → 结束实验 → 回读状态、文本、归档及关键指标。首轮空历史不计入三次；观察和没有提交候选的回合也不计入。未满足要求不会写为成功。

SDK 请求经本地宿主转发至真实 PyTRiO 服务，原始响应和提供方 ID 原样交回 `rnd.model()`。本地固定任务提示只附加一次，不改写模型决定。网络或结果未知的模型请求不自动重试。回滚演练只改变父模型副本的输出层权重，不修改数据、标签或历史产物。

## 云端保存什么

| 云端键 | 内容 |
|---|---|
| `suite/configuration` | 数据、代码、镜像摘要，SDK/控制器版本及调用预算 |
| `calls/request`、`calls/response` | 实际 SDK 请求、绑定任务消息、原始模型回复与真实响应 ID；不含授权头 |
| `normal/round_record`、`recovery/round_record` | 实际读过的历史 ID、实验配置、假设、决策理由、调用 ID 和完成确认 |
| `*/monitoring`、`*/diagnosis` | 公开特征漂移、薄弱类别、混淆及统计关联诊断 |
| `*/training_json`、`*/routing_training_json`、`*/continuation_json` | 实际训练、逐 epoch 测量、路由和父子权重信息 |
| `*/selected_macro_f1` 等指标 | 每轮实测成绩、漂移标记、策略版本、历史数量及训练曲线 |
| `evidence/bundle_manifest`、`evidence/bundle_chunk` | 完整 ZIP 的清单及可逆 base64 分块，各块与整个 ZIP 均有 SHA-256 |

完整 ZIP 包含运行源码快照、依赖锁定信息和 SDK wheel、官方公开输入/基线、完整模型及状态、SDK JSONL、真实 API 请求响应、逐条预测与验证测量。镜像 tar 单独交付，不重复上传进此 ZIP。上传前检查证据是否含当前密钥；终端、Git、环境探针关闭。

SwanLab 公有云的 `swanlab.save()` 不是这里使用的功能：依据[官方说明](https://docs.swanlab.cn/api/py-save.html)，该接口仅供私有部署。这里通过支持的 Text 媒体保存 JSON 和归档分块，因而需要下面的脚本恢复 ZIP，不能把分块表述为原生模型文件下载入口。

## 下载或重新核验，不调用模型

只用云端记录即可下载完整归档，需要有该私有实验读取权限的 `SWANLAB_API_KEY`。不新建实验、不恢复运行状态、不调用模型、不覆盖已有目标文件，也不自动解压。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/download_cloud_evidence.py `
  --run '1218/gas-b/928d73e54bbc4c46bfd28e67487d571e' `
  --output runtime/cloud-evidence-restored.zip
```

如果本地已有完整运行目录，可再次校验云端全部文本、关键指标和归档：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/download_cloud_evidence.py `
  --verify-local-run runtime/cloud-rsi-v4-20261122
.\.venv\Scripts\python.exe -B -X utf8 runtime/cloud-rsi-v4-20261122/verify_real_run.py
```

第二条只读取本地原始回复、协议、预测和摘要，重算 12 组测量及 14,928 条预测。云端媒体索引有短暂延迟时，回读器仅重查索引；完整标量接口的 `step/value` 与采样接口的 `index/data` 均可解析。测试运行已完成但回读失败时，使用此只读命令恢复核验即可，无需重新付费训练。

已保存回复的数值复现：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 runtime/cloud-rsi-v4-20261122/replay_recorded.py
```

该命令在实际 v4 镜像中重放八条原始正常阶段回复，显式禁止新的网络模型请求；比对训练权重、逐条预测和选版。重放不是新的真实调用证据。新发起真实模型调用可能产生不同选择和分数，不能要求服务端回复逐字一致。

## 生命周期与边界

每次运行在初始化前保存唯一实验 ID；只结束本次拥有的实验，并通过新 API 对象确认终态，避免把无关实验标记完成。正常异常/可处理终止执行收尾；强制杀进程或断电无法保证 Python `finally` 执行，因此未确认的终态会明确记录为待处理。

本次测试全程只用官方公开 train 的 445 条和 validation 的 1,244 条，预测仅接收特征。没有隐藏数据挂载、标签回填接口或样本标识反推。漂移基于公开 train/validation 的分布差异；诊断是统计关联，不证明物理因果。本次云端留档不替代正式平台模型绑定、隐藏流评测或《实验记录规范》的单独验收。
