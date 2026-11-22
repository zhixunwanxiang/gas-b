# PyTRiO / SwanLab 连通性检查与异常收尾

入口为 `scripts/live_platforms.py`，可用于 `gas-a` 或 `gas-b` 云端项目；项目名由 `GAS_A_SWANLAB_PROJECT` 指定。它只做一次真实模型连通性调用及对应云端记录，不代替正式 Harness 训练、比赛 SDK 调用回执或全部训练实验留档。

## 配置和只读排查

可选依赖来自 `requirements-platform.txt`。配置项见根目录 `.env.example`，复制为本地 `.env` 后填写；密钥文件不会打进正式镜像。

只检查配置是否齐全，不联网：

```powershell
.\.venv\Scripts\python.exe scripts/live_platforms.py --check-config
```

只读取已有实验状态，不创建实验、不改变状态、不调用模型；只需要 `SWANLAB_API_KEY`：

```powershell
.\.venv\Scripts\python.exe scripts/live_platforms.py --check-run qunmo/gas-b/xwiaqn6l
```

将参数换成自己的 `工作区/项目/运行ID`。返回的 `terminal` 表示云端是否已进入结束状态；`RUNNING` 不能作为工业模型仍在训练的证据。命令无法鉴权或读取状态时会非零退出，不把未知状态当作完成。

确认旧任务已经停止后，可以在 SwanLab 实验页面将其终止。上述查询命令不会结束历史记录，也不会续跑或重试它们。

## 新建一次连通性测试

下列命令会真正请求模型，可能产生费用：

```powershell
.\.venv\Scripts\python.exe scripts/live_platforms.py --output runtime/live-check-new
```

`--output` 必须是尚不存在的新目录，避免覆盖旧回执。省略时自动创建带时间和随机后缀的新目录。程序会校验账户是否可用精确配置的模型，不会回退到其他模型。

当前收尾逻辑：

1. 在任何网络操作前生成唯一运行 ID，原子保存 `receipt.json`。云端初始化使用同一 ID，即使中途失败也能定位记录。`resume=allow` 只用于这个新随机 ID，不复用历史 ID。
2. 模型检查、云端鉴权、初始化、模型请求和日志上传都位于异常处理范围。初始化中途失败时，尝试通过 SDK 获取已注册且 ID 一致的运行对象并结束；不会结束进程里其他实验。
3. 模型请求前记录 `outcome_unknown`。获得有效响应后立即保存真实响应回执，再上传指标。超时或异常不自动重试模型，避免重复请求和费用。
4. 正常退出请求 `success`，业务异常请求 `crashed`，Ctrl+C 和可处理的终止信号请求 `aborted`。保存回执失败不会跳过已有运行对象的收尾。
5. 结束调用返回后，通过 SwanLab 查询 API 重新读取云端状态，最多查询三次，每次获取新对象以避开 SDK 缓存。只有状态与本次期望一致，才标记 `swanlab_finish_completed=true`。只读轮询不会重复结束请求或模型调用。

## 如何理解回执

| 字段 | 含义 |
| --- | --- |
| `run_id` / `run_path` / `swanlab_url` | 本次运行身份与云端定位信息；早于鉴权失败时可能尚无完整路径 |
| `phase` | 当前阶段；`needs_attention` 表示云端收尾未确认，需要排查 |
| `outcome` | 整体检查结果：`succeeded`、`failed`、`aborted` 或 `needs_attention` |
| `model_call_status` | `not_started`、`outcome_unknown`、`verified_response`；未知结果不代表请求一定没发生 |
| `real_model_call` | 已得到并校验真实模型响应；为 false 时仍需结合 `model_call_status` 判断是否曾发出请求 |
| `swanlab_finish_attempted` | 已尝试结束属于本次 ID 的运行对象 |
| `swanlab_finish_returned` | SDK 的结束调用已返回，本身不能证明云端结束成功 |
| `swanlab_finish_completed` | 云端实际状态已与期望终态一致；例如失败后核验为 `CRASHED` 也表示收尾完成，不代表模型检查成功 |
| `cloud_verification` | 本次期望终态、实际状态及核验是否成功 |
| `failure` / `cleanup_error_type` | 失败阶段与异常类型，不保存可能含密钥的第三方异常正文 |

检查失败或云端完成未确认时进程非零退出。回执使用临时文件加原子替换，写入失败时保留上一版；模型回执不会因后续同步失败而主动删除。

操作系统强制杀进程、断电、SDK 在注册运行对象之前失败、网络长期不可用等情况，仍可能留下运行中的记录。此时应使用已落盘的 ID 查询和处理，不能保证 `finally` 一定执行，也不能为了补收尾再次调用模型。

## 离线回归

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_live_platforms -v
```

当前 26 项测试使用本地替身，覆盖部分初始化失败、模型超时、Ctrl+C、终止信号、结束静默失败、云端查询失败、状态不一致、磁盘写入失败、保护其他运行对象及只读排查。测试不会调用真实模型或改动云端记录。
