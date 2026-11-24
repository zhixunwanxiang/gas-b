# v8：平台 180 秒超时排查与启动/预测优化

日期：2026-11-24。作者：wyx。

目前平台只给出“镜像预检单步超过 180 秒”，没有具体命令或对应日志。本地未复现同样的 180 秒超时，不能直接认定为死循环，也不能认定已找到了平台故障的唯一原因。本次定位并去掉了实际存在的重复开销，构建新镜像，验证启动、逐行预测和既有回复下的完整多轮运行。

## 改动及依据

1. **轻量启动**。原 describe 会通过选手模块导入 A/B/C 训练、NumPy 和路由代码。现在只加载任务声明及回调包装函数，真正收到训练帧或首条预测才导入业务。测试直接阻止导入 NumPy/gas_a 后运行 describe，仍能完整输出 SDK 声明。
2. **减少逐条文件访问**。性能剖析中，C 的 2,000 条预测约 2.39 秒，主要耗在反复 resolve、stat 和 lstat。平台模型在一次预测进程中不可变；现在仅当文件解析后位于 `/model` 且 `statvfs` 确认挂载只读，才缓存已核验路径及文件身份。可写目录、只读性无法确认、路径变化或产物 ID 变化均不会盲目复用。文件身份缓存最多 4 项，不缓存样本、标签或预测结果。
3. **避免重复生成 schema**。A 不再逐条生成 128 个 JSON 路径；B/C 的固定路径缓存为不可变 tuple，对外返回独立 list，数值仍逐条检查。B 按官方字典键排序，C 先 sensors 后 settings；用打乱字段顺序的随机输入逐项对比官方 flatten。布尔、NaN/Inf、额外字段、嵌套值和错误维度继续拒绝。
4. **增加定位日志**。stderr 记录入口、数据加载、父评估、读历史、历史恢复、模型 RPC、本地训练、候选提交、状态保存和结束确认的开始/结束及耗时。日志不含密钥、逐条样本或完整提示。stdout 仍仅输出官方协议 JSONL。

没有向 stdout 插入伪造心跳，没有修改官方 SDK，没有捕获 SDK 中断后继续 RPC，也没有为结果未知的请求增加自动重试。训练预算、候选门槛、模型权重计算和标签边界保持原规则。

## 最终镜像身份

- 镜像：`gas-rsi-abc:wyx-sdk104-v8-20261124`。
- 实际运行 ID：`sha256:6f980b030dbb49bc089a0c601bbf034ac481c4d0d7d9c722e3eb36dc862ef476`。
- 镜像 tar：`output/submission-wyx-sdk104-v8-20261124/gas-rsi-abc-wyx-sdk104-v8-20261124.tar`。
- tar 大小：61,963,264 字节。
- tar SHA-256：`bd633a2585f85a106110c6bc3cec63d989747d1b64e464fdf4a62c8182cf5d04`。
- SDK 1.0.4，Linux amd64，Python 3.12.13，NumPy 1.26.4，UID 65534。
- 基于已核验的 v7 本地运行环境离线构建，关闭额外构建证明清单；归档内 4 个 OCI blob 摘要全部核验。

镜像不含凭证；本次模型调用和云端写入均为 0。v7 的真实 27B 回执、原镜像和原实验目录保留原样。

## 最终容器耗时

在禁网、根目录只读、2 CPU/4 GiB 条件下测试。describe 和交互预测在 stdin 保持打开、下一条输入及 EOF 尚未发送时就已收到回复。

| 检查 | v7 本地测量 | 最终 v8 本地测量 |
|---|---:|---:|
| 显式 describe 首帧 | 0.425 秒 | 0.320 秒 |
| v8 默认启动首帧 | — | 0.326 秒 |
| 规范 `python3 /app/harness.py describe` 首帧 | 0.398 秒 | 0.350 秒 |
| A 1,244 条完整公开预测 | 2.559 秒 | 0.823 秒 |
| B 1,679 条完整公开预测 | 2.533 秒 | 0.799 秒 |
| C 23,049 条完整公开预测 | 28.727 秒 | 2.017 秒 |

v8 三组预测首帧均小于 0.5 秒；其后交互抽测的单条最长约 0.002 秒。全集输出与原 v7 真实测试保存的预测逐字节一致。上述秒数包含本地 Docker 与挂载环境因素，不能当成正式设备上的时延承诺，也不能推出任意长度隐藏流总耗时都小于 180 秒。

168 项项目离线测试通过。最终固定镜像另使用 v7 保存的 24 条真实 27B 回复，对 A/B/C 各 4 轮重新进行实际数值训练、SDK 上传、宿主预测、选版及状态继承；具体逐文件/逐条预测一致性以 `v8-accepted-replay/verification.json` 为准。该重放不计为新模型调用。

最终重放已全部通过：16 个检查点/路由文件、27 组逐条预测和全部选版一致。12 轮记录中数据加载最长约 8.01 秒、本地训练最长约 0.39 秒、候选提交及宿主验证最长约 9.74 秒。这里模型 RPC 消费保存回复，不能用其毫秒级耗时承诺真实网关也会快速返回。导出 tar 已实际 Docker 加载并核对 ID 一致。

## 本地证据和复现

完整目录：`runtime/timeout-audit-20261124/`。

| 证据 | 内容 |
|---|---|
| `v7-baseline/results.json` | 原提交 v7 的保持 stdin 打开与全集预测测试 |
| `v8-profile.log` | 2,000 条 C 输入的性能剖析，定位文件系统调用 |
| `full-tests-final.log` | 168 项测试结果 |
| `v8-accepted-latency/results.json` | 最终镜像 ID、容器约束、响应和全集耗时 |
| `v8-accepted-replay/` | 最终固定镜像四轮重放的协议、诊断、阶段日志、候选和逐条预测 |
| `application-sha256.json` | 最终镜像应用文件与当前源码的字节核验 |
| `archive-verification.json` | tar 大小、摘要、OCI blob 与标签核验 |
| `final-verification.json` | 本次最终汇总和每阶段最大耗时 |

中间版本一次重放被并行重建同名标签打断，Docker 报 `No such image`，日志保留在 `v8-recorded-replay/`。这不是成功验收，也不是平台 180 秒问题的复现；最终验收改用固定的新 ID，运行期间不重建或换标签。早期预检因误用 BuildKit config 摘要而拒绝运行的记录也保留，不混入最终通过结果。

解压交付材料后，先运行 `restore_materials.py` 恢复一份按摘要去重的初始权重。精简包附源码、SDK、依赖、初始检查点与验证摘要；完整公开数据、逐条输出和每轮权重保留在本地 runtime，精简包不替代这些大文件。

在项目根目录用 PowerShell 验证：

```powershell
docker load -i output/submission-wyx-sdk104-v8-20261124/gas-rsi-abc-wyx-sdk104-v8-20261124.tar
docker run --rm --network none --read-only gas-rsi-abc:wyx-sdk104-v8-20261124 describe

.\.venv\Scripts\python.exe -B -X utf8 scripts/validate_image_latency.py `
  --image gas-rsi-abc:wyx-sdk104-v8-20261124 `
  --expected-image-id sha256:6f980b030dbb49bc089a0c601bbf034ac481c4d0d7d9c722e3eb36dc862ef476 `
  --recorded-run runtime/fresh-v7-full-20261123 `
  --output runtime/my-v8-latency
```

Docker 不在 PATH 时添加 `--docker-executable`，输出目录必须不存在。省略 `--recorded-run` 时只测入口，不声称已经测过预测。

建议用本目录的 **v8 tar** 新建平台提交并重新试运行。如果仍出现 180 秒提示，需要平台给出具体执行命令、超时阶段和最后一条日志；本地无法替组委会检查镜像导入队列、资源分配、宿主网关或未收到 start 的情形。训练收到 start 前按协议等待读取、模型 RPC 等待宿主回复，都应与本地死循环区分，不能靠伪造响应“绕过”超时。
