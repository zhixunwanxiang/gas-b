# C 项目平台反馈优化与 v11c 交付（2026-11-25）

本次修复测试报告中“10 轮候选全部未采用”的主要阻塞。原 v10 镜像已在 C 官方公开数据上复现相同拒绝原因；新 v11c 镜像在精简父/候选摘要下完成 10 轮实际训练、独立预测和选版，共采用 4 次合格候选。

## 报告诊断

用户报告的正式隐藏集结果为 RMSE 56.225925、MAE 41.286567、NASA 评分 42,594,777,435.81236，最终使用起始模型。每轮本地候选多次改善，但 `host_candidate_gate` 因父模型 `unknown_validation_scope` 拒绝，未形成跨轮积累。

公开数据复现实验中，旧镜像首轮父 RMSE 为 52.868572566，候选降至 51.827463563，本地门槛通过；父摘要没有 `scope/validation_sha256`，宿主门槛拒绝，最终仍选择初始版本。这证明当前代码在相同摘要缺省条件下可产生报告中的阻塞；TXT 没有包含完整原始 start 帧，因此不能断言已逐字节还原正式宿主。

## 修改内容

- 回归链路先按 SDK 清单和父检查点摘要确认实际父权重，再在本轮授权公开 validation 上独立测量。父摘要仅缺少身份字段且具名 RMSE 与本地一致时，比较基准明确记录为 `local_current_parent_public_reevaluation`；原始宿主摘要和其未核验原因分别留档，不把本地身份说成宿主提供的证明。
- 显式隐藏范围、错误数据摘要、错误父 ID/数据集/轮次/检查点、无效指标或宿主与本地分数冲突仍然拒绝。候选仍须通过本地 RMSE/MAE 门槛和真实 SDK 候选回执核验。
- 父测量确实不可比时只开放 observe，避免重复训练无法采用的候选。规划和最终记录均带父测量来源、阻塞原因。
- 更新回归提示，区分未采用的历史候选与当前父模型，说明 bias 正负方向；新增父参考策略版本进入实验指纹，使旧规则造成的拒绝不屏蔽修复后的实验。
- 本次未修改四个训练菜单、学习率、训练标签、模型结构、RMSE/MAE 保护阈值或 NASA 目标。保留实际父权重、Adam、scaler 和 seed 的续训约束。

## 公开验证结果

同一份官方 C 公开 validation，共 23,049 条样本；以下数据不能与正式隐藏集指标直接比较。

| 指标 | 起始模型 | 10 轮最终模型 | 降低 |
|---|---:|---:|---:|
| RMSE（循环） | 52.868573 | 51.614535 | 2.37% |
| MAE（循环） | 38.899830 | 38.043047 | 2.20% |
| NASA 评分（总和） | 5,823,439,325.342465 | 3,370,559,951.751916 | 42.12% |

| 轮次 | 方案 | 本轮候选 RMSE | 结果 | 选中模型 RMSE |
|---|---|---:|---|---:|
| 1 | low_rate | 51.827464 | 采用 | 51.827464 |
| 2 | standard_rate | 51.756637 | 采用 | 51.756637 |
| 3 | head_only | 51.739621 | 保留父模型 | 51.756637 |
| 4 | robust_huber | 51.856324 | 保留父模型 | 51.756637 |
| 5 | low_rate | 51.695700 | 采用 | 51.695700 |
| 6 | standard_rate | 51.614535 | 采用 | 51.614535 |
| 7 | head_only | 51.630699 | 保留父模型 | 51.614535 |
| 8 | robust_huber | 51.782684 | 保留父模型 | 51.614535 |
| 9 | low_rate | 51.743242 | 保留父模型 | 51.614535 |
| 10 | standard_rate | 51.689302 | 保留父模型 | 51.614535 |

最终选择 `candidate-6`，采用轮次为 1、2、5、6。其余 6 次候选因收益不达门槛保留实际父模型。所有轮次均核对父摘要继承，10 次训练保留 Adam/scaler/seed 等状态，9 次包含非空历史的完整实验。

## 验证范围

- 从 v10 固定源码加本次 C 修改制作独立快照，排除工作区同期其他赛题改动。修改落在 B/C 共用回归链路；A 运行代码和训练菜单保持 v10 内容，其他任务的专项优化不在此镜像中。
- 独立快照全量项目测试：193 项通过；SDK 官方测试：18 项通过、2 项跳过。
- 固定镜像 10 轮：2 CPU、4 GiB、无网络、非 root、只读根目录及授权挂载；每轮真实训练及独立预测，全部 round_ack 确认。
- 独立重算 21 组、484,029 条预测输出的 RMSE、MAE、NASA 评分，均与保存记录一致；核对每轮历史引用、调用回执、SDK 顺序、实际父权重和候选继承。
- 决策使用离线替身，真实外部模型调用 0 次。实验没有复现真实 27B 决策等待时间，也未重新测量隐藏集，不能承诺相同正式成绩或正式时限。
- 镜像归档全部 OCI blob 摘要、应用源码与 C 快照一致；9 个官方 SDK 模块及官方 harness 模板逐字节一致。镜像不包含数据、基线权重或密钥。

## 使用交付

上传 `gas-rsi-abc-wyx-sdk104-v11c-20261125.tar` 并发起新的平台运行。只重试旧 v10 提交不会更新运行镜像。

- 镜像标签：`gas-rsi-abc:wyx-sdk104-v11c-20261125`
- 不可变镜像 ID：`sha256:ed2dd57004c05473eeb2a8d76cf96632e5d6956972734002adc184c6e95a0565`
- TAR 大小：61,971,968 字节，59.10 MiB。
- TAR SHA-256：`7be0f1a06f31ccc16615d78e2417a8a0aef3132c65521a0ca54ec73b690a23cd`

本机检查：

```powershell
docker load --input .\gas-rsi-abc-wyx-sdk104-v11c-20261125.tar
docker run --rm --network none --read-only gas-rsi-abc:wyx-sdk104-v11c-20261125 describe
```

平台继续使用 SDK 标准命令 `python3 /app/harness.py describe|train|predict`，无需更换模型身份或补充平台密钥。

## 复现与证据

固定完整快照：`runtime/c-optimization-20261125/source/`。在该目录用本项目 Python 环境执行：

```powershell
python -B -X utf8 -m unittest discover -s tests -q
python -B -X utf8 scripts/validate_multitask.py --baseline-root 'D:\liyue\工业模型赛题研究' --output '..\c-repeat-new-directory' --datasets C --rounds 10 --compact-validation --image gas-rsi-abc:wyx-sdk104-v11c-20261125 --docker-executable 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
python -B -X utf8 scripts/verify_multitask_run.py '..\c-repeat-new-directory'
```

输出目录必须全新。`c-source-and-evidence.zip` 是精简审查包，包含实际镜像源码、SDK、C 回归测试、逐轮指标及验证证据，不包含公开大数据、历史回放数据和全套本地模型。完整 193 项测试使用上述完整快照；精简包可直接执行其中的 34 项 C/回归/宿主验证测试，运行命令见包内说明。

完整协议、状态、检查点、逐条预测和隔离证据保留于 `runtime/c-optimization-20261125/new-v11c-compact/`；旧镜像对照位于 `old-v10-compact/`。独立复核结果见 `independent-verification.json` 和 `continuation-verification.json`。
