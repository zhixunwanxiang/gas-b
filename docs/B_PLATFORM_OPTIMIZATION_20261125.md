# B 项目：平台报告修复与十轮验收（2026-11-25）

已修复 B 项目“每轮训练有效、十轮全部未采用”的准入问题，并生成独立 B 修复镜像。以官方公开验证集的 1,679 条样本运行十轮，新镜像实际采用两次改进，RMSE 从 **681.159832604 降至 651.885364673 分钟，降低 4.2977%**；MAE 从 **563.105841451 降至 542.722782383 分钟**。

本次是原 SDK、真实数值训练与容器预测的离线验证，实验决定由明确标注的模型替身产生。新增付费模型调用和云端写入均为 0，尚未重新提交正式平台，也未测得新的隐藏集成绩。

## 报告暴露的根因

来源：`D:\liyue\工业模型赛题研究\实验报告B.txt`。报告显示 10/10 轮完成，最终采用起始模型；各轮本地候选均改善，但宿主准入因父参考 `unknown_validation_scope` / `unverified_parent_or_candidate_measurement` 拒绝。原隐藏集 RMSE 为 720.065935 分钟、MAE 为 518.721020 分钟（200 条）；这些隐藏成绩不能与本次公开集结果混比。

报告不含原始 start/候选帧。本次以缺少 `scope` / `validation_sha256` 的具名 RMSE 摘要复现同一拒绝路径，不能据此断言平台原始字段的完整形状。

固定旧 v10 镜像做一轮对照：父公开 RMSE 681.159832604，候选 657.476355594，本地门槛通过，父宿主范围仍为 unknown，候选被拒绝。固定新镜像在同一条件下实际采用该候选。

## 实现

- 本轮实际父检查点先经 SDK 声明清单与摘要校验，再在本轮公开 validation 独立复评；校验复评所用文件未变化。
- 宿主具名 RMSE 与父复评相符、只缺身份字段时，父参考记为 `local_current_parent_public_reevaluation`，保留原始宿主摘要及其拒绝原因，不冒充宿主已声明范围。
- 显式隐藏范围、错误/空摘要、错误产物/数据集/轮次、宿主与本地分数冲突、无指标名或非法数值仍不能放行。
- 候选仍必须通过本地 RMSE/MAE 保护及 SDK 宿主候选测量门槛；父参考不影响候选冲突检查。
- 父测量真正不可核验时，只开放 observe，跳过无法采用的训练和历史恢复，把阻塞原因写入规划与状态。
- 提示词解释本地父复评与旧宿主摘要的区别；实验执行指纹加入父参考策略版本。
- 沿用父权重、Adam、seed、输入与目标 scaler；没有修改学习率菜单、优化器或原 SDK。

核心文件：`src/gas_a/regression/evaluation.py`、`submission.py`、`prompts.py`；新增 `tests/test_b_platform_admission.py` 和 `scripts/validate_b_platform_report.py`。

## 十轮容器结果

| 轮次 | 实际执行方案 | 采用新候选 | 选中版本公开 RMSE（分钟） |
| --- | --- | --- | ---: |
| 1 | robust_huber | 是 | 657.476355594 |
| 2 | standard_rate | 是 | 651.885364673 |
| 3–4 | head_only | 否 | 651.885364673 |
| 5–6 | low_rate | 否 | 651.885364673 |
| 7 | robust_huber | 否 | 651.885364673 |
| 8–9 | standard_rate | 否 | 651.885364673 |
| 10 | robust_huber | 否 | 651.885364673 |

离线替身优先请求 TXT 中对应轮次的方案，重复拒绝被屏蔽后使用当轮允许菜单中的方案。使用固定官方公开数据，没有平台各阶段的新增/漂移投影；这不是逐字重放原 27B 回复。

十轮均完成 SDK 上传、历史读取、状态保存及结束确认；9 次有非空历史的实际实验。全部续训检查和 SDK 顺序审计通过，最长单轮约 5.59 秒。模型等待由离线替身替代，该耗时不能预测正式平台真实模型等待。

隔离参数：2 CPU、4 GiB、非 root、只读根目录、禁网；预测容器只挂载模型，不挂载带标签数据。21 组逐条预测在 Windows 本地与 Linux 镜像间字节相同，最终检查点所有张量与 metadata 相同；NPZ 归档字节因 ZIP 创建系统字段不同而不完全相同。

独立 B 源码的 **191 项项目测试、18 项原 SDK 测试**通过。镜像内 **37 个应用文件、9 个 SDK 模块**逐字节核对通过，`harness.py` 与官方模板一致。

## 交付与复现

镜像：`gas-rsi-abc:wyx-b-parentfix-v11-20261125`。

镜像 ID：`sha256:1ff02b7e02e5b8c897f944b62e8a03672c04f2e3e029495c365c3911dbde409c`。

交付目录：`output/submission-b-parentfix-v11-20261125/`。镜像压缩包为 `gas-rsi-abc-wyx-b-parentfix-v11-20261125.tar.gz`，必须导入并选择这个新镜像后重新发起平台测试；重试旧镜像不能带入修复。

为了保留同一工作目录中正在进行的其他修改，本次 B 镜像以提交 `ca21867b760d187cc0563bcad0e83cb467d07517` 为基础，仅覆盖上述三个回归模块。它没有包含并行任务对 A 路由、共享分类准入或配置的其他修改。完整已测源码位于 `runtime/b-platform-release-20261125/source/`；镜像构建与测试均使用该快照。

使用该快照复现（模型费用为零）：

```powershell
$pythonExe = 'D:\aiproject\gas-rsi-four-sensor\.venv\Scripts\python.exe'
Set-Location 'D:\aiproject\gas-rsi-four-sensor\runtime\b-platform-release-20261125\source'
& $pythonExe -B -X utf8 scripts/validate_b_platform_report.py `
  --baseline-root 'D:\liyue\工业模型赛题研究' `
  --output 'D:\aiproject\gas-rsi-four-sensor\runtime\my-b-platform-check' `
  --rounds 10 --image gas-rsi-abc:wyx-b-parentfix-v11-20261125 `
  --expected-image-id 'sha256:1ff02b7e02e5b8c897f944b62e8a03672c04f2e3e029495c365c3911dbde409c' `
  --docker-executable 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe' `
  --docker-context desktop-linux
```

完整证据位于 `runtime/b-platform-release-20261125/`：`docker-10/summary.json`、逐轮 `protocol.jsonl` / `sdk-timing.json`、逐条预测、状态及权重；`v10-control-1/` 为旧版预期拒绝对照。补丁与证据 ZIP 提供指定 v10 提交上的修改和精简证据，不是独立完整工程，也不替代完整 runtime。
