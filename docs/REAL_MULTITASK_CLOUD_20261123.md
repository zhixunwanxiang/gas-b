# v5真实27B多任务云端实验

本次A/B/C三项均已完成，24次真实调用、12轮训练及全部云端回读通过，详见[结果与三个Run链接](REAL_MULTITASK_RESULTS_20261123.md)。

本次在固定v5镜像`sha256:440079707a61b3282e9d98e9dd38e502714f241be2a85a332aac5dba6a524a69`中运行A分类、B/C回归。每个数据集建立独立SwanLab Run，宿主将选手`rnd.model()`请求转交PyTRiO `Qwen/Qwen3.8-27B`，原始完整响应及提供方调用号原样交回SDK，不进入离线模型替身分支。

三个Run位于当前授权配置的`qunmo/gas-b`，与历史`1218`账号Run分别保存。每项计划至少4轮，若非空历史实验不足3次最多延长到6轮；每项最多16次实际请求，结果未知不自动重试。实际轮次、用量和结果以最终报告为准。

## 运行与回读

宿主`.env`提供`PYTRIO_API_KEY`、`SWANLAB_API_KEY`及获准的模型/项目配置。密钥不进入镜像、候选、源码快照或云端证据包。此命令会产生真实模型调用费用：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/test_multitask_cloud.py `
  --baseline-root 'D:\liyue\工业模型赛题研究' `
  --output runtime/my-new-real-cloud-abc `
  --image gas-rsi-abc:wyx-sdk104-v5-20261123 `
  --docker-executable 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe' `
  --docker-context desktop-linux --datasets A B C --max-calls 16
```

`scripts/test_multitask_cloud.py`先核对镜像ID、SDK/NumPy版本、应用字节摘要、非root、只读根目录、禁网及2 CPU/4GiB限制，再使用真实模型完成计划和选版。`scripts/validate_multitask.py`提供相同的原生训练/预测执行流程；真实模式通过独立peer截获模型RPC，不先运行离线规则决定，也不在异常时降级为替身。

每轮SDK结束确认后，记录真实历史引用、计划/选版回复、候选测量、漂移诊断、逐epoch报告、采用或拒绝理由。选中模型再通过镜像predict进程全量复测。所有预测请求只发送原生inputs，预测容器不挂载带answer的数据。

每个Run结束前，将精确源码、SDK wheel、官方公开基线与输入、候选、状态、SDK JSONL、原始提供方请求回复和逐条预测打成ZIP，经摘要校验分块写入SwanLab Text媒体。随后确认FINISHED、回读全部文本与关键标量、重组ZIP核对字节摘要。归档分块不是训练指标，也不是公有云原生模型文件上传。

无需新模型调用即可独立验证本次证据：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/verify_multitask_cloud_run.py `
  runtime/cloud-abc-v5-20261123 --output runtime/cloud-abc-v5-20261123/verification.json
```

只读云端重新下载/核验单项时，使用现有`download_cloud_evidence.py --verify-local-run runtime/cloud-abc-v5-20261123/A`，将A换成B或C。它不调用模型、不重新init或finish云端Run。

## 验证边界

这是官方公开数据、实际Docker和真实27B驱动的本地SDK宿主测试。提供方调用号不冒充组委会正式计量回执，公开验证成绩不冒充隐藏集成绩；本次正常多轮实验不另计一次故障注入回滚演练。旧v4的真实A回滚证据和v5回归回滚单元测试按各自范围保留。
