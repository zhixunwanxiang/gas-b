# v5真实27B多任务实验结果（2026-11-23）

本次使用最新v5镜像，A/B/C各自主完成4轮训练、3次“读非空历史—实际实验—依据结果决策”记录；共24次真实PyTRiO Qwen/Qwen3.8-27B调用，提供方报告总Token为129,998。没有人工指定实验或选版、没有替换原始回复、没有重启付费实验或自动重试结果未知的请求；本次完整回复均一次通过校验，自动纠正次数为0。

三个独立SwanLab Run均已确认FINISHED；使用当前授权配置的qunmo/gas-b空间，不修改旧1218或旧v4记录。

| 任务 | 主指标 | 初始公开基线 | 最终 | 改善 | 成功升级 | 云端Run |
|---|---|---:|---:|---|---:|---|
| A | macro_f1 | 0.714029722 | 0.798701650 | +8.4672个百分点 | 3次 | [查看](https://swanlab.cn/@qunmo/gas-b/runs/c8a6c2b8a7d14071967d32e449032315/chart) |
| B | rmse | 681.159832604 | 651.885364673 | RMSE下降4.2977% | 2次 | [查看](https://swanlab.cn/@qunmo/gas-b/runs/693e9edb2ec34f7198aa56ec5f7887f5/chart) |
| C | rmse | 52.868572566 | 51.763407434 | RMSE下降2.0904% | 3次 | [查看](https://swanlab.cn/@qunmo/gas-b/runs/634957dd8e6a42a28342b45aba08a3a4/chart) |

## 每轮真实行动与结果

### 任务A

| 轮次 | 历史数 | 27B选择的实际实验 | 最终产物 | 本轮选中分数 |
|---|---:|---|---|---:|
| 1 | 0 | frozen_router | candidate-1 | 0.763612635 |
| 2 | 1 | conservative_head | candidate-1 | 0.763612635 |
| 3 | 2 | five_sensor | candidate-3 | 0.793189042 |
| 4 | 3 | four_sensor | candidate-4 | 0.798701650 |


### 任务B

| 轮次 | 历史数 | 27B选择的实际实验 | 最终产物 | 本轮选中分数 |
|---|---:|---|---|---:|
| 1 | 0 | robust_huber | candidate-1 | 657.476355594 |
| 2 | 1 | standard_rate | candidate-2 | 651.885364673 |
| 3 | 2 | robust_huber | candidate-2 | 651.885364673 |
| 4 | 3 | head_only | candidate-2 | 651.885364673 |

MAE：563.105841451 → 542.722782383。

### 任务C

| 轮次 | 历史数 | 27B选择的实际实验 | 最终产物 | 本轮选中分数 |
|---|---:|---|---|---:|
| 1 | 0 | robust_huber | candidate-1 | 51.964354452 |
| 2 | 1 | low_rate | candidate-2 | 51.908401676 |
| 3 | 2 | head_only | candidate-3 | 51.763407434 |
| 4 | 3 | head_only | candidate-3 | 51.763407434 |

MAE：38.899829827 → 37.972778951。
NASA逐样本惩罚总分：5823439325.342465 → 3997162625.815919，越低越好；两次测量均为同一23,049条公开验证数据。

## 独立核验与复现

- 24条提供方原始响应与SDK model_response逐条相等，finish引用真实调用号和实际读取的历史ID。
- 27组、233,748条保存预测独立重算通过；F1、RMSE、MAE和C的NASA总分一致。
- 云端217条文本及40个关键指标点已全部回读；三个完整证据ZIP（合计1093个文件）从云端重组后摘要、CRC一致。
- 各Run的精确源码和应用字节摘要、官方公开输入/父检查点、候选及状态、模型请求/回复、SDK协议、逐条预测和逐epoch实验均保留。
- 真实运行镜像ID：`sha256:440079707a61b3282e9d98e9dd38e502714f241be2a85a332aac5dba6a524a69`，三项预检各核对36个应用及依赖文件。
- 非root、根目录只读、禁网、2 CPU/4GiB限制通过预检；模型和云端凭证只留在宿主，训练/预测容器不包含密钥或隐藏标签。
- 更新后的138项项目测试通过；这些单元测试使用替身，不计入24次真实调用。

本次三个Run均为正常多轮实验，没有额外注入故障来计一次真实回滚；既有回滚证据按其原有版本与任务范围保留。这是官方公开数据、真实镜像和真实27B的本地SDK宿主测试，尚未证明组委会正式宿主或隐藏集成绩。A/B/C各一个Run不是同一数据集的三次独立重复试验。

完整本地证据：`runtime/cloud-abc-v5-20261123/`。`verification.json`为独立核验，A/B/C各目录的`suite.json`、`cloud-receipt.json`、`cloud-verification.json`为运行与云端回执。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/verify_multitask_cloud_run.py runtime/cloud-abc-v5-20261123
```

真实运行和只读下载命令见[真实多任务实验说明](REAL_MULTITASK_CLOUD_20261123.md)。
