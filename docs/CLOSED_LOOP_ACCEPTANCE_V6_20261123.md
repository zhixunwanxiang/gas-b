# v6闭环模拟验收：正常训练、真实回滚与可复现性

这是此前开发过程的阶段验收，正常阶段采用 v5 记录。之后已用同一固定 v6 镜像从头完成 A/B/C 全流程，见[最新全新测试](FRESH_PLATFORM_TEST_20261123.md)。本文件保留原始失败和修复边界，不改写成新一次的成绩。

本报告区分正常实验、首次故障演练、修复后复测和记录回复重放，不把失败实验或旧镜像记录冒充新版本的一次性全通过。

[云端验收总览](https://swanlab.cn/@qunmo/gas-b/runs/4e82f1de652c44829ca0999f3cf4d544/chart)仅汇总指标、布尔核验及六个实验链接，状态FINISHED，新增模型调用为0，不计作额外历史实验。完整源码和复现工具保存在本地提交材料；向该新总览上传完整源码被工具自动审批拒绝，随后采用仅上传结果的低风险方案。

## 版本和修复

正常A/B/C实验来自已完成的v5真实27B实验，每项4轮、3次非空历史实验。补做故障回滚时，v5的A通过，B/C失败：模型将历史最佳版本误当当前父模型，把observe/experiment与历史target ID混搭；泛化的自动纠正提示未充分解释动作含义，B还触发调用预算中止。

v6只修改B/C的控制提示与动作身份说明，未改变输入输出、网络结构、梯度训练、指标、准入阈值或数据范围。新增`regression/prompts.py`给出：唯一实际父模型及实测分数、当前验证合格历史排序、允许的完整动作组合、观察/训练/回滚各自含义。观察和训练只能指向实际父模型，要使用历史版本必须先选择rollback、experiment=none，下一轮才能续训。模型回复仍须自主给出并通过原有校验，程序不篡改返回值、不自动放宽门槛。

v6镜像：`gas-rsi-abc:wyx-sdk104-v6-20261123`，不可变ID：`sha256:905c8e9d172d5271091b84be787922c67c14388bb75cf05b14f4b0c18dc515e7`。三个回滚Run的显示名沿用了旧工具中的“v5”字样，实际镜像以各自configuration、image-preflight.json和上述ID为准；后续脚本已移除名称里的固定版本文字。

## 正常阶段与基线比较

| 任务 | 主指标 | 初始公开基线 | 真实正常阶段最终 | 改善 | 实际采用新候选 |
|---|---|---:|---:|---|---:|
| A | Macro-F1 ↑ | 0.714029722 | 0.798701650 | +8.4672个百分点 | 3次 |
| B | RMSE ↓，分钟 | 681.159832604 | 651.885364673 | 下降4.2977% | 2次 |
| C | RMSE ↓，循环 | 52.868572566 | 51.763407434 | 下降2.0904% | 3次 |

每项正常第2、3、4轮分别读1、2、3条已完成历史，各有真实候选提交、实测结果、模型选版和SDK结束确认。这三轮才计为三次完整历史实验；回滚/观察回合不凑实验次数。正常记录见[正常阶段结果](REAL_MULTITASK_RESULTS_20261123.md)。这些是包内公开验证基线，不能替代比赛平台正式基线或隐藏评分。

## v6真实回滚及下一轮继承

故障只注入独立基线模型副本：A循环置换输出权重/偏置，B/C输出偏置增加10个原目标缩放单位。不改原模型、数据或标签。实际当前父模型以start帧的fault-parent为准，历史模型从正常阶段真实产物恢复，均须在同一当前公开验证集复评达标。

| 任务 | 故障时主指标 | 真正恢复后 | 已选历史产物 | 下一轮完整产物继承 |
|---|---:|---:|---|---|
| A Macro-F1 | 0.016877637 | 0.798701650 | candidate-4 | 通过 |
| B RMSE | 7505.554218977 | 651.885364673 | candidate-2 | 通过 |
| C RMSE | 845.564269461 | 51.763407434 | candidate-3 | 通过 |

每项2轮、4次新增真实27B调用；首轮自主选择rollback，下一轮主检查点、路由组件、配置和声明文件的完整清单及摘要与恢复目标一致。保留故障父模型会被判失败，不能仅凭reason里写“回滚”就通过。

| 任务 | 正常阶段Run | v6回滚Run |
|---|---|---|
| A | [查看](https://swanlab.cn/@qunmo/gas-b/runs/c8a6c2b8a7d14071967d32e449032315/chart) | [查看](https://swanlab.cn/@qunmo/gas-b/runs/ff00c34ae52a4a0b9b9d0ee7689f1c53/chart) |
| B | [查看](https://swanlab.cn/@qunmo/gas-b/runs/693e9edb2ec34f7198aa56ec5f7887f5/chart) | [查看](https://swanlab.cn/@qunmo/gas-b/runs/7ca077bc801a47928ad6edcd39b571cd/chart) |
| C | [查看](https://swanlab.cn/@qunmo/gas-b/runs/634957dd8e6a42a28342b45aba08a3a4/chart) | [查看](https://swanlab.cn/@qunmo/gas-b/runs/6915cd5d7dd7443eb75488d65355e11b/chart) |

六个验收Run均FINISHED，正常阶段24次与v6回滚12次构成最终通过的36次真实调用。最初v5回滚另外12次也如实计入实际消耗：A4次通过，B6次/C2次暴露问题，不能计为合格完整回合。全部阶段48个提供方ID唯一，提供方报告316,927个Token；其中本次回滚初测/复测新增24次，正常24次来自前一次已完成请求，未重复付费跑正常阶段。

## 五步闭环证据

1. 漂移监测：在每轮训练前，以固定公开train特征参考，对当前公开validation计算PSI、分箱KS与均值变化。A按传感器8维汇总，B/C逐特征；结果和阈值写入monitoring.json。正常C可能没有输入漂移，仍可根据误差实验；漂移不是唯一训练触发条件。
2. 诊断归因：A记录薄弱类别和主要混淆，B/C记录RMSE/MAE、偏高偏低、寿命区间误差及漂移特征。它们是统计线索，不证明物理原因。故障演练是模型权重退化，不能称为新输入漂移。
3. 实验优化：真实27B读历史、当前测量和有限菜单做计划；程序从宿主实际父产物续训，保留Adam、scaler、目标缩放、seed及计数。只用train更新梯度。
4. 评估上线：公开validation选epoch和版本；A用F1及逐类召回保护，B/C用RMSE减少和MAE保护；新候选还需通过同范围宿主回执校验。未达标或证据不足保留父模型，原始拒绝结果仍留档。
5. 回滚：仅恢复帧内允许的历史产物，核对摘要，在当前validation复评后才开放；真实模型决定和下一轮完整父产物校验均通过。

## 无人干预、数据与材料边界

正常训练和修复后的v6回滚运行中，没有人工逐轮指定实验、改写候选、替换模型返回或补写成绩。开发调试阶段确实根据v5失败做过提示词修复、重建和复测，因此不能说整个开发过程从未需要外部干预，也不能说v5首次故障测试全部通过。

仅使用各官方包公开train/validation，输入字节摘要与正常阶段一致。预测容器只挂载模型并接收原生特征，不挂载带标签数据；隐藏数据与标签未提供、未获取。容器为非root、根目录只读、禁网、2 CPU/4GiB；网络模型调用及SwanLab归档发生在宿主，凭证不进入镜像。容器资源和允许挂载已核验，逐次文件/socket调用没有额外插桩追踪。

云端保存真实请求/回复、历史、实验、诊断、选版、候选和状态ZIP，经Text分块归档后回读核对。v5 B/C失败原件保留在`runtime/cloud-abc-v5-recovery-20261123/`，并以prior-failure.zip随新B/C归档保存，原始记录未修改。

当前独立核验：`runtime/cloud-abc-v6-recovery-20261123/verification.json`，复核原始提供方响应与SDK、真实历史、历史恢复读取、完整父产物、保存预测、输入摘要、故障仅改输出层、ZIP摘要与CRC。项目146项测试通过。v5正常记录在v5及修复后v6镜像均已零费用重放全部通过：24条原始回复全部消费，16个检查点/路由文件、27组逐条预测和最终选版完全一致。v6重放见`runtime/recorded-replay-abc-v6-20261123/verification.json`，新增模型调用和云端写入均为0；不把重放称为新模型调用。

```powershell
# 独立核验，不调用模型或云端
.\.venv\Scripts\python.exe -B -X utf8 scripts/verify_closed_loop.py `
  --normal runtime/cloud-abc-v5-20261123 `
  --recovery runtime/cloud-abc-v6-recovery-20261123 `
  --output runtime/my-closed-loop-verification.json

# 只使用已保存真实回复，验证v6数值训练可复现；新增模型费用为0
.\.venv\Scripts\python.exe -B -X utf8 scripts/replay_multitask_cloud.py `
  --source runtime/cloud-abc-v5-20261123 --output runtime/my-recorded-v6-replay `
  --image gas-rsi-abc:wyx-sdk104-v6-20261123 `
  --docker-executable 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
```

本次验证官方公开投影上的本地SDK宿主闭环能力和基线对比；不等于组委会正式平台、隐藏测试流泛化收益或正式评分已通过。
