# 历史回放与归档重建

本页保留原 README 中的历史回放资料。当前正式入口和平台验收请先阅读 [README](../README.md) 与 [PLATFORM_VALIDATION.md](../PLATFORM_VALIDATION.md)。以下历史数据、回放指标、规则控制器及标签反馈流程不能作为正式隐藏集评测证据。

## 回放命令与数据

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B run.py replay
# 或
.\start.ps1 replay
```

入口是 `run.py replay → replay.py → FourSensorRouter`，固定读取 `config.json`、`models/` 和 `data/replay/batch3.dat` 至 `batch10.dat`。每批先预测、独占创建 `.npy` 文件、重新读取并校验预测摘要，再揭晓该批标签、计算混淆矩阵并调用 `release_labels()`。后续批次使用更新的历史路由统计。

可显式指定 `--models`、`--output`；输出目录必须尚不存在，默认保存到 `runtime/<时间戳>/`。完成后写入 `results.json` 并退出，没有网页或常驻服务。

| 数据 | 样本数 | 历史用途 |
|---|---:|---|
| `data/public/train.jsonl` | 445 | batch1：梯度训练与训练内校准 |
| `data/public/validation.jsonl` | 1,244 | batch2：公开验证、epoch 与路由选择 |
| `data/replay/batch3.dat` 至 `batch10.dat` | 12,221 | 固定顺序回放，批后读取归档标签 |

DAT 标签为 1～6，读取后转为内部 0～5；气体顺序为 Ethanol、Ethylene、Ammonia、Acetaldehyde、Acetone、Toluene。公开 JSONL 使用英文标签名。

## 回放模型与路由

`models/` 包含 `start.npz`、`reference.npz`、`four_sensor.npz`、`five_sensor.npz`、`home_trunk.npz`、`home_router.npz` 六个归档组件。每个基础 MLP 为 128→32→6，共 4,326 个神经网络参数；部分组件还包含 KNN 记忆和辅助头。

batch3 使用 `start`；从 batch4 起计算四传感器主专家、五传感器辅助专家和 home 分支：

- 四传感器使用第 3、14、15、16 号传感器，五传感器增加第 13 号；通过第一层权重零行屏蔽不用的传感器，仍接收 128 维输入。
- `signals.Flatline` 根据第 13 号传感器最近 8 条归一化读数跨度判断平线，报警样本使用四传感器分支。
- `signals.correct` 使用历史特征缓冲估计偏移，当前样本先按已有估计校正再入缓冲，不使用当前标签。
- home 融合神经网络、KNN 和训练集校准的成对头。
- 路由统计保留最近三个已揭晓批次，积累后最早从 batch7 生效。

`predict_batch()` 后必须调用 `release_labels()` 才能预测下一批。SDK 隐藏预测没有这个标签回填约定，因此正式入口采用 `frozen_router.py` 中训练阶段选定、预测阶段不依赖标签的策略。

## 历史指标与评分口径

`config.json` 的归档验收目标为：

| 批次 | 样本数 | 固定六类 Macro-F1 |
|---|---:|---:|
| batch3 | 1,586 | 0.593269 |
| batch4 | 161 | 0.685127 |
| batch5 | 197 | 0.688513 |
| batch6 | 2,300 | 0.760428 |
| batch7 | 3,613 | 0.675819 |
| batch8 | 294 | 0.505773 |
| batch9 | 470 | 0.344042 |
| batch10 | 3,600 | 0.537445 |
| 合并 | 12,221 | 0.649151222 |

该目标对应历史记录中的 `four_sensor_router_20261121a` 自然顺序 macro 分支。回放会检查六个权重文件、每批数据和预测的摘要及累计混淆矩阵，不能替换成任意新数据后继续使用原验收值。

`router.macro()` 固定平均六类；训练端 `score_predictions()` 平均真实或预测中出现的类别。合并指标从累计混淆矩阵计算，不能用八批 F1 的算术平均替代，也不能把两个入口不同口径的单批数值直接混比。

上述表格保留历史验收值，本次 README 重写没有重新执行回放。旧 README 提及的 2026-11-21 运行目录和 PDF 当前工作区未保留，因此本页不把它们作为可访问的验证报告链接。

## 归档权重重建

```powershell
$rebuiltOutput = Join-Path 'runtime' ('rebuilt-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
.\.venv\Scripts\python.exe -B rebuild_replay_models.py $rebuiltOutput

# 仅在脚本全部 SHA-256 验收通过后，使用重建目录回放
.\.venv\Scripts\python.exe -B run.py replay --models $rebuiltOutput
```

此命令与 `rebuild_models.py` 不同：前者重建固定归档配方并要求字节级摘要一致；后者调用当前动态 `train-local`，要求显式续训起点与数据。

历史 Windows 环境重建曾出现四个训练组件的字节摘要不一致，而内存中的预测诊断匹配。记录中的权重/Adam 数组差异约为 `10^-18～10^-16`，与浮点环境末位差异相符，但原始环境信息不足以确认根因。预测行为一致不代表文件字节一致；本项目没有关闭回放的严格摘要校验，也不应通过伪造摘要宣称重建通过。

## 旧的本地闭环工具

相关脚本仍保留：

| 工具 | 用途 |
|---|---|
| `scripts/closed_loop_validation.py` | 按固定历史批次执行监测、诊断、候选训练、本地版本切换与回滚演练 |
| `scripts/verify_closed_loop.py` | 对已有回放运行目录核验 |
| `scripts/build_closed_loop_report.py` | 生成历史闭环 PDF 报告 |

`closed_loop_validation.py` 需要外部 DAT 数据目录和已有上轮产物，具体参数先用 `--help` 查看；本页不再把当前不存在的上轮目录写成可直接执行的示例。报告使用独立的 `requirements-report.txt` / `.venv-report` 环境，避免改变正式推理依赖。

旧闭环使用本地规则控制器与批后标签，外部 AI 调用为零，不等于正式 SDK 模型驱动的回合。历史报告曾记录总体 F1 改善同时乙醛召回下降，不能解释为所有业务指标都改善。当前正式 Harness 的新测试及已保存证据见主 README。
