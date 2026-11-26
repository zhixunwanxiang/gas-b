# 当前组合重新测试及多轮图表

日期：2026-11-26。按用户要求，不使用历史结果作图，重新从三份官方初始检查点开始。代码为提交 `56839aa` 对应的 A 当前版本、B/C V18；各任务从空状态开始，连续续训十轮。

本次是真实数值训练和独立逐条预测，实验规划与选版由明确的离线控制器完成，真实大模型调用数为 0。使用本机 Windows Python 3.12.13、NumPy 1.26.4，数值线程固定为 1，各任务可并行运行；不冒称比赛容器限制或正式隐藏评分。一次路线十轮不是十次独立随机重复实验。

## 本次结果

| 项目 | 主指标 | 官方初始模型 | 第十轮实际选中模型 | 相对初始的改善 | 候选采用次数 |
|---|---|---:|---:|---:|---:|
| A | Macro-F1 | 0.714030 | 0.895614 | +18.1584 个百分点 | 6/10 |
| B | RMSE/分钟 | 681.159833 | 651.626661 | 降低 4.3357% | 1/10 |
| C | RMSE/循环 | 52.868573 | 51.614535 | 降低 2.3720% | 4/10 |

十轮实际选中主指标平均值分别为 A 0.874105、B 651.626661、C 51.686575。第零轮是初始模型，不计入上述十轮平均。全部 30 轮均收到完成确认，每轮两次离线模型回执。

总体收益不代表全部分组改善。图中保留 B 部分轴承退化、C 部分来源子集退化及 NASA 的非单调变化。固定公开投影没有正式阶段切换，最差轮不替代最差阶段，也不计算恢复速度。token usage 和峰值内存缺失时记未知，不填零。

## 图表与数据

共 9 组图，每组提供 240 DPI PNG 与 SVG：

1. 主指标和相对初始模型的累计提升。
2. 候选收益与实际采用收益。
3. A 准确率、Macro-F1、类别召回率及类别 F1。
4. B RMSE、MAE、偏差、P95/P99 及设备表现。
5. C RMSE、MAE、NASA、P95/P99 及高估/低估惩罚。
6. B 逐轴承及 C 逐来源子集的 RMSE 和相对变化热图。
7. 逐轮 CPU、回调墙钟和独立预测耗时。
8. 选版动作、离线回执数、候选及状态文件载荷。
9. 初始、十轮平均、最差轮和最终指标。

- [浏览全部图表](D:/aiproject/gas-rsi-four-sensor/output/fresh-multiround-a-current-bc-v18-20261126/index.html)
- [完整 PNG/SVG/CSV 图表包](D:/aiproject/gas-rsi-four-sensor/output/本次新测-ABC十轮图表-20261126.zip)
- [逐轮指标 CSV](D:/aiproject/gas-rsi-four-sensor/output/fresh-multiround-a-current-bc-v18-20261126/逐轮指标.csv)
- [完整数据与来源](D:/aiproject/gas-rsi-four-sensor/output/fresh-multiround-a-current-bc-v18-20261126/逐轮指标数据.json)

## 可追溯性与重现

运行开始前登记源码、配置、官方初始检查点及 train/validation SHA256，候选续训核对其实际选中父模型。图表生成时独立重算本次 285,692 条选中模型预测及 258,041 条候选预测，对照逐轮原始指标、回执和完成确认。PNG/SVG、所有页面链接、ZIP 内容摘要均已检查，图表已目视审查。

原始新测目录为 `runtime/fresh-multiround-a-current-bc-v18-20261126/`，包含本次注册、三个独立报告、每轮协议、产物、状态和逐条预测。任何历史运行均未作为曲线数据输入。源数据和初始状态固定，因此重新测量出现相同数值并不代表复用历史测量。

新测注册及每个工作进程：

```powershell
.venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --baseline-root 'D:/liyue/工业模型赛题研究' --output runtime/<新的运行目录>
.venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --output runtime/<新的运行目录> --dataset A
.venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --output runtime/<新的运行目录> --dataset B
.venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --output runtime/<新的运行目录> --dataset C
.venv-report/Scripts/python.exe -B -X utf8 scripts/chart_fresh_multiround.py --run runtime/<新的运行目录> --output output/<新的图表目录>
```

新测目录必须全新，不能覆盖已保存运行。后两个驱动文件是新增工具，没有修改 A 或 B/C 的业务源码与配置。
