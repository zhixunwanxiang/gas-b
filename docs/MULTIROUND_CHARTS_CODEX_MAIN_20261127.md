# 当前分支多轮测试与图表 · 2026-11-27

本次在用户更新代码后，从官方初始 checkpoint 重新运行 A/B/C 各 10 轮，共 30 轮。分支为 `codex/main-20261127`，HEAD 为 `0c0ba6ffe70e80fb0f10c05ac743bb894a4ca449`，包含注册时尚未提交的工作区改动。2026-11-27 11:39:27（北京时间）冻结了 237 个项目文件，测试从 `runtime/retest-codex-main-20261127-source/` 执行。49 个训练源码、配置和运行入口文件的 SHA-256、数据摘要及确切实验顺序保存在 `runtime/retest-codex-main-20261127/registration.json`。最终核对这些文件与工作区一致。结果不是只针对 HEAD 提交的测试。

训练使用真实数值计算与父产物连续继承。控制器使用显式离线替身，没有真实 27B 调用。按照更新后的 `dataset-defaults` 顺序执行：B 保留配置顺序，C 首先运行 `cmapss_condition_mlp`。未使用同时调整 B/C 的 `mlp-first` 备选模式，也没有拼接此前版本结果。十轮属于一条连续训练轨迹，不是十次独立重复试验。

| 指标 | 官方初始模型 | 本次第 10 轮 | 相对初始变化 |
|---|---:|---:|---:|
| A Macro-F1 | 71.402972% | 89.561384% | +18.158412 个百分点 |
| A Accuracy，准确率 | 87.861736% | 95.900322% | +8.038585 个百分点 |
| B RMSE，分钟 | 681.159833 | 575.862329 | 降低 15.458560% |
| B MAE，分钟 | 563.105841 | 479.410290 | 降低 14.863201% |
| C RMSE，循环 | 52.868573 | 49.177018 | 降低 6.982512% |
| C MAE，循环 | 38.899830 | 35.317484 | 降低 9.209156% |
| C NASA 非对称评分，公开投影逐行求和 | 5,823,439,325.342465 | 2,220,713,836.564069 | 降低 61.865940% |

A/B/C 十轮平均主指标分别为 F1 87.410538%、RMSE 639.537864 分钟、RMSE 49.282317 循环，排除初始第 0 轮。30 轮全部完成 SDK 结束确认；在该固定公开集上，实际选中模型主指标没有逐轮退化。这不构成正式稳定性得分或未知设备泛化证明。

B 的 Bearing1_4 最终 RMSE 仍比初始恶化约 9.9%，图 06 如实显示。C 的高低估风险、NASA 分方向和尾部集中度分别保留，不能只依据总体 RMSE 判断全面改善。

C 第 2 轮 NASA 为 2,155,476,497.57，低于最终模型的 2,220,713,836.56；最终模型的 RMSE 更低。两个指标各对应同一轮实际选中的模型，不能把不同轮次的最佳单项值拼成最终成绩。

项目 263 项测试全部通过；SDK 1.0.4 共运行 20 项测试，18 项离线通过、2 项真实外部调用测试跳过。2,314 次实际上传分块中，最大原始块为 65,536 字节，Base64 长度为 87,384 字符。详见 `runtime/retest-codex-main-20261127/verification.json`。传输帧均经离线 SDK 宿主核对，不代表真实平台已测量。

## 图表与数据

- 浏览入口：`output/multiround-charts-codex-main-20261127-retest/index.html`。
- 完整压缩包：`output/multiround-charts-codex-main-20261127-retest.zip`。
- 12 组图表，各有 240 DPI PNG、SVG，另附六份 CSV 与完整 JSON。首页及图 01 包含 A F1/Accuracy、B/C RMSE/MAE、C NASA 非对称评分。图 01 为 NASA 单独绘制评分总和与相对初始降幅；`各项指标汇总.csv` 汇总全部七项指标。
- 图 01–03：主指标与累计提升、候选/实际采用收益、A 分类指标。
- 图 04–06：B 误差、C NASA 与尾部、设备和来源子集分组。
- 图 07–09：CPU/墙钟/预测耗时、选版与上传文件量、平均/最差轮/最终。
- 图 10–12：高低估方向、NASA 贡献集中度、收益与累计 CPU。

初始及选中模型共 285,692 条预测、上传候选共 89,982 条预测，从原生 `harness.py predict` 输出重算主指标，并核对报告、父 checkpoint 继承链、模型回执和结束确认。未上传的本地拒绝候选没有额外原生预测复核，图 02 用斜线柱标示，测量来源与拒绝原因见 `实验与准入.csv`。完整精度数据、源码快照摘要和 263 份原始证据摘要保存在 `逐轮指标数据.json`。

固定公开投影缺少正式阶段切换，不能计算正式恢复速度，最差轮和最差公开分组也不能替代最差阶段得分。本机并行运行未强制比赛容器资源限制，CPU 仅计参与方回调；宿主测量、独立预测、大模型等成本不能混作同一口径。真实 Token、峰值内存与正式资源分缺失，图表不填零、不换算官方成绩。

## 复现入口

注册要求使用全新的目录，并锁定当时源码和配置。注册后分别运行 A/B/C，数值库线程均设为 1；所有子进程设置 UTF-8。数据根目录使用用户提供的官方基线包。

```powershell
& .venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --baseline-root 'D:\liyue\工业模型赛题研究' --output runtime/new-chart-run
& .venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --output runtime/new-chart-run --dataset A
& .venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --output runtime/new-chart-run --dataset B
& .venv/Scripts/python.exe -B -X utf8 scripts/run_fresh_multiround.py --output runtime/new-chart-run --dataset C
& .venv/Scripts/python.exe -B -X utf8 scripts/chart_fresh_multiround.py --run runtime/new-chart-run --output output/new-charts --prepare-only
```

渲染使用具备 Matplotlib/Pillow 的 Python 执行同一图表脚本，传入 `--render-only`。视觉检查后以 `--package-only --visual-review-complete` 完成来源复核、PNG/SVG 检查和 ZIP 打包。本次使用 `D:\tools\Python\python.exe` 的 Matplotlib 3.10.8；没有改变项目训练依赖。该标志只应在实际完成图面检查后设置。
