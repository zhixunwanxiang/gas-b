# 真实 27B 测试后的代码复核（2026-11-22）

后续状态：下述五项代码问题已按用户要求修复，详见 [修复说明](CODE_FIXES_20261122.md) 和 `runtime/code-fixes-20261122/`。以下保留审查当时的源码及故障证据叙述，不代表修复后仍然存在；原真实模型测试材料未被覆盖。

结论：真实 27B 驱动的主要闭环已经跑通，空 JSON 问题已修复；当前仍有上线保护、实验失败恢复和验收判定等问题。**不能把已通过的正常回合等同于正式提交已无缺陷。**

本次核对真实测试保存的 61 个文件，均与当前工作区一致。沿用同代码的 14 次实际 API 响应、6 个完成回合、16,172 条预测核算及工作区/保存源码各 85 项测试通过证据；本轮没有新增真实模型或云端请求，仅增加离线故障探针和审核材料，没有修改业务代码。

## 需要修复的代码问题

### 1. [P1，主入口，已再次复现] 宿主测量不能硬性阻止退化候选上线

[submission.py:316](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:316) 在提交前用本地结果计算门槛，[submission.py:331](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:331) 仍只用这个门槛产生可选 ID；后续校验只检查 ID 属于允许列表。宿主回执只是传给模型，没有落实为程序约束。

本轮用当前代码再次注入同一公开验证范围上的冲突：父模型宿主 F1=0.714029722，候选宿主 F1=0，但本地候选 F1=0.744330129。离线模型替身选择候选后，程序仍完成了该选择。**注入的 0 分不是实际平台评分，也不是声称真实 27B 曾做过这次错误选择。**

应按正式回执的实际结构核对指标、方向与范围；已识别的宿主退化不能仅靠提示词阻止。范围缺失或不一致时应明确处理，不能猜字段或直接混比分数。证据：[host-receipt-conflict/result.json](D:/aiproject/gas-rsi-four-sensor/runtime/post-real-code-review-20261122/host-receipt-conflict/result.json)。

### 2. [P2，主入口，已复现] 本地训练超时直接中断，失败实验没有进入决策记录

[submission.py:306](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:306) 对 `continue_candidate()` 没有本地失败恢复；`decisions.jsonl` 到 [submission.py:384](D:/aiproject/gas-rsi-four-sensor/src/gas_a/submission.py:384) 才写入。

注入一个本地优化器 `TimeoutError` 后，异常离开 `train()`，没有 `finish` 确认，也没有失败决策记录；此时 SDK 会话仍可正常读取历史，证明它不是 `RPC_OUTCOME_UNKNOWN`，具备合法恢复条件。

建议在训练前记录计划和实验身份，仅对这段不包含 RPC 的本地计算处理已知失败；把失败证据传给真实模型，让它在允许列表中决定保留父模型并完成本轮。失败尝试也应进入重复预算。**不要用一个覆盖 `rnd.model()`、提交或上传的宽泛异常处理来继续未知结果的 RPC 会话。**

### 3. [P2，测试工具，已复现] 未达到至少三次实验的目标，验收命令仍可成功退出

[test_real_sdk.py:311](D:/aiproject/gas-rsi-four-sensor/scripts/test_real_sdk.py:311) 在达到最大回合数后无条件写 `status=completed`；即使 `at_least_three_complete_records=false`，也会输出 `test_completed` 并正常返回。

离线控制场景让模型替身连续四轮只选择合法的 observe，结果完整实验数为 0、要求标志为 false，但命令仍正常返回，等价 CLI 退出码 0。这个场景使用的是合成回复，不是真实 27B 的表现；原始合成回执仅存在于临时目录，没有作为真实证据发布。

报告字段没有伪造实验数，但只看退出码或 `status` 的自动验收会误判。建议增加明确的 `requirements_passed`，目标未满足时返回非零或专门的未达标状态；回滚演练也应区分“脚本执行完成”与“回滚要求通过”，不能强行改写 AI 决定来凑次数。

### 4. [P2，配置，仍未修复] `routing.home.epochs` 不影响实际训练次数

配置声明 home 训练轮数，但 [home_training.py:157](D:/aiproject/gas-rsi-four-sensor/src/gas_a/home_training.py:157) 固定调用一次 `head_epoch()`。默认 1 与实际一致；改成其他值仍训练一次，会让实验配置与实际执行不对应。应实现该参数，或去除其可调含义。现有参数指南已经如实注明这个缺口。

## 测试隔离与验收风险

### 5. [P2，原生测试隔离，合成样本复现] 审计器没有禁止读取应用根目录的凭据文件

[simulation_audit.py:30](D:/aiproject/gas-rsi-four-sensor/scripts/simulation_audit.py:30) 只禁止部分目录；[simulation_audit.py:63](D:/aiproject/gas-rsi-four-sensor/scripts/simulation_audit.py:63) 默认允许其他读取。因此应用根目录下 `.env` 不在禁止范围。

本次在独立临时应用目录放置仅含固定合成标记的 `.env`，通过同一审计器启动程序，读取被允许。**没有读取真实 `.env`、真实密钥或进行外传；也未发现当前正式选手代码主动读取密钥。**正式 Docker 白名单不包含 `.env`，因此这项不等于正式镜像已泄露密钥。

此问题说明原生演练不能证明宿主密钥在文件系统上对选手不可见。应将宿主配置放到选手无法访问的位置，增加已知凭据文件的拒绝探针，并用真正的容器挂载/操作系统权限验证隔离；不能把 Python 审计钩子当作 OS 沙箱。

### 云端收尾仍有边界

[live_platforms.py:186](D:/aiproject/gas-rsi-four-sensor/scripts/live_platforms.py:186) 同步等待 SDK 的 `finish()`，外层没有整体收尾期限；若 SDK 阻塞，后续状态查询还未开始。没有可取回的初始化 Run 对象、强制杀进程、断电等情形也仍可能留下记录。代码现在会留 ID 和未确认状态，不再把普通返回当作云端确认，但不能保证任何故障下都自动关闭。需要在支持的接口范围内增加可观测时限和针对自有记录的清理流程，不能为了收尾重试模型请求。

### 可选在线统计校正仍待规则确认

[submission_config.json:44](D:/aiproject/gas-rsi-four-sensor/submission_config.json:44) 允许 `feature_correction=true`，[signals.py:67](D:/aiproject/gas-rsi-four-sensor/src/gas_a/signals.py:67) 会根据预测特征窗口估计偏移。它不读标签，也不修改继承的 scaler；不能称为获取隐藏标签。但与“拟合统计量只使用训练集”的边界仍需要赛方确认。最新真实回合选择的是 false，最终模型也没有这条分支，不能由此证明未来开启它的合规性。

### 云端记录、正式宿主和隐藏集仍缺证据

正式选手入口通过 SDK 保存候选、状态和回合记录；当前 SwanLab 集成仍只服务单次连通性检查，真实四轮脚本明确不创建 SwanLab 实验。不能宣称当前项目已将全部训练、调参与评估自动接入该云端链路。正式 SDK 的宿主持久化是否属于认可的云端记录平台，应以赛方要求和实际覆盖证明为准；不要在离线选手镜像里加入未经允许的外部记录客户端或密钥。

真实 PyTRiO 回复已验证，不能继续写成“没有真实调用”；但本地提供方 ID 仍不是正式宿主计量回执，Docker/资源/只读挂载和隐藏集得分也没有由本次测试证明。遵照用户既定范围，不审核《实验记录规范》的具体字段。

## 保持成立的结果

- 修复后的真实四轮：三次完整历史实验记录，公开 Macro-F1 0.714029722 → 0.798701650，提高约 8.47 个百分点。
- 真实模型在独立故障演练主动回滚，下一轮父产物匹配；原始模型响应与 SDK 回执一致。
- `enable_thinking=False` 已解决本次观察到的 2048 Token 用尽而无最终 JSON 的问题。
- 公开验证标签用于评估、选路是合法输入；未发现正式预测入口依赖隐藏标签回填。
- 当前 AI 主要从固定菜单中选择，诊断以统计关联为主。这是能力范围，不应误判为“两层 RSI 鼓励项”的确定违规。

本轮探针汇总：[offline-probes.json](D:/aiproject/gas-rsi-four-sensor/runtime/post-real-code-review-20261122/offline-probes.json)。探针复现脚本：[reproduce_review_cases.py](D:/aiproject/gas-rsi-four-sensor/runtime/post-real-code-review-20261122/reproduce_review_cases.py)。正常真实回合证据：[verification.json](D:/aiproject/gas-rsi-four-sensor/runtime/real-27b-sdk-20261122-results/verification.json)。

建议优先修复 1、2、3，再处理无效参数和原生测试隔离；随后完成认可云端记录与正式容器验收。此次只审核，没有改动上述业务逻辑。
