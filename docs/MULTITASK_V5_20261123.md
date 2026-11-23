# v5：补齐 A/B/C 原生比赛入口

旧 v4 镜像仅声明 classification，A 任务测试通过不能证明 B/C 可运行。v5 在同一 SDK 1.0.4 镜像中提供 A 分类和 B/C 回归的实际实现。

## 原生合同

| 数据集 | 输入 | 输出 | 评价 |
|---|---|---|---|
| A | `inputs.features`，128维 | `outputs: [{label, score}, ...]` | Macro-F1越高越好，参考准确率 |
| B | `inputs.features.horizontal/vertical`，各17个官方数值字段 | `outputs: {value: 非负分钟数}` | RMSE越低越好，辅助MAE |
| C | `inputs.settings`（3项）和`inputs.sensors`（21项） | `outputs: {value: 非负循环数}` | RMSE越低越好，辅助MAE和NASA非对称评分 |

`contest_submission.TASKS`声明`classification`、`regression`；`gas_a.multitask`按宿主提供的task与dataset_key分发。B/C没有改成气体分类或寿命等级分类，预测不读取answer、id、meta，不使用时间编号或完整寿命反推目标。没有预测上下文的SDK兼容调用按输入结构识别B/C，再核对模型profile和完整特征路径。

## B/C 回归闭环的实际实现

1. 读取本轮宿主父产物，SDK校验已声明文件摘要。使用项目内与B/C官方代码字节相同的`reference/model.py`加载回归MLP；保留权重、Adam一二阶矩、种子、累计epoch/step、输入标准化和目标缩放。
2. 只从`rnd.data`明确提供的公开train/validation读取数据。只允许train更新梯度；validation用于测量、诊断和选择epoch/模型。
3. 首次用公开训练特征保存固定分箱、均值和标准差参考；每轮比较当前公开validation的34维或24维特征，记录PSI、分箱KS和标准化均值变化。这是公开投影监测，不是隐藏流遥测。
4. 用公开验证残差诊断整体偏高/偏低、寿命区间中的误差和变化明显的特征；寿命区间边界由公开train计算。它提供实验线索，不证明物理故障原因。
5. 真正的比赛入口每轮通过`rnd.model()`进行计划和选版两次模型调用。27B只能选择观察、一次菜单实验，或已复评合格的历史回滚。四个回归实验为low_rate、standard_rate、head_only、robust_huber；配置含义见根目录参数说明。
6. 从实际父检查点数值续训，按batch检查训练时间预算。head_only保留隐藏层参数及其Adam矩。Huber仅改变训练损失，不改变原始目标、输出单位或验证指标。逐epoch保存真实损失、RMSE/MAE及准入检查，优先导出满足门槛且RMSE最低的已训练epoch。
7. 新候选须RMSE严格降低，减少量同时达到绝对`1e-6`和父RMSE的`0.05%`，且MAE相对增加不超过`2%`；随后还需宿主回执的公开范围、验证摘要、RMSE与本地测量一致且达到同样主指标增益。通过只是获得选择资格，模型仍可保留父版本。
8. C另计算NASA逐样本非对称惩罚之和：误差为预测减真值，高估使用`exp(error/10)-1`，低估使用`exp(-error/13)-1`。不将该总分与不同样本数的数据直接横比。若数值溢出，明确记录不可表示，不能用截断小数冒充真实总分；该辅助分数目前不是上线硬门槛。
9. 仅恢复start帧允许的历史ID，重新在当前validation评估；满足同一回归门槛才开放回滚。SDK恢复/上传/模型RPC发生结果未知时立即退出，不作为本地训练失败吞掉继续调用。
10. 保存监测、诊断、计划、逐epoch实验、宿主回执、最终决策和版本配套策略；引用实际读过的history_ids和真实model_call_ids，通过save_state/finish等待宿主确认。相同父/数据/实验/执行配置失败两次后屏蔽，回滚时恢复相应策略。

## 复现入口

以下宿主工具使用离线决策替身，运行真实SDK与数值训练，不读取密钥、不调用付费模型、不创建SwanLab实验。它不能替代真实27B或正式平台评分。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -q

# 原生进程：官方公开数据、全量验证、每项四轮。
.\.venv\Scripts\python.exe -B -X utf8 scripts/validate_multitask.py `
  --baseline-root 'D:\liyue\工业模型赛题研究' `
  --output runtime/my-native-abc-test --datasets A B C --rounds 4

# 实际Docker镜像：工作区tmpfs、/app只读、网络隔离、2 CPU/4GiB。
.\.venv\Scripts\python.exe -B -X utf8 scripts/validate_multitask.py `
  --baseline-root 'D:\liyue\工业模型赛题研究' `
  --output runtime/my-docker-abc-test --datasets A B C --rounds 4 `
  --image gas-rsi-abc:wyx-sdk104-v5-20261123 `
  --docker-executable 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe' `
  --docker-context desktop-linux
```

输出目录必须全新，已有证据不会被覆盖。每项数据保留协议JSONL、训练stderr、完整候选、状态、逐条预测、独立测量及report.json；总目录保存summary.json和镜像约束预检。

## 构建和历史结果边界

联网可用时使用标准`Dockerfile`；它固定Python基础镜像摘要、NumPy1.26.4和SDK1.0.4。此次Docker Hub连接超时，实际使用`Dockerfile.offline`，从已核验的本地v4运行环境离线复制最新应用，再合并为一个运行文件层。离线基础镜像已固定到`sha256:388fb0d9721fa11d0cb9befcb6dee6dbadb23597dfdb59415913d276b4ec25fe`，运行依赖没有重新下载或升级。

```powershell
docker build --platform linux/amd64 --provenance=false -f Dockerfile.offline `
  -t gas-rsi-abc:wyx-sdk104-v5-20261123 .
docker run --rm --network none --read-only gas-rsi-abc:wyx-sdk104-v5-20261123 describe
```

v5镜像ID为`sha256:440079707a61b3282e9d98e9dd38e502714f241be2a85a332aac5dba6a524a69`。

2026-11-22的SwanLab Run `928d73e54bbc4c46bfd28e67487d571e`是**旧v4、仅A任务的真实27B实验**。它不证明新B/C逻辑已经真实模型、云端或正式宿主验收。新v5离线结果应读取本次新目录，不能回写旧云端记录冒充新实验。

## 2026-11-23 实际v5镜像离线验收结果

| 数据集 | 主指标 | 初始公开基线 | 四轮后 | 实际采用新候选次数 |
|---|---|---:|---:|---:|
| A | macro_f1 | 0.714029722 | 0.805881336 | 3 |
| B | rmse | 681.159832604 | 651.626661250 | 1 |
| C | rmse | 52.868572566 | 51.756636508 | 2 |

三个任务各自主完成四轮原生SDK训练、三次非空历史实验；共12个训练容器、27个预测容器。独立重算27组、233,748条预测及实际历史/调用/结束确认记录一致。

A的Macro-F1提高9.1852个百分点；B的RMSE降低约4.34%，C降低约2.10%。B/C同时记录MAE，C的NASA总分由5,823,439,325.342465降低到4,528,567,324.412974。

决策全部来自明确标注的离线宿主替身，真实模型调用为0；该结果证明原生接口和实际数值闭环可执行，不是27B决策质量或正式平台/隐藏集成绩证明。未新建或改写SwanLab记录。

完整证据在`runtime/multitask-docker-v5-20261123/`；原生B/C复测在`runtime/multitask-native-v5-20261123-r2/`。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/verify_multitask_run.py runtime/multitask-docker-v5-20261123
```
