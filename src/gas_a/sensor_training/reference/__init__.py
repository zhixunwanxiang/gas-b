"""来自 cpu-a-guide-20261110-v1 的原始 model.py 和 data.py，两个文件保持原样。

为保留官方源码 SHA-256 核验能力，中文阅读说明集中放在此处：

- data.read_dataset：读取 JSONL，校验样本 ID、标签与特征 schema，返回 NumPy 数组。
- data.flatten：按确定顺序展开数值特征，并拒绝标签派生字段与非有限数值。
- Model.normalized：沿用父模型 mean/scale 标准化，并裁剪到 [-10, 10]。
- Model.forward/predict：执行 tanh 隐藏层和分类 softmax，训练与推理共用参数。
- Model.fit：使用累计 epoch 派生随机顺序，保留 Adam 一阶矩、二阶矩与 steps 以支持续训。
- Model.save/load：保存和恢复权重、Adam 状态、标准化参数及 metadata，并校验 checkpoint 格式。
"""

