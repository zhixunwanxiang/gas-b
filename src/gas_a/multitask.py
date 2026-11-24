"""按宿主原生任务分发；A 分类与 B/C 寿命回归不互相改写输入输出。"""


def train(rnd):
    # 训练帧已经给出冻结的任务类型和数据集：必须同时匹配后再交给对应模块。
    # A 的分类保护规则与 B/C 的回归误差规则不同，只改 JSON 格式不能互相替代。
    task = rnd.task.get("task")
    if task == "classification" and rnd.dataset_key == "A":
        from . import submission as classification
        return classification.train(rnd)
    if task == "regression" and rnd.dataset_key in ("B", "C"):
        from .regression import submission as regression
        return regression.train(rnd)
    raise ValueError("Expected A/classification or B,C/regression in the native task contract")


def predict(sample, model_dir):
    # 没有 prediction context 的 SDK 兼容调用仍按 task 分发；回归模块再核对输入和父模型 profile。
    # dataset_key=None 只表示调用方没有附加上下文，不代表可以跳过特征和检查点校验。
    # 这里不把样本 id、文件名或 answer 交给模型，独立预测也不等待后续标签反馈。
    if sample.task == "classification" and sample.dataset_key in (None, "A"):
        from . import submission as classification
        return classification.predict(sample, model_dir)
    if sample.task == "regression" and sample.dataset_key in (None, "B", "C"):
        from .regression import submission as regression
        return regression.predict(sample, model_dir)
    raise ValueError("Prediction task and dataset disagree")
