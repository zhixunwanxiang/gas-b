"""任务 A 的数据和模型边界：仅接收 128 维特征，隔离 train/validation 并统一评估指标。"""
# 数据边界：id 和 answer 均不能成为特征，输入固定为 128 维 features。
# train 与 validation 必须有相同特征顺序且样本 ID 不重叠；评估不修改模型状态。
import numpy as np
from pathlib import Path
from .io import CLASSES
from .reference.data import read_dataset, json_value
from .architecture import load_checkpoint, MigratedModel


def load_unlabelled(path):
    """监测仅使用 inputs，标签和 metadata 不进入特征。"""
    from pathlib import Path
    from .reference.data import flatten, MAX_BYTES
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Current data must be a regular JSONL file <=128 MiB")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json_value(line)
            except ValueError as error:
                raise ValueError(f"{Path(path).name}: line {number}: {error}") from error
            inputs = row.get("inputs") if isinstance(row, dict) else None
            if not isinstance(inputs, dict) or set(inputs) != {"features"} or not isinstance(inputs["features"], list) or len(inputs["features"]) != 128:
                raise ValueError("Current rows require inputs.features with 128 numbers")
            _, values = flatten(inputs)
            if len(values) != 128:
                raise ValueError("Current rows require 128 numeric scalar features")
            rows.append(values)
    if not rows:
        raise ValueError("Current dataset is empty")
    return np.asarray(rows, dtype=np.float64)


def load_data(path, expected_paths=None):
    # 在通用 reference 特征展开之前检查输入，仅允许 128 维 features 数组。
    """先限定任务 A 的输入结构，再由 reference 加载器校验样本 ID、标签与特征 schema。"""
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json_value(line)
            except ValueError as error:
                raise ValueError(f"{Path(path).name}: line {number}: {error}") from error
            inputs = row.get("inputs") if isinstance(row, dict) else None
            if not isinstance(inputs, dict) or set(inputs) != {"features"}:
                raise ValueError(f"{Path(path).name}: line {number}: A input must contain only inputs.features; IDs and targets are not features")
            values = inputs["features"]
            if not isinstance(values, list) or len(values) != 128:
                raise ValueError(f"{Path(path).name}: line {number}: A requires exactly 128 features")
    result = read_dataset(path, "classification", CLASSES, expected_paths)
    if result["x"].shape[1] != 128:
        raise ValueError("A requires 128 numeric features")
    return result


def load_pair(train_path, validation_path):
    """按相同特征 schema 加载 train 和 validation，并拒绝两者出现相同样本 ID。"""
    # id 只用于样本唯一性和划分交叉检查；同名或匿名标识不参与特征，也不用于推断标签。
    train = load_data(train_path)
    # validation 沿用 train 的特征路径与顺序，防止维度相同但语义错位。
    validation = load_data(validation_path, train["paths"])
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("Training and validation IDs overlap")
    return train, validation


def load_model(path, paths=None):
    """加载原参考格式或显式迁移的新架构；两者都保持任务 A 的输入输出契约。"""
    model = load_checkpoint(path)
    if (model.meta.get("input_transform", "identity") != "identity"
            or model.meta.get("active_features") is not None or model.meta.get("sensor_weights") is not None):
        raise ValueError("Checkpoint requires experimental preprocessing; it is not reference-predictor compatible.")
    return _check_model(model, paths)


def _check_model(model, paths):
    """固定任务 A 的输入输出；非原始宽度必须使用带迁移证据的新格式。"""
    if (model.meta["task"] != "classification" or model.meta.get("profile") != "A"
            or model.meta["classes"] != CLASSES or len(model.meta["paths"]) != 128):
        raise ValueError("This project accepts task A checkpoints only")
    if not isinstance(model, MigratedModel) and model.meta["hidden"] != 32:
        raise ValueError("Task A checkpoint hidden width must be 32; architecture changes require explicit weight migration")
    if paths is not None and model.meta["paths"] != paths:
        raise ValueError("Feature schema does not match parent checkpoint")
    return model


def evaluate(model, data):
    """执行一次纯预测并计算指标，不更新权重、Adam 状态或标准化参数。"""
    import time
    start = time.perf_counter()
    probabilities = model.predict(data["x"])
    return score_predictions(probabilities, data["y"], time.perf_counter() - start)


def score_predictions(probabilities, expected, prediction_seconds=0.0):
    """直接为已保存的预测计算指标，不重新推理或更新模型。"""
    # 混淆矩阵的行是真实类别、列是预测类别，F1=2TP/(真实支持数+预测数)。
    # 遵循参考口径，只平均真实或预测出现过的类别；不能把它与固定补零六类口径混算。
    probabilities = np.asarray(probabilities)
    expected = np.asarray(expected)
    if (probabilities.shape != (len(expected), 6) or not len(expected)
            or not np.isfinite(probabilities).all() or expected.ndim != 1
            or not np.issubdtype(expected.dtype, np.integer) or np.any((expected < 0) | (expected >= 6))):
        raise ValueError("Non-finite or incorrectly shaped predictions")
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((6, 6), dtype=int)
    np.add.at(matrix, (expected, predicted), 1)
    per_class, observed_f1 = {}, []
    for k, name in enumerate(CLASSES):
        support = int(matrix[k].sum())
        selected = int(matrix[:, k].sum())
        tp = int(matrix[k, k])
        f1 = 2 * tp / (support + selected) if support + selected else 0.0
        # Macro-F1 与 reference 口径一致：纳入真实或预测中出现的类别。
        # 对两者均未出现的类别，不额外补零拉低当前窗口分数。
        if support + selected:
            observed_f1.append(f1)
        per_class[name] = {"support": support, "predicted": selected, "correct": tp,
                           "precision": tp / selected if selected else 0.0,
                           "recall": tp / support if support else 0.0, "f1": f1}
    return {"accuracy": float(np.mean(predicted == expected)), "macro_f1": float(np.mean(observed_f1)),
            "macro_labels": "observed_true_or_predicted_as_reference_baseline",
            "per_class": per_class, "confusion_matrix": matrix.tolist(), "rows": len(expected),
            "prediction_seconds": prediction_seconds}

