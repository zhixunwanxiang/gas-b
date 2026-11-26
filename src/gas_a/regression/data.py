"""只读取授权投影；严格沿用官方特征顺序，不把 id/meta 或寿命信息当成输入。"""
from functools import lru_cache
import math
from ..sensor_training.reference.data import flatten, read_dataset
from ..sensor_training.reference.model import Model


BEARING_FEATURES = (
    "rms", "peak", "peak_to_peak", "mean_abs", "kurtosis", "skewness", "crest_factor",
    "shape_factor", "impulse_factor", "margin_factor", "spectral_centroid", "spectral_rms",
    "spectral_kurtosis", "dominant_freq", "band1_energy_0_2k", "band2_energy_2k_6k",
    "band3_energy_6k_12k",
)
UNITS = {"B": "minutes", "C": "cycles"}


@lru_cache(maxsize=2)
def _schema(dataset):
    # 用空值构造官方输入结构，再沿用官方 flatten 的排序生成特征路径。
    # 校验的是每一列的身份和顺序，不只是“有 34/24 个数”；换列会使父模型含义改变。
    if dataset == "B":
        inputs = {"features": {channel: dict.fromkeys(BEARING_FEATURES, 0.0)
                               for channel in ("horizontal", "vertical")}}
    elif dataset == "C":
        inputs = {"settings": [0.0] * 3, "sensors": [0.0] * 21}
    else:
        raise ValueError("Regression dataset must be B or C")
    # 缓存不可变的路径，避免每条预测重新构造 24/34 个 JSON 路径。
    return tuple(flatten(inputs)[0])


def schema(dataset):
    # 对外仍返回独立 list，调用方修改结果不会污染后续输入检查或模型加载。
    return list(_schema(dataset))


def input_values(inputs, dataset=None):
    # 结构用于识别协议，不用样本编号、文件名或 meta 推断数据集/目标。
    if dataset is None:
        dataset = "B" if isinstance(inputs, dict) and set(inputs) == {"features"} else "C"
    if dataset == "B":
        if (not isinstance(inputs, dict) or set(inputs) != {"features"}
                or not isinstance(inputs["features"], dict) or set(inputs["features"]) != {"horizontal", "vertical"}
                or any(not isinstance(inputs["features"][c], dict)
                       or set(inputs["features"][c]) != set(BEARING_FEATURES) for c in ("horizontal", "vertical"))):
            raise ValueError("B requires exactly two official 17-feature vibration channels")
        # 与官方 flatten 的“字典键排序”完全一致，不能按接收时的字典顺序取值。
        scalars = [inputs["features"][channel][name]
                   for channel in ("horizontal", "vertical") for name in sorted(BEARING_FEATURES)]
    elif dataset == "C":
        if (not isinstance(inputs, dict) or set(inputs) != {"settings", "sensors"}
                or not isinstance(inputs["settings"], list) or len(inputs["settings"]) != 3
                or not isinstance(inputs["sensors"], list) or len(inputs["sensors"]) != 21):
            raise ValueError("C requires 3 settings and 21 sensor values")
        # 官方字典排序先 sensors 后 settings，数组内部仍按下标顺序。
        scalars = inputs["sensors"] + inputs["settings"]
    else:
        raise ValueError("Regression dataset must be B or C")
    values = []
    for value in scalars:
        # 结构已严格限定，只需校验标量；布尔、嵌套字段和非有限值仍然拒绝。
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Regression inputs must contain finite numeric features only")
        values.append(float(value))
    return dataset, schema(dataset), values


def load_model(path, dataset, paths=None):
    # 权重文件自身包含任务、数据集 profile、特征路径和目标缩放。
    # 即使 B/C 都输出一个数，也不能把“分钟”的模型直接拿去预测“循环数”。
    model = Model.load(path)
    if (model.meta["task"] != "regression" or model.meta.get("profile") != dataset
            or model.meta["paths"] != (schema(dataset) if paths is None else paths)):
        raise ValueError("Checkpoint regression profile or feature schema does not match the task")
    if dataset == "B" and model.meta.get("target_unit") != "minutes":
        raise ValueError("B checkpoint must predict remaining minutes")
    if model.meta.get("target_unit", UNITS[dataset]) != UNITS[dataset]:
        raise ValueError("Checkpoint target unit does not match the task")
    return model


def load_pair(train_path, validation_path, dataset):
    # 两份文件分别读取，不把公开验证样本并回训练集；验证标签仅供后续诊断与选版。
    # id 只用于检查投影是否交叉，不进入 x 特征矩阵，也不用于反推寿命。
    train = read_dataset(train_path, "regression", [], schema(dataset))
    validation = read_dataset(validation_path, "regression", [], train["paths"])
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("Public training and validation sample IDs overlap")
    return train, validation
