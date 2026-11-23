"""只读取授权投影；严格沿用官方特征顺序，不把 id/meta 或寿命信息当成输入。"""
from ..sensor_training.reference.data import flatten, read_dataset
from ..sensor_training.reference.model import Model


BEARING_FEATURES = (
    "rms", "peak", "peak_to_peak", "mean_abs", "kurtosis", "skewness", "crest_factor",
    "shape_factor", "impulse_factor", "margin_factor", "spectral_centroid", "spectral_rms",
    "spectral_kurtosis", "dominant_freq", "band1_energy_0_2k", "band2_energy_2k_6k",
    "band3_energy_6k_12k",
)
UNITS = {"B": "minutes", "C": "cycles"}


def schema(dataset):
    # 用空值构造官方输入结构，再沿用官方 flatten 的排序生成特征路径。
    # 校验的是每一列的身份和顺序，不只是“有 34/24 个数”；换列会使父模型含义改变。
    if dataset == "B":
        inputs = {"features": {channel: dict.fromkeys(BEARING_FEATURES, 0.0)
                               for channel in ("horizontal", "vertical")}}
    elif dataset == "C":
        inputs = {"settings": [0.0] * 3, "sensors": [0.0] * 21}
    else:
        raise ValueError("Regression dataset must be B or C")
    return flatten(inputs)[0]


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
    elif dataset == "C":
        if (not isinstance(inputs, dict) or set(inputs) != {"settings", "sensors"}
                or not isinstance(inputs["settings"], list) or len(inputs["settings"]) != 3
                or not isinstance(inputs["sensors"], list) or len(inputs["sensors"]) != 21):
            raise ValueError("C requires 3 settings and 21 sensor values")
    paths, values = flatten(inputs)
    # flatten 同时负责数值有效性检查；额外字段不会被悄悄当成可训练特征。
    if paths != schema(dataset):
        raise ValueError("Regression features differ from the official B/C schema")
    return dataset, paths, values


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
