"""兼容官方 MLP 的单文件复合回归模型；新增层有独立的训练统计量。"""
from copy import deepcopy
import time
import numpy as np
from ..sensor_training.reference.model import Model, FORMAT
from .diagnostics import conditions, condition_distances, groups
from . import bearing_guard, bounded_residual, residual_basis

COMPOSITE_FORMAT = "contest.rul-correction.v1"
MAX_CORRECTIONS = 2
TRANSFORMS = {"B": ("bearing_log", "bearing_raw", "bearing_energy"),
              "C": ("condition", "condition_soft", "condition_soft_device", "cmapss_global")}
TRANSFORM_SIZES = {"bearing_log": 57, "bearing_raw": 34, "bearing_energy": 63,
                   "condition": 177, "condition_soft": 177, "condition_soft_device": 177, "cmapss_global": 24}
SOFT_CONDITION = ("condition_soft", "condition_soft_device")
CORRECTION_METHODS = ("frozen_ridge", "frozen_residual")
BOUNDED_HEADS = {"B": ("bounded_bearing", bearing_guard.METHOD),
                 "C": ("bounded_cmapss", bounded_residual.METHOD)}
BOUNDED_METHODS = tuple(method for _, method in BOUNDED_HEADS.values())


def transformed(x, kind, statistics):
    if kind in ("bearing_raw", "cmapss_global"):
        return x
    if kind in ("bearing_log", "bearing_energy"):
        logged = np.sign(x) * np.log1p(np.abs(x))
        # 输入每通道按字母排序，前三列为三个频带能量。
        bands = [np.abs(x[:, offset:offset + 3]) for offset in (0, 17)]
        fractions = [b / np.maximum(b.sum(axis=1, keepdims=True), 1e-12) for b in bands]
        extra = []
        if kind == "bearing_energy":
            total = bands[0] + bands[1]
            relative = np.divide(bands[0], total, out=np.full_like(total, .5), where=total > 0)
            extra = [np.log1p(total), relative]
        return np.column_stack([logged, *fractions, logged[:, :17] - logged[:, 17:], *extra])
    if kind not in TRANSFORMS["C"] or x.shape[1] != 24:
        raise ValueError("Unsupported regression correction transform")
    if kind in SOFT_CONDITION:
        distance = condition_distances(x)
        nearest = distance.min(axis=1, keepdims=True)
        membership = np.exp(-(distance - nearest) / (.1 ** 2))
        membership /= membership.sum(axis=1, keepdims=True)
        membership *= np.clip(1 - (nearest / (.15 ** 2)), 0., 1.)
        membership *= np.asarray(statistics["condition_support"])
        mean, scale = np.asarray(statistics["condition_mean"]), np.asarray(statistics["condition_scale"])
        local = np.clip((x[:, None, :21] - mean) / scale, -10, 10) * membership[:, :, None]
        return np.column_stack([x, local.sum(axis=1), membership, local.reshape(len(x), 126)])
    regime = conditions(x)
    local = np.zeros((len(x), 21))
    one_hot = np.zeros((len(x), 6))
    for i in range(6):
        mask = regime == i
        local[mask] = (x[mask, :21] - statistics["condition_mean"][i]) / statistics["condition_scale"][i]
        one_hot[mask, i] = 1
    # 全局特征保留，局部特征及交互负责工况残差；未知工况的局部分支为零。
    local = np.clip(local, -10, 10)
    return np.column_stack([x, local, one_hot, (one_hot[:, :, None] * local[:, None, :]).reshape(len(x), -1)])


def fit_statistics(x, kind, devices=None):
    statistics = {}
    if kind == "condition" or kind in SOFT_CONDITION:
        regime = conditions(x)
        mean, variance = x[:, :21].mean(axis=0), x[:, :21].var(axis=0)
        means, scales, support = [], [], []
        for i in range(6):
            mask = regime == i
            selected = x[mask, :21]
            # 小组向全局收缩；总方差包含组均值与混合均值的差，避免小组尺度塌缩。
            if kind == "condition_soft_device":
                if devices is None or np.any(devices == "unknown"):
                    raise ValueError("Device-based condition support requires engine trace metadata")
                count = len(np.unique(devices[mask]))
                weight = count / (count + 10.)
            else:
                weight = len(selected) / (len(selected) + 100.)
            group_mean = selected.mean(axis=0) if len(selected) else mean
            group_variance = selected.var(axis=0) if len(selected) else variance
            mixed_mean = weight * group_mean + (1 - weight) * mean
            mixed_variance = (weight * (group_variance + (group_mean - mixed_mean) ** 2)
                              + (1 - weight) * (variance + (mean - mixed_mean) ** 2))
            means.append(mixed_mean.tolist())
            scales.append(np.maximum(np.sqrt(mixed_variance), 1e-6).tolist())
            support.append(weight)
        statistics.update(condition_mean=means, condition_scale=scales)
        if kind in SOFT_CONDITION:
            statistics["condition_support"] = support
    features = transformed(x, kind, statistics)
    statistics.update(mean=features.mean(axis=0).tolist(), scale=np.maximum(features.std(axis=0), 1e-6).tolist())
    return statistics


def standardized(x, layer):
    stats = layer["statistics"]
    return np.clip((transformed(x, layer["kind"], stats) - stats["mean"]) / stats["scale"], -10, 10)


def design(x, layer):
    z = standardized(x, layer)
    return np.column_stack([np.ones(len(x)), residual_basis.features(z, layer.get("basis", {}))])


class RegressionModel(Model):
    def validate(self):
        # 复合格式不能被原始 Model.load 静默读成只剩 MLP 的模型。
        metadata = self.meta
        composite = metadata.get("format") == COMPOSITE_FORMAT
        if composite:
            self.meta = {**metadata, "format": FORMAT}
        try:
            super().validate()
        finally:
            self.meta = metadata
        layers = metadata.get("corrections", [])
        profile = metadata.get("profile")
        has_cmapss = "bounded_cmapss" in metadata
        has_bearing = "bounded_bearing" in metadata
        if (not isinstance(layers, list) or len(layers) > MAX_CORRECTIONS
                or ((bool(layers) or has_cmapss or has_bearing) != composite)
                or ("corrections" in metadata and not composite)
                or (has_cmapss and profile != "C") or (has_bearing and profile != "B")):
            raise ValueError("Invalid regression correction manifest")
        for layer in layers:
            if (not isinstance(layer, dict) or layer.get("kind") not in TRANSFORMS.get(profile, ())
                    or type(layer.get("steps")) is not int or layer["steps"] < 0
                    or type(layer.get("strength")) not in (int, float) or not 0 <= layer["strength"] <= 1
                    or not isinstance(layer.get("train_sha256"), str)
                    or len(layer["train_sha256"]) != 71 or not layer["train_sha256"].startswith("sha256:")):
                raise ValueError("Invalid regression correction identity")
            stats = layer.get("statistics")
            size = TRANSFORM_SIZES[layer["kind"]]
            if not isinstance(stats, dict):
                raise ValueError("Invalid correction statistics")
            shapes = {"mean": (size,), "scale": (size,)}
            if layer["kind"] == "condition" or layer["kind"] in SOFT_CONDITION:
                shapes.update(condition_mean=(6, 21), condition_scale=(6, 21))
            if layer["kind"] in SOFT_CONDITION:
                shapes["condition_support"] = (6,)
            for name, shape in shapes.items():
                values = np.asarray(stats.get(name), dtype=float)
                if values.shape != shape or not np.isfinite(values).all() or ("scale" in name and np.any(values <= 0)):
                    raise ValueError("Invalid correction scaler")
                if name == "condition_support" and np.any((values < 0) | (values > 1)):
                    raise ValueError("Invalid correction condition support")
            coefficients = np.asarray(layer.get("coefficients"), dtype=float)
            width = residual_basis.validate(layer.get("basis", {"kind": "linear"}), size)
            if coefficients.shape != (width + 1,) or not np.isfinite(coefficients).all():
                raise ValueError("Invalid regression correction coefficients")
        self._bounded_parameters = None
        if has_cmapss:
            self._bounded_parameters = bounded_residual.deserialize(metadata["bounded_cmapss"])
        elif has_bearing:
            self._bounded_parameters = bearing_guard.deserialize(metadata["bounded_bearing"])

    def correction(self, x):
        result = np.zeros(len(x))
        for layer in self.meta.get("corrections", []):
            result += layer["strength"] * (design(x, layer) @ layer["coefficients"])
        return result

    def predict_anchor(self, x):
        # 无层时与官方基线逐位一致；有层时也从原始 NN 输出叠加，只在最后做非负裁剪。
        if not self.meta.get("corrections"):
            return super().predict(x)
        _, output = self.forward(self.normalized(x))
        return np.maximum(0, output[:, 0] * self.meta["target_scale"] + self.meta["target_mean"] + self.correction(x))

    def predict(self, x):
        anchor = self.predict_anchor(x)
        head = self.meta.get("bounded_cmapss")
        if head is not None:
            return bounded_residual.predict(x, anchor, head, self._bounded_parameters)
        head = self.meta.get("bounded_bearing")
        if head is not None and head["strength"] > 0.:
            return bearing_guard.predict(transformed(x, "bearing_energy", {}), anchor, head, self._bounded_parameters)
        return anchor


def fit_guarded_bearing(parent, train, deadline, *, learning_rate):
    if parent.meta.get("profile") != "B":
        raise ValueError("Bearing upward guard requires B")
    candidate = deepcopy(parent)
    candidate.meta["format"] = COMPOSITE_FORMAT
    anchor = parent.predict_anchor(train["x"])
    candidate.meta["bounded_bearing"] = bearing_guard.fit(
        transformed(train["x"], "bearing_energy", {}), train["y"], anchor, groups(train, "B")["device"],
        train["sha256"], deadline, learning_rate=learning_rate)
    if time.monotonic() >= deadline:
        raise TimeoutError("B upward guard training budget exhausted")
    candidate.validate()
    return candidate


def fit_bounded_cmapss(parent, train, deadline, *, learning_rate):
    if parent.meta.get("profile") != "C":
        raise ValueError("Bounded residual requires C")
    candidate = deepcopy(parent)
    candidate.meta["format"] = COMPOSITE_FORMAT
    anchor = parent.predict_anchor(train["x"])
    candidate.meta["bounded_cmapss"] = bounded_residual.fit(
        train["x"], train["y"], anchor, train["sha256"], parent.meta["seed"], deadline,
        learning_rate=learning_rate)
    if time.monotonic() >= deadline:
        raise TimeoutError("Bounded C residual training budget exhausted")
    candidate.validate()
    return candidate


def frozen_correction(parent, train, kind, ridge, deadline, *, basis="linear"):
    """拟合一个新层，已有 MLP、Adam 和所有旧层完全冻结；返回零强度候选。"""
    if len(parent.meta.get("corrections", [])) >= MAX_CORRECTIONS:
        raise ValueError("Regression correction layer limit reached")
    devices = groups(train, "C")["device"] if kind == "condition_soft_device" else None
    stats = fit_statistics(train["x"], kind, devices)
    layer = {"kind": kind, "statistics": stats, "strength": 0., "steps": 1,
             "train_sha256": train["sha256"]}
    z = standardized(train["x"], layer)
    # NN 原始输出允许为负；拟合整体在裁剪前的残差，避免叠加层与预测定义不一致。
    _, output = parent.forward(parent.normalized(train["x"]))
    raw = output[:, 0] * parent.meta["target_scale"] + parent.meta["target_mean"] + parent.correction(train["x"])
    residual = train["y"] - raw
    if time.monotonic() >= deadline:
        raise TimeoutError("Regression correction budget exhausted")
    components, layer["coefficients"] = residual_basis.fit(z, residual, basis, ridge, parent.meta["seed"], deadline)
    if basis != "linear":
        layer["basis"] = components
    if time.monotonic() >= deadline:
        raise TimeoutError("Regression correction budget exhausted")
    candidate = deepcopy(parent)
    candidate.meta["format"] = COMPOSITE_FORMAT
    candidate.meta.setdefault("corrections", []).append(layer)
    candidate.validate()
    return candidate
