"""基于训练数据拟合 CPU 残差层的非线性基函数。"""
import time
import numpy as np
from ..sensor_training.reference.model import Model

WIDTH = 16
BASIS_KINDS = ("linear", "rbf", "tanh", "stumps")


def features(z, basis):
    kind = basis.get("kind", "linear")
    if kind == "linear":
        return z
    if kind == "rbf":
        centers = np.asarray(basis["centers"])
        squared = (z ** 2).sum(axis=1, keepdims=True) + (centers ** 2).sum(axis=1) - 2 * z @ centers.T
        return np.column_stack([z, np.exp(-np.maximum(squared, 0.) / z.shape[1])])
    if kind == "tanh":
        return np.tanh(z @ np.asarray(basis["weight"]) + basis["bias"])
    return z[:, basis["columns"]] > basis["thresholds"]


def _stumps(z, residual):
    # 用训练分位点构成有限切分集合，逐轮拟合一棵深度为 1 的回归树。
    levels = np.quantile(z, [.2, .4, .6, .8], axis=0).T.reshape(-1)
    columns = np.repeat(np.arange(z.shape[1]), 4)
    mask = (z[:, columns] > levels).astype(float)
    count = mask.sum(axis=0)
    keep = (count >= 10) & (len(z) - count >= 10)
    if not np.any(keep):
        return {"kind": "stumps", "columns": [], "thresholds": []}, [float(residual.mean())]
    mask, count, columns, levels = mask[:, keep], count[keep], columns[keep], levels[keep]
    remaining = residual - residual.mean()
    intercept = float(residual.mean())
    chosen, coefficients = [], []
    for _ in range(WIDTH):
        total = remaining.sum()
        right_sum = remaining @ mask
        left_mean, right_mean = (total - right_sum) / (len(z) - count), right_sum / count
        idx = int(np.argmax((len(z) - count) * left_mean ** 2 + count * right_mean ** 2))
        shift, delta = .25 * left_mean[idx], .25 * (right_mean[idx] - left_mean[idx])
        intercept += float(shift)
        remaining -= shift + delta * mask[:, idx]
        chosen.append(idx)
        coefficients.append(float(delta))
    return {"kind": "stumps", "columns": columns[chosen].tolist(), "thresholds": levels[chosen].tolist()}, [intercept, *coefficients]


def fit(z, residual, kind, ridge, seed, deadline):
    if kind == "tanh":
        names = [f"residual_feature_{i}" for i in range(z.shape[1])]
        model = Model("regression", names, [], z, residual, hidden=WIDTH, seed=seed,
                      learning_rate=.001, batch_size=256)
        # z 已按新增层的 train 统计量标准化，内部 MLP 沿用这个坐标系。
        model.mean.fill(0)
        model.scale.fill(1)
        for _ in range(4):
            if time.monotonic() >= deadline:
                raise TimeoutError("Regression correction budget exhausted")
            model.fit(z, residual, 1)
        scale = model.meta["target_scale"]
        intercept = model.meta["target_mean"] + model.params["b2"][0] * scale
        return {"kind": "tanh", "weight": model.params["w1"].tolist(), "bias": model.params["b1"].tolist()}, [
            float(intercept), *(model.params["w2"][:, 0] * scale).tolist()]
    if kind == "stumps":
        return _stumps(z, residual)
    basis = {"kind": kind}
    if kind == "rbf":
        rng = np.random.default_rng(seed)
        basis["centers"] = z[rng.choice(len(z), WIDTH, replace=len(z) < WIDTH)].tolist()
    matrix = np.column_stack([np.ones(len(z)), features(z, basis)])
    regularizer = np.eye(matrix.shape[1]) * ridge
    regularizer[0, 0] = 1e-8
    coefficients = np.linalg.solve(matrix.T @ matrix / len(z) + regularizer, matrix.T @ residual / len(z))
    return basis, coefficients.tolist()


def validate(basis, size):
    def finite_array(name, shape):
        value = np.asarray(basis.get(name), dtype=float)
        return value.shape == shape and np.isfinite(value).all()

    if not isinstance(basis, dict) or basis.get("kind") not in BASIS_KINDS:
        raise ValueError("Invalid residual basis")
    kind = basis["kind"]
    if kind == "linear":
        return size
    if kind == "rbf" and finite_array("centers", (WIDTH, size)):
        return size + WIDTH
    if kind == "tanh" and finite_array("weight", (size, WIDTH)) and finite_array("bias", (WIDTH,)):
        return WIDTH
    if kind == "stumps":
        columns = basis.get("columns")
        if (isinstance(columns, list) and len(columns) <= WIDTH
                and all(type(i) is int and 0 <= i < size for i in columns)
                and finite_array("thresholds", (len(columns),))):
            return len(columns)
    raise ValueError("Invalid residual basis dimensions or values")
