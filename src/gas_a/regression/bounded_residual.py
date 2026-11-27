"""C 的条件归一化及限幅残差头，训练和预测只依赖 NumPy。"""
import math
import time

import numpy as np

from .diagnostics import conditions

METHOD = "cmapss_bounded_residual"
WIDTHS = (52, 96, 48, 24, 1)
EPOCHS = 16


def _condition_features(x, stats):
    sensors = x[:, :21]
    regime = conditions(x)
    raw = np.clip((sensors - stats["global_mean"]) / stats["global_scale"], -10, 10)
    normalized = raw.copy()
    known = regime >= 0
    normalized[known] = ((sensors[known] - stats["condition_mean"][regime[known]])
                         / stats["condition_scale"][regime[known]])
    return np.column_stack([raw, np.clip(normalized, -8, 8),
                            regime[:, None] == np.arange(6), x[:, 21:] / [42., .84, 100.]])


def _inputs(x, anchor, stats):
    values = np.column_stack([_condition_features(x, stats), anchor])
    return np.clip((values - stats["mean"]) / stats["scale"], -10, 10)


def _forward(x, weights, biases):
    for weight, bias in zip(weights[:-1], biases[:-1]):
        x = x @ weight + bias
        x = x / (1 + np.exp(-np.clip(x, -75, 75)))
    return (x @ weights[-1] + biases[-1])[:, 0]


def _base_statistics(x):
    regime = conditions(x)
    sensors = x[:, :21]
    mean, var = sensors.mean(axis=0), sensors.var(axis=0)
    blocks = [sensors[regime == i] for i in range(6)]
    condition_mean = np.array([b.mean(axis=0) if len(b) else mean for b in blocks])
    condition_var = np.array([b.var(axis=0) if len(b) else var for b in blocks])
    return {"global_mean": mean, "global_scale": np.maximum(np.sqrt(var), 1e-6),
            "condition_mean": condition_mean,
            "condition_scale": np.sqrt(condition_var + .03 * np.median(condition_var, axis=0) + 1e-10)}


def fit(x, target, anchor, train_sha, seed, deadline, *, learning_rate):
    stats = _base_statistics(x)
    values = np.column_stack([_condition_features(x, stats), anchor])
    stats.update(mean=values.mean(axis=0), scale=np.maximum(values.std(axis=0), .005))
    z = np.clip((values - stats["mean"]) / stats["scale"], -10, 10).astype(np.float32)
    scale = 84.
    residual = ((target - anchor) / scale).astype(np.float32)[:, None]
    rng_init = np.random.default_rng(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(WIDTHS, WIDTHS[1:]):
        radius = 1 / math.sqrt(fan_in)
        weights.append(rng_init.uniform(-radius, radius, (fan_in, fan_out)).astype(np.float32))
        biases.append(rng_init.uniform(-radius, radius, fan_out).astype(np.float32))
    weights[-1].fill(0.)
    biases[-1].fill(0.)
    params = weights + biases
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(seed)
    step = 0
    lr = learning_rate
    for _ in range(EPOCHS):
        order = rng.permutation(len(z))
        for ii in np.array_split(order, (len(order) + 511) // 512):
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded C residual training budget exhausted")
            activations, slopes = [z[ii]], []
            for weight, bias in zip(weights[:-1], biases[:-1]):
                value = activations[-1] @ weight + bias
                sigmoid = 1 / (1 + np.exp(-np.clip(value, -75, 75)))
                activations.append(value * sigmoid)
                slopes.append(sigmoid * (1 + value * (1 - sigmoid)))
            delta = (activations[-1] @ weights[-1] + biases[-1] - residual[ii]) * (2 / len(ii))
            dw, db = [None] * len(weights), [None] * len(weights)
            for j in reversed(range(len(weights))):
                dw[j] = activations[j].T @ delta
                db[j] = delta.sum(axis=0)
                if j:
                    delta = (delta @ weights[j].T) * slopes[j - 1]
            step += 1
            for p, m, v, grad in zip(params, first, second, dw + db):
                m *= .9
                m += .1 * grad
                v *= .999
                v += .001 * grad ** 2
                p *= 1 - lr * .005
                p -= (lr / (1 - .9 ** step)) * m / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
    lower, upper = np.quantile(target, [.90, .97]).tolist()
    # 在寿命标签近似常数的小投影上，保留连续的修正带。
    upper = max(lower + 1., upper)
    return {"format": METHOD, "train_sha256": train_sha, "strength": 0., "steps": step,
            "statistics": {key: value.tolist() for key, value in stats.items()},
            "weights": [w.tolist() for w in weights], "biases": [b.tolist() for b in biases],
            "lower": lower, "upper": upper, "target_scale": scale}


def deserialize(state):
    if (not isinstance(state, dict) or state.get("format") != METHOD
            or type(state.get("steps")) is not int or state["steps"] < 1
            or not isinstance(state.get("train_sha256"), str) or len(state["train_sha256"]) != 71
            or not state["train_sha256"].startswith("sha256:")):
        raise ValueError("Invalid bounded C residual header")
    for key in ("lower", "upper", "strength", "target_scale"):
        value = state.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Invalid bounded C residual parameters")
    if not (0 <= state["strength"] <= 1 and state["target_scale"] > 0 and 0 <= state["lower"] < state["upper"]):
        raise ValueError("Invalid bounded C residual thresholds")

    def array(value, shape):
        result = np.asarray(value, dtype=float)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError("Invalid bounded C residual tensors")
        return result

    shapes = {"global_mean": (21,), "global_scale": (21,), "condition_mean": (6, 21),
              "condition_scale": (6, 21), "mean": (52,), "scale": (52,)}
    stored = state.get("statistics")
    if not isinstance(stored, dict):
        raise ValueError("Invalid bounded C residual scaler")
    stats = {key: array(stored.get(key), shape) for key, shape in shapes.items()}
    if any(np.any(stats[key] <= 0) for key in ("global_scale", "condition_scale", "scale")):
        raise ValueError("Invalid bounded C residual scaler")
    for key in ("weights", "biases"):
        if not isinstance(state.get(key), list) or len(state[key]) != len(WIDTHS) - 1:
            raise ValueError("Invalid bounded C residual depth")
    weights = [array(w, (a, b)) for w, a, b in zip(state["weights"], WIDTHS, WIDTHS[1:])]
    biases = [array(bias, (width,)) for bias, width in zip(state["biases"], WIDTHS[1:])]
    return stats, weights, biases


def predict(x, anchor, state, cached):
    strength = state["strength"]
    if strength == 0:
        return anchor
    stats, weights, biases = cached
    raw = _forward(_inputs(x, anchor, stats), weights, biases) * state["target_scale"]
    delta = 40. * np.tanh(raw / 40.)
    lower, upper = state["lower"], state["upper"]
    upward = np.clip((upper - anchor) / (upper - lower), 0., 1.)
    adjusted = anchor + np.minimum(delta, 0.) + upward * np.maximum(delta, 0.)
    adjusted -= .20 * np.maximum(adjusted - lower, 0.)
    return np.maximum(0., anchor + strength * (adjusted - anchor))
