"""B residual MLP with vibration support and limits on upward corrections."""
import time

import numpy as np

METHOD = "bearing_upward_guard"
WIDTHS = (64, 64, 32, 1)
MAX_EPOCHS = 90
MAX_PROTOTYPES = 1024
TARGET_SCALE = 785.
MAX_CORRECTION = 450.
LOW_CUTOFF = 230.
HIGH_CUTOFF = 759.
HIGH_REDUCTION_LIMIT = 30.


def guarded_delta(anchor, delta):
    upward_weight = np.clip((LOW_CUTOFF - anchor) / (LOW_CUTOFF / 2.), 0., 1.)
    downward_weight = np.clip((anchor - LOW_CUTOFF) / (HIGH_CUTOFF - LOW_CUTOFF), 0., 1.)
    upward = np.maximum(delta, 0.) * upward_weight
    downward = np.minimum(delta, 0.) * downward_weight
    high_reduction = -np.clip(.20 * (anchor - HIGH_CUTOFF), 0., HIGH_REDUCTION_LIMIT)
    return upward + np.minimum(downward, high_reduction)


def _distances(points, references):
    squared_norm = np.sum(points ** 2, axis=1, keepdims=True)
    nearest = [(squared_norm + norms - 2 * points @ rows.T).min(axis=1) for rows, norms in references]
    return np.sqrt(np.maximum(0., np.column_stack(nearest)))


def fit_support(values, bearings):
    names = np.unique(bearings)
    if len(names) < 2 or "unknown" in names or len(names) > MAX_PROTOTYPES:
        raise ValueError("B upward guard requires identifiable training rows from at least two bearings")
    low, high = np.quantile(values, [.1, .9], axis=0)
    center = np.median(values, axis=0)
    scale = np.maximum(high - low, .03)
    normalized = np.clip((values - center) / scale, -8., 8.)
    rng = np.random.default_rng(42)
    per_bearing = min(80, MAX_PROTOTYPES // len(names))
    prototypes = []
    for name in names:
        indices = np.flatnonzero(bearings == name)
        prototypes.append(normalized[rng.choice(indices, min(per_bearing, len(indices)), replace=False)])
    sample = np.concatenate(prototypes)
    origin = np.repeat(np.arange(len(prototypes)), [len(rows) for rows in prototypes])
    references = [(rows, np.sum(rows ** 2, axis=1)) for rows in prototypes]
    cross_bearing = _distances(sample, references)
    cross_bearing[np.arange(len(sample)), origin] = np.inf
    lower, upper = np.quantile(cross_bearing.min(axis=1), [.75, .95])
    return {"center": center.tolist(), "scale": scale.tolist(), "prototypes": [rows.tolist() for rows in prototypes],
            "lower": float(lower), "upper": float(upper)}


def load_support(state):
    if not isinstance(state, dict) or not isinstance(state.get("prototypes"), list):
        raise ValueError("Invalid B upward guard support")
    shape = (WIDTHS[0] - 1,)
    center = _array(state.get("center"), shape)
    scale = _array(state.get("scale"), shape)
    lower, upper = state.get("lower"), state.get("upper")
    if (np.any(scale <= 0.) or not all(type(v) in (int, float) and np.isfinite(v) for v in (lower, upper))
            or not 0. <= lower <= upper or not 2 <= len(state["prototypes"]) <= MAX_PROTOTYPES):
        raise ValueError("Invalid B upward guard support thresholds")
    references = []
    for block in state["prototypes"]:
        if not isinstance(block, list) or not 1 <= len(block) <= 80:
            raise ValueError("Invalid B upward guard support prototypes")
        rows = _array(block, (len(block), *shape))
        if np.any(np.abs(rows) > 8.):
            raise ValueError("Invalid B upward guard support prototypes")
        references.append((rows, np.sum(rows ** 2, axis=1)))
    if sum(len(rows) for rows, _ in references) > MAX_PROTOTYPES:
        raise ValueError("Too many B upward guard prototypes")
    return center, scale, tuple(references), lower, upper


def support_weight(values, support):
    center, scale, references, lower, upper = support
    points = np.clip((values - center) / scale, -8., 8.)
    distances = _distances(points, references)
    second_closest = np.partition(distances, 1, axis=1)[:, 1]
    return np.clip((upper - second_closest) / max(upper - lower, .1), 0., 1.)


def _activations(inputs, weights, biases):
    values = [inputs]
    for weight, bias in zip(weights[:-1], biases[:-1]):
        values.append(np.maximum(values[-1] @ weight + bias, 0.))
    values.append(values[-1] @ weights[-1] + biases[-1])
    return values


def _fit_head(x, y, learning_rate, deadline):
    # Match the validated 64/32 ReLU residual fit with one RNG for initialization and shuffling.
    rng = np.random.RandomState(42)
    weights, biases = [], []
    for fan_in, fan_out in zip(WIDTHS, WIDTHS[1:]):
        radius = np.sqrt(6. / (fan_in + fan_out))
        weights.append(rng.uniform(-radius, radius, (fan_in, fan_out)))
        biases.append(rng.uniform(-radius, radius, fan_out))
    params = weights + biases
    first = [np.zeros_like(param) for param in params]
    second = [np.zeros_like(param) for param in params]
    order = np.arange(len(x))
    best_loss, stale, steps = np.inf, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(order)
        total_loss = 0.
        for start in range(0, len(order), 256):
            if time.monotonic() >= deadline:
                raise TimeoutError("B upward guard training budget exhausted")
            rows = order[start:start + 256]
            values = _activations(x[rows], weights, biases)
            error = values[-1] - y[rows, None]
            total_loss += .5 * np.sum(error ** 2) + .02 * sum(np.dot(w.ravel(), w.ravel()) for w in weights)
            dw, db = [None] * len(weights), [None] * len(biases)
            for layer in reversed(range(len(weights))):
                dw[layer] = (values[layer].T @ error + .04 * weights[layer]) / len(rows)
                db[layer] = error.sum(axis=0) / len(rows)
                if layer:
                    error = error @ weights[layer].T
                    error[values[layer] <= 0.] = 0.
            steps += 1
            rate = learning_rate * np.sqrt(1 - .999 ** steps) / (1 - .9 ** steps)
            for param, moment, variance, gradient in zip(params, first, second, dw + db):
                moment *= .9
                moment += (1 - .9) * gradient
                variance *= .999
                variance += (1 - .999) * gradient ** 2
                param -= rate * moment / (np.sqrt(variance) + 1e-8)
        loss = total_loss / len(x)
        if not np.isfinite(loss):
            raise ValueError("Nonfinite B upward guard training loss")
        stale = stale + 1 if loss > best_loss - 1e-4 else 0
        best_loss = min(best_loss, loss)
        if stale > 10:
            break
    return weights, biases, epoch, steps


def fit(features, target, anchor, bearings, train_sha, deadline, *, learning_rate):
    values = np.column_stack([features, anchor])
    mean = np.mean(values, axis=0)
    centered = values - mean
    variance = (np.sum(centered ** 2, axis=0) - np.sum(centered, axis=0) ** 2 / len(values)) / len(values)
    variance = np.maximum(variance, 0.)
    eps = np.finfo(np.float64).eps
    constant = variance <= (len(values) * eps * variance + (len(values) * mean * eps) ** 2)
    scale = np.where(constant, 1., np.sqrt(variance))
    support = fit_support(features, bearings)
    w, b, epochs, steps = _fit_head(centered / scale, (target - anchor) / TARGET_SCALE, learning_rate, deadline)
    return {"format": METHOD, "train_sha256": train_sha, "strength": 0., "epochs": epochs, "steps": steps,
            "mean": mean.tolist(), "scale": scale.tolist(), "support": support,
            "weights": [v.tolist() for v in w], "biases": [v.tolist() for v in b]}


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid B upward guard tensors")
    return result


def deserialize(state):
    if (not isinstance(state, dict) or state.get("format") != METHOD
            or type(state.get("steps")) is not int or state["steps"] < 1
            or type(state.get("epochs")) is not int or not 1 <= state["epochs"] <= MAX_EPOCHS
            or not isinstance(state.get("train_sha256"), str) or len(state["train_sha256"]) != 71
            or not state["train_sha256"].startswith("sha256:")
            or type(state.get("strength")) not in (int, float) or not 0 <= state["strength"] <= 1.):
        raise ValueError("Invalid B upward guard header")
    mean, scale = [_array(state.get(key), (WIDTHS[0],)) for key in ("mean", "scale")]
    if np.any(scale <= 0.):
        raise ValueError("Invalid B upward guard scaler")
    for key in ("weights", "biases"):
        if not isinstance(state.get(key), list) or len(state[key]) != len(WIDTHS) - 1:
            raise ValueError("Invalid B upward guard depth")
    weights = [_array(w, (a, b)) for w, a, b in zip(state["weights"], WIDTHS, WIDTHS[1:])]
    biases = [_array(bias, (width,)) for bias, width in zip(state["biases"], WIDTHS[1:])]
    return mean, scale, load_support(state.get("support")), weights, biases


def predict(features, anchor, state, cached):
    mean, scale, support, weights, biases = cached
    inputs = (np.column_stack([features, anchor]) - mean) / scale
    residual = _activations(inputs, weights, biases)[-1][:, 0] * TARGET_SCALE
    delta = MAX_CORRECTION * np.tanh(residual / MAX_CORRECTION)
    return np.maximum(0., anchor + state["strength"] * support_weight(features, support) * guarded_delta(anchor, delta))
