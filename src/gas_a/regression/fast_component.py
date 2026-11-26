"""One fixed-size train-only RBF correction; no tree accumulation or new dependency."""
from copy import deepcopy
import math
import numpy as np

FORMAT = "rul.fixed-rbf.v1"
PREDICTOR_FORMAT = "contest.cpu-rul-fast.v1"
KEY = "regression_fast_component"


def validate(state, profile, dimensions):
    if (not isinstance(state, dict) or state.get("format") != FORMAT
            or state.get("profile") != profile or profile not in ("B", "C")
            or type(state.get("width")) is not int or state["width"] not in (8, 16)
            or state.get("space") not in ("linear", "log")
            or profile == "C" and state["space"] != "linear"
            or type(state.get("updates")) is not int or not 0 <= state["updates"] <= 1000000000):
        raise ValueError("Invalid fixed RBF component identity")
    for key in ("cap", "bandwidth"):
        value = state.get(key)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1000000
        except OverflowError:
            valid = False
        if not valid or key == "bandwidth" and value == 0:
            raise ValueError("Invalid fixed RBF component scale")
    try:
        centers, weights = (np.asarray(state[key], dtype=np.float64) for key in ("centers", "weights"))
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        raise ValueError("Invalid fixed RBF component arrays") from error
    k = len(centers) if centers.ndim == 2 else 0
    count = centers.size + weights.size + 2
    if (not 1 <= k <= state["width"] or centers.shape != (k, dimensions)
            or weights.shape != (k + 1,) or not np.isfinite(centers).all()
            or not np.isfinite(weights).all() or np.any(np.abs(centers) > 10)
            or type(state.get("parameter_count")) is not int or state["parameter_count"] != count):
        raise ValueError("Invalid fixed RBF component shape or count")
    return centers, weights


def design(x, centers, bandwidth):
    if len(x) == 1:
        distance = np.sum((centers - x[0]) ** 2, axis=1)[None, :] / x.shape[1]
    else:
        distance = np.maximum(0., np.sum(x * x, axis=1)[:, None]
                              + np.sum(centers * centers, axis=1) - 2 * x @ centers.T) / x.shape[1]
    return np.column_stack((np.ones(len(x)), np.exp(-distance / bandwidth)))


def initialize(model, train, spec):
    x = model.normalized(train["x"])
    sample = x[np.linspace(0, len(x) - 1, min(len(x), 512), dtype=int)]
    centers = sample[[int(np.argmin(np.sum((sample - sample.mean(0)) ** 2, axis=1)))]]
    for _ in range(spec["basis_width"] - 1):
        distance = np.min(np.sum((sample[:, None] - centers) ** 2, axis=2), axis=1)
        if distance.max() < 1e-10:
            break
        centers = np.vstack((centers, sample[int(distance.argmax())]))
    distance = np.sum((centers[:, None] - centers) ** 2, axis=2) / x.shape[1]
    nonzero = distance[distance > 1e-10]
    bandwidth = max(.25, float(np.median(nonzero)) if len(nonzero) else 1.)
    prediction = model.predict(train["x"])
    error = ((train["y"] - prediction) / model.meta["target_scale"] if spec["residual_space"] == "linear"
             else np.log1p(train["y"]) - np.log1p(prediction))
    return {"format": FORMAT, "profile": model.meta["profile"], "width": spec["basis_width"],
            "space": spec["residual_space"], "updates": 0, "initial_train_sha256": train["sha256"],
            "cap": float(.25 * np.quantile(np.abs(error), .9)), "bandwidth": bandwidth,
            "centers": centers.tolist(), "weights": np.zeros(len(centers) + 1).tolist(),
            "parameter_count": centers.size + len(centers) + 3}


def update(model, train, state, sample_weights, deadline):
    import time
    centers, old_weights = validate(state, model.meta["profile"], len(model.meta["paths"]))
    k = len(old_weights)
    gram, rhs = np.zeros((k, k)), np.zeros(k)
    for start in range(0, len(train["y"]), 2048):
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed RBF fitting budget exhausted")
        batch = train["x"][start:start + 2048]
        y, weights = train["y"][start:start + 2048], sample_weights[start:start + 2048]
        prediction = model.predict(batch)
        error = ((y - prediction) / model.meta["target_scale"] if state["space"] == "linear"
                 else np.log1p(y) - np.log1p(prediction))
        features = design(model.normalized(batch), centers, state["bandwidth"])
        gram += features.T @ (weights[:, None] * features)
        rhs += features.T @ (weights * error)
    coefficients = old_weights + .35 * np.linalg.solve(gram + .02 * len(train["y"]) * np.eye(k), rhs)
    result = deepcopy(state)
    result.update(weights=coefficients.tolist(), updates=state["updates"] + 1,
                  last_train_sha256=train["sha256"], solver="weighted_ridge_closed_form",
                  ridge_per_train_row=.02, update_shrink=.35)
    return result
