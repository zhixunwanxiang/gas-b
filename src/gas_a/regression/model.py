"""Complete regression predictor: inherited MLP plus an optional fixed-size C head."""
from copy import deepcopy
import hashlib
import json
import math
import numpy as np
from ..sensor_training.reference.model import Model as ReferenceModel, FORMAT as NEURAL_FORMAT
from .grouping import fit_groups

COMPONENT_FORMAT = "rul.condition-ridge.v1"
PREDICTOR_FORMAT = "contest.cpu-rul-components.v1"


def component_digest(component):
    return "sha256:" + hashlib.sha256(json.dumps(component, sort_keys=True, allow_nan=False).encode()).hexdigest()


def basis(x, mean, scale):
    z = np.clip((x - mean) / scale, -5., 5.)
    return np.column_stack((np.ones(len(z)), z, z * z))


class Model(ReferenceModel):
    def validate(self):
        head = self.meta.get("regression_components")
        complete = self.meta.get("format") == PREDICTOR_FORMAT
        if complete:
            if head is None or self.meta.get("neural_format") != NEURAL_FORMAT:
                raise ValueError("Incomplete regression predictor components")
            self.meta["format"] = NEURAL_FORMAT
            try:
                super().validate()
            finally:
                self.meta["format"] = PREDICTOR_FORMAT
        else:
            super().validate()
            if head is not None:
                raise ValueError("Regression components require the explicit complete predictor format")
        self.component = None
        if self.meta.get("rul_residual_v1") is not None:
            raise ValueError("Unsupported regression component; a complete compatible loader is required")
        if head is None:
            if self.meta.get("prediction_parameter_count", self.meta["parameter_count"]) != self.meta["parameter_count"]:
                raise ValueError("Incomplete regression predictor components")
            return
        try:
            valid_cap = isinstance(head, dict) and type(head.get("cap")) in (int, float) and math.isfinite(head["cap"]) and 0 <= head["cap"] <= 1000000
        except OverflowError:
            valid_cap = False
        if (not isinstance(head, dict) or head.get("format") != COMPONENT_FORMAT
                or self.meta.get("profile") != "C" or len(self.meta["paths"]) != 24
                or type(head.get("updates")) is not int or not 0 <= head["updates"] <= 1000000000
                or not valid_cap):
            raise ValueError("Unsupported or invalid regression component format")
        try:
            arrays = {key: np.asarray(head[key], dtype=np.float64)
                      for key in ("setting_mean", "setting_scale", "centers", "means", "scales", "weights")}
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid regression component tensors") from error
        k = len(arrays["centers"]) if arrays["centers"].ndim else 0
        if (not 1 <= k <= 7 or arrays["centers"].shape != (k, 3)
                or arrays["setting_mean"].shape != (3,) or arrays["setting_scale"].shape != (3,)
                or arrays["means"].shape != (k + 1, 21) or arrays["scales"].shape != (k + 1, 21)
                or arrays["weights"].shape != (k + 1, 43)
                or any(not np.isfinite(v).all() for v in arrays.values())
                or np.any(arrays["setting_scale"] <= 0) or np.any(arrays["scales"] <= 0)
                or type(head.get("parameter_count")) is not int
                or head["parameter_count"] != arrays["weights"].size
                or self.meta.get("prediction_parameter_count") != self.meta["parameter_count"] + arrays["weights"].size):
            raise ValueError("Invalid regression component shape, values or parameter count")
        self.component = arrays
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            q = arrays["weights"][:, 22:] / arrays["scales"] ** 2
            b = arrays["weights"][:, 1:22] / arrays["scales"] - 2 * arrays["means"] * q
            a = (arrays["weights"][:, 0] - np.sum(arrays["means"] * arrays["weights"][:, 1:22] / arrays["scales"], axis=1)
                 + np.sum(arrays["means"] ** 2 * q, axis=1))
            self.compiled = {"quadratic": q + q[0], "linear": b + b[0], "intercept": a + a[0],
                "low": np.maximum(arrays["means"] - 5 * arrays["scales"], arrays["means"][0] - 5 * arrays["scales"][0]),
                "high": np.minimum(arrays["means"] + 5 * arrays["scales"], arrays["means"][0] + 5 * arrays["scales"][0])}
        if not all(np.isfinite(v).all() for v in self.compiled.values()):
            raise ValueError("Nonfinite compiled regression component")

    def component_groups(self, normalized):
        c = self.component
        setting = np.clip((normalized[:, 21:] - c["setting_mean"]) / c["setting_scale"], -10., 10.)
        return ((setting[:, None] - c["centers"]) ** 2).sum(2).argmin(1) + 1

    def forward(self, x):
        hidden, output = super().forward(x)
        c = self.component
        if c is not None:
            groups = self.component_groups(x)
            sensors, fast = x[:, :21], self.compiled
            if len(x) == 1:
                group, sensor = int(groups[0]), sensors[0]
                if np.all((sensor >= fast["low"][group]) & (sensor <= fast["high"][group])):
                    correction = float(fast["intercept"][group] + sensor @ (fast["linear"][group] + sensor * fast["quadratic"][group]))
                    if not math.isfinite(correction):
                        raise ValueError("Nonfinite regression component correction")
                    cap = self.meta["regression_components"]["cap"]
                    output[0, 0] += max(-cap, min(cap, correction))
                    return hidden, output
            in_range = ((sensors >= fast["low"][groups]) & (sensors <= fast["high"][groups])).all(axis=1)
            correction = fast["intercept"][groups] + np.sum(sensors * (fast["linear"][groups] + sensors * fast["quadratic"][groups]), axis=1)
            if not in_range.all():
                selected = ~in_range
                g = groups[selected]
                global_z = np.clip((sensors[selected] - c["means"][0]) / c["scales"][0], -5., 5.)
                local_z = np.clip((sensors[selected] - c["means"][g]) / c["scales"][g], -5., 5.)
                w = c["weights"]
                correction[selected] = (w[0, 0] + np.sum(global_z * w[0, 1:22] + global_z ** 2 * w[0, 22:], axis=1)
                    + w[g, 0] + np.sum(local_z * w[g, 1:22] + local_z ** 2 * w[g, 22:], axis=1))
            if not np.isfinite(correction).all():
                raise ValueError("Nonfinite regression component correction")
            output[:, 0] += np.clip(correction, -self.meta["regression_components"]["cap"], self.meta["regression_components"]["cap"])
        return hidden, output

    def initialize_component(self, train):
        if self.component is not None:
            return False
        if self.meta.get("profile") != "C":
            raise ValueError("Condition ridge requires the C profile")
        normalized = self.normalized(train["x"])
        # Grouping statistics belong to the new component; old MLP scalers stay unchanged.
        grouping = fit_groups({"x": normalized, "y": train["y"], "sha256": train["sha256"]}, "C")
        settings = np.clip((normalized[:, 21:] - np.asarray(grouping["mean"])) / np.asarray(grouping["scale"]), -10., 10.)
        centers = np.asarray(grouping["centers"])
        groups = ((settings[:, None] - centers) ** 2).sum(2).argmin(1) + 1
        sensors = normalized[:, :21]
        means = np.tile(sensors.mean(0), (len(centers) + 1, 1))
        scales = np.tile(np.maximum(sensors.std(0), .15), (len(centers) + 1, 1))
        for group in np.unique(groups):
            mask = groups == group
            if mask.sum() >= 64:
                means[group], scales[group] = sensors[mask].mean(0), np.maximum(sensors[mask].std(0), .15)
        cap = float(np.quantile(np.abs(self.predict(train["x"]) - train["y"]), .9) * .25 / self.meta["target_scale"])
        self.meta["regression_components"] = {"format": COMPONENT_FORMAT, "updates": 0, "cap": cap,
            "initial_train_sha256": train["sha256"], "parameter_count": (len(centers) + 1) * 43,
            "setting_mean": grouping["mean"], "setting_scale": grouping["scale"], "centers": centers.tolist(),
            "means": means.tolist(), "scales": scales.tolist(), "weights": np.zeros((len(centers) + 1, 43)).tolist()}
        self.meta["prediction_parameter_count"] = self.meta["parameter_count"] + (len(centers) + 1) * 43
        self.meta["neural_format"] = NEURAL_FORMAT
        self.meta["format"] = PREDICTOR_FORMAT
        self.validate()
        return True

    def update_component(self, train):
        if self.component is None:
            raise ValueError("Initialize a zero component before fitting")
        x = self.normalized(train["x"])
        groups = self.component_groups(x)
        c = self.component
        error = (train["y"] - self.predict(train["x"])) / self.meta["target_scale"]
        def ridge(design, residual):
            return np.linalg.solve(design.T @ design + .02 * len(design) * np.eye(43), design.T @ residual)
        design = basis(x[:, :21], c["means"][0], c["scales"][0])
        global_update = ridge(design, error)
        weights = c["weights"].copy()
        weights[0] += .5 * global_update
        remaining = error - design @ global_update
        for group in np.unique(groups):
            mask = groups == group
            if mask.sum() >= 64:
                weights[group] += .5 * ridge(basis(x[mask, :21], c["means"][group], c["scales"][group]), remaining[mask])
        head = deepcopy(self.meta["regression_components"])
        head.update(weights=weights.tolist(), updates=head["updates"] + 1, last_train_sha256=train["sha256"],
                    solver="ridge_closed_form", ridge_per_train_row=.02, update_shrink=.5)
        self.meta["regression_components"] = head
        self.validate()
