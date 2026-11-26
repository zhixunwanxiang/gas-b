"""Shrunk train-only residuals in the parent model's normalized target space."""
import numpy as np
from .reference_model import Model


def assign(x, centers):
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def centers_from_train(settings, seed):
    rng = np.random.default_rng(seed)
    sample = settings[rng.choice(len(settings), min(4096, len(settings)), replace=False)]
    centers = sample[:1].copy()
    for _ in range(6):
        distances = ((sample[:, None, :] - centers) ** 2).sum(axis=2).min(axis=1)
        if distances.max() < 1e-8:
            break
        centers = np.vstack((centers, sample[distances.argmax()]))
    for _ in range(8):
        groups = assign(sample, centers)
        centers = np.stack([sample[groups == g].mean(axis=0) for g in np.unique(groups)])
    return centers


def basis(x, mean, scale):
    z = np.clip((x - mean) / scale, -5., 5.)
    return np.column_stack((np.ones(len(z)), z, z * z))


class EngineModel(Model):
    supports_engine_residual = True
    def validate(self):
        super().validate()
        self.residual = None
        state = self.meta.get("rul_residual_v1")
        if state is None:
            return
        centers, means, scales, weights = [np.asarray(state[k], dtype=float)
                                           for k in ("centers", "means", "scales", "weights")]
        count = len(centers)
        if (not 1 <= count <= 7 or centers.shape != (count, 3)
                or means.shape != (count + 1, 21) or scales.shape != means.shape
                or weights.shape != (count + 1, 43) or (scales <= 0).any()
                or not all(np.isfinite(a).all() for a in (centers, means, scales, weights))):
            raise ValueError("Invalid engine residual checkpoint")
        self.residual = centers, means, scales, weights

    def forward(self, x):
        hidden, output = super().forward(x)
        if self.residual is not None:
            centers, means, scales, weights = self.residual
            group = assign(x[:, 21:], centers) + 1
            global_basis = basis(x[:, :21], means[0], scales[0])
            local_basis = basis(x[:, :21], means[group], scales[group])
            output[:, 0] += global_basis @ weights[0] + np.einsum("ij,ij->i", local_basis, weights[group])
        return hidden, output

    def update_residual(self, train, mode):
        if mode not in ("global", "condition"):
            raise ValueError("Unknown residual mode")
        x = self.normalized(train["x"])
        sensors = x[:, :21]
        if self.residual is None:
            centers = centers_from_train(x[:, 21:], self.meta["seed"])
            groups = assign(x[:, 21:], centers) + 1
            means = np.tile(sensors.mean(axis=0), (len(centers) + 1, 1))
            scales = np.tile(np.maximum(sensors.std(axis=0), .15), (len(centers) + 1, 1))
            weights = np.zeros((len(means), 43))
            for group in np.unique(groups):
                rows = sensors[groups == group]
                if len(rows) >= 64:
                    means[group] = rows.mean(axis=0)
                    scales[group] = np.maximum(rows.std(axis=0), .15)
        else:
            centers, means, scales, previous = self.residual
            groups = assign(x[:, 21:], centers) + 1
            weights = previous.copy()
        _, prediction = self.forward(x)
        error = (train["y"] - self.meta["target_mean"]) / self.meta["target_scale"] - prediction[:, 0]
        def ridge(design, residual):
            return np.linalg.solve(design.T @ design + .02 * len(design) * np.eye(43), design.T @ residual)
        design = basis(sensors, means[0], scales[0])
        global_update = ridge(design, error)
        weights[0] += .5 * global_update
        if mode == "condition":
            remaining = error - design @ global_update
            for group in np.unique(groups):
                mask = groups == group
                if mask.sum() >= 64:
                    local = basis(sensors[mask], means[group], scales[group])
                    weights[group] += .5 * ridge(local, remaining[mask])
        self.meta["rul_residual_v1"] = {k: a.tolist() for k, a in zip(
            ("centers", "means", "scales", "weights"), (centers, means, scales, weights))}
        self.validate()
