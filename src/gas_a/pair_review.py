"""Frozen A second opinion built from log and relative sensor responses."""
from dataclasses import dataclass
import time
from zipfile import BadZipFile

import numpy as np

FORMAT = "gas-a.pair-review.v1"
FILENAME = "routing/pair_review.npz"
VIEW_SIZES = (128, 224, 352)
MAX_REFERENCES = 512


def sensor_views(x):
    logs = (np.sign(x) * np.log1p(np.abs(x))).reshape(-1, 16, 8)
    relative = logs - np.median(logs, axis=1, keepdims=True)
    response = logs[:, :, 2:] - logs[:, :, :1]
    absolute = logs.reshape(len(x), 128)
    shape = np.column_stack([relative.reshape(len(x), 128), response.reshape(len(x), 96)])
    return absolute, shape, np.column_stack([absolute, shape])


def kernel(x, reference, squared_norm, gamma):
    distance = np.sum(x ** 2, axis=1, keepdims=True) + squared_norm - 2 * x @ reference.T
    return np.exp(-gamma * np.maximum(0., distance))


def softmax(values):
    numerator = np.exp(values - values.max(axis=1, keepdims=True))
    return numerator / numerator.sum(axis=1, keepdims=True)


@dataclass
class View:
    center: np.ndarray
    scale: np.ndarray
    reference: np.ndarray
    coefficients: np.ndarray
    gamma: float

    def __post_init__(self):
        self.squared_norm = np.sum(self.reference ** 2, axis=1)

    def scores(self, x):
        # Chunk validation predictions so matrix memory is bounded even for a large projection.
        outputs = [kernel((chunk - self.center) / self.scale, self.reference, self.squared_norm, self.gamma)
                   @ self.coefficients for chunk in np.array_split(x, max(1, (len(x) + 255) // 256))]
        logits = 6. * np.concatenate(outputs)
        return softmax(logits[:, :6]), softmax(logits[:, 6:])


@dataclass
class PairReview:
    views: tuple[View, ...]

    @classmethod
    def fit(cls, train, deadline):
        x, y = train["x"], train["y"]
        if set(y) != set(range(6)):
            raise ValueError("Pair review needs all six gases in the authorized training projection")
        if len(y) > MAX_REFERENCES:
            rng = np.random.default_rng(42)
            by_class = [np.flatnonzero(y == label) for label in range(6)]
            rows = np.sort(np.concatenate([rng.choice(indices, min(len(indices), MAX_REFERENCES // 6), replace=False)
                                          for indices in by_class]))
            x, y = x[rows], y[rows]
        weights = len(y) / (6. * np.bincount(y, minlength=6)[y])
        pair = y < 2
        views = []
        for values in sensor_views(x):
            if time.monotonic() >= deadline:
                raise TimeoutError("Pair review exceeded the candidate budget")
            center = np.median(values, axis=0)
            low, high = np.quantile(values, [.10, .90], axis=0)
            scale = np.where(high - low > 1e-12, high - low, 1.)
            reference = (values - center) / scale
            gamma = .5 / max(reference.shape[1] * reference.var(), 1e-12)
            gram = kernel(reference, reference, np.sum(reference ** 2, axis=1), gamma)
            coefficients = np.zeros((len(y), 8))
            coefficients[:, :6] = np.linalg.solve(gram + np.diag(.1 / weights), np.eye(6)[y])
            pair_gram = gram[np.ix_(pair, pair)]
            coefficients[pair, 6:] = np.linalg.solve(pair_gram + .1 * np.eye(pair.sum()), np.eye(2)[y[pair]])
            views.append(View(center, scale, reference, coefficients, gamma))
        if time.monotonic() >= deadline:
            raise TimeoutError("Pair review exceeded the candidate budget")
        return cls(tuple(views))

    def evidence(self, x):
        view_scores = [view.scores(values) for view, values in zip(self.views, sensor_views(x))]
        multiclass = np.stack([all_classes for all_classes, _ in view_scores])
        binary = np.stack([eth_pair for _, eth_pair in view_scores])
        average = multiclass.mean(axis=0)
        target = average.argmax(axis=1)
        pair_average = binary.mean(axis=0)
        pair_target = pair_average.argmax(axis=1)
        votes = (multiclass.argmax(axis=2) == target).sum(axis=0)
        pair_votes = (binary.argmax(axis=2) == pair_target).sum(axis=0)
        ethylene_evidence = (target == pair_target) & (pair_average.max(axis=1) >= .70) & (pair_votes >= 2) & (votes >= 2)
        toluene_evidence = (multiclass[1:].argmax(axis=2) == target).all(axis=0) & (average.max(axis=1) >= .60)
        return target, ethylene_evidence, toluene_evidence

    @staticmethod
    def apply(base, evidence):
        target, ethylene_evidence, toluene_evidence = evidence
        initial = base.argmax(axis=1)
        eligible = ((initial < 2) & ethylene_evidence) | (((initial == 5) | (target == 5)) & toluene_evidence)
        rows = np.flatnonzero(eligible & (initial != target))
        revised = base.copy()
        # Keep the original score spectrum while switching the top two involved classes.
        revised[rows, target[rows]] = base[rows, initial[rows]]
        revised[rows, initial[rows]] = base[rows, target[rows]]
        return revised

    def predict(self, x, base):
        return self.apply(base, self.evidence(x))

    @property
    def parameter_count(self):
        return sum(view.center.size + view.scale.size + view.reference.size + view.coefficients.size + 1
                   for view in self.views)

    def save(self, path):
        arrays = {}
        for index, view in enumerate(self.views):
            for name in ("center", "scale", "reference", "coefficients", "gamma"):
                arrays[f"{name}_{index}"] = getattr(view, name)
        with path.open("xb") as output:
            np.savez_compressed(output, **arrays)

    @classmethod
    def load(cls, path):
        views = []
        try:
            with path.open("rb") as source, np.load(source, allow_pickle=False) as payload:
                for index, size in enumerate(VIEW_SIZES):
                    view = [payload[f"{key}_{index}"] for key in ("center", "scale", "reference", "coefficients", "gamma")]
                    center, scale, reference, coefficients, gamma = view
                    rows = len(reference)
                    if (not 1 <= rows <= MAX_REFERENCES or center.shape != (size,) or scale.shape != (size,)
                            or reference.shape != (rows, size) or coefficients.shape != (rows, 8) or gamma.shape != ()
                            or any(a.dtype.kind != "f" or not np.isfinite(a).all() for a in view)
                            or np.any(scale <= 0.) or gamma <= 0.
                            or (views and rows != len(views[0].reference))):
                        raise ValueError("Invalid pair review tensor shapes or values")
                    views.append(View(center, scale, reference, coefficients, float(gamma)))
        except (BadZipFile, KeyError, OSError, TypeError) as error:
            raise ValueError("Invalid pair review checkpoint") from error
        return cls(tuple(views))
