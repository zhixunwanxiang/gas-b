"""Stateful four-sensor route from the natural-order macro-F1 experiment."""
from collections import deque
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .model import Model, checkpoint_digest
from .signals import Flatline, correct, home_probabilities


PAIRS = tuple(combinations(range(6), 2))


def confusion(y, predicted):
    return np.bincount(6 * y + predicted, minlength=36).reshape(6, 6)


def macro(matrix):
    return float(np.mean(2 * matrix.diagonal() / np.maximum(1, matrix.sum(0) + matrix.sum(1))))


class HomeExpert:
    def __init__(self, trunk, router):
        self.trunk = trunk
        self.router = router

    def inputs(self, x):
        z = self.trunk.normalized(x)
        heads = self.router.meta["private_joint_rescue"]
        odds = []
        for cls in ("0", "1"):
            head = heads[cls]
            odds.append(z[:, head["features"]] @ np.asarray(head["w"]) + head["b"])
        return self.trunk.predict(x), self.router.predict(x), odds


def pair_codes(left, right):
    return 6 * np.minimum(left, right) + np.maximum(left, right)


def weight_table(counts):
    table = np.full(36, .75)
    for i, (a, b) in enumerate(PAIRS):
        home, sensor = counts[:, i, :].T
        total = home.sum() + sensor.sum()
        if total < 20:
            continue
        rate = sensor.sum() / total
        if max(rate, 1 - rate) < .6 - 1e-12:
            continue
        winning = sensor > home if rate > .5 else home > sensor
        if np.count_nonzero(winning) >= 2:
            table[6*a+b] = .9 if rate > .5 else .4
    return table


class FourSensorRouter:
    """For each chronological batch call predict_batch, then release_labels after commitment."""
    def __init__(self, model_dir, config):
        models = {}
        for name, signature in config["model_sha256"].items():
            path = Path(model_dir) / (name + ".npz")
            if checkpoint_digest(path) != signature:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
            models[name] = Model.load(path)
        self.parent = models["start"]
        self.reference = models["reference"]
        self.home = HomeExpert(models["home_trunk"], models["home_router"])
        self.four = models["four_sensor"]
        self.five = models["five_sensor"]
        self.cutoffs = config["cutoffs"]
        self.state = (deque(maxlen=128), np.zeros(128), 0)
        self.detector = Flatline(self.reference.scale[96:104])
        self.history = deque(maxlen=3)
        self.weights = np.full(36, .75)
        self.active_pairs = set()
        self.batch_count = 0
        self.waiting = None

    def predict_batch(self, x):
        if self.waiting is not None:
            raise ValueError("The previous batch is still awaiting feedback")
        if self.batch_count == 0:
            predicted = self.parent.predict(x).argmax(1)
            self.waiting = (len(x), None)
            return predicted
        alarm = self.detector.predict(x[:, 96:104])
        adjusted, self.state = correct(x, self.reference, self.state)
        home = home_probabilities(self.home, adjusted, self.cutoffs)
        five = (self.five.predict(x) + self.five.predict(adjusted)) * .5
        four = (self.four.predict(x) + self.four.predict(adjusted)) * .5
        h, s = home.argmax(1), five.argmax(1)
        inner_codes = pair_codes(h, s)
        weights = self.weights[inner_codes, None]
        challenger = home * (1 - weights) + five * weights
        challenger[alarm] = four[alarm]
        base, other = four.argmax(1), challenger.argmax(1)
        outer_codes = pair_codes(base, other)
        alternative = (.4 * four + .6 * challenger).argmax(1)
        self.waiting = (len(x), (h, s, inner_codes, base, other, outer_codes, alternative))
        return np.where(~alarm & np.isin(outer_codes, list(self.active_pairs)), alternative, base)

    def release_labels(self, y):
        y = np.asarray(y)
        if self.waiting is None:
            raise ValueError("No batch is awaiting feedback")
        count, feedback = self.waiting
        if y.shape != (count,) or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= 6)):
            raise ValueError("Expected one six-class label per committed prediction")
        if feedback is not None:
            h, s, inner, base, other, outer, alternative = feedback
            counts, decisive, deltas = [], [], []
            matrix = confusion(y, base)
            for a, b in PAIRS:
                code = 6*a+b
                mask = inner == code
                counts.append([np.sum(mask & (h == y)), np.sum(mask & (s == y))])
                mask = outer == code
                decisive.append(np.sum(mask & (base == y)) + np.sum(mask & (other == y)))
                deltas.append(confusion(y[mask], alternative[mask]) - confusion(y[mask], base[mask]))
            self.history.append((counts, np.asarray(decisive), deltas, matrix))
            if len(self.history) == self.history.maxlen:
                self.weights = weight_table(np.array([r[0] for r in self.history]))
                self.active_pairs = set()
                pooled = sum(r[3] for r in self.history)
                for i, (a, b) in enumerate(PAIRS):
                    if sum(r[1][i] for r in self.history) < 20:
                        continue
                    wins = sum(macro(r[3] + r[2][i]) > macro(r[3]) + 1e-12 for r in self.history)
                    if wins >= 2 and macro(pooled + sum(r[2][i] for r in self.history)) >= macro(pooled) + .0005:
                        self.active_pairs.add(6*a+b)
        self.waiting = None
        self.batch_count += 1
