"""Feature-only DAT reads, delayed labels, causal drift correction, and expert fusion."""
from collections import deque
from pathlib import Path
import numpy as np
from .model import checkpoint_digest as digest


def dat_inputs(path, paths):
    """仅读取特征字段，有意跳过标签字段的解析。"""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Expected a regular public DAT file")
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        tokens = line.split()[1:]
        values = np.empty(128, dtype=np.float64)
        indices = set()
        for token in tokens:
            index, value = token.split(":", 1)
            index = int(index) - 1
            if index in indices or not 0 <= index < 128:
                raise ValueError("DAT feature index is duplicated or out of range")
            indices.add(index)
            values[index] = float(value)
        if len(indices) != 128 or not np.isfinite(values).all():
            raise ValueError("DAT requires 128 finite numeric features")
        rows.append(values)
    if not rows:
        raise ValueError("Empty DAT window")
    return {"x": np.asarray(rows), "paths": paths, "sha256": digest(path),
            "ids": [f"{path.stem}-{i + 1:05d}" for i in range(len(rows))]}


def reveal_labels(path, features):
    """确认 DAT 文件与预测提交时的摘要一致后再解析标签，防止评分数据被替换。"""
    if digest(path) != features["sha256"]:
        raise ValueError("Window changed after prediction commitment")
    labels = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            label = int(line.split()[0]) - 1
            if not 0 <= label < 6:
                raise ValueError("DAT label outside the six reference classes")
            labels.append(label)
    if len(labels) != len(features["x"]):
        raise ValueError("Window label count changed")
    return {**features, "y": np.asarray(labels, dtype=np.int64)}


class Flatline:
    """Causal detector, using the span of the most recent readings in training SD units."""
    def __init__(self, scale):
        self.scale = scale
        self.window = deque(maxlen=8)

    def predict(self, readings):
        out = np.zeros(len(readings), dtype=bool)
        for i, row in enumerate(readings):
            self.window.append(row / self.scale)
            if len(self.window)==self.window.maxlen:
                out[i] = np.ptp(self.window,axis=0).max() <= .001
        return out


def correct(x, reference, state):
    buf, shift, count = state
    corrected = np.empty_like(x, dtype=float)
    for i, z in enumerate((x - reference.mean) / reference.scale):
        corrected[i] = reference.mean + reference.scale * (z - .75 * shift)
        buf.append(z)
        count += 1
        if len(buf) == 128 and count % 64 == 0:
            shift = .5 * shift + .5 * np.median(buf, axis=0)
    return corrected, (buf, shift, count)


def home_probabilities(bundle, adjusted, cutoffs):
    left, right, logits = bundle.inputs(adjusted)
    probabilities = (left + right) / 2
    initial = probabilities.argmax(1)
    # Preserve the current classifier's top class and confidence spectrum on rescued rows.
    for cls, odds in enumerate(logits):
        cut = cutoffs[str(cls)]["cut"]
        rows = np.flatnonzero((initial == cls) & (odds >= np.log(cut / (1-cut))))
        probabilities[rows, cls], probabilities[rows, 3] = (
            probabilities[rows, 3].copy(), probabilities[rows, cls].copy())
    return probabilities
