"""Bounded, NumPy-only inference shared by desktop and native regression predictors."""
import numpy as np


def features(x, state):
    if str(state["task"]) == "B":
        h, v = np.split(x, 2, axis=1)
        return np.column_stack((x, (h-v)/(np.abs(h)+np.abs(v)+1e-9)))
    if str(state["method"]) == "condition_tree":
        setting = (x[:, 21:] - state["setting_mean"]) / state["setting_scale"]
        g = ((setting[:, None] - state["centers"])**2).sum(2).argmin(1)
        centered = (x[:, :21]-state["means"][g])/state["scales"][g]
        return np.column_stack((x, np.clip(centered, -12, 12)))
    return x


class Forest:
    def __init__(self, state, width):
        self.method = str(state["method"])
        self.intercept = float(state["intercept"])
        self.value, self.threshold = (np.asarray(state[k]) for k in ("value", "threshold"))
        self.feature, self.left, self.right, self.starts = (
            np.asarray(state[k]) for k in ("feature", "left", "right", "starts"))
        n = self.value.size
        roots = self.starts
        if (not 1 <= n <= 250000 or self.value.shape != (n,)
                or not all(a.shape == (n,) for a in (self.left,self.right,self.threshold,self.feature))
                or any(a.dtype.kind not in "iu" for a in (self.left,self.right,self.feature,roots))
                or roots.ndim != 1 or not 1 <= len(roots) <= 128 or roots[0] != 0
                or np.any(roots >= n) or np.any(roots < 0) or np.any(roots[1:] <= roots[:-1])
                or not np.isfinite(self.intercept) or not np.isfinite(self.value).all()
                or not np.isfinite(self.threshold).all()):
            raise ValueError("Invalid forest dimensions, roots or values")
        active = self.left != -1
        parents = np.flatnonzero(active)
        if (np.any((self.right != -1) != active) or np.any(self.feature[active] < 0)
                or np.any(self.feature[active] >= width)):
            raise ValueError("Invalid split or unpaired children")
        ends = np.repeat(np.r_[roots[1:], n], np.diff(np.r_[roots, n]))
        children = np.concatenate((self.left[active], self.right[active])).astype(np.int64)
        if (np.any(children <= np.tile(parents, 2))
                or np.any(children >= np.tile(ends[active], 2))):
            raise ValueError("Non-forward or cross-tree child index")
        counts = np.bincount(children, minlength=n)
        counts[roots] += 1
        if np.any(counts != 1):
            raise ValueError("Invalid forest ownership or disconnected node")
        depths = np.zeros(n, dtype=np.int32)
        for j in parents:
            depths[self.left[j]] = depths[self.right[j]] = depths[j] + 1
        self.depth = int(depths.max())
        if self.depth > 96:
            raise ValueError("Tree exceeds depth budget")
        # Leaf slots may contain arbitrary sentinel features. Keep all vectorized reads in bounds.
        self.feature = np.where(active, self.feature, 0)
        self.single_nodes = list(zip(self.feature.tolist(), self.threshold.tolist(),
                                     self.left.tolist(), self.right.tolist()))
        self.single_roots = self.starts.tolist()

    def predict(self, x):
        if self.method == "extra_trees":
            x = x.astype(np.float32).astype(np.float64)
        out = np.empty(len(x))
        if len(x) == 1:
            row = x[0].tolist()
            leaves = []
            for node in self.single_roots:
                feature, cut, left, right = self.single_nodes[node]
                while left >= 0:
                    node = left if row[feature] <= cut else right
                    feature, cut, left, right = self.single_nodes[node]
                leaves.append(node)
            out[0] = self.intercept + self.value[leaves].sum()
            return out
        # Batched developer evaluation: walking one tree at a time bounds temporary memory.
        out.fill(self.intercept)
        indices = np.arange(len(x))
        for start in self.starts:
            node = np.full(len(x), start, dtype=np.int64)
            for _ in range(self.depth):
                rows = indices[self.left[node] >= 0]
                if not len(rows):
                    break
                at = node[rows]
                node[rows] = np.where(x[rows,self.feature[at]] <= self.threshold[at], self.left[at], self.right[at])
            out += self.value[node]
        return out


class Layer:
    def __init__(self, state, dataset):
        if str(state["task"]) != dataset:
            raise ValueError("Tree layer dataset mismatch")
        self.dataset = dataset
        self.state = state
        if dataset == "B":
            if str(state["method"]) != "bearing_mix":
                raise ValueError("Unsupported bearing layer")
            self.weights = np.asarray(state["weights"])
            if (self.weights.shape != (2,) or not np.isfinite(self.weights).all()
                    or min(self.weights) < 0 or sum(self.weights) > 1):
                raise ValueError("Invalid bearing weights")
            self.trees = []
            for prefix in ("arithmetic_", "log_"):
                child = {k[len(prefix):]: v for k,v in state.items() if k.startswith(prefix)}
                if str(child["method"]) != "extra_trees":
                    raise ValueError("Invalid bearing component")
                self.trees.append(Forest(child, 51))
        else:
            if dataset != "C" or str(state["method"]) != "condition_tree":
                raise ValueError("Unsupported engine layer")
            centers = state["centers"]
            if centers.ndim != 2 or not 1 <= len(centers) <= 7 or centers.shape[1] != 3:
                raise ValueError("Invalid engine centroids")
            for key, shape in (("means",(len(centers),21)), ("scales",(len(centers),21)),
                               ("setting_mean",(3,)), ("setting_scale",(3,))):
                if state[key].shape != shape or not np.isfinite(state[key]).all():
                    raise ValueError("Invalid engine normalizer")
            self.strength = float(state["strength"])
            self.cap = float(state["positive_cap"])
            self.positive_scale = float(state.get("positive_scale", 1.))
            if not 0 < self.positive_scale <= 1:
                raise ValueError("Invalid positive residual scale")
            if (not np.isfinite(centers).all() or np.any(state["scales"] <= 0)
                    or np.any(state["setting_scale"] <= 0) or not 0 < self.strength <= 1
                    or not np.isfinite(self.cap) or self.cap < 0):
                raise ValueError("Invalid engine correction parameters")
            self.tree = Forest(state, 45)

    def correction(self, x):
        delta = self.strength * self.tree.predict(features(x, self.state))
        if self.positive_scale != 1:
            delta = np.where(delta > 0, self.positive_scale * delta, delta)
        return np.minimum(self.cap, delta)

    def apply(self, x, previous):
        if self.dataset == "B":
            data = features(x, self.state)
            a,b = self.weights
            return np.maximum(0, (1-a-b)*previous + a*self.trees[0].predict(data)
                              + b*np.expm1(self.trees[1].predict(data)))
        return np.maximum(0, previous + self.correction(x))

