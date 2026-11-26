"""Bounded CPU tree fits on the current authorized training projection."""
import io
import time
import numpy as np
from .engine import centers_from_train, assign
from .forests import features, Forest


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Tree candidate exceeded the numerical training budget")


def fit_forest(x, y, rng, deadline, *, boosted, count=12):
    # Refit shallow trees using train rows only. Each step checks the same candidate deadline.
    intercept = float(y.mean()) if boosted else 0.
    residual = y - intercept if boosted else y
    starts, feature, threshold, left, right, value = [], [], [], [], [], []
    for _ in range(count):
        check_time(deadline)
        start = len(value)
        starts.append(start)
        ids = rng.choice(len(y), min(3072, len(y)), replace=False)
        def grow(rows, depth):
            check_time(deadline)
            node = len(value)
            feature.append(0); threshold.append(0.); left.append(-1); right.append(-1)
            value.append(float(residual[rows].mean()) * (.18 if boosted else 1/count))
            if depth >= 5 or len(rows) < 24:
                return node
            best_gain, split = 0., None
            for f in rng.choice(x.shape[1], min(8, x.shape[1]), replace=False):
                col = x[rows, f]
                cut = float(np.quantile(col, rng.uniform(.2, .8)))
                mask = col <= cut
                n = int(mask.sum())
                if min(n, len(rows)-n) < 8:
                    continue
                gain = n*(len(rows)-n)/len(rows) * (residual[rows[mask]].mean()-residual[rows[~mask]].mean())**2
                if gain > best_gain:
                    best_gain, split = gain, (int(f), cut, mask)
            if split is not None:
                f, cut, mask = split
                feature[node], threshold[node] = f, cut
                left[node] = grow(rows[mask], depth+1)
                right[node] = grow(rows[~mask], depth+1)
            return node
        grow(ids, 0)
        if boosted:
            # Each new tree explains the errors left by earlier trees in this layer.
            tree = dict(method="condition_tree", intercept=0., starts=np.array([0]),
                        feature=np.array(feature[start:]), threshold=np.array(threshold[start:]),
                        left=np.array([i-start if i>=0 else -1 for i in left[start:]]),
                        right=np.array([i-start if i>=0 else -1 for i in right[start:]]), value=np.array(value[start:]))
            residual = residual - Forest(tree, x.shape[1]).predict(x)
    return dict(intercept=intercept, starts=np.array(starts), feature=np.array(feature),
                threshold=np.array(threshold), left=np.array(left), right=np.array(right), value=np.array(value))


def fit_layer(model, train, task, deadline, positive_scale=1.):
    x, y = train["x"], train["y"]
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"],model.meta["epochs"]]))
    state = dict(task=task)
    if task == "B":
        state.update(method="bearing_mix", strength=1., weights=np.array([.1,.2]))
        data = features(x,state).astype(np.float32).astype(np.float64)
        for prefix,target in (("arithmetic_",y), ("log_",np.log1p(y))):
            tree = fit_forest(data,target,rng,deadline,boosted=False,count=8)
            tree["method"] = "extra_trees"
            state.update({prefix+k:v for k,v in tree.items()})
        radius = None
    else:
        settings = x[:,21:]
        state.update(method="condition_tree",setting_mean=settings.mean(0),setting_scale=np.maximum(settings.std(0),1e-7))
        scaled = (settings-state["setting_mean"])/state["setting_scale"]
        centers = centers_from_train(scaled,model.meta["seed"])
        group = assign(scaled,centers)
        state.update(centers=centers,means=np.stack([x[group==i,:21].mean(0) for i in range(len(centers))]),
                     scales=np.maximum(1e-5,np.stack([x[group==i,:21].std(0) for i in range(len(centers))])))
        radius = float(.05*np.diff(np.quantile(y,[.25,.75]))[0])
        state.update(strength=.35 if model.tree_layers else .7, positive_cap=radius)
        if positive_scale != 1:
            state["positive_scale"] = positive_scale
        state.update(fit_forest(features(x,state),y-model.predict(x),rng,deadline,boosted=True))
    check_time(deadline)
    content = io.BytesIO()
    np.savez_compressed(content, **state)
    model.append_layer(content.getvalue(),radius)

