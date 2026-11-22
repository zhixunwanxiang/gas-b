"""Train-only decision calibration and a relative-sensor ethanol/acetaldehyde head."""
import hashlib

import numpy as np


def training_parts(train):
    labels = np.asarray(train["y"])
    ids = train["ids"]
    calibration = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        positions = np.flatnonzero(labels == cls)
        order = sorted(positions, key=lambda i: hashlib.sha256(("gas-cal-v1|" + str(ids[i])).encode()).digest())
        calibration[order[:len(order) // 5]] = True
    def subset(mask):
        return {**train, "x":train["x"][mask], "y":labels[mask],
                "ids":[ids[i] for i in np.flatnonzero(mask)]}
    return subset(~calibration), subset(calibration)


def relative_response(z):
    sensors = z.reshape(len(z), 16, 8)
    relative = sensors - sensors.mean(axis=1, keepdims=True)
    return relative.reshape(len(z), 128)


TRIO = np.array([0, 3, 5])


def fit_trio(z, y, relative):
    features = relative_response(z) if relative else z
    groups = [features[y == cls] for cls in TRIO]
    if min(map(len, groups)) < 2:
        return None
    centers = np.stack([g.mean(0) for g in groups])
    residuals = np.concatenate([g-c for g,c in zip(groups, centers)])
    columns = np.argsort(-centers.var(0)/(residuals.var(0)+.03), kind="stable")[:20]
    cov = np.cov(residuals[:,columns], rowvar=False)
    cov.flat[::len(columns)+1] += .3*max(float(np.trace(cov))/len(columns), .01)
    means = centers[:,columns]
    weights = np.linalg.solve(cov, means.T)
    offsets = -.5*np.sum(means*weights.T, axis=1)
    logits = np.concatenate(groups)[:,columns] @ weights + offsets
    margins = np.sort(logits,axis=1)[:,-2:]
    scale = max(float(np.median(margins[:,1]-margins[:,0]))/3.,1.)
    return dict(columns=columns.tolist(), weights=(weights/scale).tolist(), bias=(offsets/scale).tolist(),
                relative=relative, strength=.25)


def blend_trio(p, z, head):
    rows = np.flatnonzero(np.isin(p.argmax(1), TRIO))
    if not len(rows):
        return p
    x = relative_response(z[rows]) if head["relative"] else z[rows]
    logits = x[:,head["columns"]] @ np.asarray(head["weights"]) + head["bias"]
    q = np.exp(logits - logits.max(1,keepdims=True))
    q /= q.sum(1,keepdims=True)
    idx = np.ix_(rows, TRIO)
    old = p[idx]
    p[idx] = (1-head["strength"])*old + head["strength"]*old.sum(1, keepdims=True)*q
    return p


def calibrate_trio(model, fit_head, calibration):
    model.meta.pop("ade_threeway", None)
    if fit_head is None or not len(calibration["y"]):
        return
    p = model.predict(calibration["x"])
    z = model.normalized(calibration["x"])
    target = calibration["y"] == 3
    if not target.any():
        return
    best = -1.
    for strength in (.25, .5, .75, 1.):
        head = dict(fit_head, strength=strength)
        positive = blend_trio(p.copy(), z, head).argmax(1) == 3
        f1 = 2 * (positive & target).sum() / (positive.sum()+target.sum())
        if f1 > best:
            model.meta["ade_threeway"] = head
            best = f1


def fit_pair(z, y, strength):
    x = relative_response(z)
    groups = [x[y == i] for i in (0, 3)]
    if any(len(g) < 3 for g in groups):
        return None
    ethanol, ade = (g.mean(axis=0) for g in groups)
    residual = np.vstack((groups[0]-ethanol, groups[1]-ade))
    variance = np.mean(residual**2, axis=0)
    rank = (ade-ethanol)**2 / (variance + .01)
    cols = np.argsort(-rank, kind="stable")[:16]
    cov = residual[:,cols].T @ residual[:,cols] / (len(residual)-2)
    cov.flat[::len(cols)+1] += .15 * max(float(np.trace(cov)/len(cols)), 1e-3)
    weights = np.linalg.solve(cov, (ade-ethanol)[cols])
    bias = -float((ethanol+ade)[cols] @ weights) / 2
    return dict(columns=cols.tolist(),weights=weights.tolist(),bias=bias,strength=strength)


def apply_decisions(p, z, meta):
    head = meta.get("ethanol_ade_head")
    if head is not None:
        rows = np.flatnonzero(np.isin(p.argmax(axis=1), (0,3)))
        if len(rows):
            x = relative_response(z[rows])[:,head["columns"]]
            linear = x @ np.asarray(head["weights"]) + head["bias"]
            q = 1 / (1 + np.exp(-np.clip(linear,-30,30)))
            mass = p[rows,0] + p[rows,3]
            p[rows,3] = (1-head["strength"])*p[rows,3] + head["strength"]*mass*q
            p[rows,0] = mass - p[rows,3]
    if "logit_bias" in meta:
        p *= np.exp(meta["logit_bias"])
        p /= p.sum(axis=1,keepdims=True)
    if "ade_threeway" in meta:
        p = blend_trio(p, z, meta["ade_threeway"])
    return p


def fit_bias(model, calibration):
    model.meta.pop("logit_bias", None)
    y = calibration["y"]
    if not len(y):
        return
    logits = np.log(np.maximum(model.predict(calibration["x"]),1e-30))
    bias = np.zeros(6)
    def objective(b):
        predicted = (logits+b).argmax(axis=1)
        cm = np.bincount(6*y+predicted,minlength=36).reshape(6,6)
        f1 = 2*cm.diagonal()/np.maximum(1,cm.sum(0)+cm.sum(1))
        return float(f1.mean()) - .001 * float(np.abs(b).mean())
    best = objective(bias)
    for step in (.5,.25):
        for cls in np.unique(y):
            selected = bias[cls]
            for value in np.clip(bias[cls] + step*np.array([-2,-1,1,2]), -1.5, 1.5):
                trial = bias.copy()
                trial[cls] = value
                score = objective(trial)
                if score > best+1e-12:
                    selected, best = float(value), score
            bias[cls] = selected
    model.meta["logit_bias"] = bias.tolist()
