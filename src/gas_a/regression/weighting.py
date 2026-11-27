"""只从 train 计算有限权重；整体均值为 1，保持学习率可比较。"""
import numpy as np
from .diagnostics import groups


def sample_weights(train, spec, dataset):
    weights = np.ones(len(train["y"]))
    alpha = spec.get("device_balance", 0.)
    if alpha:
        values = groups(train, dataset)["device"]
        if np.any(values == "unknown"):
            raise ValueError("Device balancing requires complete public trace metadata")
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        weights *= (len(values) / (len(counts) * counts[inverse])) ** alpha
    boost = spec.get("tail_boost", 1.)
    if boost != 1:
        cutoff = float(np.quantile(train["y"], .75))
        weights[train["y"] >= cutoff] *= boost
    return weights / weights.mean()


def resolve_training_spec(parent, train, spec):
    resolved = dict(spec)
    if "huber_quantile" in spec:
        residual = np.abs(parent.predict(train["x"]) - train["y"]) / parent.meta["target_scale"]
        resolved["huber_delta"] = float(np.clip(np.quantile(residual, spec["huber_quantile"]), .1, 2.))
    if "short_overestimate_boost" in spec:
        resolved["short_rul_cut"] = float(np.quantile(train["y"], .25))
    return resolved


def stratified_order(rng, strata):
    """把每个分层的样本均匀铺到 epoch 内，每行仍恰好使用一次。"""
    order = rng.permutation(len(strata))
    grouped = order[np.argsort(strata[order], kind="stable")]
    _, counts = np.unique(strata, return_counts=True)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    rank = np.arange(len(order)) - starts
    position = (rank + rng.random(len(order))) / np.repeat(counts, counts)
    return grouped[np.argsort(position, kind="stable")]


def nasa_proxy(error, target_scale, mix=.35):
    """平滑 NASA 代理损失及梯度，指数段后接切线以限制梯度。"""
    scale = 6. * np.where(error < 0, 13., 10.) / target_scale
    magnitude = np.abs(error) / scale
    capped = np.minimum(magnitude, 2.)
    slope = np.expm1(capped)
    loss = np.expm1(capped) - capped + slope * (magnitude - capped)
    return ((1 - mix) * error ** 2 + mix * loss,
            2 * (1 - mix) * error + mix * np.sign(error) * slope / scale)
