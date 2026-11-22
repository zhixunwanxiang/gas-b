"""Training and calibration of the full-input auxiliary expert."""
# 全输入辅助专家训练：输出头、KNN 记忆和成对逻辑回归头仅使用公开 train。
# 特征选择和二分类阈值的交叉验证也在 train 内完成；外层公开 validation 负责整体路由选版。
import math
import time
from copy import deepcopy
import numpy as np


def train_fisher_feature_mask(model, x, y, keep_features):
    """Choose discriminative KNN dimensions from training rows and labels only."""
    # 用训练标签计算类间/类内方差比，稳定排序选特征；不用漂移值或验证标签拟合 KNN 特征。
    normalized = model.normalized(np.asarray(x, dtype=np.float64))
    labels = np.asarray(y, dtype=np.int64)
    feature_count = normalized.shape[1]
    class_count = len(model.meta["classes"])
    if (normalized.ndim != 2 or labels.shape != (len(normalized),) or not len(labels)
            or type(keep_features) is not int or not 16 <= keep_features <= feature_count
            or np.any(labels < 0) or np.any(labels >= class_count)):
        raise ValueError("Invalid training-only Fisher feature selection inputs")
    overall_mean = normalized.mean(axis=0)
    between = np.zeros(feature_count, dtype=np.float64)
    within = np.zeros(feature_count, dtype=np.float64)
    for class_index in range(class_count):
        group = normalized[labels == class_index]
        if not len(group):
            continue
        group_mean = group.mean(axis=0)
        between += len(group) * (group_mean - overall_mean) ** 2
        residual = group - group_mean
        within += np.sum(residual * residual, axis=0)
    # Fisher 分数偏好类间差异大、类内波动小的维度；稳定排序让相同分数下的选择可复现。
    scores = between / (within + 1e-12)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite training-only feature score")
    ranked = np.argsort(-scores, kind="stable")
    mask = np.zeros(feature_count, dtype=bool)
    mask[ranked[:keep_features]] = True
    return mask


def pair_fisher_order(x, labels):
    """Rank pairwise features using only the two classes in the training split."""
    # 对两个训练类别做同样的 Fisher 排序；任一类别缺失则不制造不可训练的二分类头。
    labels = np.asarray(labels, dtype=np.int64)
    overall = x.mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    within = np.zeros(x.shape[1], dtype=np.float64)
    for class_index in (0, 1):
        group = x[labels == class_index]
        if not len(group):
            raise ValueError("Pairwise head requires both classes in training data")
        mean = group.mean(axis=0)
        between += len(group) * (mean - overall) ** 2
        within += np.sum((group - mean) ** 2, axis=0)
    return np.argsort(-(between / (within + 1e-12)), kind="stable")


def fit_pairwise_logistic(x, labels, positive_weight, l2):
    """Fit a small weighted logistic head with Newton steps on training data only."""
    # 使用加权逻辑回归与 L2 正则，通过牛顿步求解小型成对头；不涉及公开 validation。
    n, feature_count = x.shape
    labels_float = np.asarray(labels, dtype=np.float64)
    sample_weight = np.where(labels_float == 1, positive_weight, 1.0)
    theta = np.zeros(feature_count + 1, dtype=np.float64)
    prevalence = labels_float.mean()
    theta[-1] = math.log((prevalence + 1e-6) / (1.0 - prevalence + 1e-6))
    design = np.column_stack((x, np.ones(n, dtype=np.float64)))
    penalty = np.diag(np.r_[np.full(feature_count, l2), 1e-8])
    for _ in range(80):
        logits = np.clip(design @ theta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weight * (probability - labels_float)) / n + penalty @ theta
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / n + penalty
        step = np.linalg.solve(hessian, gradient)
        # 牛顿步使用梯度/Hessian 的局部曲率，较小步长范数作为收敛条件，不按验证成绩更新权重。
        theta -= step
        if np.linalg.norm(step) < 1e-7:
            break
    if not np.all(np.isfinite(theta)):
        raise ValueError("Non-finite pairwise head parameters")
    return theta


rank_features = pair_fisher_order
fit_logistic = fit_pairwise_logistic


def head_epoch(model, fit, spec):
    """Adam on the output layer only. Keep both hidden-layer Adam moments bit-identical."""
    # 只对输出层 w2/b2 更新 Adam；隐藏层权重和已有一二阶矩都保持原样。
    # 全局 step 延续父模型，用于 Adam 偏差校正，不能因冻结隐藏层而清零。
    x,y = model.normalized(fit["x"]),fit["y"]
    freq = np.bincount(y,minlength=6)
    weights = (len(y)/(6*np.maximum(freq,1)))**spec["balance"]
    weights /= float(weights[y].mean())
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"],model.meta["epochs"]]))
    order = rng.permutation(len(y))
    rate = spec["learning_rate"]
    model.meta["learning_rate"] = rate
    batch_size = model.meta["batch_size"]
    smoothing = spec["label_smoothing"]
    epoch_loss = 0.
    for offset in range(0,len(order),batch_size):
        ids = order[offset:offset+batch_size]
        hidden,logits = model.forward(x[ids])
        p = np.exp(logits-logits.max(axis=1,keepdims=True))
        p /= p.sum(axis=1,keepdims=True)
        target = y[ids]
        logp = np.log(np.maximum(p,1e-300))
        loss = -(1-smoothing)*logp[np.arange(len(ids)),target]-smoothing*logp.mean(1)
        epoch_loss += float(weights[target] @ loss)
        delta = p - smoothing/6
        delta[np.arange(len(ids)),target] -= 1-smoothing
        delta *= weights[target,None]/len(ids)
        grads = dict(w2=hidden.T@delta+spec["l2"]*model.params["w2"], b2=delta.sum(0))
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        model.meta["steps"] += 1
        step = model.meta["steps"]
        for k,g in grads.items():
            g *= min(1.,5./max(float(norm),1e-12))
            model.first[k] = .9*model.first[k]+.1*g
            model.second[k] = .999*model.second[k]+.001*g**2
            m = model.first[k]/(1-.9**step)
            v = model.second[k]/(1-.999**step)
            model.params[k] -= rate*m/(np.sqrt(v)+1e-8)
    model.meta["epochs"] += 1
    model.validate()
    return dict(epoch=model.meta["epochs"],steps=model.meta["steps"],loss=epoch_loss/len(y))


def binary_cutoffs(parent,x,y):
    """Five-fold B1-only binary expert scores. Refit feature ranking inside every fold."""
    # 在公开 train 内做分层五折验证，每折只用训练折重排特征并拟合成对头。
    # 候选阈值按折外 Macro-F1 选择，防止先用全部样本选特征再交叉验证造成泄漏。
    z=parent.normalized(x)
    output={}
    for cls in (0,1):
        idx=np.flatnonzero(np.isin(y,[cls,3]))
        t=(y[idx]==3).astype(int)
        fold=np.zeros(len(t),int)
        rng=np.random.default_rng(1627)
        for v in (0,1):
            loc=rng.permutation(np.flatnonzero(t==v))
            fold[loc]=np.arange(len(loc))%5
        odds=np.zeros(len(t))
        for i in range(5):
            a,b=fold!=i,fold==i
            # 每一折重新选特征，不能用验证折标签先筛维度，否则折外分数会乐观偏高。
            cols=rank_features(z[idx[a]],t[a])[:8]
            w=fit_logistic(z[idx[a]][:,cols],t[a],1.,.03)
            odds[b]=z[idx[b]][:,cols]@w[:-1]+w[-1]
        scores={}
        for cut in (.5,.65,.8,.9):
            pred=(odds>=np.log(cut/(1-cut))).astype(int)
            m=np.bincount(2*t+pred,minlength=4).reshape(2,2)
            scores[str(cut)]=float((2*m.diagonal()/np.maximum(1,m.sum(0)+m.sum(1))).mean())
        selected=max(scores,key=lambda k:(scores[k],-abs(float(k)-.8)))
        output[str(cls)]=dict(cut=float(selected),fold_macro_f1=scores,n=len(t))
    return output


def train_home(parent, train, spec, dest, *, max_epochs=32, deadline=None):
    """Train the full-input branches on training rows only; preserve parent Adam/scaler."""
    # 从父模型复制输出头续训后再构建训练样本 KNN 与成对头；保存 trunk/router 两份完整组件。
    from .decisions import training_parts
    epochs = spec.get("epochs", 1)
    if type(epochs) is not int or not 1 <= epochs <= max_epochs:
        raise ValueError("home.epochs must be an integer within the epoch budget")
    fit, reserve = training_parts(train)
    trunk = deepcopy(parent)
    trunk.meta.pop("prediction_bundle", None)
    history = []
    for _ in range(epochs):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Home expert exceeded the candidate wall-time budget")
        history.append(head_epoch(trunk, fit, spec))
    trunk.save(dest / "home_trunk.npz")
    router = deepcopy(trunk)
    x = np.vstack([fit["x"], reserve["x"]])
    y = np.concatenate([fit["y"], reserve["y"]])
    mask = train_fisher_feature_mask(router, x, y, 120)
    # 这里保存的记忆标签全部来自公开 train；预测时查询训练记忆不等于获得当前/隐藏样本真实标签。
    router.attach_drift_pruned_knn(x, y, mask, neighbors=3, distance_power=3., class_balance=.25,
                                 mlp_blend=.4, drift_prune_features=8, class_adjustments=[1.,1.,1.,2.,1.,1.])
    z = router.normalized(x)
    heads = {}
    for cls in (0, 1):
        pair = np.isin(y, [cls, 3])
        target = (y[pair] == 3).astype(int)
        columns = rank_features(z[pair], target)[:8]
        theta = fit_logistic(z[pair][:, columns], target, 1., .03)
        heads[str(cls)] = dict(features=columns.tolist(), w=theta[:-1].tolist(), b=float(theta[-1]))
    router.meta["private_joint_rescue"] = heads
    router.save(dest / "home_router.npz")
    return {"epochs_requested": epochs, "epochs_executed": len(history), "epoch_history": history}
