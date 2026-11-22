"""Continue an explicitly supplied baseline/resume checkpoint; retain train_home for local analysis."""
import os
from pathlib import Path
import sys
from copy import deepcopy

ROOT = Path(__file__).resolve().parent
for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gas_a.decisions import training_parts
from gas_a.home_training import head_epoch, train_fisher_feature_mask, rank_features, fit_logistic


def train_home(parent, train, spec, dest):
    fit, reserve = training_parts(train)
    trunk = deepcopy(parent)
    head_epoch(trunk, fit, spec)
    trunk.save(dest / "home_trunk.npz")
    router = deepcopy(trunk)
    x = np.vstack([fit["x"], reserve["x"]])
    y = np.concatenate([fit["y"], reserve["y"]])
    mask = train_fisher_feature_mask(router, x, y, 120)
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


def main():
    from gas_a.local import train_main
    return train_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
