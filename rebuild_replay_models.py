"""Rebuild the component weights from the two supplied initial checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from copy import deepcopy

ROOT = Path(__file__).resolve().parent
for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gas_a.model import Model, checkpoint_digest
from gas_a.decisions import training_parts
from gas_a.home_training import head_epoch, train_fisher_feature_mask, rank_features, fit_logistic, binary_cutoffs
from gas_a.sensor_training.data import load_pair, load_model, evaluate
from gas_a.sensor_training.optimize import train_candidate


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for rebuilt weights")
    dest = parser.parse_args().output
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data = ROOT / "data" / "public"
    train, validation = load_pair(data / "train.jsonl", data / "validation.jsonl")
    dest.mkdir(parents=True, exist_ok=False)
    for role in ("reference", "start"):
        source = ROOT / "models" / (role + ".npz")
        if checkpoint_digest(source) != cfg["model_sha256"][role]:
            raise ValueError(f"Wrong parent for {role}")
        shutil.copyfile(source, dest / source.name)
    parent = Model.load(dest / "start.npz")
    train_home(parent, train, cfg["training"]["home"], dest)
    cuts = binary_cutoffs(parent, train["x"], train["y"])
    if {k:v["cut"] for k,v in cuts.items()} != {k:v["cut"] for k,v in cfg["cutoffs"].items()}:
        raise ValueError("Different calibration cutoffs")
    baseline = dest / "reference.npz"
    limits = dict(cfg["training"]["limits"], checkpoint_reference=evaluate(load_model(baseline), validation),
                  checkpoint_reference_sha256=checkpoint_digest(baseline))
    for name, spec in cfg["training"]["sensors"].items():
        trace = dest / "training_log" / name
        trace.parent.mkdir(parents=True, exist_ok=True)
        train_candidate(baseline, train, validation, trace, spec, {"weakest_class": "Acetaldehyde"}, limits)
        shutil.copyfile(trace / "checkpoint.npz", dest / (name + ".npz"))
    for name, signature in cfg["model_sha256"].items():
        if checkpoint_digest(dest / (name + ".npz")) != signature:
            raise ValueError(f"Rebuilt checkpoint differs from the archived bytes: {name}")
    print("All six component files match the archived SHA-256 digests.")


if __name__ == "__main__":
    main()
