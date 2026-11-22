"""Replay the released four-sensor policy on batch3 through batch10."""
# 历史批次回放入口：先提交本批预测，再揭示已授权的公开标签并更新后续路由。
# 该反馈流程用于开发回放；正式平台入口是 harness.py，不会调用此回放循环。
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gas_a.signals import dat_inputs, reveal_labels
from gas_a.router import FourSensorRouter, confusion, macro


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="new directory for locked predictions")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    args = parser.parse_args(argv)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    engine = FourSensorRouter(args.models, config)
    out = args.output or ROOT / "runtime" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True, exist_ok=False)
    pooled = np.zeros((6, 6), dtype=int)
    results = []
    for batch in range(3, 11):
        path = ROOT / "data" / "replay" / f"batch{batch}.dat"
        features = dat_inputs(path, engine.parent.meta["paths"])
        expected = config["acceptance"][str(batch)]
        if features["sha256"] != expected["data_sha256"]:
            raise ValueError(f"Input differs from archived batch {batch}")
        predicted = engine.predict_batch(features["x"])
        seal = out / f"batch{batch:02d}.npy"
        with seal.open("xb") as stream:
            np.save(stream, predicted.astype(np.uint8), allow_pickle=False)
        locked = np.load(seal, allow_pickle=False)
        signature = hashlib.sha256(locked.tobytes()).hexdigest()
        if signature != expected["prediction_sha256"]:
            raise ValueError(f"Predictions differ from the original batch {batch}")
        labels = reveal_labels(path, features)["y"]
        matrix = confusion(labels, locked)
        pooled += matrix
        engine.release_labels(labels)
        results.append(dict(batch=batch, rows=len(locked), macro_f1=macro(matrix),
                            prediction_sha256=signature))
        print(f"batch{batch:02d}  n={len(locked):4d}  Macro-F1={macro(matrix):.6f}")
    if pooled.tolist() != config["acceptance_matrix"]:
        raise ValueError("Pooled confusion matrix differs from the archive")
    summary = dict(rows=int(pooled.sum()), pooled_macro_f1=macro(pooled),
                   confusion_matrix=pooled.tolist(), batches=results)
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Pooled Macro-F1={macro(pooled):.9f}; rows={pooled.sum()}")
    print(out.resolve())


if __name__ == "__main__":
    main()
