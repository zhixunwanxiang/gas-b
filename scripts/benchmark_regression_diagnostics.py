"""比较逐 epoch 完整诊断与轻量晋级计算，检查导出的模型字节一致。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from gas_a.regression.data import load_pair
from gas_a.regression.training import continue_candidate
from gas_a.sensor_training.io import digest, write_json
from scripts.validate_multitask import BASELINES


def _peak_memory():
    if sys.platform == "win32":
        value = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
                                        f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64"], text=True)
        return int(value.strip())
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def worker(base, dataset, output, detailed):
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))["regression"]
    spec = next(s for s in config["experiments"] if s["name"] == "standard_rate")
    checkpoint = base / "checkpoint.npz"
    train, validation = load_pair(base / "public/train.jsonl", base / "public/validation.jsonl", dataset)
    wall, cpu = time.perf_counter(), time.process_time()
    fit = continue_candidate(checkpoint, train, validation, output, spec,
                             config["limits"], config["controller"], dataset, digest(checkpoint), detailed_epochs=detailed)
    metrics = {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
               "peak_rss_bytes": _peak_memory(), "candidate_history_bytes": (output / "history.jsonl").stat().st_size,
               "checkpoint_sha256": fit["checkpoint_sha256"], "parent_checkpoint_sha256": fit["parent_checkpoint_sha256"],
               "train_sha256": fit["train_sha256"], "validation_sha256": fit["validation_sha256"],
               "measured_epochs": fit["executed_epochs"], "selected_epoch": fit["selected_epoch"],
               "rmse": fit["validation_after"]["rmse"], "nasa_score": fit["validation_after"].get("nasa_score")}
    write_json(output / "benchmark.json", metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("B", "C"))
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()
    if args.dataset:
        worker(args.baseline_root / BASELINES[args.dataset], args.dataset, args.output, args.detailed)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for dataset in ("B", "C"):
        timings = {}
        for label in ("detailed", "light"):
            target = args.output / f"{dataset}-{label}"
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                            "--baseline-root", str(args.baseline_root), "--output", str(target),
                            "--dataset", dataset, *(["--detailed"] if label == "detailed" else [])], check=True, cwd=ROOT)
            timings[label] = json.loads((target / "benchmark.json").read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "parent_checkpoint_sha256", "validation_sha256", "train_sha256",
                    "measured_epochs", "selected_epoch", "rmse", "nasa_score"):
            if timings["detailed"][key] != timings["light"][key]:
                raise ValueError(f"{dataset} diagnostic benchmark changed selection: {key}")
        results[dataset] = timings
    write_json(args.output / "summary.json", {"scope": "local_training_two_isolated_processes_per_dataset",
        "identical_checkpoint_verified": True, "benchmark": results})


if __name__ == "__main__":
    main()
