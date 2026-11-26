"""Fresh A/B/C SDK rounds using the active menu, official initial parents and empty state."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--dataset", choices=list("ABC"))
    args = parser.parse_args()
    output = args.output.resolve()
    registration = output / "registration.json"
    if args.dataset is None:
        if args.baseline_root is None:
            parser.error("--baseline-root required for registration")
        output.mkdir(parents=True, exist_ok=False)
        config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
        menu = [s["name"] for s in config["regression"]["experiments"]]
        is_xiaoyang = any(s.get("tree_update") for s in config["regression"]["experiments"])
        if not is_xiaoyang:
            assert menu == ["low_rate", "standard_rate", "head_only", "robust_huber"]
        sources = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "contest_submission.py", ROOT / "harness.py",
                   ROOT / "submission_config.json", ROOT / "scripts/validate_multitask.py", Path(__file__).resolve()]
        sources += [ROOT / name for name in ("tests/test_submission.py", "tests/test_regression.py", "scripts/validate_platform.py")]
        inputs = {}
        for dataset, directory in BASELINES.items():
            base = args.baseline_root.resolve() / directory
            inputs[dataset] = {"base": str(base), "sha256": {name: digest(base / name) for name in
                ("checkpoint.npz", "baseline-info.json", "public/train.jsonl", "public/validation.jsonl")}}
        write(registration, {"scope": "fresh_native_offline_sdk_numeric_test", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "timezone": "Asia/Shanghai", "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "code_versions": {"A": "current; preserved", "B": "xiaoyang" if is_xiaoyang else "V18", "C": "xiaoyang" if is_xiaoyang else "V18"},
            "sources_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
            "inputs": inputs, "configuration": config, "rounds_per_dataset": 10,
            "experiment_sequences": {"A": ["frozen_router", "frozen_router", "conservative_head", "frozen_router", "conservative_head",
                "four_sensor", "conservative_head", "five_sensor", "conservative_head", "frozen_router"],
                "B": (["bearing_tree_mix", "robust_huber", "bearing_tree_mix", "robust_huber", "standard_rate",
                       "head_only", "robust_huber", "low_rate", "robust_huber", "low_rate"] if is_xiaoyang else (menu * 3)[:10]),
                "C": (["engine_tail_tree", "engine_cumulative_tree", "engine_tail_tree", "robust_huber", "condition_residual",
                       "engine_tail_tree", "engine_cumulative_tree", "engine_tail_tree", "engine_cumulative_tree", "standard_rate"] if is_xiaoyang else (menu * 3)[:10])},
            "controller": "explicit_offline_stub; no provider calls", "real_model_calls": 0,
            "initial_state": "empty per dataset; official initial checkpoint only", "historical_results_used": False,
            "runtime": "native Windows Python; one numerical thread per process; datasets may run concurrently",
            "container_resource_limits_enforced": False, "official_score_verified": False})
        print(str(registration), flush=True)
        return
    record = json.loads(registration.read_text(encoding="utf-8"))
    for name, expected in record["sources_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered code changed: " + name)
    dataset = args.dataset
    base = Path(record["inputs"][dataset]["base"])
    for name, expected in record["inputs"][dataset]["sha256"].items():
        if digest(base / name) != expected:
            raise ValueError("Registered initial input changed: " + name)
    sys.path.insert(0, str(ROOT))
    import contest_submission
    from scripts.validate_multitask import run_dataset
    started = time.monotonic()
    report = run_dataset(base, output / dataset, dataset, 10, None, max_rounds=10,
        compact_validation=True, aggregate_parent=True,
        experiment_sequence=record["experiment_sequences"][dataset], exchange_options={"round_timeout": 120})
    assert len(report["rounds"]) == 10 and all(r["finish_acknowledged"] for r in report["rounds"])
    write(output / f"{dataset}-completion.json", {"dataset": dataset, "wall_seconds": time.monotonic() - started,
        "fresh_run": True, "rounds": 10, "real_model_calls": 0, "official_score_verified": False})


if __name__ == "__main__":
    main()
