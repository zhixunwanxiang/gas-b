"""Dynamic task-A entry: platform commands, local continuation, and explicit replay."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "describe"
    if command in ("describe", "train", "predict"):
        from contest_harness.cli import main as harness_main
        return harness_main([str(ROOT / "harness.py"), command, *argv[1:]])
    if command == "train-local":
        from gas_a.local import train_main
        return train_main(argv[1:])
    if command == "predict-local":
        from gas_a.local import predict_main
        return predict_main(argv[1:])
    if command == "evaluate":
        from gas_a.local import evaluate_main
        return evaluate_main(argv[1:])
    if command == "replay":
        from replay import main as replay_main
        return replay_main(argv[1:])
    if command in ("-h", "--help"):
        print(__doc__)
        print("Commands: describe, train, predict, train-local, predict-local, evaluate, replay")
        print("Use <command> --help for local command options; train requires the platform host.")
        return 0
    raise SystemExit(f"Unknown command {command!r}; use --help")


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
