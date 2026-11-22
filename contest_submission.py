"""Task A participant module loaded by contest_harness 1.0.4."""
import os
from pathlib import Path
import sys

for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gas_a.submission import predict, train

TASKS = ("classification",)
