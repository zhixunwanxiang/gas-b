"""Task A participant module loaded by contest_harness 1.0.4."""
# SDK 加载的选手模块：先固定数值库线程数，再导入训练和预测回调。
# TASKS 声明任务类型 classification；它与数据集代号 A 是两个不同的概念。
import os
from pathlib import Path
import sys

for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gas_a.submission import predict, train

TASKS = ("classification",)
