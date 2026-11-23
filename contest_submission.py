"""Native A classification and B/C regression callbacks for SDK 1.0.4."""
# SDK 加载的选手模块：先固定数值库线程数，再导入训练和预测回调。
# TASKS 声明任务类型；数据集 A/B/C 由宿主合同决定，不能把 B/C 当成分类。
import os
from pathlib import Path
import sys

for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    # 必须在导入 NumPy 相关业务模块前设置，防止 BLAS 后台线程突破容器 CPU 预算。
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gas_a.multitask import predict, train

TASKS = ("classification", "regression")
