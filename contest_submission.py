"""Native A classification and B/C regression callbacks for SDK 1.0.4."""
# SDK 加载的选手模块：先固定数值库线程数；真正收到任务后才加载数值模块。
# TASKS 声明任务类型；数据集 A/B/C 由宿主合同决定，不能把 B/C 当成分类。
import os
from pathlib import Path
import sys

for name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    # 必须在导入 NumPy 相关业务模块前设置，防止 BLAS 后台线程突破容器 CPU 预算。
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

TASKS = ("classification", "regression")


def train(rnd):
    # describe 只需检查声明和函数是否可调用，不应启动 NumPy、训练或读取数据。
    # SDK 已收到 start 后才走到此处；不提前等待 stdin EOF，也不另建协议。
    from gas_a.multitask import train as run
    from gas_a.progress import resource_scope
    try:
        with resource_scope(dataset=rnd.dataset_key, round_index=rnd.round_index):
            return run(rnd)
    except Exception as error:
        from gas_a.progress import training_failure
        training_failure(error, dataset=rnd.dataset_key, round_index=rnd.round_index)
        raise


def predict(sample, model_dir):
    # 保留 SDK 逐行调用方式，首条输入到来时再加载相应预测器。
    from gas_a.multitask import predict as run
    from gas_a.progress import prediction_resource
    with prediction_resource():
        return run(sample, model_dir)
