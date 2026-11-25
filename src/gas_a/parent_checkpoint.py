"""按 SDK 父产物清单核验文件，区分产物身份与单文件摘要。"""
from pathlib import Path


def verified_checkpoints(rnd):
    """逐文件使用 SDK 的清单摘要核验；不推测宿主的聚合摘要算法。"""
    names = rnd.parent.descriptor.get("checkpoint_paths")
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Parent must declare distinct checkpoint paths")
    # parent_checkpoint_sha256 可以标识整组检查点，不能拿它与单个文件比较。
    # rnd.parent.path 对每个文件检查：路径已声明、普通文件、清单 SHA-256 一致。
    return tuple(rnd.parent.path(name) for name in names)


def verify_selected_checkpoint(rnd, checkpoint):
    """确认实际加载的主文件属于本轮已逐文件核验的父产物。"""
    paths = verified_checkpoints(rnd)
    if Path(checkpoint).resolve() not in {path.resolve() for path in paths}:
        raise ValueError("Parent reevaluation must use the SDK-declared checkpoint")
    return paths


def primary_checkpoint(rnd):
    """单文件直接读取；A 多组件包以自身的主模型标记定位，拒绝歧义。"""
    paths = verified_checkpoints(rnd)
    if len(paths) == 1:
        return paths[0]
    # 四组件包只有 four_sensor 主模型携带 prediction_bundle 标记。
    # 不依赖文件名、清单顺序或聚合摘要；后续 load_predictor 还会验证路由清单及全部组件。
    from .sensor_training.data import load_model
    from .frozen_router import FORMAT
    anchors = []
    for path in paths:
        try:
            model = load_model(path)
        except ValueError:
            # Home 组件并非主 MLP 格式。这里只识别主文件，不代替完整包的校验。
            continue
        marker = model.meta.get("prediction_bundle")
        if isinstance(marker, dict) and marker.get("format") == FORMAT:
            anchors.append(path)
    if len(anchors) != 1:
        raise ValueError("Multi-file parent must declare exactly one supported bundle anchor")
    return anchors[0]
