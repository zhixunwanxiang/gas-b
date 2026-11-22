"""Small NumPy MLP + Adam. Checkpoints preserve every state needed to resume."""
# 扩展 NumPy 模型：保存神经网络、完整 Adam 状态和可选的训练样本记忆/辅助分类头。
# 新建模型时才计算 scaler；平台续训应使用 load 恢复父状态，不能重新初始化这份模型。
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

_THREADS = int(os.environ.get("BASELINE_CPU_THREADS", "1"))
if not 1 <= _THREADS <= 8:
    raise ValueError("BASELINE_CPU_THREADS must be between 1 and 8")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(_THREADS)
import numpy as np
from .decisions import apply_decisions

PARAMETERS = ("w1", "b1", "w2", "b2")
OPTIONAL_MEMORY_ARRAYS = ("memory_x", "memory_y", "feature_mask")
FORMAT = "contest.cpu-mlp.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


class Model:
    def __init__(self, task, paths, classes, x, y, *, hidden=32, seed=42, learning_rate=.003, batch_size=128):
        # 只用于显式新建模型；同时初始化权重、Adam 矩、scaler 和训练元数据。平台续训使用 load。
        if task not in ("classification", "regression") or not 1 <= hidden <= 128:
            raise ValueError("invalid model configuration")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError("classification needs distinct classes")
        count = x.shape[1] * hidden + hidden + hidden * output + output
        if count >= 100000:
            raise ValueError("reference baseline must have fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {"w1": rng.normal(0, math.sqrt(1 / x.shape[1]), (x.shape[1], hidden)),
                       "b1": np.zeros(hidden), "w2": rng.normal(0, math.sqrt(1 / hidden), (hidden, output)),
                       "b2": np.zeros(output)}
        self.first = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.memory_x = self.memory_y = self.feature_mask = None
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.meta = {"format": FORMAT, "device": "cpu", "task": task, "paths": paths,
                     "classes": list(classes), "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
                     "learning_rate": learning_rate, "batch_size": batch_size, "parameter_count": count,
                     "target_mean": float(y.mean()) if task == "regression" else 0.0,
                     "target_scale": max(float(y.std()), 1.0) if task == "regression" else 1.0}
        self.validate()

    def validate(self):
        # 检查元数据、张量维度/dtype、有限值、Adam 二阶矩及辅助预测器字段，拒绝不一致检查点。
        m = self.meta
        if m.get("format") != FORMAT or m.get("device") != "cpu" or m.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        for name in ("seed", "epochs", "steps", "hidden", "batch_size", "parameter_count"):
            if type(m.get(name)) is not int or m[name] < 0:
                raise ValueError("invalid checkpoint integer")
        if not 1 <= m["hidden"] <= 128 or not 1 <= m["batch_size"] <= 65536 or not 0 < m["parameter_count"] < 100000:
            raise ValueError("invalid checkpoint architecture")
        if not 0 <= m["seed"] < 2**32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if not isinstance(m.get("paths"), list) or not 1 <= len(m["paths"]) <= 4096 or any(not isinstance(p, str) for p in m["paths"]) or len(set(m["paths"])) != len(m["paths"]):
            raise ValueError("invalid checkpoint feature schema")
        classes = m.get("classes")
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid checkpoint classes")
        if (m["task"] == "classification" and len(classes) < 2) or (m["task"] == "regression" and classes):
            raise ValueError("invalid checkpoint classes")
        for name in ("learning_rate", "target_mean", "target_scale"):
            if isinstance(m.get(name), bool) or not isinstance(m.get(name), (float, int)) or not math.isfinite(m[name]):
                raise ValueError("invalid checkpoint numeric state")
        if not 0 < m["learning_rate"] <= 1 or m["target_scale"] <= 0:
            raise ValueError("invalid optimizer/scaler state")
        input_size, hidden = len(m["paths"]), m["hidden"]
        output = len(classes) if m["task"] == "classification" else 1
        shapes = {"w1": (input_size, hidden), "b1": (hidden,), "w2": (hidden, output), "b2": (output,)}
        for group in (self.params, self.first, self.second):
            if set(group) != set(PARAMETERS):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor shape or values invalid")
        if sum(value.size for value in self.params.values()) != m["parameter_count"]:
            raise ValueError("checkpoint parameter count mismatch")
        if self.mean.dtype != np.dtype("float64") or self.scale.dtype != np.dtype("float64") or self.mean.shape != (input_size,) or self.scale.shape != (input_size,) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("checkpoint normalization invalid")
        if any(np.any(value < 0) for value in self.second.values()):
            raise ValueError("invalid Adam second moment")
        predictor = m.get("predictor", "mlp")
        if "logit_bias" in m:
            b = np.asarray(m["logit_bias"],dtype=float)
            if b.shape != (6,) or not np.isfinite(b).all() or np.max(np.abs(b)) > 1.5:
                raise ValueError("Invalid calibration offsets")
        if "ethanol_ade_head" in m:
            pair = m["ethanol_ade_head"]
            cols, weights = np.asarray(pair["columns"]), np.asarray(pair["weights"],dtype=float)
            if (input_size != 128 or output != 6 or cols.shape != (16,) or
                    not np.issubdtype(cols.dtype,np.integer) or np.any(cols<0) or np.any(cols>=128)
                    or weights.shape != cols.shape or not np.isfinite(weights).all()
                    or not np.isfinite(pair["bias"]) or not 0 < pair["strength"] <= 1):
                raise ValueError("Invalid ethanol/acetaldehyde specialist")
        trio = m.get("ade_threeway")
        if trio is not None:
            cols = np.asarray(trio["columns"])
            w, b = np.asarray(trio["weights"]), np.asarray(trio["bias"])
            if (cols.shape != (20,) or not np.issubdtype(cols.dtype,np.integer) or len(np.unique(cols))!=20
                    or (cols<0).any() or (cols>=input_size).any() or w.shape!=(20,3) or b.shape!=(3,)
                    or not np.isfinite(w).all() or not np.isfinite(b).all() or type(trio["relative"]) is not bool
                    or not 0 < trio["strength"] <= 1 or m["task"]!="classification" or input_size!=128 or output!=6):
                raise ValueError("Invalid three-class specialist")
        head = m.get("discriminant")
        if head is not None:
            if not isinstance(head, dict) or predictor != "mlp" or m["task"] != "classification":
                raise ValueError("Incompatible discriminant head")
            w, b = np.asarray(head.get("weights"), dtype=float), np.asarray(head.get("bias"), dtype=float)
            if (w.shape != (input_size, output) or b.shape != (output,) or
                    not np.isfinite(w).all() or not np.isfinite(b).all() or
                    not 0 < head["blend"] <= 1):
                raise ValueError("Invalid discriminant head")
        if predictor not in ("mlp", "drift_pruned_knn"):
            raise ValueError("invalid predictor")
        memory = (getattr(self, "memory_x", None), getattr(self, "memory_y", None),
                  getattr(self, "feature_mask", None))
        if predictor == "mlp":
            if any(value is not None for value in memory):
                raise ValueError("MLP checkpoint contains unexpected memory")
        else:
            memory_x, memory_y, feature_mask = memory
            if (not isinstance(memory_x, np.ndarray) or memory_x.dtype != np.dtype("float64")
                    or memory_x.ndim != 2 or not len(memory_x) or not np.all(np.isfinite(memory_x))):
                raise ValueError("invalid KNN memory")
            if (not isinstance(memory_y, np.ndarray) or memory_y.dtype != np.dtype("int64")
                    or memory_y.shape != (len(memory_x),) or np.any(memory_y < 0) or np.any(memory_y >= output)):
                raise ValueError("invalid KNN labels")
            if (not isinstance(feature_mask, np.ndarray) or feature_mask.dtype != np.dtype("bool")
                    or feature_mask.shape != (input_size,) or int(feature_mask.sum()) != memory_x.shape[1]
                    or int(feature_mask.sum()) < 16):
                raise ValueError("invalid KNN feature mask")
            neighbors = m.get("knn_neighbors")
            prune = m.get("drift_prune_features")
            if type(neighbors) is not int or not 1 <= neighbors <= len(memory_x):
                raise ValueError("invalid KNN neighbor count")
            if type(prune) is not int or not 0 <= prune <= input_size - 16:
                raise ValueError("invalid drift pruning count")
            for name, lo, hi in (("knn_distance_power", 0, 4), ("knn_class_balance", 0, 1),
                                 ("knn_mlp_blend", 0, 1)):
                value = m.get(name)
                if (isinstance(value, bool) or not isinstance(value, (float, int))
                        or not math.isfinite(value) or not lo <= value <= hi):
                    raise ValueError(f"invalid {name}")
            adjustments = m.get("knn_class_adjustments")
            if (not isinstance(adjustments, list) or len(adjustments) != output
                    or any(isinstance(value, bool) or not isinstance(value, (float, int))
                           or not math.isfinite(value) or not 0.01 <= value <= 5 for value in adjustments)):
                raise ValueError("invalid KNN class adjustments")

    def normalized(self, x):
        # 沿用已保存的均值和尺度，并裁剪极端标准化值；推理不会重新拟合 scaler。
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        return np.clip((x - self.mean) / self.scale, -10, 10)

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def clear_auxiliary_predictor(self):
        # 清理下面明确列出的记忆和辅助校准字段，再校验剩余模型状态；其他专家元数据由上层构建流程管理。
        self.memory_x = self.memory_y = self.feature_mask = None
        for name in ("predictor", "knn_neighbors", "knn_distance_power", "knn_class_balance",
                     "knn_mlp_blend", "drift_prune_features", "knn_class_adjustments", "discriminant",
                     "ethanol_ade_head", "logit_bias", "ade_threeway"):
            self.meta.pop(name, None)
        self.validate()

    def attach_drift_pruned_knn(self, x, y, feature_mask, *, neighbors, distance_power,
                                class_balance, mlp_blend, drift_prune_features,
                                class_adjustments=None):
        # 把授权训练样本的标准化特征、标签和特征掩码写入模型，供无标签推理时的近邻融合。
        feature_mask = np.asarray(feature_mask, dtype=bool)
        if feature_mask.shape != (len(self.meta["paths"]),):
            raise ValueError("KNN feature mask shape mismatch")
        memory = self.normalized(np.asarray(x, dtype=np.float64))[:, feature_mask]
        norms = np.linalg.norm(memory, axis=1, keepdims=True)
        self.memory_x = np.asarray(memory / np.maximum(norms, 1e-12), dtype=np.float64)
        self.memory_y = np.asarray(y, dtype=np.int64).copy()
        self.feature_mask = feature_mask.copy()
        adjustments = ([1.0] * len(self.meta["classes"]) if class_adjustments is None
                       else [float(value) for value in class_adjustments])
        self.meta.update(predictor="drift_pruned_knn", knn_neighbors=int(neighbors),
                         knn_distance_power=float(distance_power), knn_class_balance=float(class_balance),
                         knn_mlp_blend=float(mlp_blend), drift_prune_features=int(drift_prune_features),
                         knn_class_adjustments=adjustments)
        self.validate()

    def memory_predict(self, normalized):
        # 在训练时已固定的特征掩码上计算当前输入与训练记忆的距离和加权投票，不读取当前真实标签。
        query = normalized[:, self.feature_mask]
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        result = np.empty((len(query), len(self.meta["classes"])), dtype=np.float64)
        k = self.meta["knn_neighbors"]
        power = self.meta["knn_distance_power"]
        counts = np.bincount(self.memory_y, minlength=len(self.meta["classes"]))
        class_weights = (len(self.memory_y) / (len(counts) * np.maximum(counts, 1))) ** self.meta["knn_class_balance"]
        class_weights *= np.asarray(self.meta["knn_class_adjustments"], dtype=np.float64)
        for start in range(0, len(query), 128):
            batch = query[start:start + 128]
            # Both sides are L2-normalized, so this computes mean squared distance
            # without allocating a [batch, train_rows, features] temporary.
            # 两侧向量已做 L2 归一化，平方距离可由 2-2<x,y> 得到，避免分配 batch×训练样本×特征 的大数组。
            distance = np.maximum(2.0 - 2.0 * (batch @ self.memory_x.T), 0.0) / self.memory_x.shape[1]
            indices = np.argpartition(distance, k - 1, axis=1)[:, :k]
            selected_distance = np.take_along_axis(distance, indices, axis=1)
            weights = np.ones_like(selected_distance) if power == 0 else 1 / np.maximum(selected_distance, 1e-12) ** power
            votes = np.zeros((len(batch), len(counts)), dtype=np.float64)
            labels = self.memory_y[indices]
            for class_index in range(len(counts)):
                votes[:, class_index] = np.sum(weights * (labels == class_index), axis=1) * class_weights[class_index]
            result[start:start + len(batch)] = votes / votes.sum(axis=1, keepdims=True)
        return result

    def fit(self, x, y, epochs):
        # 基本 MLP 训练保留累计 step/epoch 与已有 Adam 状态；正式受限实验另由 optimize.fit_weighted 管理。
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        x = self.normalized(x)
        if y.shape != (len(x),) or not len(x):
            raise ValueError("target shape mismatch")
        if self.meta["task"] == "regression":
            target = (y - self.meta["target_mean"]) / self.meta["target_scale"]
        else:
            if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                raise ValueError("unknown class")
            target = y
        epoch_losses = []
        for _ in range(epochs):
            # Epoch-indexed shuffling makes split/resumed execution reproduce a continuous run.
            rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                indices = order[start:start + self.meta["batch_size"]]
                batch, expected = x[indices], target[indices]
                hidden, logits = self.forward(batch)
                if self.meta["task"] == "classification":
                    logits -= logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    loss_sum -= float(np.log(np.maximum(probabilities[np.arange(len(indices)), expected], 1e-300)).sum())
                    delta = probabilities
                    delta[np.arange(len(indices)), expected] -= 1
                    delta /= len(indices)
                else:
                    error = logits[:, 0] - expected
                    loss_sum += float(np.sum(error ** 2))
                    delta = (2 * error / len(indices))[:, None]
                gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
                hidden_delta = (delta @ self.params["w2"].T) * (1 - hidden ** 2)
                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                self.meta["steps"] += 1
                for name, gradient in gradients.items():
                    gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                    first = self.first[name] / (1 - .9 ** self.meta["steps"])
                    second = self.second[name] / (1 - .999 ** self.meta["steps"])
                    self.params[name] -= self.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
                if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                    raise ValueError("nonfinite model state")
            self.meta["epochs"] += 1
            epoch_losses.append(loss_sum / len(x))
        self.validate()
        return epoch_losses

    def predict(self, x):
        # 根据已存元数据应用神经分数、可选记忆融合和辅助修正，推理不更新参数。
        normalized = self.normalized(x)
        _, logits = self.forward(normalized)
        if self.meta["task"] == "classification":
            head = self.meta.get("discriminant")
            if head is not None:
                direction = normalized / np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
                linear = direction @ np.asarray(head["weights"]) + np.asarray(head["bias"])
                logits = (1 - head["blend"]) * logits + head["blend"] * linear
            logits -= logits.max(axis=1, keepdims=True)
            mlp = np.exp(logits)
            mlp /= mlp.sum(axis=1, keepdims=True)
            if self.meta.get("predictor", "mlp") == "mlp":
                return apply_decisions(mlp, normalized, self.meta)
            blend = self.meta["knn_mlp_blend"]
            result = (1 - blend) * self.memory_predict(normalized) + blend * mlp
            # Preserve deterministic unique ordering even for a tied neighbor vote.
            result += 1e-12 * mlp
            return apply_decisions(result / result.sum(axis=1, keepdims=True), normalized, self.meta)
        return np.maximum(0, logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"])

    def save(self, path):
        # 把权重、Adam、scaler 和辅助记忆写进普通 NPZ；完整恢复不能只保存 params。
        self.validate()
        arrays = {**self.params, **{"adam_first_" + k: v for k, v in self.first.items()},
                  **{"adam_second_" + k: v for k, v in self.second.items()}, "mean": self.mean, "scale": self.scale,
                  "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
        if self.meta.get("predictor", "mlp") == "drift_pruned_knn":
            arrays.update(memory_x=self.memory_x, memory_y=self.memory_y, feature_mask=self.feature_mask)
        # Fixed ZIP metadata avoids changing the identity solely because of wall-clock time.
        with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        # 限制检查点大小并拒绝对象 pickle，读取数值数组后校验整份状态，避免加载任意可执行对象。
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS), *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        with zipfile.ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist()]
            actual = {name.removesuffix(".npy") for name in names if name.endswith(".npy")}
            allowed = (expected, expected | set(OPTIONAL_MEMORY_ARRAYS))
            if (len(names) != len(actual) or actual not in allowed
                    or set(names) != {name + ".npy" for name in actual}
                    or sum(e.file_size for e in archive.infolist()) > MAX_CHECKPOINT_BYTES):
                raise ValueError("invalid checkpoint archive")
        with np.load(path, allow_pickle=False) as arrays:
            model = cls.__new__(cls)
            metadata = arrays["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("invalid checkpoint metadata")
            model.meta = json.loads(str(metadata))
            model.params = {k: arrays[k].copy() for k in PARAMETERS}
            model.first = {k: arrays["adam_first_" + k].copy() for k in PARAMETERS}
            model.second = {k: arrays["adam_second_" + k].copy() for k in PARAMETERS}
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
            model.memory_x = arrays["memory_x"].copy() if "memory_x" in arrays else None
            model.memory_y = arrays["memory_y"].copy() if "memory_y" in arrays else None
            model.feature_mask = arrays["feature_mask"].copy() if "feature_mask" in arrays else None
        model.validate()
        return model


def checkpoint_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
