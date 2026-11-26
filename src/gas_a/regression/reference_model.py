"""Xiaoyang regression checkpoint adapter, isolated from the unchanged task-A model."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile
import zlib

_THREADS = int(os.environ.get("BASELINE_CPU_THREADS", "1"))
if not 1 <= _THREADS <= 8:
    raise ValueError("BASELINE_CPU_THREADS must be between 1 and 8")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(_THREADS)
import numpy as np

PARAMETERS = ("w1", "b1", "w2", "b2")
FORMAT = "contest.cpu-mlp.v1"
COMPLETE_FORMAT = "contest.cpu-rul-xiaoyang.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


class Model:
    def __init__(self, task, paths, classes, x, y, *, hidden=32, seed=42, learning_rate=.003, batch_size=128):
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
        m = self.meta
        if m.get("format") not in (FORMAT, COMPLETE_FORMAT) or m.get("device") != "cpu" or m.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        if m.get("format") == COMPLETE_FORMAT and not any(k in m for k in ("rul_tree_bundle_v1", "rul_residual_v1")):
            raise ValueError("Incomplete xiaoyang regression predictor")
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
            value = m.get(name)
            try:
                valid = not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)
            except OverflowError:
                # JSON 整数可以远大于浮点数范围，isfinite 转换时也可能溢出。
                # 统一为本地检查点格式错误：坏历史可排除，实际父模型仍会停止。
                valid = False
            if not valid:
                raise ValueError("invalid checkpoint numeric state: " + name)
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

    def normalized(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        return np.clip((x - self.mean) / self.scale, -10, 10)

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def fit(self, x, y, epochs):
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
        _, logits = self.forward(self.normalized(x))
        if self.meta["task"] == "classification":
            logits -= logits.max(axis=1, keepdims=True)
            result = np.exp(logits)
            return result / result.sum(axis=1, keepdims=True)
        return np.maximum(0, logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"])

    def save(self, path):
        # Source-package legacy composite checkpoints remain readable. New exports
        # use an explicit format so older plain-MLP loaders cannot drop components.
        if any(k in self.meta for k in ("rul_tree_bundle_v1", "rul_residual_v1")):
            self.meta["format"] = COMPLETE_FORMAT
        self.validate()
        arrays = {**self.params, **{"adam_first_" + k: v for k, v in self.first.items()},
                  **{"adam_second_" + k: v for k, v in self.second.items()}, "mean": self.mean, "scale": self.scale,
                  "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
        # Fixed ZIP metadata avoids changing the identity solely because of wall-clock time.
        from contextlib import nullcontext
        target = nullcontext(path) if isinstance(path, io.BytesIO) else Path(path).open("xb")
        with target as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        if isinstance(path, io.BytesIO):
            if path.getbuffer().nbytes > MAX_CHECKPOINT_BYTES:
                raise ValueError("Embedded checkpoint too large")
        else:
            path = Path(path)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS), *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if (len(names) != len(expected) or set(names) != {name + ".npy" for name in expected}
                        or sum(e.file_size for e in entries) > MAX_CHECKPOINT_BYTES
                        or any(e.flag_bits & 1 for e in entries)):
                    raise ValueError("invalid checkpoint archive")
            if isinstance(path, io.BytesIO):
                path.seek(0)
            with np.load(path, allow_pickle=False) as arrays:
                def array(name):
                    # ZIP 清单正确时，内部条目仍可能不是 NPY；NumPy 此时返回 bytes。
                    # 在访问 ndim/copy 前检查类型，格式错误仍统一走 ValueError。
                    value = arrays[name]
                    if not isinstance(value, np.ndarray):
                        raise ValueError("invalid checkpoint array: " + name)
                    return value
                model = cls.__new__(cls)
                metadata = array("metadata")
                if metadata.ndim != 0 or metadata.dtype.kind != "U":
                    raise ValueError("invalid checkpoint metadata")
                model.meta = json.loads(str(metadata))
                # 能解析成 JSON 不代表元数据可用：数组、null、字符串等没有 .get。
                # 在读取张量及调用 validate 前明确拒绝，避免泄漏 AttributeError。
                if not isinstance(model.meta, dict):
                    raise ValueError("checkpoint metadata must be a JSON object")
                if "rul_tree_bundle_v1" in model.meta and not getattr(cls, "supports_tree_bundle", False):
                    raise ValueError("This checkpoint requires the complete regression tree predictor")
                if "rul_residual_v1" in model.meta and not getattr(cls, "supports_engine_residual", False):
                    raise ValueError("This checkpoint requires the engine residual predictor")
                model.params = {k: array(k).copy() for k in PARAMETERS}
                model.first = {k: array("adam_first_" + k).copy() for k in PARAMETERS}
                model.second = {k: array("adam_second_" + k).copy() for k in PARAMETERS}
                model.mean, model.scale = array("mean").copy(), array("scale").copy()
        except (zipfile.BadZipFile, EOFError, zlib.error, NotImplementedError) as error:
            # 只统一本地文件格式错误，不兜底捕获任意程序异常或 SDK/RPC 错误。
            # 历史复评可据此排除旧候选；实际父模型和预测入口仍会抛错并终止。
            raise ValueError("invalid checkpoint archive") from error
        model.validate()
        return model


def checkpoint_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
