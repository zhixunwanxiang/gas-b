"""Keep the entire resumable predictor inside the SDK's verified checkpoint.npz."""
import base64
import hashlib
import io
import zipfile
import numpy as np
from .engine import EngineModel
from .forests import Layer
from .reference_model import Model, MAX_CHECKPOINT_BYTES

KEY = "rul_tree_bundle_v1"
MAX_LAYERS = {"B": 2, "C": 5}


def blob(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode("ascii")}


def unpack(item):
    if not isinstance(item, dict) or not isinstance(item.get("data"), str) or len(item["data"]) > 4_000_000:
        raise ValueError("Invalid embedded component")
    raw = base64.b64decode(item["data"], validate=True)
    if hashlib.sha256(raw).hexdigest() != item.get("sha256"):
        raise ValueError("Embedded component digest mismatch")
    return raw


def read_layer(raw, task):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) > 64 or not entries or sum(e.file_size for e in entries) > MAX_CHECKPOINT_BYTES
                    or len({e.filename for e in entries}) != len(entries)):
                raise ValueError("Invalid embedded update archive")
        with np.load(io.BytesIO(raw), allow_pickle=False) as source:
            state = dict(source)
        return Layer(state, task)
    except (KeyError, IndexError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("Invalid tree state") from error


class TreePredictor:
    supports_tree_bundle = True
    @classmethod
    def load(cls, path):
        model = super().load(path)
        model.tree_layers = []
        if KEY not in model.meta:
            return model
        bundle = model.meta[KEY]
        task = model.meta["profile"]
        if (not isinstance(bundle, dict) or task not in MAX_LAYERS or bundle.get("task") != task or not isinstance(bundle.get("layers"), list)
                or not 1 <= len(bundle["layers"]) <= MAX_LAYERS[task]):
            raise ValueError("Invalid tree bundle manifest")
        model.tree_layers = [read_layer(unpack(item), task) for item in bundle["layers"]]
        if task == "C":
            model.radius = bundle.get("radius")
            if type(model.radius) not in (float, int) or not np.isfinite(model.radius) or model.radius < 0:
                raise ValueError("Invalid cumulative engine radius")
            model.anchor = EngineModel.load(io.BytesIO(unpack(bundle.get("anchor"))))
            if (KEY in model.anchor.meta or model.anchor.meta.get("profile") != "C"
                    or model.anchor.meta["paths"] != model.meta["paths"] or model.anchor.meta["seed"] != model.meta["seed"]):
                raise ValueError("Incompatible cumulative engine anchor")
        return model

    def append_layer(self, raw, radius=None):
        task = self.meta["profile"]
        layer = read_layer(raw, task)
        if len(self.tree_layers) >= MAX_LAYERS[task]:
            raise ValueError("Tree layer budget exhausted")
        if not self.tree_layers:
            bundle = {"task": task, "layers": []}
            if task == "C":
                anchor = io.BytesIO()
                self.save(anchor)
                bundle["anchor"] = blob(anchor.getvalue())
                self.anchor = EngineModel.load(io.BytesIO(anchor.getvalue()))
                self.radius = float(radius)
                if not np.isfinite(self.radius) or self.radius < 0:
                    raise ValueError("Invalid first engine radius")
                bundle["radius"] = self.radius
            self.meta[KEY] = bundle
        self.meta[KEY]["layers"].append(blob(raw))
        self.tree_layers.append(layer)

    def predict(self, x):
        value = super().predict(x)
        if not self.tree_layers:
            return value
        if self.meta["profile"] == "B":
            for layer in self.tree_layers:
                value = layer.apply(x, value)
            return value
        first = self.tree_layers[0].correction(x)
        anchor = np.maximum(0, self.anchor.predict(x) + first)
        lower, upper = np.maximum(0, anchor-self.radius), anchor+self.radius
        value = np.clip(np.maximum(0, value + first), lower, upper)
        for layer in self.tree_layers[1:]:
            value = np.clip(layer.apply(x,value), lower, upper)
        return value


class BearingPredictor(TreePredictor, Model):
    pass


class EnginePredictor(TreePredictor, EngineModel):
    pass

