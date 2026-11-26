"""Train-fitted input groups; public device IDs are trace/weighting fields only."""
import hashlib
import json
import numpy as np

VERSION = "rul-groups-train-only-v1"


def fit_groups(train, dataset):
    x = train["x"][:, 21:] if dataset == "C" else train["x"]
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-7)
    z = np.clip((x - mean) / scale, -10, 10)
    # Keep small development fixtures in one supported group. Centers use train only.
    count = min(7 if dataset == "C" else 3, max(1, len(z) // 256))
    sample = z[np.linspace(0, len(z) - 1, min(4096, len(z)), dtype=int)]
    centers = sample[:1].copy()
    for _ in range(1, count):
        distance = ((sample[:, None] - centers) ** 2).sum(2).min(1)
        if distance.max() <= 1e-12:
            break
        centers = np.vstack((centers, sample[distance.argmax()]))
    for _ in range(8):
        labels = ((sample[:, None] - centers) ** 2).sum(2).argmin(1)
        centers = np.stack([sample[labels == i].mean(0) for i in np.unique(labels)])
    result = {"version": VERSION, "dataset": dataset, "train_sha256": train["sha256"],
              "mean": mean.tolist(), "scale": scale.tolist(), "centers": centers.tolist()}
    result["sha256"] = "sha256:" + hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def attach_groups(data, dataset, fitted):
    """A proxy group is explicitly distinguished from an actual public device."""
    trace = data.get("metadata", {}).get("bearing", []) if dataset == "B" else []
    usable = len(trace) == len(data["y"]) and all(isinstance(v, str) and v.strip() for v in trace)
    if dataset == "B" and usable:
        data["groups"] = np.asarray(trace)
        source = "public_bearing_metadata"
    else:
        x = data["x"][:, 21:] if dataset == "C" else data["x"]
        z = np.clip((x - np.asarray(fitted["mean"])) / np.asarray(fitted["scale"]), -10, 10)
        labels = ((z[:, None] - np.asarray(fitted["centers"])) ** 2).sum(2).argmin(1)
        data["groups"] = np.asarray([f"input-condition-{i}" for i in labels])
        source = "train_settings_conditions" if dataset == "C" else "train_input_proxy"
    data["grouping"] = {"version": VERSION, "source": source, "fit_sha256": fitted["sha256"],
                        "fit_train_sha256": fitted["train_sha256"],
                        "device_identity_available": dataset == "B" and usable,
                        "missing_device_rows": (0 if usable or dataset == "C" else len(data["y"]) -
                            sum(isinstance(v, str) and bool(v.strip()) for v in trace))}
    return data
