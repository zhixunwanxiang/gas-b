"""Bounded controller policy and comparable public-validation promotion gates."""
from copy import deepcopy
import hashlib
import json
import math


def validate_controller(config):
    for key, low, high in (("json_corrections", 0, 2), ("max_history_restores", 0, 2),
                           ("max_repeat_rejections", 1, 8), ("minimum_class_support", 1, 1000000)):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("Controller limit outside supported bounds: " + key)
    for key in ("minimum_macro_f1_gain", "maximum_class_recall_drop"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid promotion limit: " + key)
    thresholds = config["drift_thresholds"]
    if type(thresholds.get("minimum_rows")) is not int or thresholds["minimum_rows"] < 1:
        raise ValueError("Invalid monitoring sample floor")
    for key in ("psi", "binned_ks", "absolute_mean_z"):
        value = thresholds.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid drift threshold: " + key)


def fingerprint(parent_sha256, train_sha256, validation_sha256, spec):
    return hashlib.sha256(json.dumps([parent_sha256, train_sha256, validation_sha256, spec],
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def initial_policy():
    return {"format": "gas-a.strategy.v1", "version": 1, "revision": 1, "preferred_experiment": None,
            "attempts": [], "artifact_policies": {}, "last_measurement": None}


def load_policy(value, menu):
    if value is None:
        return initial_policy()
    if (not isinstance(value, dict) or value.get("format") != "gas-a.strategy.v1"
            or type(value.get("version")) is not int or value["version"] < 1
            or value.get("preferred_experiment") not in (None, *menu)
            or not isinstance(value.get("attempts"), list)
            or not isinstance(value.get("artifact_policies"), dict)):
        raise ValueError("Invalid bounded strategy state")
    for snapshot in value["artifact_policies"].values():
        if (not isinstance(snapshot, dict) or set(snapshot) != {"version", "preferred_experiment"}
                or type(snapshot["version"]) is not int or snapshot["version"] < 1
                or snapshot["preferred_experiment"] not in (None, *menu)):
            raise ValueError("Invalid artifact-paired strategy")
    versions = [value["version"], *(item["version"] for item in value["artifact_policies"].values())]
    if type(value.get("revision")) is not int or value["revision"] < max(versions):
        raise ValueError("Invalid strategy revision counter")
    return value


def available_experiments(policy, menu, rnd, train, validation, config):
    allowed, blocked = {}, []
    for name, spec in menu.items():
        identity = fingerprint(rnd.parent.checkpoint_sha256, train["sha256"], validation["sha256"], spec)
        rejected = sum(a["fingerprint"] == identity and not a["adopted"] for a in policy["attempts"])
        if rejected >= config["max_repeat_rejections"]:
            blocked.append(name)
        else:
            allowed[name] = spec
    return allowed, blocked


def promotion_gate(before, after, config):
    gain = after["macro_f1"] - before["macro_f1"]
    losses = {name: item["recall"] - after["per_class"][name]["recall"]
              for name, item in before["per_class"].items()
              if item["support"] >= config["minimum_class_support"]
              and item["recall"] - after["per_class"][name]["recall"] > config["maximum_class_recall_drop"] + 1e-12}
    return {"eligible": bool(gain > 1e-12 and gain >= config["minimum_macro_f1_gain"] and not losses),
            "macro_f1_gain": gain, "class_recall_losses": losses}


def advance_policy(policy, *, selected, parent_id, candidate_id, plan, attempt_key, gain, validation_sha256, selected_f1):
    result = deepcopy(policy)
    parent_snapshot = {k: policy[k] for k in ("version", "preferred_experiment")}
    result["artifact_policies"].setdefault(parent_id, parent_snapshot)
    if candidate_id:
        adopted = selected == candidate_id
        result["attempts"].append({"fingerprint": attempt_key, "experiment": plan["experiment"],
                                   "adopted": adopted, "macro_f1_gain": gain})
        result["attempts"] = result["attempts"][-128:]
        if adopted:
            result["revision"] += 1
            result["version"] = result["revision"]
            result["preferred_experiment"] = plan["experiment"]
    if selected not in (parent_id, candidate_id):
        # The model artifact is authoritative. Restore its paired strategy when
        # known; an external historical artifact gets the conservative default.
        saved = result["artifact_policies"].get(selected, {"version": 1, "preferred_experiment": None})
        result.update(saved)
    result["artifact_policies"][selected] = {k: result[k] for k in ("version", "preferred_experiment")}
    result["artifact_policies"] = dict(list(result["artifact_policies"].items())[-64:])
    result["last_measurement"] = {"validation_sha256": validation_sha256, "selected_macro_f1": selected_f1}
    return result
