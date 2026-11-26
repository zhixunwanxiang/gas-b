"""Bounded, identity-bound measurements and stage evidence; no official score formula."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from .evaluation import evaluate, EVALUATION_VERSION
from ..sensor_training.io import digest, write_json


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("regression/assessment.py", "regression/evaluation.py", "regression/data.py", "regression/model.py",
             "regression/grouping.py", "sensor_training/reference/model.py")
    return "sha256:" + hashlib.sha256(b"".join((root / n).read_bytes() for n in names)).hexdigest()


class Measurements:
    def __init__(self, state, dataset, validation, round_index):
        self.path = Path(state) / "regression_measurements.json"
        self.dataset, self.data, self.round_index = dataset, validation, round_index
        self.implementation = implementation_identity()
        self.entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if not isinstance(self.entries, dict) or len(self.entries) > 16:
            raise ValueError("Invalid regression measurement cache")
        self.events = []

    def measure(self, path, model):
        binding = {"checkpoint_sha256": digest(path), "validation_sha256": self.data["sha256"],
                   "implementation_sha256": self.implementation, "dataset": self.dataset,
                   "grouping": self.data.get("grouping"), "prediction_state": "stateless-empty-v1"}
        key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        cached = self.entries.get(key)
        if cached is not None:
            if (not isinstance(cached, dict) or cached.get("binding") != binding
                    or not isinstance(cached.get("metrics"), dict)
                    or cached["metrics"].get("rows") != len(self.data["y"])
                    or cached.get("metrics_sha256") != hashlib.sha256(json.dumps(
                        cached["metrics"], sort_keys=True, allow_nan=False).encode()).hexdigest()):
                raise ValueError("Invalid cached regression measurement")
            result = deepcopy(cached["metrics"])
            source = {"source": "cached_identical_evaluation", "origin_round": cached["round"],
                      "measured_this_round": False, "identity_verified_this_round": True, "binding": binding}
        else:
            result = evaluate(model, self.data, self.dataset)
            self.entries[key] = {"binding": binding, "round": self.round_index, "metrics": result,
                "metrics_sha256": hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()}
            source = {"source": "current_round_evaluation", "origin_round": self.round_index,
                      "measured_this_round": True, "identity_verified_this_round": True, "binding": binding}
        self.events.append(source)
        return result, source

    def save(self):
        write_json(self.path, dict(list(self.entries.items())[-16:]))


class StageEvidence:
    def __init__(self, state, rnd, train, validation, before, implementation):
        self.path = Path(state) / "regression_stage_evidence.json"
        self.value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "format": "rul.stage-evidence.v1", "rounds": [], "recorded_rounds": 0,
            "relative_gain_sum": 0., "worst_relative_gain": None}
        if (not isinstance(self.value, dict) or self.value.get("format") != "rul.stage-evidence.v1"
                or not isinstance(self.value.get("rounds"), list)):
            raise ValueError("Invalid regression stage evidence")
        host_stage = rnd.task.get("stage_id")
        formal = isinstance(host_stage, str) and bool(host_stage.strip())
        key = ("host:" + host_stage if formal else "projection:" + train["sha256"] + ":" + validation["sha256"])
        self.scope = {"validation_sha256": validation["sha256"], "implementation_sha256": implementation,
                      "grouping": validation.get("grouping")}
        previous = self.value.get("stage")
        changed = not previous or previous["key"] != key
        if changed:
            target = previous.get("last_relative_gain") if previous else None
            self.value["stage"] = {"key": key, "scope_kind": "host_stage" if formal else "data_projection_event",
                "first_round": rnd.round_index, "detected_unix_seconds": time.time(),
                "reference_artifact_id": rnd.parent.artifact_id, "reference_checkpoint_sha256": rnd.parent.checkpoint_sha256,
                "reference_scope": self.scope, "reference_metrics": before,
                "recovery_target": target if target is not None and target > 0 else None,
                "recovered_round": None, "confirmed_round": None, "first_positive_round": None, "active_wall_seconds": 0.}
        self.stage = self.value["stage"]
        self.changed = changed
        self.reference = self.stage["reference_metrics"] if self.stage["reference_scope"] == self.scope else None

    def bind_reference(self, before, parent_id, restored):
        if self.reference is None:
            if self.stage["reference_artifact_id"] == parent_id:
                self.reference = before
            else:
                item = restored.get(self.stage["reference_artifact_id"], {})
                if (item.get("validation_sha256") == self.scope["validation_sha256"]
                        and item.get("measurement_source", {}).get("binding", {}).get("implementation_sha256") == self.scope["implementation_sha256"]):
                    self.reference = item.get("metrics")
            if self.reference is not None:
                self.stage.update(reference_scope=self.scope, reference_metrics=self.reference)
        return self.reference

    def record(self, rnd, selected, metrics, resources, action):
        reference = self.reference
        gain = ((reference["rmse"] - metrics["rmse"]) / max(reference["rmse"], 1e-12)) if reference else None
        self.stage["last_relative_gain"] = gain
        if gain is not None and gain > 0 and self.stage["first_positive_round"] is None:
            self.stage["first_positive_round"] = rnd.round_index
        self.stage["active_wall_seconds"] += resources.get("wall_seconds", 0.)
        target = self.stage["recovery_target"]
        recovered = gain is not None and target is not None and gain >= target - 1e-12
        if recovered and self.stage["recovered_round"] is None:
            self.stage["recovered_round"] = rnd.round_index
        elif recovered and rnd.round_index > self.stage["recovered_round"]:
            self.stage["confirmed_round"] = rnd.round_index
        elif not recovered:
            self.stage.update(recovered_round=None, confirmed_round=None)
        row = {"round": rnd.round_index, "selected_artifact_id": selected, "selection_action": action,
               "stage_key": self.stage["key"], "scope_kind": self.stage["scope_kind"],
               "metrics": metrics, "relative_gain": gain, "resources": resources,
               "stage_reference_artifact_id": self.stage["reference_artifact_id"],
               "recovery_target": target, "recovered_round": self.stage["recovered_round"],
               "first_positive_round": self.stage["first_positive_round"],
               "confirmed_round": self.stage["confirmed_round"],
               "rounds_since_detection": rnd.round_index - self.stage["first_round"],
               "elapsed_since_detection_seconds": time.time() - self.stage["detected_unix_seconds"],
               "active_wall_seconds": self.stage["active_wall_seconds"],
               "recovery_definition": "prior_stage_positive_relative_gain_restored_and_next_checkpoint_confirmed"}
        self.value["rounds"].append(row)
        self.value["rounds"] = self.value["rounds"][-128:]
        self.value["recorded_rounds"] += 1
        if gain is not None:
            self.value["relative_gain_sum"] += gain
            self.value["comparable_rounds"] = self.value.get("comparable_rounds", 0) + 1
            worst = self.value["worst_relative_gain"]
            self.value["worst_relative_gain"] = gain if worst is None else min(worst, gain)
            self.value["mean_relative_gain"] = self.value["relative_gain_sum"] / self.value["comparable_rounds"]
        self.value["official_score_verified"] = False
        self.value["finish_acknowledged"] = None
        self.value["resource_cutoff"] = "before_save_state_and_finish; participant process only"
        write_json(self.path, self.value)
        return row
