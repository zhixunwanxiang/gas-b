"""Chronological local closed-loop experiment; no external services or future labels.

The policy is written before any replay labels are revealed. Each run owns its
models, registry, frozen predictions, timing log and source snapshot.
"""
import argparse
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for key in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np
from gas_a.model import Model, checkpoint_digest
from gas_a.router import FourSensorRouter, confusion, macro
from gas_a.signals import dat_inputs, reveal_labels
from gas_a.sensor_training.data import load_pair, load_model, evaluate
from gas_a.sensor_training.io import CLASSES
from gas_a.sensor_training.optimize import train_candidate
from rebuild_models import train_home

TZ = timezone(timedelta(hours=8))
POLICY = {
    "controller": "deterministic_local_rules_v1", "external_ai_calls": 0,
    "monitor": {"psi": .2, "ks": .2, "mean_z": 1., "minimum_rows": 30},
    "training_data": "public batch1 only; batch2 validation selects epochs",
    "selection_data": "completed windows only; last three weights 0.6/0.3/0.1",
    "cold_start": "activate reconstructed released router after batch3 if fixed F1 gain >=.002, accuracy drop <=.02, acetaldehyde recall >0",
    "gate": {"fixed_f1_floor": "reference.npz", "max_fixed_f1_drop": .02,
             "max_fixed_accuracy_drop": .02, "min_current_f1_gain": .002,
             "max_current_accuracy_drop": .02, "recent_switch_margin": .005,
             "max_target_recall_drop": .02, "minimum_class_support": 20},
    "rollback": "evaluate previously deployed versions on completed last three windows; margin .005 and fixed validation gate",
    "adaptive_recipe": {"learning_rate": .0005, "epochs": 8, "minimum_epochs": 4,
                        "balance": .35, "weak_boost": 1.5, "l2": .001,
                        "anchor_l2": .01, "batch_size": 128, "class_boost": 1.5},
    "terminal_window": "monitor, score, diagnose and recovery check; no training or future deployment",
    "performance_claim_scope": "historical replay; previously studied data; no untouched future holdout",
}


def now():
    return datetime.now(TZ).isoformat(timespec="microseconds")


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, predicted):
    m = confusion(y, predicted)
    return {"rows": int(m.sum()), "macro_f1": macro(m), "accuracy": float(m.trace() / m.sum()),
            "confusion_matrix": m.tolist(), "recall": (m.diagonal() / np.maximum(1, m.sum(1))).tolist(),
            "support": m.sum(1).tolist()}


def monitor(reference, x):
    output = []
    for k in range(128):
        a, b = reference[:, k], x[:, k]
        bins = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        edges = np.r_[-np.inf, bins[1:-1], np.inf]
        p = np.maximum(np.histogram(a, edges)[0] / len(a), 1e-6)
        q = np.maximum(np.histogram(b, edges)[0] / len(b), 1e-6)
        psi = float(np.sum((q - p) * np.log(q / p)))
        points = np.sort(np.r_[a, b])
        ks = float(np.max(np.abs(np.searchsorted(np.sort(a), points, side="right") / len(a)
                                  - np.searchsorted(np.sort(b), points, side="right") / len(b))))
        mean = float(abs(a.mean() - b.mean()) / max(float(a.std()), 1e-12))
        output.append({"feature": k + 1, "sensor": k // 8 + 1, "psi": psi, "ks": ks,
                       "mean_z": mean, "flagged": psi > .2 or ks > .2 or mean > 1})
    sensor = [float(np.mean([r["psi"] for r in output[s*8:(s+1)*8]])) for s in range(16)]
    return {"reference_rows": len(reference), "rows": len(x), "flagged_features": sum(r["flagged"] for r in output),
            "drift_detected": len(x) >= 30 and any(r["flagged"] for r in output),
            "sensor_psi": sensor, "top_features": sorted(output, key=lambda r: -r["psi"])[:8],
            "features": output, "thresholds": POLICY["monitor"]}


class ParentEngine:
    def __init__(self):
        self.model = Model.load(ROOT / "models/start.npz")
        self.waiting = None
        self.batch_count = 0

    def predict_batch(self, x):
        if self.waiting is not None:
            raise ValueError("Previous batch awaits labels")
        self.waiting = len(x)
        return self.model.predict(x).argmax(1)

    def release_labels(self, y):
        if self.waiting != len(y):
            raise ValueError("Wrong feedback length")
        self.waiting = None
        self.batch_count += 1


def router_state(engine):
    if isinstance(engine, ParentEngine):
        return {"kind": "parent", "batch_count": engine.batch_count}
    if engine.waiting is not None:
        raise ValueError("Only committed, feedback-complete states can be deployed")
    buf, shift, count = engine.state
    return {"kind": "router", "batch_count": engine.batch_count, "drift_buffer": list(buf),
            "drift_shift": shift, "drift_count": count, "flatline_buffer": list(engine.detector.window),
            "history": list(engine.history), "weights": engine.weights,
            "active_pairs": sorted(engine.active_pairs)}


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


class Experiment:
    def __init__(self, out, dataset, previous):
        self.out, self.dataset, self.previous = out, dataset, previous
        self.cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.events, self.spans, self.windows, self.candidates, self.deployments = [], [], [], [], []
        self.data, self.registry = [], {}
        self.fixed = {}
        self.active_id, self.active = "start_only", ParentEngine()
        self.registry[self.active_id] = {"kind": "parent", "bundle": None, "config": None}
        self.control = FourSensorRouter(ROOT / "models", self.cfg)
        self.train, self.validation = load_pair(ROOT / "data/public/train.jsonl", ROOT / "data/public/validation.jsonl")
        self.reference_fixed = metrics(self.validation["y"], Model.load(ROOT / "models/reference.npz").predict(self.validation["x"]).argmax(1))

    def event(self, event, **fields):
        obj = {"index": len(self.events), "at": now(), "event": event, **fields}
        self.events.append(obj)
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(obj, ensure_ascii=False) + "\n")

    @contextmanager
    def span(self, phase, batch=None, **fields):
        start, clock, cpu = now(), time.perf_counter(), time.process_time()
        status = "ok"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            self.spans.append({"id": len(self.spans), "phase": phase, "batch": batch,
                               "started_at": start, "ended_at": now(),
                               "wall_seconds": time.perf_counter() - clock,
                               "cpu_seconds": time.process_time() - cpu, "status": status, **fields})
            write(self.out / "phase-timing.json", self.spans)

    def engine(self, version):
        spec = self.registry[version]
        return ParentEngine() if spec["kind"] == "parent" else FourSensorRouter(spec["bundle"], spec["config"])

    def replay(self, version):
        engine, scored, predictions = self.engine(version), [], []
        for record in self.data:
            pred = engine.predict_batch(record["x"])
            scored.append(metrics(record["y"], pred))
            predictions.append(pred)
            engine.release_labels(record["y"])
        return engine, scored, predictions

    def fixed_metric(self, version):
        if version not in self.fixed:
            engine = self.engine(version)
            if isinstance(engine, FourSensorRouter):
                engine.batch_count = 1  # fixed independent validation state; never mutate the operational engine
            self.fixed[version] = metrics(self.validation["y"], engine.predict_batch(self.validation["x"]))
        return self.fixed[version]

    def recent(self, scored):
        scores = [s["macro_f1"] for s in scored[-3:]][::-1]
        weights = np.asarray([.6, .3, .1][:len(scores)])
        return float(np.dot(scores, weights) / weights.sum())

    def register_bundle(self, version, directory, origin):
        cfg = deepcopy(self.cfg)
        cfg["model_sha256"] = {name: checkpoint_digest(directory / f"{name}.npz") for name in cfg["model_sha256"]}
        write(directory / "runtime-config.json", cfg)
        self.registry[version] = {"kind": "router", "bundle": str(directory.resolve()), "config": cfg,
                                  "origin": origin, "created_at": now()}
        write(self.out / "registry.json", self.registry)

    def train_bundle(self, batch, diagnostic, initial=False):
        version = "released_retrained" if initial else f"batch{batch:02d}_focus_{diagnostic['weakest_class'].lower()}"
        directory = self.out / "versions" / version
        directory.mkdir(parents=True)
        parent_directory = ROOT / "models" if initial or self.active_id == "start_only" else Path(self.registry[self.active_id]["bundle"])
        for name in self.cfg["model_sha256"]:
            if name not in ("four_sensor", "five_sensor"):
                shutil.copyfile(parent_directory / f"{name}.npz", directory / f"{name}.npz")
        if initial:
            # train_home uses exclusive saves; keep the copied initial components only.
            for name in ("home_trunk", "home_router"):
                (directory / f"{name}.npz").unlink()
            train_home(Model.load(directory / "start.npz"), self.train, self.cfg["training"]["home"], directory)
        limits = dict(self.cfg["training"]["limits"], checkpoint_reference=evaluate(load_model(ROOT / "models/reference.npz"), self.validation),
                      checkpoint_reference_sha256=checkpoint_digest(ROOT / "models/reference.npz"))
        components = []
        for role in ("four_sensor", "five_sensor"):
            if initial:
                spec = deepcopy(self.cfg["training"]["sensors"][role])
                parent = ROOT / "models/reference.npz"
            else:
                recipe = POLICY["adaptive_recipe"]
                spec = {k: v for k, v in recipe.items() if k != "class_boost"}
                spec.update(name=f"focus_{diagnostic['weakest_class'].lower()}",
                            kept_sensors=self.cfg["training"]["sensors"][role]["kept_sensors"],
                            class_boosts={diagnostic["weakest_class"]: recipe["class_boost"]})
                parent = parent_directory / f"{role}.npz"
            trial = directory / "training" / role
            report = train_candidate(parent, self.train, self.validation, trial, spec,
                                     {"weakest_class": diagnostic["weakest_class"]}, limits)
            shutil.copyfile(trial / "checkpoint.npz", directory / f"{role}.npz")
            components.append({"role": role, "parent_sha256": checkpoint_digest(parent), "parameters": spec,
                               "selected_epoch": report["final_epochs"], "executed_epochs": report["executed_epochs"] - report["initial_epochs"],
                               "validation_f1": report["validation_after"]["macro_f1"],
                               "checkpoint_sha256": checkpoint_digest(directory / f"{role}.npz")})
        self.register_bundle(version, directory, "released_recipe_retrained" if initial else "diagnostic_rule_retrained")
        return version, components

    def gate(self, version, cold=False):
        engine, scored, _ = self.replay(version)
        _, incumbent, _ = self.replay(self.active_id)
        fixed, old_fixed = self.fixed_metric(version), self.fixed_metric(self.active_id)
        current, old_current = scored[-1], incumbent[-1]
        reasons, rules = [], POLICY["gate"]
        if fixed["macro_f1"] < self.reference_fixed["macro_f1"] - 1e-12:
            reasons.append("below_reference_validation")
        if fixed["macro_f1"] < old_fixed["macro_f1"] - rules["max_fixed_f1_drop"] - 1e-12:
            reasons.append("fixed_f1_regression")
        if fixed["accuracy"] < old_fixed["accuracy"] - rules["max_fixed_accuracy_drop"] - 1e-12:
            reasons.append("fixed_accuracy_regression")
        if fixed["recall"][3] <= 0:
            reasons.append("zero_fixed_acetaldehyde_recall")
        if not cold and fixed["recall"][3] < old_fixed["recall"][3] - .02 - 1e-12:
            reasons.append("fixed_acetaldehyde_regression")
        if cold:
            if fixed["macro_f1"] < old_fixed["macro_f1"] + .002 - 1e-12:
                reasons.append("cold_start_validation_gain_below_minimum")
        else:
            if current["macro_f1"] < old_current["macro_f1"] + .002 - 1e-12:
                reasons.append("current_gain_below_minimum")
            if self.recent(scored) < self.recent(incumbent) + .005 - 1e-12:
                reasons.append("recent_gain_below_margin")
            if current["accuracy"] < old_current["accuracy"] - .02 - 1e-12:
                reasons.append("current_accuracy_regression")
            if current["support"][3] >= 20 and current["recall"][3] < old_current["recall"][3] - .02 - 1e-12:
                reasons.append("current_acetaldehyde_regression")
        record = {"version": version, "incumbent": self.active_id, "accepted": not reasons,
                  "reasons": reasons or ["all_gates_passed"], "cold_start": cold,
                  "fixed": fixed, "incumbent_fixed": old_fixed, "current_shadow": current,
                  "incumbent_current_shadow": old_current, "recent_f1": self.recent(scored),
                  "incumbent_recent_f1": self.recent(incumbent), "selection_cutoff_batch": self.data[-1]["batch"]}
        return record, engine

    def deploy(self, version, engine, batch, reason):
        prior = self.active_id
        state_path = self.out / "deployment" / f"after-batch{batch:02d}.json"
        write(state_path, {"version": version, "state": serializable(router_state(engine)),
                           "effective_from_batch": batch + 1})
        pointer = {"version": version, "state_path": str(state_path.resolve()), "state_sha256": digest(state_path),
                   "registry_entry_sha256": hashlib.sha256(json.dumps(self.registry[version], sort_keys=True).encode()).hexdigest(),
                   "effective_from_batch": batch + 1, "updated_at": now()}
        write(self.out / "deployment/active.json", pointer)
        self.active_id, self.active = version, engine
        event = {"batch": batch, "from": prior, "to": version, "reason": reason,
                 "effective_from_batch": batch + 1, "at": now()}
        self.deployments.append(event)
        self.event("deployment_switched", **event)

    def integrity(self):
        pointer_file = self.out / "deployment/active.json"
        if not pointer_file.exists():
            return
        pointer = json.loads(pointer_file.read_text(encoding="utf-8"))
        if pointer["version"] != self.active_id or digest(pointer["state_path"]) != pointer["state_sha256"]:
            raise ValueError("Active deployment state integrity failed")
        entry = self.registry[self.active_id]
        if entry["kind"] == "router":
            for name, signature in entry["config"]["model_sha256"].items():
                if checkpoint_digest(Path(entry["bundle"]) / f"{name}.npz") != signature:
                    raise ValueError("Active model integrity failed")

    def rollback_check(self, batch):
        _, active_scores, _ = self.replay(self.active_id)
        checked, winner, restored = [], self.active_id, None
        best = self.recent(active_scores)
        old_fixed = self.fixed_metric(self.active_id)
        previous_versions = list(dict.fromkeys([d["from"] for d in self.deployments]))
        for version in previous_versions:
            engine, scored, _ = self.replay(version)
            fixed = self.fixed_metric(version)
            eligible = self.rollback_quality(fixed, old_fixed)
            recent = self.recent(scored)
            checked.append({"version": version, "recent_f1": recent, "eligible": eligible})
            if eligible and recent >= best + .005 - 1e-12:
                winner, restored, best = version, engine, recent
        result = {"active": self.active_id, "active_recent_f1": self.recent(active_scores),
                  "checked": checked, "selected": winner, "switched": winner != self.active_id}
        if restored is not None and batch < 10:
            with self.span("rollback_switch", batch):
                self.deploy(winner, restored, batch, "natural_performance_rollback")
        return result

    def rollback_quality(self, fixed, old_fixed):
        return (fixed["macro_f1"] >= self.reference_fixed["macro_f1"]
                and fixed["macro_f1"] >= old_fixed["macro_f1"] - .02
                and fixed["accuracy"] >= old_fixed["accuracy"] - .02
                and fixed["recall"][3] >= old_fixed["recall"][3] - .02)

    def diagnose(self, y, pred, drift):
        score = metrics(y, pred)
        eligible = [k for k in range(6) if score["support"][k] >= 20]
        weak = min(eligible, key=lambda k: (score["recall"][k], k))
        matrix = np.asarray(score["confusion_matrix"])
        errors = matrix.copy()
        np.fill_diagonal(errors, 0)
        a, b = np.unravel_index(errors.argmax(), errors.shape)
        ranked = np.argsort(-np.asarray(drift["sensor_psi"]))[:3] + 1
        return {"weakest_class": CLASSES[weak], "weakest_recall": score["recall"][weak],
                "top_confusion": {"true": CLASSES[a], "predicted": CLASSES[b], "rows": int(errors[a, b])},
                "top_drift_sensors": ranked.tolist(), "class_counts": score["support"],
                "hypothesis": "feature distribution shift and class-specific confusion; physical fault causality unproven",
                "decision": "keep four/five sensor structure, strengthen weakest supported class on public training only"}

    def run(self):
        self.event("policy_locked", policy_sha256=digest(self.out / "policy.json"))
        for batch in range(3, 11):
            window = self.out / "windows" / f"batch{batch:02d}"
            window.mkdir(parents=True)
            with self.span("integrity_check", batch):
                self.integrity()
            with self.span("input_read", batch):
                features = dat_inputs(self.dataset / f"batch{batch}.dat", self.control.parent.meta["paths"])
                if features["sha256"] != self.cfg["acceptance"][str(batch)]["data_sha256"]:
                    raise ValueError("Dataset differs from matched comparison population")
            with self.span("drift_monitor", batch):
                drift = monitor(self.train["x"], features["x"])
                write(window / "drift.json", drift)
                self.event("drift_monitored_without_labels", batch=batch, flagged_features=drift["flagged_features"])
            prediction_version = self.active_id
            with self.span("predict_commit", batch):
                predicted = self.active.predict_batch(features["x"])
                control = self.control.predict_batch(features["x"])
                for name, values in (("predictions", predicted), ("packaged-control", control)):
                    with (window / f"{name}.npy").open("xb") as stream:
                        np.save(stream, values.astype(np.uint8), allow_pickle=False)
                predicted = np.load(window / "predictions.npy", allow_pickle=False)
                signature = hashlib.sha256(control.astype(np.uint8).tobytes()).hexdigest()
                if signature != self.cfg["acceptance"][str(batch)]["prediction_sha256"]:
                    raise ValueError("Packaged policy control did not reproduce")
                commit = {"batch": batch, "version": prediction_version, "rows": len(predicted),
                          "source_sha256": features["sha256"], "file_sha256": digest(window / "predictions.npy"),
                          "committed_at": now(), "labels_revealed": False}
                write(window / "prediction_commit.json", commit)
                self.event("prediction_committed_before_labels", batch=batch, **{"commit": commit})
            with self.span("label_reveal_score", batch):
                labels = reveal_labels(self.dataset / f"batch{batch}.dat", features)["y"]
                score = metrics(labels, predicted)
                self.event("labels_revealed", batch=batch, rows=len(labels))
                self.active.release_labels(labels)
                self.control.release_labels(labels)
                self.data.append({"batch": batch, "x": features["x"], "y": labels})
            with self.span("diagnosis", batch):
                diagnosis = self.diagnose(labels, predicted, drift)
                write(window / "diagnosis.json", diagnosis)
                self.event("diagnosis_completed", batch=batch, weakest_class=diagnosis["weakest_class"])
            with self.span("rollback_check", batch):
                recovery = self.rollback_check(batch)
            candidate = None
            if batch < 10 and drift["drift_detected"]:
                with self.span("experiment_training", batch):
                    version, components = self.train_bundle(batch, diagnosis, initial=batch == 3)
                    self.event("candidate_trained", batch=batch, version=version, training_rows=len(self.train["y"]),
                               stream_rows_used_for_gradient=0, components=components)
                with self.span("evaluation_gate", batch):
                    candidate, candidate_engine = self.gate(version, cold=self.active_id == "start_only")
                    candidate.update(batch=batch, components=components, selected=False)
                    self.event("candidate_evaluated", batch=batch, version=version, accepted=candidate["accepted"], reasons=candidate["reasons"])
                if candidate["accepted"]:
                    with self.span("deployment_switch", batch):
                        self.deploy(version, candidate_engine, batch, "cold_start_activation" if candidate["cold_start"] else "candidate_passed_all_gates")
                        candidate["selected"] = True
                self.candidates.append(candidate)
                write(window / "candidate.json", candidate)
            elif batch == 10:
                self.event("terminal_training_skipped", batch=batch, reason="no future batch within evaluation")
            with self.span("state_checkpoint", batch):
                state = serializable(router_state(self.active))
                write(window / "next-state.json", {"version": self.active_id, "state": state})
                row = {"batch": batch, "rows": len(labels), "prediction_version": prediction_version,
                       "next_version": self.active_id, "operational": score, "packaged_control": metrics(labels, control),
                       "drift": {k:v for k,v in drift.items() if k != "features"}, "diagnosis": diagnosis,
                       "rollback": recovery, "candidate": candidate, "route_state": {
                           "active_pairs": sorted(getattr(self.active, "active_pairs", [])),
                           "nondefault_weights": int(np.count_nonzero(getattr(self.active, "weights", np.full(36,.75)) != .75)),
                           "feedback_batches": len(getattr(self.active, "history", []))}}
                self.windows.append(row)
                write(window / "result.json", row)
                write(self.out / "history.json", self.windows)
            print(f"batch{batch:02d} n={len(labels)} F1={score['macro_f1']:.6f} active={prediction_version} next={self.active_id}", flush=True)
        self.drills()
        return self.analyze()

    def drills(self):
        directory = self.out / "rollback_drills"
        directory.mkdir()
        untouched = digest(self.out / "deployment/active.json")
        with self.span("isolated_integrity_rollback", None, scope="drill"):
            healthy = directory / "healthy-state.json"
            write(healthy, {"version": self.active_id, "state": serializable(router_state(self.active))})
            broken = directory / "broken-state.json"
            shutil.copyfile(healthy, broken)
            expected = digest(broken)
            with broken.open("ab") as stream:
                stream.write(b"CORRUPTION_INJECTION")
            pointer = {"state_path": str(broken), "expected_sha256": expected, "version": "injected_corruption"}
            write(directory / "integrity-active.json", pointer)
            detected = digest(broken) != pointer["expected_sha256"]
            if detected:
                write(directory / "integrity-active.json", {"state_path": str(healthy), "expected_sha256": digest(healthy), "version": self.active_id})
            restored = json.loads((directory / "integrity-active.json").read_text(encoding="utf-8"))
            restored_engine = restore_engine(self.registry[restored["version"]], json.loads(healthy.read_text(encoding="utf-8"))["state"])
            probe = self.data[-1]["x"][:32]
            expected_pred = deepcopy(self.active).predict_batch(probe)
            recovered_pred = restored_engine.predict_batch(probe)
            integrity = {"injected": True, "detected": detected, "restored_version": restored["version"],
                         "restored_predictions_equal": bool(np.array_equal(expected_pred, recovered_pred)),
                         "physical_pointer_switched": restored["state_path"] == str(healthy), "scope": "isolated copy only"}
            write(directory / "integrity-result.json", integrity)
        with self.span("isolated_performance_rollback", None, scope="drill"):
            # Use real observed scores and a controlled degraded deployment, not invented metrics.
            _, good_scores, _ = self.replay(self.active_id)
            _, poor_scores, _ = self.replay("start_only")
            good, poor = self.recent(good_scores), self.recent(poor_scores)
            pointer_path = directory / "performance-active.json"
            write(pointer_path, {"version": "start_only", "injected_degraded_deployment": True})
            quality_passed = self.rollback_quality(self.fixed_metric(self.active_id), self.fixed_metric("start_only"))
            eligible = good >= poor + .005 and quality_passed
            if eligible:
                write(pointer_path, {"version": self.active_id, "reason": "observed_recent_margin_passed"})
            restored = json.loads(pointer_path.read_text(encoding="utf-8"))["version"]
            replayed, _, _ = self.replay(restored)
            equality = np.array_equal(replayed.predict_batch(self.data[-1]["x"][:32]), deepcopy(self.active).predict_batch(self.data[-1]["x"][:32]))
            performance = {"injected_degraded_version": "start_only", "recovery_version": self.active_id,
                           "observed_degraded_recent_f1": poor, "observed_recovery_recent_f1": good,
                           "required_margin": .005, "fixed_validation_gate_passed": quality_passed, "threshold_triggered": eligible,
                           "physical_pointer_switched": restored == self.active_id,
                           "restored_predictions_equal": bool(equality), "scope": "isolated injected deployment; not natural rollback"}
            write(directory / "performance-result.json", performance)
        self.drill_results = {"integrity": integrity, "performance": performance,
                              "operational_pointer_unchanged": digest(self.out / "deployment/active.json") == untouched}
        write(directory / "summary.json", self.drill_results)

    def analyze(self):
        ys, active, control, previous = [], [], [], []
        prior_sha = []
        for record in self.data:
            batch = record["batch"]
            window = self.out / "windows" / f"batch{batch:02d}"
            old_path = self.previous / "windows" / f"batch{batch:02d}" / "predictions.npz"
            old_commit = json.loads((old_path.parent / "prediction_commit.json").read_text(encoding="utf-8"))
            if old_commit["source_sha256"] != self.cfg["acceptance"][str(batch)]["data_sha256"]:
                raise ValueError("Previous population mismatch")
            with np.load(old_path, allow_pickle=False) as archive:
                old = archive["active"].argmax(1)
            y = record["y"]
            current = np.load(window / "predictions.npy", allow_pickle=False)
            packaged = np.load(window / "packaged-control.npy", allow_pickle=False)
            assert len(old) == len(y)
            ys.append(y); active.append(current); control.append(packaged); previous.append(old)
            self.windows[batch-3]["previous_pdf"] = metrics(y, old)
            prior_sha.append({"batch": batch, "file_sha256": digest(old_path)})
        y, pred, packaged, old = map(np.concatenate, (ys, active, control, previous))
        current_metrics, packaged_metrics, prior_metrics = metrics(y, pred), metrics(y, packaged), metrics(y, old)
        original_summary = json.loads((self.previous / "summary.json").read_text(encoding="utf-8"))
        assert abs(prior_metrics["macro_f1"] - original_summary["aggregate_frozen_operational"]["operational"]["macro_f1"]) < 1e-12
        rng = np.random.default_rng(20261121)
        cm_current = np.array([w["operational"]["confusion_matrix"] for w in self.windows])
        cm_previous = np.array([w["previous_pdf"]["confusion_matrix"] for w in self.windows])
        cm_control = np.array([w["packaged_control"]["confusion_matrix"] for w in self.windows])
        deltas_previous, deltas_control = [], []
        for indices in rng.integers(0, 8, size=(3000, 8)):
            value = macro(cm_current[indices].sum(0))
            deltas_previous.append(value - macro(cm_previous[indices].sum(0)))
            deltas_control.append(value - macro(cm_control[indices].sum(0)))
        bootstrap = {"unit": "paired batch clusters", "replicates": 3000, "seed": 20261121,
                     "delta_vs_previous_95pct": np.quantile(deltas_previous, [.025,.975]).tolist(),
                     "delta_vs_packaged_95pct": np.quantile(deltas_control, [.025,.975]).tolist(),
                     "interpretation": "descriptive resampling of eight studied batches; not untouched future generalization"}
        newly_correct, newly_wrong = int(np.sum((pred == y) & (old != y))), int(np.sum((pred != y) & (old == y)))
        summary = {"format": "gas-four-sensor.closed-loop-local.v1", "created_at": now(), "dataset": str(self.dataset),
                   "previous_run": str(self.previous), "current": current_metrics, "packaged_control": packaged_metrics,
                   "previous_pdf": prior_metrics, "f1_delta_vs_previous": current_metrics["macro_f1"] - prior_metrics["macro_f1"],
                   "accuracy_delta_vs_previous": current_metrics["accuracy"] - prior_metrics["accuracy"],
                   "f1_delta_vs_packaged": current_metrics["macro_f1"] - packaged_metrics["macro_f1"],
                   "newly_correct_vs_previous": newly_correct, "newly_wrong_vs_previous": newly_wrong,
                   "monitor_windows": len(self.windows), "diagnosis_windows": len(self.windows),
                   "optimization_windows": len(self.candidates), "candidate_bundles": len(self.candidates),
                   "sensor_training_trials": 2 * len(self.candidates), "home_training_trials": 1,
                   "promotions": sum(d["reason"] != "natural_performance_rollback" for d in self.deployments),
                   "natural_rollbacks": sum(d["reason"] == "natural_performance_rollback" for d in self.deployments),
                   "deployments": self.deployments, "rollback_drills": self.drill_results, "bootstrap": bootstrap,
                   "external_ai_calls": 0, "official_competition_result": False,
                   "scope": "local historical replay with deterministic rules; no cloud or HTTP deployment",
                   "starting_checkpoint_matches_previous": self.cfg["model_sha256"]["start"] == "sha256:" + digest(ROOT / "models/reference.npz"),
                   "prior_predictions": prior_sha}
        write(self.out / "summary.json", summary)
        write(self.out / "history.json", self.windows)
        write(self.out / "candidates.json", self.candidates)
        write(self.out / "registry.json", self.registry)
        for name, rows in (("phase-timing", self.spans), ("batch-comparison", [{"batch":w["batch"],"rows":w["rows"],
                 "current_f1":w["operational"]["macro_f1"],"previous_pdf_f1":w["previous_pdf"]["macro_f1"],
                 "packaged_f1":w["packaged_control"]["macro_f1"],"current_accuracy":w["operational"]["accuracy"],
                 "prediction_version":w["prediction_version"],"next_version":w["next_version"]} for w in self.windows])):
            with (self.out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                fields = list(dict.fromkeys(k for r in rows for k in r))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
        return summary


def restore_engine(entry, state):
    if state["kind"] == "parent":
        engine = ParentEngine()
        engine.batch_count = state["batch_count"]
        return engine
    engine = FourSensorRouter(entry["bundle"], entry["config"])
    engine.batch_count = state["batch_count"]
    engine.state = (deque((np.array(r) for r in state["drift_buffer"]), maxlen=128),
                    np.array(state["drift_shift"]), state["drift_count"])
    engine.detector.window = deque((np.array(r) for r in state["flatline_buffer"]), maxlen=8)
    engine.history = deque(((r[0], np.array(r[1]), [np.array(d) for d in r[2]], np.array(r[3])) for r in state["history"]), maxlen=3)
    engine.weights = np.array(state["weights"])
    engine.active_pairs = set(state["active_pairs"])
    return engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "policy.json", POLICY)
    sources = [ROOT / "config.json", ROOT / "run.py", ROOT / "rebuild_models.py", *sorted((ROOT / "src").rglob("*.py")), Path(__file__).resolve()]
    protected = sources + list((ROOT / "models").glob("*.npz")) + [args.dataset / f"batch{b}.dat" for b in range(3,11)]
    protected += [args.previous / name for name in ("closed-loop-validation.pdf", "summary.json", "history.jsonl")]
    protected += [args.previous / "windows" / f"batch{b:02d}" / "predictions.npz" for b in range(3,11)]
    before = {str(p.resolve()): digest(p) for p in protected}
    write(out / "protected-hashes-before.json", before)
    for p in sources:
        target = out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    experiment = Experiment(out, args.dataset.resolve(), args.previous.resolve())
    summary = experiment.run()
    after = {p: digest(p) for p in before}
    write(out / "protected-hashes-after.json", after)
    assert before == after, "Protected inputs changed"
    print(json.dumps({k:summary[k] for k in ("f1_delta_vs_previous","f1_delta_vs_packaged","promotions","natural_rollbacks")}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
