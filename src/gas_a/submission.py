"""Workspace-v2 training and feature-only JSONL inference for task A.

The platform supplies the checkpoint and public projection. The local replay
router, archived batches, and packaged experiment weights are not used here.
"""
from functools import lru_cache
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import numpy as np
from contest_harness.errors import HarnessError

from .sensor_training.data import evaluate, load_pair
from .sensor_training.io import CLASSES, append_jsonl, digest, write_json
from .sensor_training.reference.data import flatten, json_value
from .continuation import continue_candidate
from .frozen_router import load_predictor, checkpoint_paths as artifact_checkpoint_paths
from .monitoring import fit_reference, monitor, diagnose
from .evolution import (load_policy, available_experiments, promotion_gate,
                        fingerprint, advance_policy, validate_controller)

ROOT = Path(__file__).resolve().parents[2]
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]


def _check_task(task, dataset_key):
    if task != "classification" or dataset_key not in (None, "A"):
        raise ValueError("This submission supports task A classification only")


@lru_cache(maxsize=1)
def _prediction_model(path, file_identity=None):
    # Cache only within this prediction process. Each measurement gets a new one.
    return load_predictor(Path(path))


def _selected_checkpoint(sample, model_dir):
    explicit = os.environ.get("CONTEST_ARTIFACT_PATH")
    if explicit:
        return Path(explicit).resolve()
    context = getattr(sample, "context", None) or {}
    paths = context.get("artifact", {}).get("checkpoint_paths", [])
    if not paths:
        return (Path(model_dir) / "checkpoint.npz").resolve()
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
        raise ValueError("CONTEST_ARTIFACT_PATH is required to select among multiple checkpoint files")
    relative = PurePosixPath(paths[0])
    if (relative.is_absolute() or "\\" in paths[0] or not relative.parts
            or any(part in ("..", ".") for part in paths[0].split("/"))):
        raise ValueError("Prediction checkpoint path must be relative to the model artifact")
    return Path(model_dir).joinpath(*relative.parts).resolve()


def predict(sample, model_dir):
    _check_task(sample.task, sample.dataset_key)
    inputs = sample.inputs
    if not isinstance(inputs, dict) or set(inputs) != {"features"}:
        raise ValueError("Task A requires only inputs.features")
    features = inputs["features"]
    if (not isinstance(features, list) or len(features) != 128
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in features)):
        raise ValueError("Task A requires 128 finite numeric features")
    checkpoint = _selected_checkpoint(sample, model_dir)
    stat = checkpoint.stat()
    model = _prediction_model(str(checkpoint), (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
    paths, values = flatten(inputs)
    if paths != model.meta["paths"]:
        raise ValueError("Input schema differs from selected checkpoint")
    inference = getattr(model, "predict_one", model.predict)
    scores = inference(np.asarray([values], dtype=np.float64))[0].copy()
    if scores.shape != (6,) or not np.isfinite(scores).all() or np.any(scores < 0) or scores.sum() <= 0:
        raise ValueError("Invalid classifier probabilities")
    scores /= scores.sum()
    # The SDK requires a unique top score, including for degenerate checkpoints.
    top = int(scores.argmax())
    if np.count_nonzero(scores == scores[top]) > 1:
        scores *= 1 - 1e-12
        scores[top] += 1e-12
    return [{"label": label, "score": float(score)} for label, score in zip(CLASSES, scores)]


def _parent_checkpoint(rnd):
    expected = rnd.parent.checkpoint_sha256.removeprefix("sha256:")
    for relative in rnd.parent.descriptor["checkpoint_paths"]:
        checkpoint = rnd.parent.path(relative)  # SDK validates declaration and file digest.
        if digest(checkpoint).removeprefix("sha256:") == expected:
            return checkpoint
    raise ValueError("No declared parent checkpoint matches parent_checkpoint_sha256")


def _decision(rnd, request, fields, token_limit, validate, corrections=1):
    call_ids, failures = [], []
    for attempt in range(corrections + 1):
        # Never catch an RPC failure: its outcome may be unknown to the SDK.
        receipt = rnd.model(
            [{"role": "user", "content": json.dumps(request, ensure_ascii=False, allow_nan=False)}],
            max_tokens=min(token_limit, rnd.max_output_tokens), temperature=0,
        )
        call_ids.append(receipt.call_id)
        try:
            raw = receipt.text.strip()
            if raw.startswith("```json\n") and raw.endswith("```"):
                raw = raw[8:-3].strip()
            value = json_value(raw)
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ValueError("Model decision does not match the requested JSON schema")
            if any(not isinstance(value[k], str) or not value[k].strip()
                   or len(value[k].encode("utf-8")) > 4000 for k in fields):
                raise ValueError("Model decision fields must be nonempty text <=4000 bytes")
            validate(value)
            return value, {"model_call_ids": call_ids, "corrections": failures}
        except ValueError as error:
            failures.append({"call_id": receipt.call_id, "error": str(error), "response_excerpt": receipt.text[:2000]})
            if attempt == corrections:
                raise
            request = {**request, "correction": {"attempt": attempt + 1, "error": str(error),
                       "instruction": "The previous completed response failed validation. Return a corrected JSON object using only the supplied schema and choices."}}


def _read_state(state, name):
    path = state / name
    return json_value(path.read_text(encoding="utf-8")) if path.is_file() else None


def _restore_history(rnd, history, run, validation, before, controller):
    receipts = {item["artifact_id"]: item for item in rnd.artifacts}
    order = [item.get("selected_artifact_id") for item in reversed(_history_view(history))]
    order.extend(reversed(list(receipts)))
    restored, seen = {}, {rnd.parent.artifact_id}
    for identifier in order:
        if identifier in seen or identifier not in receipts:
            continue
        seen.add(identifier)
        if len(restored) >= controller["max_history_restores"]:
            break
        receipt = receipts[identifier]
        # Only this submission's feature-only predictor format can be evaluated
        # locally. No historical source is imported or executed.
        if receipt.get("artifact", {}).get("predict_argv") != PREDICT_ARGV:
            continue
        directory = rnd.restore_artifact(identifier, run / f"restored-{len(restored) + 1}")
        try:
            names = set(receipt["artifact"]["checkpoint_paths"] + receipt["artifact"]["code_paths"])
            def resolve_file(path):
                relative = Path(path).resolve().relative_to(directory.resolve()).as_posix()
                if relative not in names:
                    raise ValueError("Historical predictor references an undeclared file")
                return directory / relative
            # Our artifact contract declares the primary checkpoint first.
            primary = directory / receipt["artifact"]["checkpoint_paths"][0]
            model = load_predictor(primary, validation["paths"], resolve_file=resolve_file)
            metric = evaluate(model, validation)
            restored[identifier] = {"metrics": metric, "gate": promotion_gate(before, metric, controller),
                                    "checkpoint_sha256": digest(primary), "seed": model.meta["seed"]}
        except (ValueError, OSError, KeyError) as error:
            restored[identifier] = {"error": str(error), "gate": {"eligible": False}}
    return restored


def _history_view(records):
    result = []
    for item in records[-10:]:
        record = item.get("record", item)
        if not isinstance(record, dict):
            record = item
        result.append({"id": item["id"], **{k: record[k] for k in (
            "round_index", "selected_artifact_id", "reason", "configuration") if k in record}})
    return result


def _snapshot(candidate):
    # Explicit code sources only: no local models, data, replay outputs, or secrets.
    paths = [ROOT / name for name in ("harness.py", "contest_submission.py", "submission_config.json")]
    paths.extend(sorted((ROOT / "src" / "gas_a").rglob("*.py")))
    names = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        names.append(relative.as_posix())
    return names


def train(rnd):
    _check_task(rnd.task.get("task"), rnd.dataset_key)
    # 1.0.4 has no inline fallback. Fail before data loading or model calls.
    if rnd.limits.get("artifact_upload_protocol") != "chunked-v1":
        raise HarnessError("CHUNKED_UPLOAD_UNSUPPORTED", "SDK 1.0.4 requires a host with chunked-v1 uploads")
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
    validate_controller(config["controller"])
    work = Path(os.environ.get("CONTEST_WORK", "/workspace")).resolve()
    protected = [ROOT, rnd.parent.root, Path(rnd.data.train_path).parent,
                 Path(rnd.data.validation_path).parent]
    if rnd.state_path:
        protected.append(Path(rnd.state_path))
    protected.extend(Path(p) for p in ("/data", "/previous", "/state", "/model"))
    if any(work.is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError("CONTEST_WORK must be outside read-only code, data, model, and state mounts")
    work.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"round-{rnd.round_index:02d}-", dir=work))
    checkpoint = _parent_checkpoint(rnd)
    train_data, validation = load_pair(rnd.data.train_path, rnd.data.validation_path)
    def parent_file(path):
        return rnd.parent.path(Path(path).resolve().relative_to(rnd.parent.root.resolve()).as_posix())
    parent = load_predictor(checkpoint, train_data["paths"], resolve_file=parent_file)
    before = evaluate(parent, validation)
    history = rnd.history()
    menu = {spec["name"]: spec for spec in config["experiments"]}
    controller = config["controller"]
    state = rnd.state_workspace(run / "state")
    reference = _read_state(state, "monitor_reference.json")
    if reference is None:
        reference = fit_reference(train_data)
    write_json(state / "monitor_reference.json", reference)
    drift = monitor(reference, validation, controller["drift_thresholds"])
    policy = load_policy(_read_state(state, "strategy.json"), menu)
    # State can describe a rejected candidate or a different artifact. It never
    # overrides the host-selected model or carries its strategy across blindly.
    paired = policy["artifact_policies"].get(rnd.parent.artifact_id)
    if paired:
        policy.update(paired)
    elif policy["artifact_policies"]:
        policy.update(version=1, preferred_experiment=None)
    diagnosis = diagnose(before, drift, validation["sha256"], policy.get("last_measurement"))
    allowed_menu, blocked = available_experiments(policy, menu, rnd, train_data, validation, controller)
    restored = _restore_history(rnd, history, run, validation, before, controller)
    rollback_ids = [identifier for identifier, item in restored.items() if item["gate"]["eligible"]]
    actions = ["observe"] + (["experiment"] if allowed_menu else []) + (["rollback"] if rollback_ids else [])

    def validate_plan(value):
        if value["action"] not in actions:
            raise ValueError("Model selected an unavailable action")
        if value["action"] == "experiment" and value["experiment"] not in allowed_menu:
            raise ValueError("Model selected an experiment outside the configured menu or repetition budget")
        if value["action"] != "experiment" and value["experiment"] != "none":
            raise ValueError("Observe and rollback must use experiment=none")
        if value["action"] == "rollback" and value["target_artifact_id"] not in rollback_ids:
            raise ValueError("Rollback target has not passed current-scope validation")
        if value["action"] != "rollback" and value["target_artifact_id"] != rnd.parent.artifact_id:
            raise ValueError("Observe and experiment must refer to the actual parent")

    plan, planning_calls = _decision(rnd, {
        "phase": "plan", "round_index": rnd.round_index,
        "request": "Use drift, diagnosis, history and bounded strategy evidence to choose observation, one continuation experiment, or a validated historical rollback. Observation is appropriate when no useful experiment remains.",
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "parent_artifact_id": rnd.parent.artifact_id,
        "validation_before": before, "history": _history_view(history), "experiments": allowed_menu,
        "monitoring": {key: value for key, value in drift.items() if key != "features"},
        "diagnosis": diagnosis, "historical_validation": restored,
        "strategy": {"version": policy["version"], "preferred_experiment": policy["preferred_experiment"],
                     "blocked_repeated_experiments": blocked},
        "response_schema": {"action": actions, "experiment": ["none", *allowed_menu],
                            "target_artifact_id": [rnd.parent.artifact_id, *rollback_ids],
                            "hypothesis": "testable hypothesis", "reason": "cite measured evidence and uncertainty"},
        "rules": "Return only JSON. Only public training rows update gradients; validation selects epochs and versions. Preserve parent Adam/scaler/steps. Hidden-stream labels or telemetry are unavailable. Do not infer physical causes from PSI alone.",
    }, ("action", "experiment", "target_artifact_id", "hypothesis", "reason"),
       config["max_output_tokens"], validate_plan, controller["json_corrections"])
    spec = allowed_menu[plan["experiment"]] if plan["action"] == "experiment" else None
    report, submitted, candidate_gate = None, None, None
    configuration = {"action": plan["action"], "experiment": spec, "round_index": rnd.round_index,
                     "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
                     "train_sha256": train_data["sha256"], "validation_sha256": validation["sha256"],
                     "local_validation_before": before["macro_f1"],
                     "controller_model_name": rnd.model_name, "strategy_version_before": policy["version"],
                     "drift_detected": drift["drift"], "monitor_reference_train_sha256": reference["train_sha256"],
                     "validation_scope": "authorized_public_validation", "promotion_rules": controller}
    change = "Observed authorized public features and validation; retained the actual parent without training."
    if spec is not None:
        weak_class = diagnosis["weak_classes"][0]["label"]
        candidate = run / "candidate"
        report = continue_candidate(checkpoint, train_data, validation, candidate, spec,
                                    {"weakest_class": weak_class}, config["limits"],
                                    expected_parent_sha256=rnd.parent.checkpoint_sha256,
                                    parent_artifact_id=rnd.parent.artifact_id)
        code_paths = _snapshot(candidate)
        if (candidate / "routing.json").is_file():
            code_paths.append("routing.json")
            manifest = json.loads((candidate / "routing.json").read_text(encoding="utf-8"))
            configuration.update(routing_policy=manifest["policy"], routing_manifest_sha256=digest(candidate / "routing.json"))
        configuration["local_validation_after"] = report["validation_after"]["macro_f1"]
        candidate_gate = promotion_gate(before, report["validation_after"], controller)
        configuration["candidate_gate"] = candidate_gate
        change = "Continued the actual parent using %s; preserved Adam/scaler and exported epoch %d, step %d." % (
            spec["name"], report["final_epochs"], report["final_steps"])
        write_json(candidate / "planning.json", {"decision": plan, **planning_calls})
        write_json(candidate / "diagnosis.json", diagnosis)
        write_json(candidate / "monitoring.json", drift)
        submitted = rnd.submit_candidate(
            candidate, hypothesis=plan["hypothesis"], change_summary=change,
            configuration=configuration, seed=parent.meta["seed"],
            checkpoint_paths=artifact_checkpoint_paths(candidate / "checkpoint.npz"),
            code_paths=code_paths, predict_argv=PREDICT_ARGV,
        )
    eligible = [rnd.parent.artifact_id]
    if submitted and candidate_gate["eligible"]:
        eligible.append(submitted["artifact_id"])
    if plan["action"] == "rollback":
        eligible.append(plan["target_artifact_id"])

    def validate_selection(value):
        if value["selected_artifact_id"] not in eligible:
            raise ValueError("Model selected an unknown or regressing artifact")

    selection, selection_calls = _decision(rnd, {
        "phase": "select", "round_index": rnd.round_index,
        "request": "Confirm a guarded experiment promotion, observation, or actual historical rollback using the current public validation. Retain the parent if evidence is insufficient.",
        "action": plan["action"], "parent_artifact_id": rnd.parent.artifact_id,
        "candidate_artifact_id": submitted["artifact_id"] if submitted else None,
        "same_local_validation": {"parent": before, "candidate": report["validation_after"] if report else None,
                                  "historical": restored},
        "host_parent_validation": rnd.frame.get("current_validation", {}),
        "host_candidate_receipt": submitted,
        "response_schema": {"selected_artifact_id": eligible,
                            "reason": "explain measured results and choice"},
        "rules": "Return only JSON. Compare the same metric and dataset scope. Never call local metrics hidden-test results. Prefer retaining the parent on a tie.",
    }, ("selected_artifact_id", "reason"), config["max_output_tokens"], validate_selection, controller["json_corrections"])
    selected = selection["selected_artifact_id"]
    candidate_id = submitted["artifact_id"] if submitted else None
    selected_metric, selected_sha256 = before, rnd.parent.checkpoint_sha256
    selected_seed = parent.meta["seed"]
    if selected == candidate_id:
        selected_metric, selected_sha256 = report["validation_after"], digest(candidate / "checkpoint.npz")
    elif selected != rnd.parent.artifact_id:
        selected_metric, selected_sha256 = restored[selected]["metrics"], restored[selected]["checkpoint_sha256"]
        selected_seed = restored[selected]["seed"]
        change = "Restored and selected historical artifact %s after evaluating its complete predictor on the current public validation." % selected
    elif submitted:
        change = "Retained the actual parent checkpoint; the tested candidate was not adopted."
    next_policy = advance_policy(policy, selected=selected, parent_id=rnd.parent.artifact_id,
                                 candidate_id=candidate_id, plan=plan,
                                 attempt_key=fingerprint(rnd.parent.checkpoint_sha256, train_data["sha256"], validation["sha256"], spec) if spec else None,
                                 gain=candidate_gate["macro_f1_gain"] if candidate_gate else None,
                                 validation_sha256=validation["sha256"], selected_f1=selected_metric["macro_f1"])
    configuration.update(selected_checkpoint_sha256=selected_sha256,
                         selected_local_macro_f1=selected_metric["macro_f1"], strategy_version_after=next_policy["version"])
    write_json(state / "strategy.json", next_policy)
    write_json(state / "monitoring.json", drift)
    write_json(state / "diagnosis.json", diagnosis)
    record = {
        "round_index": rnd.round_index, "parent_artifact_id": rnd.parent.artifact_id,
        "candidate_artifact_id": candidate_id, "selected_artifact_id": selected,
        "plan": plan, "selection": selection, "configuration": configuration,
        "model_call_ids": list(rnd.call_ids), "planning_calls": planning_calls, "selection_calls": selection_calls,
        "host_candidate_receipt": submitted, "historical_validation": restored,
    }
    append_jsonl(state / "decisions.jsonl", record)
    write_json(run / "round_summary.json", record)
    rnd.save_state(state)
    rnd.finish(
        selected_artifact_id=selected, hypothesis=plan["hypothesis"],
        change_summary=change,
        reason=selection["reason"], configuration=configuration, seed=selected_seed,
        history_ids=[item["id"] for item in history], model_call_ids=list(rnd.call_ids),
    )
