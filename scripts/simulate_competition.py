"""Autonomous public-data acceptance rehearsal with an explicit offline model peer.

Uses the actual SDK and participant unchanged. No real model or hidden data.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import platform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_submission import OfflineHost
from scripts.validate_platform import exchange, receipt
from gas_a.frozen_router import load_predictor, checkpoint_paths
from gas_a.sensor_training.data import load_pair, score_predictions
from gas_a.sensor_training.io import CLASSES, digest, write_json, append_jsonl
from gas_a.sensor_training.reference.model import Model
import numpy as np


def environment():
    env = {k: v for k,v in os.environ.items()
           if not k.startswith(("CONTEST_", "PYTRIO_", "SWANLAB_")) and k != "OPENAI_API_KEY"}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


def adapter(command, directory, work, audit, dataset_root, data=None, state=None):
    argv = [sys.executable, "-B", str(ROOT / "scripts/simulation_audit.py"), command,
            "--model-root", str(directory), "--workspace", str(work), "--audit", str(audit),
            "--dataset-root", str(dataset_root)]
    if data:
        argv.extend(["--data-root", str(data)])
    if state:
        argv.extend(["--state-root", str(state)])
    return argv


def artifact_receipt(directory, identifier, descriptor=None, primary="checkpoint.npz"):
    if descriptor is None:
        model = load_predictor(directory / primary)
        manifest = getattr(model, "manifest_path", None)
        descriptor = {"checkpoint_paths": checkpoint_paths(directory / primary),
                      "code_paths": [manifest.relative_to(directory).as_posix()] if manifest else [],
                      "predict_argv": ["python3", "/app/harness.py", "predict"]}
    return receipt(directory, identifier, descriptor)


class Measurements:
    """Host-owned labels; the predictor receives features only, one row at a time."""
    def __init__(self, root, validation, dataset_root, *, prediction_command=None, prediction_cleanup=None):
        self.root, self.validation, self.dataset_root = root, validation, dataset_root
        self.prediction_command, self.prediction_cleanup = prediction_command, prediction_cleanup
        self.cache, self.by_id = {}, {}
        self.counter = 0
        (root / "measurements").mkdir()

    def measure(self, directory, descriptor, identifier):
        primary = directory / descriptor["checkpoint_paths"][0]
        model = load_predictor(primary)
        files = list(getattr(model, "component_paths", {"primary": primary}).values())
        manifest = getattr(model, "manifest_path", None)
        if manifest:
            files.append(manifest)
        key = tuple(digest(p) for p in files)
        if key in self.cache:
            result = self.cache[key]
            self.by_id[identifier] = result
            return result
        self.counter += 1
        output = self.root / "measurements" / f"measurement-{self.counter:03d}"
        output.mkdir()
        started = time.perf_counter()
        probabilities = []
        with tempfile.TemporaryDirectory(prefix="gas-predict-audit-") as tmp, \
                (output / "stderr.log").open("w", encoding="utf-8") as stderr, \
                (output / "requests.jsonl").open("w", encoding="utf-8") as requests, \
                (output / "outputs.jsonl").open("w", encoding="utf-8") as outputs:
            env = environment()
            env["CONTEST_ARTIFACT_PATH"] = str(primary)
            argv = (self.prediction_command(directory, primary, output) if self.prediction_command else
                    adapter("predict", directory, Path(tmp), output / "access.jsonl", self.dataset_root))
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8")
            lines = queue.Queue()
            def reader():
                for line in process.stdout:
                    lines.put(line)
                lines.put(None)
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            try:
                for x in self.validation["x"]:
                    request = json.dumps({"task": "classification", "inputs": {"features": x.tolist()}}, allow_nan=False)
                    requests.write(request + "\n")
                    process.stdin.write(request + "\n")
                    process.stdin.flush()
                    line = lines.get(timeout=30)
                    if line is None:
                        raise RuntimeError("Predictor exited early; inspect " + str(output))
                    row = json.loads(line)
                    scores = row["outputs"]
                    p = np.asarray([s["score"] for s in scores], dtype=float)
                    if ([s["label"] for s in scores] != CLASSES or p.shape != (6,)
                            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
                            or abs(float(p.sum()) - 1) > 1e-9 or np.count_nonzero(p == p.max()) != 1):
                        raise ValueError("Invalid prediction contract")
                    outputs.write(line)
                    probabilities.append(p)
                process.stdin.close()
                if process.wait(timeout=15) != 0:
                    raise RuntimeError("Prediction failed; inspect " + str(output))
                thread.join(timeout=2)
                if not lines.empty() and lines.get_nowait() is not None:
                    raise ValueError("Extra prediction output")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                process.stdin.close()
                process.stdout.close()
                if self.prediction_cleanup:
                    self.prediction_cleanup(argv)
        metric = score_predictions(np.asarray(probabilities), self.validation["y"], time.perf_counter() - started)
        result = {"metrics": metric, "measured_at": datetime.now(timezone.utc).isoformat(), "directory": str(output.relative_to(self.root)),
                  "validation_sha256": self.validation["sha256"], "checkpoint_sha256": digest(primary),
                  "model_file_sha256": {p.relative_to(directory).as_posix(): digest(p) for p in files},
                  "predictions_sha256": digest(output / "outputs.jsonl"),
                  "all_rows_replied_before_next_input_or_eof": True, "label_input_to_predictor": False}
        write_json(output / "metrics.json", result)
        self.cache[key] = result
        self.by_id[identifier] = result
        return result


class EvidenceHost(OfflineHost):
    def __init__(self, *args, measurements, recovery=False, observe_only=False, artifact_namespace="", **kwargs):
        super().__init__(*args, **kwargs)
        self.measurements, self.recovery = measurements, recovery
        self.observe_only = observe_only
        self.artifact_namespace = artifact_namespace
        self.frame.update(model_name="offline-evidence-controller-v1", prompt_version="offline-acceptance-v1",
                          prompt="离线仿真任务提示：只使用当轮授权公开数据和实际父产物，读取历史，做受限实验，按同口径测量决策。不得访问隐藏标签。")
        self.history_read_ids = []
        self.last_candidate_request = None

    def plan(self, request):
        available = request["experiments"]
        attempted = set()
        for item in request["history"]:
            cfg = item.get("configuration", {})
            if cfg.get("parent_checkpoint_sha256") == request["parent_checkpoint_sha256"] and cfg.get("experiment"):
                attempted.add(cfg["experiment"]["name"])
        ordered = [n for n in ("conservative_head", "five_sensor", "four_sensor", "frozen_router") if n in available and n not in attempted]
        rollback = [identifier for identifier in request["response_schema"]["target_artifact_id"]
                    if identifier != request["parent_artifact_id"]]
        weak = request["diagnosis"]["weak_classes"][0]
        if self.observe_only:
            action, experiment, target = "observe", "none", request["parent_artifact_id"]
            hypothesis = "在故障演练的下一轮核验并保留已恢复的父产物，不进行新的梯度更新。"
        elif self.recovery and "rollback" in request["response_schema"]["action"]:
            action, experiment = "rollback", "none"
            target = max(rollback, key=lambda i: request["historical_validation"][i]["metrics"]["macro_f1"])
            hypothesis = "恢复经本轮公开验证重新测量达标的完整历史产物，可改善故障注入后的模型表现。"
        elif ordered and "experiment" in request["response_schema"]["action"]:
            action, experiment, target = "experiment", ordered[0], request["parent_artifact_id"]
            hypothesis = f"从实际父模型续训 {experiment}，检验弱类 {weak['label']} 的错误能否改善，同时满足 Macro-F1 和类别召回保护。"
        else:
            action, experiment, target = "observe", "none", request["parent_artifact_id"]
            hypothesis = "当前父产物与数据范围内的菜单实验已测量，保留父模型等待新的授权证据。"
        reason = (f"读取 {len(request['history'])} 条历史；本父模型已试 {sorted(attempted)}；"
                  f"公开特征漂移={request['monitoring']['drift']}；弱类 {weak['label']} 的 F1={weak['f1']:.6f}。"
                  "先尝试低改动输出头，再尝试尚未测过的传感器/路由方案；分布偏移不证明物理因果。")
        return {"action": action, "experiment": experiment, "target_artifact_id": target,
                "hypothesis": hypothesis, "reason": reason}

    def selection(self, request):
        parent = request["parent_artifact_id"]
        eligible = request["response_schema"]["selected_artifact_id"]
        parent_score = self.measurements.by_id[parent]["metrics"]["macro_f1"]
        scores = {identifier: self.measurements.by_id[identifier]["metrics"]["macro_f1"] for identifier in eligible}
        selected = max(eligible, key=lambda identifier: (scores[identifier], identifier == parent))
        gate = self.last_candidate_request["configuration"]["candidate_gate"] if self.last_candidate_request else None
        candidate_id = request["candidate_artifact_id"]
        candidate_score = self.measurements.by_id[candidate_id]["metrics"]["macro_f1"] if candidate_id else None
        reason = (f"宿主替身通过真实逐行预测接口测量当前公开验证集：父模型 Macro-F1={parent_score:.9f}，"
                  f"候选={candidate_score}；最终门槛={json.dumps(gate, ensure_ascii=False)}。"
                  f"仅在已通过门槛的 {eligible} 中选择 {selected}，分数={scores[selected]:.9f}；同分保留父产物。")
        return {"selected_artifact_id": selected, "reason": reason}

    def write(self, text):
        frame = json.loads(text)
        kind = frame["type"]
        append_jsonl(self.store / "events.jsonl", {"at": datetime.now(timezone.utc).isoformat(),
                     "event": "rpc_received", "type": kind, "id": frame.get("id")})
        request = json.loads(frame["messages"][0]["content"]) if kind == "model_request" else None
        if kind == "candidate_request":
            self.last_candidate_request = frame
        result = super().write(text)
        if kind == "history_request":
            self.history_read_ids = [item["id"] for item in self.history]
        elif kind == "candidate_request":
            identifier = f"{self.artifact_namespace}candidate-{self.frame['round_index']}"
            measured = self.measurements.measure(self.candidate, frame["artifact"], identifier)
            self.pending[-1]["artifact_id"] = identifier
            if abs(measured["metrics"]["macro_f1"] - self.candidate_metrics["macro_f1"]) > 1e-12:
                raise ValueError("Batch evaluation differs from actual streaming predictor")
            self.candidate_metrics = measured["metrics"]
            self.pending[-1]["summary"] = {"metrics": {k: self.candidate_metrics[k] for k in ("macro_f1", "accuracy")},
                                            "validation_sha256": measured["validation_sha256"], "scope": "offline_public_validation"}
        elif kind == "model_request":
            decision = self.plan(request) if request["phase"] == "plan" else self.selection(request)
            call_id = "offline-" + self.calls[-1]
            self.calls[-1] = call_id
            reply = self.pending[-1]
            reply.update(call_id=call_id, response={"choices": [{"message": {"content": json.dumps(decision, ensure_ascii=False)}}]})
            metadata = {k: self.frame[k] for k in ("dataset_key", "round_index", "task_contract", "prompt_version")}
            bound = [{"role": "system", "content": self.frame["prompt"] + "\n" + json.dumps(metadata, ensure_ascii=False)}, *frame["messages"]]
            append_jsonl(self.store / "model-calls.jsonl", {"at": datetime.now(timezone.utc).isoformat(), "phase": request["phase"], "call_id": call_id,
                         "source": "offline_rule_based_model_peer", "real_platform_model_call": False,
                         "raw_messages": frame["messages"], "bound_messages": bound, "decision": decision})
        elif kind == "round_complete":
            if self.record["history_ids"] != self.history_read_ids or self.record["model_call_ids"] != self.calls:
                raise ValueError("Round cites unread history or fabricated calls")
            measured = self.measurements.by_id[self.record["selected_artifact_id"]]["metrics"]["macro_f1"]
            if abs(measured - self.record["configuration"]["selected_local_macro_f1"]) > 1e-12:
                raise ValueError("Selected metric differs from host measurement")
        append_jsonl(self.store / "events.jsonl", {"at": datetime.now(timezone.utc).isoformat(),
                     "event": "rpc_response_ready", "type": self.pending[-1].get("type")})
        return result


def run_round(host, directory, state, dataset_root):
    with tempfile.TemporaryDirectory(prefix="gas-training-audit-") as tmp:
        argv = adapter("train", directory, Path(tmp), host.store / "access.jsonl", dataset_root,
                       data=host.root / "data", state=state)
        exchange(host, argv, environment(), host.store)
    append_jsonl(host.root / "rounds.jsonl", {"round_index": host.frame["round_index"], "record": host.record,
                 "finished_at": datetime.now(timezone.utc).isoformat(),
                 "history_read_ids": host.history_read_ids, "candidate_submitted": host.candidate is not None,
                 "protocol": str((host.store / "protocol.jsonl").relative_to(host.root))})


def session(root, baseline, source_data, rounds, dataset_root):
    root.mkdir()
    shutil.copytree(source_data, root / "data")
    shutil.copytree(baseline, root / "initial")
    _, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl")
    measurement = Measurements(root, validation, dataset_root)
    descriptor = artifact_receipt(root / "initial", "initial")
    registry = {"initial": (root / "initial", descriptor)}
    baseline_metric = measurement.measure(root / "initial", descriptor["artifact"], "initial")
    selected, history, state, records = "initial", [], None, []
    for index in range(1, rounds + 1):
        directory, parent_receipt = registry[selected]
        parent = directory / parent_receipt["artifact"]["checkpoint_paths"][0]
        host = EvidenceHost(root, parent, index, history, state, measurements=measurement)
        host.frame["parent_artifact"] = parent_receipt
        host.frame["artifacts"] = [item[1] for identifier,item in registry.items() if identifier != selected]
        host.artifact_dirs = {identifier:item[0] for identifier,item in registry.items()}
        host.frame["current_validation"] = {"metrics": measurement.by_id[selected]["metrics"],
                                            "validation_sha256": measurement.validation["sha256"],
                                            "scope": "offline_public_validation"}
        run_round(host, directory, state, dataset_root)
        if host.candidate:
            candidate_descriptor = next(f["artifact"] for f in host.frames if f["type"] == "candidate_request")
            identifier = f"candidate-{index}"
            registry[identifier] = (host.candidate, artifact_receipt(host.candidate, identifier, candidate_descriptor))
        selected = host.record["selected_artifact_id"]
        state = host.saved_state
        history.append({"id": f"offline-history-{index}", **host.record})
        record = {"round": index, "action": host.record["configuration"]["action"],
                  "experiment": (host.record["configuration"]["experiment"] or {}).get("name"),
                  "history_records_read": len(host.history_read_ids), "selected_artifact_id": selected,
                  "candidate_submitted": host.candidate is not None,
                  "host_candidate_macro_f1": host.candidate_metrics["macro_f1"] if host.candidate_metrics else None,
                  "candidate_model_file_sha256": measurement.by_id[f"candidate-{index}"]["model_file_sha256"] if host.candidate else None,
                  "candidate_predictions_sha256": measurement.by_id[f"candidate-{index}"]["predictions_sha256"] if host.candidate else None,
                  "selected_macro_f1": measurement.by_id[selected]["metrics"]["macro_f1"],
                  "selected_checkpoint_sha256": host.record["configuration"]["selected_checkpoint_sha256"],
                  "candidate_gate": host.record["configuration"].get("candidate_gate"),
                  "model_call_ids": host.calls, "finish_acknowledged": True}
        records.append(record)
        write_json(root / "session-summary.json", {"baseline": baseline_metric, "rounds": records,
                   "selected_artifact_id": selected, "selected": measurement.by_id[selected]})
        print(json.dumps({"session": root.name, **{k:record[k] for k in ("round","action","experiment","selected_artifact_id","selected_macro_f1")}}, ensure_ascii=False), flush=True)
    return {"root": root, "registry": registry, "history": history, "state": state,
            "selected": selected, "measurement": measurement, "records": records}


def recovery_drill(root, session_result, source_data, dataset_root):
    root.mkdir()
    shutil.copytree(source_data, root / "data")
    (root / "fault-parent").mkdir()
    source_dir, source_receipt = session_result["registry"]["initial"]
    damaged = Model.load(source_dir / "checkpoint.npz")
    # A cyclic output-map fault avoids a constant-class failure that has 100%
    # recall for one class and can legitimately block all historical recoveries.
    damaged.params["w2"] = np.roll(damaged.params["w2"], 1, axis=1)
    damaged.params["b2"] = np.roll(damaged.params["b2"], 1)
    damaged.save(root / "fault-parent/checkpoint.npz")
    fault_receipt = artifact_receipt(root / "fault-parent", "fault-injected-parent")
    _, validation = load_pair(root / "data/train.jsonl", root / "data/validation.jsonl")
    measurement = Measurements(root, validation, dataset_root)
    measurement.measure(root / "fault-parent", fault_receipt["artifact"], "fault-injected-parent")
    for identifier,(directory, rec) in session_result["registry"].items():
        measurement.measure(directory, rec["artifact"], identifier)
    host = EvidenceHost(root, root / "fault-parent/checkpoint.npz", 1, session_result["history"], session_result["state"],
                        measurements=measurement, recovery=True, artifact_namespace="recovery-")
    host.frame["parent_artifact"] = fault_receipt
    host.frame["artifacts"] = [item[1] for item in session_result["registry"].values()]
    host.artifact_dirs = {identifier:item[0] for identifier,item in session_result["registry"].items()}
    run_round(host, root / "fault-parent", session_result["state"], dataset_root)
    selected = host.record["selected_artifact_id"]
    followup = None
    if selected in session_result["registry"]:
        directory, selected_receipt = session_result["registry"][selected]
        next_host = EvidenceHost(root, directory / selected_receipt["artifact"]["checkpoint_paths"][0], 2,
                                 [*session_result["history"], {"id": "fault-recovery", **host.record}], host.saved_state,
                                 measurements=measurement, observe_only=True)
        next_host.frame["parent_artifact"] = selected_receipt
        run_round(next_host, directory, host.saved_state, dataset_root)
        followup = {"selected_artifact_id": next_host.record["selected_artifact_id"],
                    "parent_checkpoint_sha256": next_host.record["configuration"]["parent_checkpoint_sha256"],
                    "matches_restored_parent": next_host.record["configuration"]["parent_checkpoint_sha256"] == measurement.by_id[selected]["checkpoint_sha256"],
                    "finish_acknowledged": True}
    result = {"scope": "isolated_weight_fault_injection_with_unmodified_official_public_data",
              "fault": "Cyclically permuted output weights and biases of a separate baseline copy; not a natural deployment regression.",
              "data_labels_modified": False, "selected_artifact_id": selected,
              "rollback_selected": selected != "fault-injected-parent" and host.record["configuration"]["action"] == "rollback",
              "historical_read_chunks": sum(f["type"] == "artifact_read_request" for f in host.frames),
              "fault_macro_f1": measurement.by_id["fault-injected-parent"]["metrics"]["macro_f1"],
              "recovered_macro_f1": measurement.by_id[selected]["metrics"]["macro_f1"],
              "next_round": followup,
              "record": host.record}
    write_json(root / "recovery-summary.json", result)
    return result


def audit_summary(root):
    files = list(root.rglob("access.jsonl"))
    denied, network, metadata_queries, data_reads = [], [], [], set()
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["type"] == "denied" or event.get("allowed") is False:
                item = {"file": str(p.relative_to(root)), "event": event}
                if event.get("category") == "blocked_os_version_query":
                    metadata_queries.append(item)
                else:
                    denied.append(item)
            if event.get("event", "").startswith("socket."):
                network.append(event)
            if event["type"] == "file" and not event["write"] and Path(event["path"]).suffix.lower() in (".jsonl", ".dat", ".npz"):
                data_reads.add(event["path"])
    return {"process_logs": len(files), "denied_accesses": denied, "network_attempts": network,
            "blocked_os_version_queries": metadata_queries,
            "data_and_checkpoint_read_paths": sorted(data_reads), "os_level_container_isolation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10, choices=range(4,11))
    parser.add_argument("--replicas", type=int, default=2, choices=(1,2))
    args = parser.parse_args()
    base, root = args.baseline_dir.resolve(), args.output.resolve()
    info = json.loads((base / "baseline-info.json").read_text(encoding="utf-8"))
    if info["dataset_key"] != "A" or info["task"] != "classification" or digest(base / "checkpoint.npz") != info["checkpoint_sha256"]:
        raise ValueError("Official task-A baseline identity mismatch")
    root.mkdir(parents=True, exist_ok=False)
    (root / "official-baseline").mkdir()
    for name in ("checkpoint.npz", "baseline-info.json", "START.md", "TASK.md", "TRAINING_GUIDE.md"):
        if (base / name).is_file():
            shutil.copyfile(base / name, root / "official-baseline" / name)
    # Keep the runtime parent artifact minimal; provenance documents are separate.
    (root / "initial-artifact").mkdir()
    shutil.copyfile(base / "checkpoint.npz", root / "initial-artifact/checkpoint.npz")
    (root / "official-public").mkdir()
    for name in ("train.jsonl", "validation.jsonl"):
        shutil.copyfile(base / "public" / name, root / "official-public" / name)
    shutil.copytree(root / "official-public", root / "official-baseline/public")
    sources = [p for p in (ROOT / "src/gas_a").rglob("*.py")]
    sources.extend(ROOT / name for name in ("harness.py", "contest_submission.py", "submission_config.json", "requirements.txt", "Dockerfile",
                                            ".dockerignore", ".python-version", "scripts/validate_platform.py", "scripts/simulate_competition.py", "scripts/simulation_audit.py", "tests/test_submission.py", "tests/__init__.py"))
    source_hashes = {}
    for p in sources:
        relative = p.relative_to(ROOT)
        target = root / "reproduction" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        source_hashes[relative.as_posix()] = digest(p)
    wheel = ROOT / "vendor/contest_harness-1.0.4-py3-none-any.whl"
    (root / "reproduction/vendor").mkdir(parents=True)
    shutil.copyfile(wheel, root / "reproduction/vendor" / wheel.name)
    dataset_root = base.parent
    runs = [session(root / f"replica-{i}", root / "initial-artifact", root / "official-public", args.rounds, dataset_root)
            for i in range(1,args.replicas + 1)]
    primary = runs[0]
    recovery = recovery_drill(root / "recovery-drill", primary, root / "official-public", dataset_root)
    signatures = [[{k:r[k] for k in ("action", "experiment", "selected_artifact_id", "selected_checkpoint_sha256", "selected_macro_f1",
                                    "host_candidate_macro_f1", "candidate_model_file_sha256", "candidate_predictions_sha256")}
                   for r in run["records"]] for run in runs]
    reproducible = signatures[0] == signatures[1] if len(signatures) == 2 else None
    records = primary["records"]
    experiments = [r for r in records if r["candidate_submitted"] and r["history_records_read"] > 0]
    baseline_score = primary["measurement"].by_id["initial"]["metrics"]["macro_f1"]
    final_score = primary["measurement"].by_id[primary["selected"]]["metrics"]["macro_f1"]
    audit = audit_summary(root)
    report = {"scope": "offline_public_data_simulation_of_current_code", "created_at": datetime.now(timezone.utc).isoformat(),
              "baseline_id": info["id"], "baseline_checkpoint_sha256": info["checkpoint_sha256"],
              "data": {name:digest(root / "official-public" / name) for name in ("train.jsonl","validation.jsonl")},
              "baseline_macro_f1": baseline_score, "final_macro_f1": final_score, "gain": final_score - baseline_score,
              "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "cpu_threads": 1},
              "requirements": {"no_manual_intervention_between_rounds": True, "history_experiment_decision_records_with_nonempty_history": len(experiments),
                               "at_least_three_complete_records": len(experiments) >= 3,
                               "not_below_official_checkpoint_on_same_public_validation": final_score + 1e-12 >= baseline_score,
                               "strictly_better_on_public_validation": final_score > baseline_score + 1e-12,
                               "real_historical_rollback_in_separate_fault_drill": recovery["rollback_selected"],
                               "next_round_used_recovered_parent": bool(recovery["next_round"] and recovery["next_round"]["matches_restored_parent"]),
                               "reproduced_candidate_weights_predictions_selections_and_scores": reproducible,
                               "no_data_access_violations_or_network_attempts": not audit["denied_accesses"] and not audit["network_attempts"],
                               "normal_candidate_promotions": sum(r["selected_artifact_id"] == f"candidate-{r['round']}" for r in records),
                               "normal_historical_rollbacks": sum(r["action"] == "rollback" for r in records),
                               "physical_causal_attribution_verified": False},
              "rounds": records, "recovery": recovery, "audit": audit, "source_sha256": source_hashes,
              "sdk_wheel_sha256": digest(wheel), "real_platform_model_calls": 0,
              "official_hidden_score": None, "docker_or_official_host_verified": False,
              "limitations": ["All selection and scores use authorized public validation; repeated tuning is not an independent hidden/generalization test.",
                              "Model decisions and call IDs are explicitly simulated; this does not satisfy the formal real-model-call requirement.",
                              "Native process audit is not a Docker or official host sandbox.",
                              "Fault recovery uses a separate injected model anomaly, not a naturally occurring performance rollback."]}
    write_json(root / "simulation-report.json", report)
    table = "\n".join(f"| {r['round']} | {r['history_records_read']} | {r['experiment'] or r['action']} | {r['host_candidate_macro_f1']} | {r['selected_artifact_id']} | {r['selected_macro_f1']:.9f} |" for r in records)
    md = ("# 当前代码离线平台模拟报告\n\n仅使用官方公开 train/validation，决策模型和宿主为离线替身，真实平台模型调用为 0。\n\n"
          f"静态基线 Macro-F1：{baseline_score:.9f}；最终：{final_score:.9f}；差值：{final_score-baseline_score:+.9f}。\n\n"
          f"含非空历史的完整实验决策记录：{len(experiments)} 次；独立重复结果一致：{reproducible}。\n\n"
          f"应用审计：数据/网络违规 {len(audit['denied_accesses']) + len(audit['network_attempts'])} 次；另行拦截 Windows 版本查询 {len(audit['blocked_os_version_queries'])} 次，均保留原始日志，未放行子进程。\n\n"
          "| 回合 | 已读历史条数 | 实验/动作 | 宿主候选 F1 | 选中产物 | 选中 F1 |\n|---|---:|---|---:|---|---:|\n" + table + "\n\n"
          f"独立故障演练实际回滚：{recovery['rollback_selected']}，分块恢复 {recovery['historical_read_chunks']} 次；故障分数 {recovery['fault_macro_f1']:.9f}，恢复后 {recovery['recovered_macro_f1']:.9f}。公开样本及标签未修改。\n\n"
          "完整证据见 simulation-report.json、replica-*/rounds.jsonl、host-*/protocol.jsonl、model-calls.jsonl、access.jsonl、measurements/*、候选与 state。reproduction/ 保存源码、配置与 SDK wheel；official-baseline/ 和 official-public/ 保存官方输入及摘要对应文件。\n\n"
          "上述公开验证结果不能推断隐藏集成绩；离线调用号不是平台真实模型回执。未执行 Docker、官方宿主或真实 PyTRiO 验证。回滚分数来自独立权重故障注入，不能混入正常训练成绩。\n")
    (root / "SIMULATION_REPORT.md").write_text(md, encoding="utf-8")
    reproduction_guide = """# 复现本次模拟

本目录包含正式源码、仿真宿主与审计器、锁定依赖及 SDK wheel。上级 official-baseline/ 含官方检查点、摘要说明和原始公开 train/validation；所有标签均来自该公开部分。

在本 reproduction/ 目录新建 Python 3.12 环境并安装 requirements.txt，然后运行：

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -B scripts/simulate_competition.py --baseline-dir ../official-baseline --output runtime/reproduced --rounds 10 --replicas 2
```

输出目录必须尚不存在。NumPy 的 wheel 需预先准备或从可用包源安装；执行模拟本身不访问模型服务。严格权重/预测摘要一致性已在报告所记的同机环境验证，跨操作系统或 BLAS 环境的逐字节一致性尚未验证。

正常运行和故障注入分目录记录，故障权重不进入正常成绩。模型决策和调用号为明确标注的离线替身，不能提交为平台真实模型调用证据。单次 JSONL 宿主等待超过预设时限会终止演练，不会重试结果未知的 RPC。
"""
    (root / "reproduction/REPRODUCE.md").write_text(reproduction_guide, encoding="utf-8")
    print(json.dumps({"report": str(root / "simulation-report.json"), "requirements": report["requirements"],
                      "baseline_macro_f1": baseline_score, "final_macro_f1": final_score}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
