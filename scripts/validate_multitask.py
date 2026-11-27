"""A/B/C 原生入口与实际数值训练的离线验收；模型决策使用明确标注的宿主替身，不计真实27B成绩。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from tests.test_submission import OfflineHost
from tests.test_regression import RegressionHost
from scripts.validate_platform import BOOTSTRAP, exchange, receipt
from scripts.docker_rehearsal import DockerRuntime, child_environment
from gas_a.sensor_training.io import digest, write_json
from gas_a.regression.data import load_pair as regression_pair
from gas_a.regression.schedule import ordered_experiments
from gas_a.regression.evaluation import evaluate as regression_evaluate
from gas_a.sensor_training.data import load_pair as classification_pair, evaluate as classification_evaluate
from gas_a.frozen_router import load_predictor

BASELINES = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]


def complete_experiment(row):
    """Count measured model decisions; a program guard is not a closed loop."""
    cfg = row["record"]["configuration"]
    locally_measured = (cfg.get("candidate_upload_skipped") == "local_promotion_gate_rejected"
                        and isinstance(cfg.get("validation_after"), dict) and isinstance(cfg.get("candidate_gate"), dict)
                        and cfg["candidate_gate"].get("eligible") is False and not cfg.get("experiment_failure"))
    return (bool(row["history_ids"]) and (row["candidate_submitted"] or locally_measured)
            and cfg.get("action") == "experiment"
            and cfg.get("controller_decisions") == {"plan": "model", "select": "model"})


def measure(runtime, directory, descriptor, validation_path, validation, dataset, out):
    """通过实际 predict 进程测量；只发 inputs，预测容器不挂载带 answer 的数据。"""
    out.mkdir()
    started = time.perf_counter()
    task = "classification" if dataset == "A" else "regression"
    context = {"protocol": "contest.predict-context.v1", "dataset_key": dataset, "task": task,
               "artifact_id": "measured", "artifact": descriptor}
    env = child_environment()
    if runtime:
        argv = runtime.predict(directory, directory / descriptor["checkpoint_paths"][0], out, context=context)
    else:
        argv = [sys.executable, "-B", str(ROOT / "harness.py"), "predict"]
        env.update(CONTEST_ARTIFACT_PATH=str(directory / descriptor["checkpoint_paths"][0]),
                   CONTEST_PREDICTION_CONTEXT=json.dumps(context))
    rows = [json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_json(out / "validation-scope.json", {"validation_sha256": digest(validation_path), "sample_ids": [row["id"] for row in rows],
        "canonical_rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    # 标签只留在宿主用于算分，stdin 发送 task 和 inputs；预测容器只挂载模型，不挂载 validation 文件。
    inputs = "".join(json.dumps({"task": task, "inputs": row["inputs"]}) + "\n" for row in rows)
    try:
        prediction_started = time.perf_counter()
        result = subprocess.run(argv, input=inputs, text=True, encoding="utf-8", capture_output=True, env=env, timeout=120)
        prediction_seconds = time.perf_counter() - prediction_started
        (out / "stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Native prediction process failed; inspect " + str(out / "stderr.log"))
    finally:
        if runtime:
            runtime.cleanup_prediction(argv)
    (out / "predictions.jsonl").write_text(result.stdout, encoding="utf-8")
    outputs = [json.loads(line)["outputs"] for line in result.stdout.splitlines()]
    if len(outputs) != len(rows):
        raise ValueError("Native prediction row count mismatch")
    if dataset == "A":
        local = load_predictor(directory / descriptor["checkpoint_paths"][0])
        expected = local.predict(validation["x"])
        values = np.asarray([[item["score"] for item in row] for row in outputs])
        if not np.allclose(values, expected, rtol=1e-8, atol=1e-10):
            raise ValueError("Container classification predictions differ from local reevaluation")
        metrics = classification_evaluate(local, validation)
    else:
        if any(not isinstance(row, dict) or set(row) != {"value"} or isinstance(row["value"], bool) for row in outputs):
            raise ValueError("Native regression output contract invalid")
        values = np.asarray([row["value"] for row in outputs])
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Native RUL predictions must be finite and nonnegative")
        class Measured:
            def predict(self, _):
                return values
        metrics = regression_evaluate(Measured(), validation, dataset)
    write_json(out / "metrics.json", metrics)
    write_json(out / "timing.json", {"dataset": dataset, "rows": len(rows),
        "prediction_process_seconds": prediction_seconds, "total_measurement_seconds": time.perf_counter() - started,
        "scope": "local_host_preparation_prediction_and_metric_evaluation"})
    return metrics


def run_dataset(base, out, dataset, rounds, runtime, *, peer_factory=None, journal=None, max_rounds=None, exchange_options=None,
                compact_validation=False, aggregate_parent=False, experiment_sequence=None, projection_callback=None,
                follow_menu_priority=False):
    if follow_menu_priority and (peer_factory is not None or experiment_sequence is not None):
        raise ValueError("Menu priority replay requires an unscripted offline peer")
    # 一个数据集的多轮宿主：registry 保存真实产物目录，current 永远是上轮最终选中的 ID。
    # peer_factory 决定模型来源（真实回复或记录重放）；缺省才使用明确标识的离线模型替身。
    out.mkdir()
    shutil.copytree(base / "public", out / "data")
    (out / "initial").mkdir()
    shutil.copyfile(base / "checkpoint.npz", out / "initial/checkpoint.npz")
    info = json.loads((base / "baseline-info.json").read_text(encoding="utf-8"))
    if info["dataset_key"] != dataset or digest(out / "initial/checkpoint.npz") != info["checkpoint_sha256"]:
        raise ValueError("Official baseline identity does not match dataset")
    _, validation = (classification_pair(out / "data/train.jsonl", out / "data/validation.jsonl") if dataset == "A"
                     else regression_pair(out / "data/train.jsonl", out / "data/validation.jsonl", dataset))
    initial_descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": PREDICT_ARGV}
    registry = {"initial": (out / "initial", receipt(out / "initial", "initial", initial_descriptor))}
    current, state, history, records, measurement_index = "initial", None, [], [], 0

    def measured(directory, descriptor):
        # 基线、新候选与选中版本都走同一预测入口并各自保存逐条输出，避免口径混比。
        nonlocal measurement_index
        measurement_index += 1
        return measure(runtime, directory, descriptor, out / "data/validation.jsonl", validation, dataset,
                       out / f"measurement-{measurement_index}")

    score = baseline = measured(out / "initial", initial_descriptor)
    for index in range(1, (max_rounds or rounds) + 1):
        directory, parent_receipt = registry[current]
        if projection_callback and projection_callback(index, out / "data"):
            _, validation = (classification_pair(out / "data/train.jsonl", out / "data/validation.jsonl") if dataset == "A"
                             else regression_pair(out / "data/train.jsonl", out / "data/validation.jsonl", dataset))
            score = measured(directory, parent_receipt["artifact"])
        parent = directory / parent_receipt["artifact"]["checkpoint_paths"][0]
        host = (peer_factory(out, parent, dataset, index, history, state) if peer_factory else
                OfflineHost(out, parent, index, history=history, state=state) if dataset == "A" else
                RegressionHost(out, parent, dataset, index, history=history, state=state))
        host.frame["parent_artifact"] = parent_receipt
        host.frame["parent_checkpoint_sha256"] = digest(parent)
        if aggregate_parent and (aggregate_parent != "multi" or len(parent_receipt["artifact"]["checkpoint_paths"]) > 1):
            # 测试专用整组身份，刻意不等于任一单文件摘要；官方聚合算法由宿主持有。
            from tests.test_submission import simulated_checkpoint_identity
            host.frame["parent_checkpoint_sha256"] = simulated_checkpoint_identity(parent_receipt)
        host.frame["current_validation"] = {"metrics": score, "scope": "authorized_public_validation",
                                            "validation_sha256": digest(out / "data/validation.jsonl")}
        if compact_validation:
            # 平台反馈复现：start 只有具名指标，不含冗余范围/摘要；原始帧仍完整留档。
            host.frame["current_validation"] = {"metrics": score}
        config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
        names = ([spec["name"] for spec in config["experiments"]] if dataset == "A"
                 else list(ordered_experiments(config["regression"], dataset)))
        if follow_menu_priority:
            host.experiment = None  # Offline peer follows the first permitted experiment in the actual request.
        elif peer_factory is None:
            host.experiment = (experiment_sequence[(index - 1) % len(experiment_sequence)] if experiment_sequence
                               else names[(index - 1) % len(names)])
            if host.experiment not in names:
                raise ValueError("Validation sequence requests an unconfigured experiment")
        for identifier, (source, _) in registry.items():
            host.add_artifact(identifier, source)
        # The host remeasures every newly uploaded candidate through the same native prediction entry.
        original_write = host.write
        def receive(text):
            result = original_write(text)
            frame = json.loads(text)
            if frame["type"] == "candidate_request":
                metric = measured(host.candidate, frame["artifact"])
                host.candidate_metrics = metric
                host.pending[-1]["summary"]["metrics"] = {k: metric[k] for k in ("macro_f1", "rmse", "mae") if k in metric}
                if compact_validation:
                    host.pending[-1]["summary"] = {"metrics": host.pending[-1]["summary"]["metrics"]}
            return result
        host.write = receive
        env = child_environment()
        if runtime:
            host.frame["task_contract"]["public_projection"].update(train_path="/data/train.jsonl", validation_path="/data/validation.jsonl")
            host.frame["state_path"] = "/state" if state else None
            argv, name = runtime.train(directory, out / "data", state, host.store / "container.jsonl")
            exchange(host, argv, env, host.store, name, docker_argv=runtime.cli, **(exchange_options or {}))
        else:
            with tempfile.TemporaryDirectory(prefix="native-rul-work-") as work:
                env.update(CONTEST_WORK=work, GAS_A_TEST_ROOT=str(ROOT), GAS_A_TEST_PARENT=str(directory))
                exchange(host, [sys.executable, "-B", "-c", BOOTSTRAP], env, host.store, **(exchange_options or {}))
        if host.candidate:
            # 提交的新候选先登记，不直接变成父模型；只有 finish.selected_artifact_id 决定下轮起点。
            descriptor = next(frame["artifact"] for frame in host.frames if frame["type"] == "candidate_request")
            identifier = f"candidate-{index}"
            registry[identifier] = (host.candidate, receipt(host.candidate, identifier, descriptor))
        current = host.record["selected_artifact_id"]
        directory, selected_receipt = registry[current]
        score = measured(directory, selected_receipt["artifact"])
        host.selected_metrics = score
        state = host.saved_state
        history.append({"id": f"{'real-local' if peer_factory else 'offline'}-{dataset}-history-{index}", **host.record})
        row = {"round": index, "history_ids": host.record["history_ids"], "candidate_submitted": host.candidate is not None,
               "model_call_ids": host.record["model_call_ids"], "selected_artifact_id": current, "metrics": score,
               "finish_acknowledged": True, "record": host.record}
        if peer_factory is None and dataset in ("B", "C"):
            plan_request = json.loads(host.messages[0][0]["content"])
            configuration = host.record["configuration"]
            actual = (configuration.get("experiment") or {}).get("name")
            unavailable = plan_request.get("strategy", {}).get("unavailable_experiments", {})
            requested = (plan_request.get("strategy", {}).get("scheduled_experiment_preference")
                         if follow_menu_priority else host.experiment)
            row["experiment_schedule"] = {"requested": requested, "executed": actual,
                "substitution_reason": None if actual == requested else unavailable.get(
                    requested, "observed_or_selected_other_action")}
        records.append(row)
        if journal:
            journal.after_round(host)
        print(json.dumps({"dataset": dataset, "round": index, "history_count": len(row["history_ids"]),
                          "candidate_submitted": row["candidate_submitted"], "selected_artifact_id": current,
                          "metrics": {k: v for k, v in score.items() if k in ("macro_f1", "rmse", "mae", "nasa_score")}}), flush=True)
        complete_count = sum(complete_experiment(r) for r in records)
        # 不能用“总共有四轮”代替“有三次完整历史实验”；观察/回滚不会增加这个计数。
        if index >= rounds and complete_count >= 3:
            break
    metric = "macro_f1" if dataset == "A" else "rmse"
    result = {"dataset": dataset, "task": "classification" if dataset == "A" else "regression",
              "scope": "real_27b_native_sdk_in_docker" if peer_factory else "authorized_public_validation_with_offline_model_stub",
              "real_model_calls": len(host.gateway.verified) if peer_factory else 0,
              "model_call_attempts": len(host.gateway.attempts) if peer_factory else 0,
              "selected_artifact_id": current,
              "baseline": baseline, "final": score, "rounds": records,
              "complete_history_experiments": sum(complete_experiment(r) for r in records),
              "not_worse_than_public_baseline": score[metric] >= baseline[metric] - 1e-9 if dataset == "A" else score[metric] <= baseline[metric] + 1e-9,
              "official_platform_score_verified": False}
    result["validation_receipt_format"] = "compact" if compact_validation else "explicit"
    write_json(out / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", choices=("A", "B", "C"), nargs="+", default=["A", "B", "C"])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--compact-validation", action="store_true", help="Reproduce compact start/candidate validation summaries")
    parser.add_argument("--image")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        parser.error("rounds must be 1..10")
    out = args.output.resolve()
    out.mkdir()  # 不覆盖任何历史实验。
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                            required_tasks=("classification", "regression")) if args.image else None
    if runtime:
        write_json(out / "image-preflight.json", runtime.preflight())
    reports = [run_dataset(args.baseline_root / BASELINES[key], out / key, key, args.rounds, runtime,
                           compact_validation=args.compact_validation) for key in args.datasets]
    audit = runtime.audit() if runtime else None
    checks = {
        "all_requested_rounds_completed": all(len(r["rounds"]) == args.rounds for r in reports),
        "not_worse_than_public_baselines": all(r["not_worse_than_public_baseline"] for r in reports),
        "history_experiments_each_dataset": all(r["complete_history_experiments"] >= args.rounds - 1 for r in reports),
        "model_stub_calls_recorded_each_round": all(row["model_call_ids"] for r in reports for row in r["rounds"]),
        "docker_isolation_verified": bool(audit and audit["container_policy_verified"]) if runtime else True,
    }
    passed = all(checks.values())
    write_json(out / "summary.json", {"status": "completed" if passed else "requirements_unmet", "checks": checks,
                                      "real_model_calls": 0, "reports": reports, "runtime": audit or "native"})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
