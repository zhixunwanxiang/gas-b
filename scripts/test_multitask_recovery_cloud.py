"""v5 A/B/C 真实回滚补充验收：复用已核验正常记录，只修改独立父模型副本。会产生费用。"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from scripts.test_multitask_cloud import RealPeerFactory, snapshot, preflight_source
from scripts.test_real_sdk import MODEL
from scripts.test_cloud_suite import DEFAULT_IMAGE, DEFAULT_IMAGE_ID
from scripts.verify_multitask_cloud_run import verify_dataset, read, rows
from scripts.validate_multitask import measure, classification_pair, regression_pair
from scripts.validate_platform import receipt, exchange
from scripts.docker_rehearsal import DockerRuntime, child_environment
from scripts.cloud_experiment import CloudJournal
from scripts.live_platforms import load_env, configuration, ensure_model, termination_handler, write_receipt
from gas_a.sensor_training.io import digest
from gas_a.sensor_training.reference.model import Model


def model_manifest(directory):
    # 核对完整产物，不能只比较主权重；路由的所有组件、配置和声明代码均计入。
    return {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def preserve_failure(source, output):
    if not source or not (source / "suite.json").is_file() or read(source / "suite.json")["status"] == "completed":
        return None
    target = output / "reproduction/prior-failure.zip"
    count = 0
    secrets = [os.environ[k].encode() for k in ("PYTRIO_API_KEY", "SWANLAB_API_KEY") if os.environ.get(k)]
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or any(p in ("swanlog", "__pycache__") for p in path.parts):
                continue
            if path.name.startswith(".env") or path.name.endswith(".zip"):
                continue
            data = path.read_bytes()
            if any(secret in data for secret in secrets):
                raise ValueError("Refusing to archive credential-containing prior failure")
            archive.writestr(path.relative_to(source).as_posix(), data)
            count += 1
    result = {"original_suite": read(source / "suite.json"), "zip_sha256": digest(target), "files": count,
              "original_records_modified": False}
    write_receipt(output / "previous-failure.json", result)
    return result


def prepare(source, recovery, dataset):
    # 从已独立核验的正常阶段复制历史产物、公开数据和状态，不借用其他批次的实验成绩。
    # 每个历史产物复制后核对全部文件摘要，故障副本与原始产物分开保存。
    recovery.mkdir()
    report = read(source / "normal/report.json")
    registry, history = {}, []
    history_root = recovery / "history-artifacts"
    history_root.mkdir()
    shutil.copytree(source / "normal/data", recovery / "data")
    descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                  "predict_argv": ["python3", "/app/harness.py", "predict"]}

    def remember(identifier, directory, artifact):
        target = history_root / identifier
        shutil.copytree(directory, target)
        if model_manifest(target) != model_manifest(directory):
            raise ValueError("Historical model changed during recovery preparation")
        registry[identifier] = (target, receipt(target, identifier, artifact))

    remember("initial", source / "normal/initial", descriptor)
    for row in report["rounds"]:
        index = row["round"]
        frames = [event["frame"] for event in rows(source / f"normal/host-{index}/protocol.jsonl")]
        candidate = next((frame for frame in frames if frame.get("type") == "candidate_request"), None)
        if candidate:
            reply = next(frame for frame in frames if frame.get("type") == "candidate_response")
            remember(reply["artifact_id"], source / f"normal/host-{index}/candidate", candidate["artifact"])
        history.append({"id": f"real-local-{dataset}-history-{index}", **row["record"]})
    prior_state = recovery / "prior-state"
    shutil.copytree(source / f"normal/host-{report['rounds'][-1]['round']}/state", prior_state)
    fault = recovery / "fault-parent"
    fault.mkdir()
    model = Model.load(source / "official-baseline/checkpoint.npz")
    # 故障方式在脚本中预设：A 置换输出类别，B/C 增加标准化目标偏置。
    # 只改内存副本的输出参数，既不改原始权重，也不改任何训练/验证样本或标签。
    if dataset == "A":
        model.params["w2"] = np.roll(model.params["w2"], 1, axis=1)
        model.params["b2"] = np.roll(model.params["b2"], 1)
        fault_kind = "Cyclic output class weight/bias permutation of an isolated baseline copy"
    else:
        model.params["b2"] += 10.0
        fault_kind = "Added 10 normalized target units to output bias of an isolated baseline copy"
    model.save(fault / "checkpoint.npz")
    registry["fault-parent"] = (fault, receipt(fault, "fault-parent", descriptor))
    write_receipt(recovery / "source-normal-report.json", report)
    write_receipt(recovery / "fault.json", {"kind": fault_kind, "dataset": dataset,
        "original_checkpoint_sha256": digest(source / "official-baseline/checkpoint.npz"),
        "fault_checkpoint_sha256": digest(fault / "checkpoint.npz"), "data_or_labels_modified": False})
    return registry, history, prior_state, report


def recovery_phase(source, out, dataset, runtime, factory, journal):
    # 真正考验两件事：第一轮自主选回历史，第二轮实际从恢复后的完整产物继续运行。
    # 比较对象仍是相同公开 validation；故障演练不能当成隐藏流上的自然漂移成绩。
    registry, history, state, normal = prepare(source, out, dataset)
    _, validation = (classification_pair(out / "data/train.jsonl", out / "data/validation.jsonl") if dataset == "A" else
                     regression_pair(out / "data/train.jsonl", out / "data/validation.jsonl", dataset))
    counter = 0

    def measured(identifier):
        nonlocal counter
        counter += 1
        directory, artifact = registry[identifier]
        return measure(runtime, directory, artifact["artifact"], out / "data/validation.jsonl", validation,
                       dataset, out / f"measurement-{counter}")

    current = "fault-parent"
    fault_metrics = score = measured(current)
    history_models = set(registry) - {"fault-parent"}
    records = []
    start_index = normal["rounds"][-1]["round"] + 1
    recovered_manifest = None
    next_parent_matches = False
    for offset in range(2):
        index = start_index + offset
        parent_id = current
        directory, artifact = registry[parent_id]
        if offset == 1 and recovered_manifest is not None:
            # 不能只比较 checkpoint 主权重：路由组件、配置和声明代码也必须与恢复目标逐一一致。
            next_parent_matches = model_manifest(directory) == recovered_manifest
        parent = directory / artifact["artifact"]["checkpoint_paths"][0]
        host = factory(out, parent, dataset, index, history, state)
        host.frame["parent_artifact"] = artifact
        host.frame["parent_checkpoint_sha256"] = digest(parent)
        host.frame["current_validation"] = {"metrics": score, "scope": "authorized_public_validation",
                                            "validation_sha256": digest(out / "data/validation.jsonl")}
        # 使用真实原始产物描述；模型只可恢复本帧列出的ID。
        for identifier, (saved, saved_receipt) in registry.items():
            if identifier != parent_id:
                host.frame["artifacts"].append(saved_receipt)
                host.artifact_dirs[identifier] = saved
        original_write = host.write

        def receive(text):
            nonlocal counter
            result = original_write(text)
            frame = json.loads(text)
            if frame["type"] == "candidate_request":
                counter += 1
                metric = measure(runtime, host.candidate, frame["artifact"], out / "data/validation.jsonl", validation,
                                 dataset, out / f"measurement-{counter}")
                host.candidate_metrics = metric
                host.pending[-1]["summary"]["metrics"] = {k: metric[k] for k in ("macro_f1", "rmse", "mae") if k in metric}
            return result

        host.write = receive
        host.frame["task_contract"]["public_projection"].update(train_path="/data/train.jsonl", validation_path="/data/validation.jsonl")
        host.frame["state_path"] = "/state"
        argv, name = runtime.train(directory, out / "data", state, host.store / "container.jsonl")
        exchange(host, argv, child_environment(), host.store, name, docker_argv=runtime.cli)
        if host.candidate:
            descriptor = next(f["artifact"] for f in host.frames if f["type"] == "candidate_request")
            identifier = f"candidate-{index}"
            registry[identifier] = (host.candidate, receipt(host.candidate, identifier, descriptor))
        current = host.record["selected_artifact_id"]
        score = measured(current)
        host.selected_metrics = score
        state = host.saved_state
        decision = rows(state / "decisions.jsonl")[-1]
        real_rollback = (offset == 0 and decision["plan"]["action"] == "rollback"
                         and current in history_models and current != parent_id)
        if real_rollback:
            recovered_manifest = model_manifest(registry[current][0])
        record = {"round": index, "parent_artifact_id": parent_id, "parent_checkpoint_sha256": digest(parent),
                  "selected_artifact_id": current, "metrics": score, "actual_historical_rollback": real_rollback,
                  "model_call_ids": host.record["model_call_ids"], "finish_acknowledged": True,
                  "candidate_submitted": host.candidate is not None, "plan": decision["plan"], "record": host.record}
        records.append(record)
        history.append({"id": f"real-recovery-{dataset}-history-{index}", **host.record})
        journal.after_round(host)
        print(json.dumps({"dataset": dataset, "recovery_round": offset + 1, "selected": current,
                          "actual_historical_rollback": real_rollback,
                          "metrics": {k: v for k, v in score.items() if k in ("macro_f1", "rmse", "mae", "nasa_score")}}), flush=True)
    key = "macro_f1" if dataset == "A" else "rmse"
    restored = records[0]["metrics"]
    not_worse = (restored[key] >= normal["baseline"][key] if dataset == "A" else restored[key] <= normal["baseline"][key])
    improved_fault = (restored[key] > fault_metrics[key] if dataset == "A" else restored[key] < fault_metrics[key])
    checks = {"two_rounds_acknowledged": len(records) == 2 and all(r["finish_acknowledged"] for r in records),
              "actual_historical_rollback": records[0]["actual_historical_rollback"],
              "next_round_uses_complete_restored_artifact": next_parent_matches,
              "fault_performance_restored": improved_fault,
              "recovered_not_worse_than_public_baseline": not_worse,
              "real_model_receipts_each_round": all(r["model_call_ids"] for r in records)}
    result = {"dataset": dataset, "checks": checks, "requirements_passed": all(checks.values()),
              "fault_metrics": fault_metrics, "restored_metrics": restored, "normal_final": normal["final"],
              "baseline": normal["baseline"], "rounds": records, "real_model_calls": len(factory.gateway.verified),
              "provider_reported_total_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in factory.gateway.verified),
              "restored_model_manifest": recovered_manifest, "official_host_or_hidden_score_verified": False}
    write_receipt(out / "report.json", result)
    return result


def run_one(args, dataset, config):
    source = args.source.resolve() / dataset
    source_verified = verify_dataset(source, dataset)
    out = args.output.resolve() / dataset
    out.mkdir()
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                            expected_id=args.expected_image_id, required_tasks=("classification", "regression"))
    probe = runtime.preflight()
    hashes = snapshot(out, source / "official-baseline")
    previous_failure = preserve_failure(args.previous_failures.resolve() / dataset if args.previous_failures else None, out)
    probe["application_source_files_matched"] = preflight_source(runtime, hashes)
    write_receipt(out / "image-preflight.json", probe)
    write_receipt(out / "source-normal-verification.json", source_verified)
    import swanlab
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    journal = CloudJournal(swanlab, config, out, {"dataset_key": dataset, "model": MODEL,
        "image_id": runtime.image, "purpose": "weight_fault_historical_recovery_and_next_parent_verification",
        "source_normal_cloud_url": source_verified["cloud_url"], "source_normal_evidence_sha256": source_verified["evidence_zip_sha256"],
        "previous_failure": previous_failure,
        "maximum_real_calls": args.max_calls, "public_validation_only": True, "hidden_labels_available": False},
        f"gas-{dataset.lower()}-real-recovery-{stamp}")
    journal.phase = "recovery"
    factory = RealPeerFactory(config, os.environ["PYTRIO_API_KEY"], journal, args.max_calls)
    result = {"dataset": dataset, "status": "starting", "image_id": runtime.image, "real_model_calls": 0,
              "cloud_verified": False, "source_normal_cloud_url": source_verified["cloud_url"]}
    error = None
    finish_state = "crashed"
    try:
        journal.begin()
        result.update(status="running", cloud_url=journal.receipt["url"])
        write_receipt(out / "suite.json", result)
        print(json.dumps({"event": "real_recovery_cloud_started", "dataset": dataset, "url": result["cloud_url"]}), flush=True)
        report = recovery_phase(source, out / "recovery", dataset, runtime, factory, journal)
        result.update(requirements_passed=report["requirements_passed"], checks=report["checks"],
            real_model_calls=report["real_model_calls"], provider_reported_total_tokens=report["provider_reported_total_tokens"],
            fault_metrics=report["fault_metrics"], restored_metrics=report["restored_metrics"], runtime=runtime.audit())
        journal.emit(texts={"recovery/summary": report}, scalars={"suite/recovery_passed": int(report["requirements_passed"])})
        write_receipt(out / "suite.json", result)
        journal.archive()
        finish_state = "success" if report["requirements_passed"] else "crashed"
    except BaseException as caught:
        error = caught
        result.update(status="failed", error_type=type(caught).__name__)
        if factory.gateway:
            result.update(real_model_calls=len(factory.gateway.verified), attempted_calls=len(factory.gateway.attempts))
        finish_state = "aborted" if isinstance(caught, (KeyboardInterrupt, SystemExit)) else "crashed"
    finally:
        write_receipt(out / "suite.json", result)
        journal.finish(finish_state)
    if error is None:
        try:
            verified = journal.verify_cloud()
            result.update(cloud_verified=True, cloud_state=verified["state"], cloud_url=verified["url"])
        except Exception as caught:
            result["cloud_verification_error_type"] = type(caught).__name__
    result["status"] = "completed" if result.get("requirements_passed") and result["cloud_verified"] else "needs_attention"
    write_receipt(out / "suite.json", result)
    print(json.dumps({"event": "real_recovery_dataset_finished", **result}, ensure_ascii=False), flush=True)
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--expected-image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    parser.add_argument("--datasets", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--previous-failures", type=Path, help="Preserve prior failed run evidence in the new archive without modifying it")
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 6:
        parser.error("Recovery max-calls must be 1..6 per dataset")
    load_env(args.env_file)
    config = configuration()
    if any(not os.environ.get(k, "").strip() for k in ("PYTRIO_API_KEY", "SWANLAB_API_KEY")):
        raise ValueError("Host credentials required")
    if config["model"] != MODEL or config["base_url"] != "https://pytrio.com":
        raise ValueError("Exact authorized model and endpoint required")
    args.output.resolve().mkdir(parents=True, exist_ok=False)
    ensure_model(config, os.environ["PYTRIO_API_KEY"])
    results = []
    with termination_handler():
        for dataset in args.datasets:
            results.append(run_one(args, dataset, config))
            write_receipt(args.output.resolve() / "summary.json", {"results": results,
                "real_model_calls": sum(r["real_model_calls"] for r in results),
                "status": "completed" if len(results) == len(args.datasets) and all(r["status"] == "completed" for r in results) else "running_or_needs_attention"})
    return 0 if all(r["status"] == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
