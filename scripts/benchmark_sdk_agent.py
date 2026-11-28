"""SDK 1.0.5 local agent timing: real containers/training, offline decisions.

Compare sequential A/B/C scheduling with three independent dataset processes.
No provider API, cloud account, or official platform host is contacted.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    from gas_a.sensor_training.io import write_json
    write_json(path, value)


def frozen_sources():
    names = ["harness.py", "contest_submission.py", "submission_config.json", "scripts/benchmark_sdk_agent.py",
             "scripts/sdk_protocol_audit.py", "scripts/validate_platform.py", "scripts/validate_multitask.py",
             "scripts/docker_rehearsal.py", "tests/test_submission.py", "tests/test_regression.py"]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}


def max_overlap(intervals):
    changes = [(item["start"], 1) for item in intervals] + [(item["end"], -1) for item in intervals]
    active = maximum = 0
    for _, delta in sorted(changes):  # End before start at an identical timestamp.
        active += delta
        maximum = max(maximum, active)
    return maximum


def summarize_worker(out, dataset, started, execution_started, report, runtime):
    rounds = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "run").glob("host-*/sdk-timing.json")]
    rounds.sort(key=lambda r: r["round_index"])
    measurements = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "run").glob("measurement-*/timing.json")]
    times = [r["process_exit_seconds"] for r in rounds]
    if not rounds or any(not r["order_valid"] or r["status"] != "completed" or not r["acknowledged"] for r in rounds):
        raise ValueError("A round failed SDK ordering, acknowledgement, or process completion")
    waits, stages, counts = defaultdict(float), defaultdict(float), Counter()
    for row in rounds:
        for kind, value in row["host_wait_seconds_by_request"].items(): waits[kind] += value
        counts.update(row["request_counts"])
        for stage in row["application_stages"]:
            if stage.get("completed"): stages[stage["stage"]] += stage["seconds"]
    return {"dataset": dataset, "status": "completed", "started_at_unix": started,
        "execution_started_at_unix": execution_started, "ended_at_unix": time.time(),
        "worker_wall_seconds": time.time() - started, "execution_wall_seconds": time.time() - execution_started,
        "rounds_completed": len(rounds), "round_mean_seconds": statistics.mean(times), "round_max_seconds": max(times),
        "round_min_seconds": min(times), "round_wall_seconds_sum": sum(times),
        "first_protocol_request_max_seconds": max(r["first_protocol_request_seconds"] for r in rounds),
        "round_trips_total": sum(r["round_trips"] for r in rounds), "request_counts": dict(counts),
        "host_wait_seconds_by_request": dict(waits), "application_stage_seconds": dict(stages),
        "measurement_count": len(measurements),
        "measurement_wall_seconds_sum": sum(m["total_measurement_seconds"] for m in measurements),
        "measurement_time_overlaps_candidate_rpc_wait": True,
        "all_rounds_order_valid": True, "all_rounds_acknowledged": True,
        "final_selected_artifact_id": report["selected_artifact_id"], "scope": "local_offline_model_decisions_in_fixed_docker_image",
        "complete_history_experiments": report["complete_history_experiments"],
        "not_worse_than_public_baseline": report["not_worse_than_public_baseline"],
        "baseline": report["baseline"], "final": report["final"],
        "real_model_calls": 0, "runtime_audit": runtime.audit(), "rounds": rounds}


def worker(config, dataset, out):
    import contest_submission
    from scripts.docker_rehearsal import DockerRuntime
    from scripts.validate_multitask import BASELINES, run_dataset
    started = time.time()
    out.mkdir(parents=True)
    save(out / "status.json", {"status": "starting", "dataset": dataset, "started_at_unix": started})
    try:
        if frozen_sources() != config["source_sha256"]:
            raise ValueError("Host or application source changed before worker startup")
        runtime = DockerRuntime(config["image"], executable=config["docker_executable"], context=config["docker_context"],
            expected_id=config["image_id"], required_tasks=("classification", "regression"))
        save(out / "image-preflight.json", runtime.preflight())
        execution_started = time.time()
        save(out / "status.json", {"status": "running", "dataset": dataset, "started_at_unix": started,
                                   "execution_started_at_unix": execution_started})
        report = run_dataset(Path(config["baseline_root"]) / BASELINES[dataset], out / "run", dataset,
            config["rounds"], runtime, compact_validation=config.get("compact_validation", False),
            aggregate_parent=config.get("aggregate_parent", False), exchange_options={"round_timeout": config["round_timeout"],
                "rpc_delay_seconds": config["rpc_delay_seconds"], "model_delay_seconds": config["model_delay_seconds"]})
        if frozen_sources() != config["source_sha256"]:
            raise ValueError("Execution source changed while testing")
        result = summarize_worker(out, dataset, started, execution_started, report, runtime)
        if result["rounds_completed"] != config["rounds"]:
            raise ValueError("Worker did not complete all requested rounds")
        if config["rounds"] >= 4 and result["complete_history_experiments"] < 3:
            raise ValueError("Fewer than three nonempty-history experiments completed")
        if not result["not_worse_than_public_baseline"]:
            raise ValueError("Final public validation is worse than the same-scope baseline")
        save(out / "result.json", result)
        save(out / "status.json", {k: result[k] for k in ("dataset", "status", "started_at_unix", "ended_at_unix", "rounds_completed")})
        return 0
    except BaseException as error:
        save(out / "result.json", {"dataset": dataset, "status": "failed", "error_type": type(error).__name__,
            "message": str(error)[:1200], "started_at_unix": started, "ended_at_unix": time.time(), "real_model_calls": 0})
        raise


def verify_sdk_and_image(args, out):
    import contest_harness
    from scripts.docker_rehearsal import DockerRuntime
    if importlib.metadata.version("contest-harness") != "1.0.5":
        raise ValueError("Installed SDK must be 1.0.5")
    package = Path(contest_harness.__file__).parent
    sdk_expected, template = {}, None
    with tarfile.open(args.sdk_archive, "r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if member.isfile() and "contest_harness" in parts and member.name.endswith(".py"):
                relative = Path(*parts[parts.index("contest_harness") + 1:])
                raw = archive.extractfile(member).read()
                if (package / relative).read_bytes() != raw:
                    raise ValueError("Installed SDK differs from user supplied source: " + str(relative))
                sdk_expected[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
            if member.isfile() and member.name.endswith("/template/harness.py"):
                template = archive.extractfile(member).read()
    if not sdk_expected or template != (ROOT / "harness.py").read_bytes():
        raise ValueError("Harness does not equal the original SDK template")
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                            expected_id=args.expected_image_id, required_tasks=("classification", "regression"))
    proof = runtime.preflight()
    paths = [Path(n) for n in ("harness.py", "contest_submission.py", "submission_config.json", "requirements.txt",
                                "vendor/contest_harness-1.0.5-py3-none-any.whl")]
    paths += [p.relative_to(ROOT) for p in (ROOT / "src").rglob("*.py")]
    app_expected = {p.as_posix(): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    script = """import hashlib,json,pathlib,sys,contest_harness
req=json.loads(sys.argv[1]); root=pathlib.Path(contest_harness.__file__).parent
print(json.dumps({'app':{p:hashlib.sha256((pathlib.Path('/app')/p).read_bytes()).hexdigest() for p in req['app']},'sdk':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in req['sdk']}}))"""
    actual = json.loads(runtime._run(["run", "--rm", "--pull=never", "--network", "none", "--read-only", "--entrypoint", "python3",
        runtime.image, "-c", script, json.dumps({"app": list(app_expected), "sdk": list(sdk_expected)})]).stdout)
    if actual != {"app": app_expected, "sdk": sdk_expected}:
        raise ValueError("Fixed image SDK or application does not match the tested source")
    proof.update(original_sdk_modules_matched=len(sdk_expected), application_files_matched=len(app_expected),
                 original_template_matched=True, application_sha256=app_expected)
    save(out / "image-preflight.json", proof)
    resources = json.loads(runtime._run(["info", "--format", "{{json .}} "]).stdout)
    save(out / "docker-resources.json", {"cpus": resources["NCPU"], "memory_bytes": resources["MemTotal"]})
    return runtime


def run_mode(mode, config_path, config, root):
    from scripts.docker_rehearsal import child_environment
    root.mkdir()
    launched, snapshots, started = [], [], time.time()

    def launch(dataset):
        log = (root / (dataset + ".log")).open("w", encoding="utf-8")
        command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--worker", dataset,
                   "--worker-config", str(config_path), "--output", str(root / dataset)]
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=log, stderr=subprocess.STDOUT)
        item = {"dataset": dataset, "process": process, "log": log, "launched_at_unix": time.time()}
        launched.append(item)
        return item

    def wait(items):
        last_log = 0
        with (root / "progress.jsonl").open("a", encoding="utf-8") as stream:
            while any(item["process"].poll() is None for item in items):
                now = time.time()
                progress = {}
                for item in launched:
                    files = list((root / item["dataset"] / "run").glob("host-*/sdk-timing.json"))
                    progress[item["dataset"]] = sum(json.loads(p.read_text(encoding="utf-8"))["status"] == "completed" for p in files)
                row = {"at_unix": now, "elapsed_seconds": now - started, "completed_rounds": progress}
                stream.write(json.dumps(row) + "\n"); stream.flush(); snapshots.append(row)
                if now - last_log >= 20:
                    print(json.dumps({"mode": mode, **row}), flush=True)
                    last_log = now
                for item in items:
                    if item["process"].poll() is None and now - item["launched_at_unix"] > config["worker_timeout"]:
                        item["cancelled"] = True
                        item["process"].terminate()
                        item["process"].wait(timeout=15)
                        raise TimeoutError("Dataset worker timeout: " + item["dataset"])
                time.sleep(1)
        for item in items:
            item["process"].wait()
            item["log"].close()
            if item["process"].returncode:
                raise RuntimeError("Dataset worker failed: " + item["dataset"] + "; inspect " + str(root / (item["dataset"] + ".log")))

    try:
        if mode == "serial":
            for dataset in "ABC": wait([launch(dataset)])
        else:
            wait([launch(dataset) for dataset in "ABC"])
    finally:
        for item in launched:
            if item["process"].poll() is None:
                item["cancelled"] = True
                item["process"].terminate(); item["process"].wait(timeout=15)
            if not item["log"].closed: item["log"].close()
        cancelled = [item for item in launched if item.get("cancelled")]
        if cancelled:
            from scripts.docker_rehearsal import DockerRuntime
            cleanup = DockerRuntime(config["image"], executable=config["docker_executable"], context=config["docker_context"])
            owned = set()
            for item in cancelled:
                for path in (root / item["dataset"] / "run").rglob("container.jsonl"):
                    for line in path.read_text(encoding="utf-8").splitlines():
                        row = json.loads(line); argv = row["argv"]
                        owned.add(argv[argv.index("--name") + 1])
            running = cleanup._run(["ps", "--filter", "label=gas-rsi-real-rehearsal=true", "--format", "{{.Names}}"])
            for name in set(running.stdout.splitlines()) & owned: cleanup.cleanup(name)
    results = [json.loads((root / dataset / "result.json").read_text(encoding="utf-8")) for dataset in "ABC"]
    intervals = [{"start": row["execution_started_at_unix"], "end": row["ended_at_unix"]} for row in results]
    result = {"mode": mode, "wall_seconds": time.time() - started, "datasets": results,
        "max_overlapping_dataset_workers": max_overlap(intervals), "new_real_model_calls": 0, "cloud_writes": 0}
    save(root / "summary.json", result)
    return result


def docker_overlaps(events_path, modes, root):
    owned = {}
    for mode in modes:
        for p in (root / mode).glob("*/run/**/container.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                row = json.loads(line); argv = row["argv"]
                owned[argv[argv.index("--name") + 1]] = {"mode": mode, "phase": row["command"], "dataset": p.relative_to(root / mode).parts[0]}
    containers = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        event = json.loads(line); actor = event.get("Actor", {})
        name = actor.get("Attributes", {}).get("name")
        if name not in owned or event.get("Action") not in ("start", "die"): continue
        item = containers.setdefault(actor["ID"], {**owned[name], "name": name})
        item["start" if event["Action"] == "start" else "end"] = event.get("timeNano", event["time"] * 1_000_000_000) / 1e9
    complete = [row for row in containers.values() if "start" in row and "end" in row]
    result = {}
    for mode in modes:
        selected = [row for row in complete if row["mode"] == mode]
        result[mode] = {"owned_containers_expected": sum(row["mode"] == mode for row in owned.values()),
            "container_start_die_intervals_captured": len(selected),
            "max_overlapping_train_containers": max_overlap([r for r in selected if r["phase"] == "train"]),
            "max_overlapping_predict_containers": max_overlap([r for r in selected if r["phase"] == "predict"]),
            "max_overlapping_all_containers": max_overlap(selected), "intervals": selected}
    return result


def compare_results(serial, parallel):
    result = []
    for dataset in "ABC":
        left, right = serial / dataset / "run", parallel / dataset / "run"
        source, target = {}, {}
        for root, values in ((left, source), (right, target)):
            for p in root.glob("host-*/candidate/**/*.npz"):
                values[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("measurement-*/predictions.jsonl"):
                values[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("host-*/candidate/routing.json"):
                values[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        result.append({"dataset": dataset, "files_checked": len(source), "model_and_predictions_byte_equal": bool(source) and source == target})
    return result


def write_outputs(root, config, results, docker):
    rows = []
    for mode in results:
        for dataset in mode["datasets"]:
            for row in dataset["rounds"]:
                waits = row["host_wait_seconds_by_request"]
                rows.append({"mode": mode["mode"], "dataset": dataset["dataset"], "round": row["round_index"],
                    "process_seconds": row["process_exit_seconds"], "to_ack_seconds": row["elapsed_seconds_to_ack"],
                    "first_request_seconds": row["first_protocol_request_seconds"], "round_trips": row["round_trips"],
                    "model_wait_seconds": waits.get("model_request", 0), "candidate_host_seconds": waits.get("candidate_request", 0),
                    "history_read_seconds": waits.get("artifact_read_request", 0),
                    "order_valid": row["order_valid"], "acknowledged": row["acknowledged"]})
    with (root / "round-timings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    equality = compare_results(root / "serial", root / "parallel") if {r["mode"] for r in results} == {"serial", "parallel"} else []
    summary = {"format": "sdk105.agent-benchmark.v1", "status": "completed", "sdk_version": "1.0.5",
        "image_id": config["image_id"], "rounds_per_dataset": config["rounds"], "scenarios": results,
        "docker_event_concurrency": docker, "serial_parallel_comparison": equality,
        "all_order_checks_passed": all(row["order_valid"] and row["acknowledged"] for row in rows),
        "docker_event_capture_complete": all(d["owned_containers_expected"] > 0 and d["owned_containers_expected"] == d["container_start_die_intervals_captured"] for d in docker.values()),
        "new_real_model_calls": 0, "cloud_writes": 0,
        "platform_concurrency_confirmed": False, "platform_parallel_mode_shown": True,
        "platform_scheduler_source_available": False,
        "platform_inference": "The provided platform screenshot explicitly shows parallel mode. Its actual container overlap, resource allocation and timeout boundary are not visible.",
        "sdk_fact": "One Round serializes its RPCs with a lock; the SDK does not define cross-dataset scheduling.",
        "repository_runner_fact": "test_fresh_platform.execute_phases currently loops through A,B,C sequentially.",
        "timing_scope": "Real numerical training and SDK host prediction; model decisions are offline stubs, not fresh 27B latency."}
    if equality and not all(row["model_and_predictions_byte_equal"] for row in equality):
        summary["status"] = "comparison_failed"
    if not summary["docker_event_capture_complete"]:
        summary["status"] = "evidence_incomplete"
    if len(results) == 2:
        by_mode = {r["mode"]: r for r in results}
        summary["serial_to_parallel_wall_speedup"] = by_mode["serial"]["wall_seconds"] / by_mode["parallel"]["wall_seconds"]
    save(root / "summary.json", summary)
    lines = ["SDK 1.0.5 测试 agent 计时与并发验证", "",
        "固定镜像：" + config["image_id"], "每个数据集轮数：" + str(config["rounds"]),
        "真实模型新增调用：0；云端写入：0。使用离线决策，真实执行训练、SDK 传输和预测。", ""]
    for mode in results:
        proof = docker[mode["mode"]]
        lines.append("%s：总耗时 %.3f 秒；Docker 同时运行的训练容器最多 %d 个。" %
            (mode["mode"], mode["wall_seconds"], proof["max_overlapping_train_containers"]))
        for dataset in mode["datasets"]:
            lines.append("  %s：完成 %d 轮；数据集执行 %.3f 秒；每轮平均 %.3f 秒，最长 %.3f 秒；RPC 往返 %d 次。" %
                (dataset["dataset"], dataset["rounds_completed"], dataset["execution_wall_seconds"], dataset["round_mean_seconds"], dataset["round_max_seconds"], dataset["round_trips_total"]))
    lines += ["", "顺序检查：start；连续 RPC id；单个未完成请求；对应类型/ID 回复；上传偏移、摘要与完成确认；候选/状态事务在上传完成后；真实已读历史和模型回执引用；最终 round_ack；进程退出。",
        "宿主等待包含传输、宿主处理和预测；participant_gap 包含管道调度等待，并不等于纯 CPU 时间。候选评测时预测容器与等待中的训练容器重叠是正常嵌套，不能据此误判为跨数据集并行。",
        "统计 measurement 总时长与 candidate_request 等待存在重叠，不能把两者直接相加。", "",
        "平台结论：用户提供的截图明确显示“并行”，本次按 A/B/C 三路并行验收。SDK 源码只规定单会话 RPC 串行，不包含组委会调度器。平台实际容器时间交集、资源池和超时边界不可见；本地容器并行由 Docker start/die 事件独立核验。", "",
        "本地通过不保证平台 180 秒预检通过；真实网关等待、排队及跨任务资源争用未由离线模型替身覆盖。", "",
        "逐轮数据：round-timings.csv；逐 RPC 数据：各 mode/dataset/run/host-N/sdk-timing.json；Docker start/die 证据：docker-events.jsonl 与 summary.json。"]
    (root / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--sdk-archive", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--expected-image-id")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--modes", nargs="+", choices=("serial", "parallel"), default=["serial", "parallel"])
    parser.add_argument("--round-timeout", type=float, default=180)
    parser.add_argument("--worker-timeout", type=float, default=1800)
    parser.add_argument("--rpc-delay-seconds", type=float, default=0)
    parser.add_argument("--model-delay-seconds", type=float, default=0)
    parser.add_argument("--compact-validation", action="store_true")
    parser.add_argument("--aggregate-parent", action="store_true",
                        help="Use a test-only opaque checkpoint-group identity, not an individual file digest")
    parser.add_argument("--worker", choices=("A", "B", "C"))
    parser.add_argument("--worker-config", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(json.loads(args.worker_config.read_text(encoding="utf-8")), args.worker, args.output.resolve())
    if not args.image or not args.expected_image_id:
        parser.error("--image and --expected-image-id are required for SDK 1.0.5 image validation")
    if not args.baseline_root or not args.sdk_archive or not 1 <= args.rounds <= 10:
        parser.error("baseline-root, sdk-archive and rounds=1..10 are required")
    if len(set(args.modes)) != len(args.modes): parser.error("Duplicate modes")
    import math
    for value in (args.round_timeout, args.worker_timeout):
        if not math.isfinite(value) or value <= 0: parser.error("Timeouts must be finite and positive")
    if not 0 <= args.rpc_delay_seconds <= 2 or not 0 <= args.model_delay_seconds <= 30:
        parser.error("rpc-delay must be 0..2 and model-delay 0..30 seconds")
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    import contest_submission
    from scripts.docker_rehearsal import child_environment
    runtime = verify_sdk_and_image(args, root)
    config = {"image": args.image, "image_id": runtime.image, "docker_executable": args.docker_executable,
        "docker_context": args.docker_context, "baseline_root": str(args.baseline_root.resolve()), "rounds": args.rounds,
        "round_timeout": args.round_timeout, "worker_timeout": args.worker_timeout,
        "rpc_delay_seconds": args.rpc_delay_seconds, "model_delay_seconds": args.model_delay_seconds,
        "compact_validation": args.compact_validation, "aggregate_parent": args.aggregate_parent,
        "source_sha256": frozen_sources()}
    save(root / "config.json", config)
    started = time.time(); results = []
    events_path = root / "docker-events.jsonl"
    with events_path.open("w", encoding="utf-8") as events, (root / "docker-events-stderr.log").open("w", encoding="utf-8") as error:
        monitor = subprocess.Popen([*runtime.cli, "events", "--since", str(int(started)), "--filter", "type=container",
            "--filter", "label=gas-rsi-real-rehearsal=true", "--format", "{{json .}}"],
            stdout=events, stderr=error, env=child_environment())
        try:
            for mode in args.modes: results.append(run_mode(mode, root / "config.json", config, root / mode))
        except BaseException as failure:
            save(root / "failure.json", {"status": "failed", "error_type": type(failure).__name__, "message": str(failure)[:1200], "new_real_model_calls": 0})
            raise
        finally:
            time.sleep(.25)  # Let the final Docker die event reach the local stream.
            monitor.terminate(); monitor.wait(timeout=15)
    if frozen_sources() != config["source_sha256"]: raise ValueError("Source changed during benchmark")
    summary = write_outputs(root, config, results, docker_overlaps(events_path, args.modes, root))
    print(json.dumps({"status": summary["status"], "output": str(root), "rounds": sum(r["rounds_completed"] for mode in results for r in mode["datasets"]),
                      "new_real_model_calls": 0, "platform_concurrency_confirmed": False}), flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
