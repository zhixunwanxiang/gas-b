"""Exercise the actual Harness with an offline model stub, natively or in Docker.

This is a transport/sandbox rehearsal, never an official model or score run.
The script is a host-side tool and is excluded from the submission image.
"""
# 离线协议验收：将宿主 start、模型/上传/状态 RPC 和结束确认与真实入口串接。
# 支持原生进程和镜像测试，关注逐行刷新、跨轮状态及协议行为，不替代正式平台成绩。
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_submission import OfflineHost
from gas_a.sensor_training.io import digest, write_json
import numpy as np


BOOTSTRAP = """
import os, runpy, sys
from pathlib import Path
from contest_harness.round import ParentView
original = ParentView.__init__
def init(self, frame):
    original(self, frame)
    self.root = Path(os.environ['GAS_A_TEST_PARENT'])
ParentView.__init__ = init
root = Path(os.environ['GAS_A_TEST_ROOT'])
sys.path.insert(0, str(root))
sys.argv = [str(root / 'harness.py'), 'train']
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def receipt(directory, identifier, descriptor):
    files = [{"path": p.relative_to(directory).as_posix(), "sha256": digest(p), "size_bytes": p.stat().st_size}
             for p in sorted(directory.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    value = {"files": files, "artifact": descriptor}
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return {"artifact_id": identifier, **value, "artifact_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest()}


def exchange(host, command, env, output, container_name=None, docker_argv=None):
    # 写入一帧 start 后边读选手输出边发送相应宿主帧，不能等 stdin 结束才开始交互。
    # 收到 round_complete 后发确认，再等待进程退出；异常也清理本次拥有的容器和管道。
    with (output / "stderr.log").open("w", encoding="utf-8") as stderr, \
            (output / "protocol.jsonl").open("w", encoding="utf-8") as transcript:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8")
        lines = queue.Queue()
        def read_lines():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        try:
            def respond():
                response = host.readline()
                transcript.write(json.dumps({"direction": "host", "frame": json.loads(response)}) + "\n")
                process.stdin.write(response)
                process.stdin.flush()
            respond()
            while host.record is None:
                line = lines.get(timeout=120)
                if line is None:
                    raise RuntimeError("Harness exited before round acknowledgement; inspect stderr.log")
                frame = json.loads(line)
                transcript.write(json.dumps({"direction": "harness", "frame": frame}) + "\n")
                if frame.get("type") == "round_failed":
                    raise RuntimeError("Harness round failed: " + line)
                host.write(line)
                respond()
            process.stdin.close()
            if process.wait(timeout=15) != 0:
                raise RuntimeError("Harness exited unsuccessfully; inspect stderr.log")
            reader.join(timeout=2)
            if not lines.empty() and lines.get_nowait() is not None:
                raise RuntimeError("Unexpected output after round acknowledgement")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for stream in (process.stdin, process.stdout):
                stream.close()
            if container_name:
                # Only the uniquely named container created by this invocation.
                subprocess.run([*(docker_argv or ["docker"]), "rm", "-f", container_name],
                               env=env, capture_output=True, timeout=30)


def check_prediction(root, directory, descriptor, image):
    # 验证逐条输入在下一条/EOF 前已有响应，并核对实际选中产物的输出，排除只在退出时批量打印。
    from gas_a.frozen_router import load_predictor
    from gas_a.sensor_training.io import CLASSES
    primary = descriptor["checkpoint_paths"][0]
    model = load_predictor(directory / primary)
    name = "gas-a-predict-" + uuid.uuid4().hex if image else None
    env = {key: value for key, value in os.environ.items() if not key.startswith("CONTEST_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CONTEST_ARTIFACT_PATH=str(directory / primary))
    if image:
        command = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only",
                   "--cpus", "2", "--memory", "4g", "--pids-limit", "128", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
                   "--mount", f"type=bind,source={directory},target=/model,readonly",
                   "-e", f"CONTEST_ARTIFACT_PATH=/model/{primary}", image, "predict"]
    else:
        command = [sys.executable, "-B", str(ROOT / "harness.py"), "predict"]
    rows = [json.loads(line) for line in (root / "data/validation.jsonl").read_text(encoding="utf-8").splitlines()][:10]
    with (root / "prediction-stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, encoding="utf-8")
        lines = queue.Queue()
        def read_lines():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        outputs = []
        try:
            for row in rows:
                process.stdin.write(json.dumps({"task": "classification", "inputs": row["inputs"]}) + "\n")
                process.stdin.flush()
                # The next input and EOF are withheld until this prediction arrives.
                line = lines.get(timeout=15)
                if line is None:
                    raise RuntimeError("Prediction process closed stdout early")
                value = json.loads(line)
                scores = value["outputs"]
                probabilities = np.asarray([item["score"] for item in scores])
                expected = getattr(model, "predict_one", model.predict)(np.asarray([row["inputs"]["features"]]))[0]
                if (len(scores) != 6 or [item["label"] for item in scores] != CLASSES
                        or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
                        or abs(float(probabilities.sum()) - 1) > 1e-9
                        or np.count_nonzero(probabilities == probabilities.max()) != 1
                        or not np.allclose(probabilities, expected, rtol=0, atol=1e-10)):
                    raise RuntimeError("Prediction contract or selected-artifact equality check failed")
                outputs.append(value)
            process.stdin.close()
            if process.wait(timeout=15) != 0:
                raise RuntimeError("Prediction process failed; inspect prediction-stderr.log")
            reader.join(timeout=2)
            if not lines.empty() and lines.get_nowait() is not None:
                raise RuntimeError("Unexpected extra prediction output")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
            if name:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
    result = {"rows": len(outputs), "replies_before_next_row_and_eof": True,
              "matches_selected_artifact": True, "training_data_mounted_in_prediction_container": False if image else None}
    write_json(root / "prediction-check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="Primary checkpoint of the complete initial artifact")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrites a previous run")
    parser.add_argument("--rounds", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--experiment", default="four_sensor")
    parser.add_argument("--image", help="Built image tag. Omitting this runs the real SDK natively.")
    args = parser.parse_args()
    if args.image and not shutil.which("docker"):
        parser.error("Docker is unavailable; container validation has not run")
    source = args.parent.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "data").mkdir()
    shutil.copyfile(args.train, root / "data/train.jsonl")
    shutil.copyfile(args.validation, root / "data/validation.jsonl")
    # Copy only this predictor's files, including its verified router sidecars.
    # A baseline package may also contain datasets that must not become models.
    from gas_a.frozen_router import checkpoint_paths, load_predictor
    model = load_predictor(source)
    declared = checkpoint_paths(source)
    manifest = getattr(model, "manifest_path", None)
    code_paths = [manifest.relative_to(source.parent).as_posix()] if manifest else []
    for name in declared + code_paths:
        destination = root / "initial" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / name, destination)
    parent = root / "initial" / source.name
    descriptor = {"checkpoint_paths": declared, "code_paths": code_paths,
                  "predict_argv": ["python3", "/app/harness.py", "predict"]}
    initial = receipt(parent.parent, "initial", descriptor)
    registry = {"initial": (parent.parent, initial)}
    selected, history, state, summaries = "initial", [], None, []
    for index in range(1, args.rounds + 1):
        directory, parent_receipt = registry[selected]
        host = OfflineHost(root, parent, index, history, state, experiment=args.experiment)
        host.frame["parent_artifact"] = parent_receipt
        host.frame["artifacts"] = [item[1] for identifier, item in registry.items() if identifier != selected]
        host.artifact_dirs = {identifier: item[0] for identifier, item in registry.items()}
        # Explicit offline controller scenarios, not real LLM decisions.
        if index == 3:
            host.action = "observe"
        if index == 4:
            host.malformed_responses = 1
        # Once a menu item is blocked, the stub chooses observation.
        original_write = host.write
        def write_frame(text):
            frame = json.loads(text)
            if frame.get("type") == "model_request":
                request = json.loads(frame["messages"][0]["content"])
                if request["phase"] == "plan" and args.experiment not in request["experiments"]:
                    host.action = "observe"
            return original_write(text)
        host.write = write_frame
        env = {key: value for key, value in os.environ.items() if not key.startswith("CONTEST_")}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                   GAS_A_TEST_ROOT=str(ROOT), GAS_A_TEST_PARENT=str(directory))
        name = None
        if args.image:
            name = "gas-a-check-" + uuid.uuid4().hex
            command = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only",
                       "--cpus", "2", "--memory", "4g", "--pids-limit", "128", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--tmpfs", "/workspace:rw,nosuid,nodev,size=4g,mode=1777",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777", "--tmpfs", "/output:rw,nosuid,nodev,size=1g,mode=1777",
                       "--mount", f"type=bind,source={directory},target=/previous,readonly",
                       "--mount", f"type=bind,source={root / 'data'},target=/data,readonly"]
            projection = host.frame["task_contract"]["public_projection"]
            projection.update(train_path="/data/train.jsonl", validation_path="/data/validation.jsonl")
            if state:
                command.extend(["--mount", f"type=bind,source={state},target=/state,readonly"])
                host.frame["state_path"] = "/state"
            command.extend([args.image, "train"])
        else:
            command = [sys.executable, "-B", "-c", BOOTSTRAP]
        # Native execution keeps all participant writes outside the /app stand-in.
        # SDK uploads preserve reviewable candidates/state in the host output.
        with tempfile.TemporaryDirectory(prefix="gas-sdk-rehearsal-") as workspace:
            env["CONTEST_WORK"] = workspace
            exchange(host, command, env, host.store, name)
        if host.candidate:
            artifact = next(f["artifact"] for f in host.frames if f["type"] == "candidate_request")
            identifier = f"candidate-{index}"
            registry[identifier] = (host.candidate, receipt(host.candidate, identifier, artifact))
        selected = host.record["selected_artifact_id"]
        parent = registry[selected][0] / registry[selected][1]["artifact"]["checkpoint_paths"][0]
        state = host.saved_state
        history.append({"id": f"offline-history-{index}", **host.record})
        summary = {"round": index, "action": host.record["configuration"]["action"],
                   "selected_artifact_id": selected, "configuration": host.record["configuration"],
                   "model_calls": len(host.calls), "artifact_read_chunks": sum(f["type"] == "artifact_read_request" for f in host.frames)}
        summaries.append(summary)
        print(json.dumps({"round": index, "action": summary["action"], "selected": selected}, ensure_ascii=False), flush=True)
        write_json(root / "validation-summary.json", {
            "scope": "offline_model_stub_container_sandbox" if args.image else "offline_model_stub_native_sdk",
            "sdk_version": "1.0.4", "image": args.image, "rounds_completed": index,
            "real_approved_controller_verified": False, "official_hidden_score": None, "rounds": summaries})
    prediction = check_prediction(root, registry[selected][0], registry[selected][1]["artifact"], args.image)
    summary_path = root / "validation-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prediction"] = prediction
    sources = list((ROOT / "src/gas_a").rglob("*.py"))
    sources.extend(ROOT / name for name in ("harness.py", "contest_submission.py", "submission_config.json",
                                            "Dockerfile", "requirements.txt", "scripts/validate_platform.py"))
    summary["source_sha256"] = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
