"""Local SDK 1.0.5 host rehearsal backed by real PyTRiO Qwen3.8-27B replies.

This host-side tool is excluded from the submission image. It spends real model
credits; it is not the organizer's host and its IDs are provider receipt IDs.
"""
# 真实模型协议桥接：选手 rnd.model → SDK JSONL → 宿主 PyTRiO 请求 → 原始回复 → SDK 回执。
# 实际训练和预测可运行在固定 Docker 镜像中；该脚本本身不创建云端 Run，云端由外层测试总控负责。
# 提供方响应 ID 不能冒充正式比赛计量回执；未知结果不自动重试，也不补写模型决定。
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.live_platforms import load_env, configuration, ensure_model, request_json
from scripts.simulate_competition import (EvidenceHost, Measurements, artifact_receipt,
                                          adapter, environment, audit_summary)
from scripts.validate_platform import exchange
from scripts.docker_rehearsal import DockerRuntime
from gas_a.sensor_training.data import load_pair
from gas_a.sensor_training.io import digest, write_json, append_jsonl
from gas_a.sensor_training.reference.model import Model
import numpy as np

MODEL = "Qwen/Qwen3.8-27B"
LOCAL_PROMPT = (
    "这是本地 SDK 1.0.5 协议测试的固定任务说明，不是赛方正式宿主提示。"
    "任务 A：使用128维气体传感器特征预测六类气体，主指标 Macro-F1 越高越好。"
    "仅公开训练集可更新梯度，公开验证集可评估和选版；不得获取或推断隐藏标签。"
    "从实际父产物续训，保留 Adam 和标准化状态。"
    "阅读选手消息中的历史、测量、允许动作及 response_schema，按其字段返回决策。"
    "结果未执行前只是假设，不能捏造提升；模型决定必须由实际实验结果支持。"
)


def request_phase(messages):
    # 从选手结构化用户消息识别 plan/select，仅用于证据归类，不改变请求内容。
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            try:
                value = json.loads(message["content"])
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("phase") in ("plan", "select"):
                return value["phase"]
    return "unknown"


class RealGateway:
    def __init__(self, config, key, root, max_calls=16):
        self.config, self.key, self.root, self.max_calls = config, key, root, max_calls
        self.attempts = []
        self.verified = []
        (root / "provider-calls").mkdir()

    def invoke(self, frame, start):
        # 发送前计入尝试预算并写 outcome_unknown；收到完整合法响应后才记 verified_response。
        # 宿主只附加一次固定任务提示，透传选手采样参数；服务超时/中断不自动重试。
        if len(self.attempts) >= self.max_calls:
            raise RuntimeError("Real model call budget exhausted; no request sent")
        if self.config["model"] != MODEL or self.config["base_url"] != "https://pytrio.com":
            raise ValueError("Only the configured PyTRiO Qwen3.8-27B endpoint is allowed")
        messages = [dict(item) for item in frame["messages"]]
        metadata = {key:start[key] for key in ("dataset_key", "round_index", "task_contract", "prompt_version")}
        prefix = start["prompt"] + "\n" + json.dumps(metadata, ensure_ascii=False)
        if messages[0]["role"] == "system":
            messages[0]["content"] = prefix + "\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": prefix})
        # 保留选手实际传入的 max_tokens/temperature/enable_thinking 等采样参数，宿主不偷偷改写。
        options = {k:v for k,v in frame.items() if k not in ("type", "id", "messages")}
        payload = {**options, "model": MODEL, "messages": messages, "stream": False}
        phase = request_phase(frame["messages"])
        number = len(self.attempts) + 1
        folder = self.root / "provider-calls" / f"call-{number:03d}"
        folder.mkdir()
        entry = {"number": number, "round": start["round_index"], "phase": phase,
                 "status": "outcome_unknown", "requested_model": MODEL,
                 "started_at": datetime.now(timezone.utc).isoformat(),
                 "call_id_source": "pytrio_provider_response_not_official_contest_receipt"}
        # 发请求前就计入预算；超时也可能已被提供方处理，不能因缺少响应而退还这次尝试额度。
        self.attempts.append(entry)
        write_json(folder / "request.json", payload)  # Never persist authorization headers.
        write_json(folder / "receipt.json", entry)
        print(json.dumps({"event":"real_model_request_started", "call":number,
                          "round":start["round_index"], "phase":phase}), flush=True)
        started = time.monotonic()
        response_received = False
        try:
            response = request_json(self.config["base_url"] + "/api/openai/v1/chat/completions", self.key, payload)
            response_received = True
            write_json(folder / "response.json", response)
            call_id = response.get("id")
            if (response.get("model") != MODEL or not isinstance(call_id, str) or not call_id
                    or any(item["call_id"] == call_id for item in self.verified)):
                raise ValueError("Provider returned an unexpected model or invalid/duplicate request ID")
            if not isinstance(response.get("choices"), list) or not response["choices"]:
                raise ValueError("Provider returned no choices")
            entry.update(status="verified_response", call_id=call_id,
                         latency_seconds=time.monotonic()-started, usage=response.get("usage", {}))
            self.verified.append(entry)
            write_json(folder / "receipt.json", entry)
            return call_id, response, messages
        except BaseException as error:
            entry.update(error_type=type(error).__name__, latency_seconds=time.monotonic()-started)
            if response_received:
                entry["status"] = "response_unverified"
            write_json(folder / "receipt.json", entry)
            # Never retry a request with an unknown outcome or fabricate a reply.
            raise


class RealModelHost(EvidenceHost):
    def __init__(self, *args, gateway, runtime=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gateway, self.runtime = gateway, runtime
        self.frame.update(model_name=MODEL, prompt=LOCAL_PROMPT,
                          prompt_version="local-real-27b-sdk-1.0.5-v1")
        if runtime is not None:
            self.frame["task_contract"]["public_projection"].update(
                train_path="/data/train.jsonl", validation_path="/data/validation.jsonl")
            if self.frame.get("state_path"):
                self.frame["state_path"] = "/state"

    def write(self, text):
        # 单独接管 model_request 并调用真实网关，不能先走离线父类的模型规则分支。
        # 其他上传、状态和历史 RPC 继续复用宿主协议实现，响应只绑定实际提供方 call_id。
        frame = json.loads(text)
        if frame["type"] != "model_request":
            result = super().write(text)
            if frame["type"] == "candidate_request":
                self.pending[-1]["summary"]["scope"] = "authorized_public_validation"
            return result
        # Do not invoke the offline peer's model branch, even as an intermediate.
        if frame["id"] != self.next_sequence:
            raise ValueError("Unexpected SDK RPC sequence")
        self.next_sequence += 1
        self.frames.append(frame)
        self.messages.append(frame["messages"])
        # 从真实网关得到原始 response 后原样放进 SDK model_response，JSON 解析与行动决定仍在选手代码中。
        call_id, response, bound = self.gateway.invoke(frame, self.frame)
        self.calls.append(call_id)
        append_jsonl(self.store / "model-calls.jsonl", {
            "phase":request_phase(frame["messages"]), "call_id":call_id, "source":"real_pytrio_provider",
            "real_remote_model_call":True, "official_contest_model_receipt":False,
            "raw_messages":frame["messages"], "bound_messages":bound, "response":response})
        self.pending.append({"type":"model_response", "id":frame["id"],
                             "call_id":call_id, "response":response})
        return len(text)


def run_native_round(host, directory, state, dataset_root):
    # 剥离模型/云端凭证后启动选手进程；Docker 模式使用容器内投影路径并在退出时清理本轮容器。
    with tempfile.TemporaryDirectory(prefix="gas-real-sdk-") as work:
        env = environment()
        for name in list(env):
            if name.startswith(("PYTRIO_", "SWANLAB_")) or name == "OPENAI_API_KEY":
                env.pop(name)
        if host.runtime is not None:
            argv, name = host.runtime.train(directory, host.root / "data", state,
                                             host.store / "container.jsonl")
            exchange(host, argv, env, host.store, name, docker_argv=host.runtime.cli)
        else:
            argv = adapter("train", directory, Path(work), host.store / "access.jsonl", dataset_root,
                           data=host.root / "data", state=state)
            exchange(host, argv, env, host.store)


def measurements(out, validation, dataset_root, runtime):
    return Measurements(out, validation, dataset_root,
                        prediction_command=runtime.predict if runtime else None,
                        prediction_cleanup=runtime.cleanup_prediction if runtime else None)


def audit(out, runtime):
    return runtime.audit() if runtime else audit_summary(out)


def finalize_acceptance(report, *, recovery=False, minimum_rounds=4):
    """Separate execution completion from the requirements this test claims."""
    # 同时检查回合确认、真实调用、成绩、历史实验数和实际隔离证据；回滚还要求真的选回历史并延续父产物。
    rows = report.get("rounds", [])
    audit = report.get("audit", {})
    checks = {
        "acknowledged_rounds": bool(rows) and all(row.get("finish_acknowledged") for row in rows),
        "model_receipts_each_round": bool(rows) and all(row.get("model_call_ids") for row in rows),
    }
    if audit.get("mode") == "docker_enforced_mounts_and_network":
        checks["docker_mount_network_resource_policy_verified"] = (
            audit.get("container_policy_verified") is True
            and audit.get("train_containers", 0) > 0 and audit.get("prediction_containers", 0) > 0)
    else:
        checks["no_observed_data_or_network_violations"] = (
            "denied_accesses" in audit and "network_attempts" in audit
            and not audit["denied_accesses"] and not audit["network_attempts"])
    baseline = report.get("baseline_macro_f1")
    score = report.get("recovered_macro_f1" if recovery else "final_macro_f1")
    comparable_scores = all(not isinstance(v, bool) and isinstance(v, (int, float)) and 0 <= v <= 1 and math.isfinite(v)
                            for v in (baseline, score))
    checks["not_below_public_baseline"] = comparable_scores and score + 1e-12 >= baseline
    if recovery:
        checks.update(two_rounds_completed=len(rows) == 2,
                      historical_rollback=report.get("rollback_selected") is True,
                      recovered_parent_used=report.get("next_round_used_recovered_parent") is True)
    else:
        checks.update(requested_rounds_completed=len(rows) >= minimum_rounds,
                      three_history_experiments=report.get("complete_history_experiment_decision_records", 0) >= 3)
    passed = all(checks.values())
    report.update(acceptance_scope="local_authorized_public_validation_only", requirements=checks,
                  requirements_passed=passed, status="completed" if passed else "requirements_unmet")
    return 0 if passed else 1


def recovery_test(source, out, base, gateway, report, runtime=None):
    """Inject weights only; real 27B chooses both actions without forced replies."""
    # 在正常运行副本上只注入模型权重故障，保存原公开数据和历史产物，两个动作都交给真实模型决定。
    original = json.loads((source/"live-report.json").read_text(encoding="utf-8"))
    if original["status"] != "completed" or original["model"] != MODEL:
        raise ValueError("Recovery requires a completed real-model session")
    shutil.copytree(source/"data",out/"data")
    for name,expected in original["data_sha256"].items():
        if digest(out/"data"/name) != expected:
            raise ValueError("Recovery public input digest mismatch")
    registry = {"initial":(source/"initial",artifact_receipt(source/"initial","initial"))}
    for row in original["rounds"]:
        if not row["candidate_submitted"]:
            continue
        directory = source/f"host-{row['round']}/candidate"
        frames = [json.loads(line) for line in (source/f"host-{row['round']}/protocol.jsonl").read_text(encoding="utf-8").splitlines()]
        descriptor = next(e["frame"]["artifact"] for e in frames if e["direction"] == "harness" and e["frame"].get("type") == "candidate_request")
        identifier = f"candidate-{row['round']}"
        registry[identifier] = (directory,artifact_receipt(directory,identifier,descriptor))
    history = [{"id":f"local-real-history-{row['round']}",**row["record"]} for row in original["rounds"]]
    fault = out/"fault-parent"
    fault.mkdir()
    model = Model.load(source/"initial/checkpoint.npz")
    model.params["w2"] = np.roll(model.params["w2"],1,axis=1)
    model.params["b2"] = np.roll(model.params["b2"],1)
    model.save(fault/"checkpoint.npz")
    selected = "fault-injected-parent"
    registry[selected] = (fault,artifact_receipt(fault,selected))
    _,validation = load_pair(out/"data/train.jsonl",out/"data/validation.jsonl")
    measured = measurements(out,validation,base.parent,runtime)
    for identifier,(directory,receipt) in registry.items():
        measured.measure(directory,receipt["artifact"],identifier)
    report.update(status="running", scope="real_27b_sdk_separate_weight_fault_recovery",
                  baseline_macro_f1=original["baseline_macro_f1"],
                  fault_macro_f1=measured.by_id[selected]["metrics"]["macro_f1"],
                  fault="Cyclic output weight/bias permutation of a separate baseline copy; data and labels unchanged",
                  source_session=str(source), forced_model_decisions=False)
    state = source/"selected-state"
    recovered_sha = None
    for index in (1,2):
        directory,receipt = registry[selected]
        host = RealModelHost(out,directory/receipt["artifact"]["checkpoint_paths"][0],index,history,state,
                             measurements=measured,gateway=gateway,runtime=runtime,artifact_namespace="recovery-")
        host.frame["parent_artifact"] = receipt
        host.frame["artifacts"] = [value[1] for identifier,value in registry.items() if identifier != selected]
        host.artifact_dirs = {identifier:value[0] for identifier,value in registry.items()}
        host.frame["current_validation"] = {"metrics":measured.by_id[selected]["metrics"],
            "validation_sha256":validation["sha256"],"scope":"authorized_public_validation"}
        run_native_round(host,directory,state,base.parent)
        if host.candidate:
            descriptor = next(f["artifact"] for f in host.frames if f["type"] == "candidate_request")
            identifier = f"recovery-candidate-{index}"
            registry[identifier] = (host.candidate,artifact_receipt(host.candidate,identifier,descriptor))
        selected,state = host.record["selected_artifact_id"],host.saved_state
        row = {"round":index,"action":host.record["configuration"]["action"],
               "selected_artifact_id":selected,"macro_f1":measured.by_id[selected]["metrics"]["macro_f1"],
               "model_call_ids":list(host.calls),"finish_acknowledged":True,"record":host.record}
        report["rounds"].append(row)
        history.append({"id":f"real-recovery-history-{index}",**host.record})
        if index == 1:
            rollback_selected = row["action"]=="rollback" and selected!="fault-injected-parent"
            recovered_sha = measured.by_id[selected]["checkpoint_sha256"] if rollback_selected else None
            report.update(rollback_selected=rollback_selected,
                          recovered_macro_f1=row["macro_f1"] if rollback_selected else None,
                          recovered_checkpoint_sha256=recovered_sha,
                          historical_read_chunks=sum(f["type"]=="artifact_read_request" for f in host.frames))
        else:
            report["next_round_used_recovered_parent"] = bool(recovered_sha and
                host.record["configuration"]["parent_checkpoint_sha256"] == recovered_sha)
        report.update(model_calls=gateway.attempts,real_verified_model_calls=len(gateway.verified))
        write_json(out/"live-report.json",report)
        print(json.dumps({"event":"recovery_round_finished",**{k:row[k] for k in ("round","action","selected_artifact_id","macro_f1")}},ensure_ascii=False),flush=True)
    directory, selected_receipt = registry[selected]
    shutil.copytree(directory, out/"selected-model")
    shutil.copytree(state, out/"selected-state")
    write_json(out/"selected-artifact.json", selected_receipt)
    report.update(status="completed",audit=audit(out,runtime), selected_artifact_id=selected,
                  finished_at=datetime.now(timezone.utc).isoformat())


def main():
    # 有调用上限的宿主测试入口：每轮把实际选择作为下一轮父产物，保留原始请求/响应和逐条测量。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT/".env")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-model-calls", type=int, default=16)
    parser.add_argument("--image", help="Use this Docker image for both train and predict")
    parser.add_argument("--expected-image-id", help="Refuse an image different from the delivered image ID")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context")
    parser.add_argument("--recovery-from",type=Path,
                        help="Separate two-round fault drill using a completed real-model session")
    args = parser.parse_args()
    if not 4 <= args.rounds <= args.max_rounds <= 8 or not 1 <= args.max_model_calls <= 32:
        raise ValueError("Use 4..8 rounds and 1..32 model calls")
    base, out = args.baseline_dir.resolve(), args.output.resolve()
    load_env(args.env_file)
    config = configuration()
    if config["model"] != MODEL or config["base_url"] != "https://pytrio.com":
        raise ValueError("This test requires the exact PyTRiO Qwen/Qwen3.8-27B model")
    key = os.environ.get("PYTRIO_API_KEY")
    if not key:
        raise ValueError("PYTRIO_API_KEY is required on the host side")
    info = json.loads((base/"baseline-info.json").read_text(encoding="utf-8"))
    if info["dataset_key"] != "A" or digest(base/"checkpoint.npz") != info["checkpoint_sha256"]:
        raise ValueError("Official baseline identity mismatch")
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope":"real_27b_local_sdk_1.0.5_host_rehearsal", "status":"preflight",
              "model":MODEL, "official_host_verified":False, "official_hidden_score":None,
              "cloud_experiment_record_created":False, "rounds":[], "model_calls":[]}
    write_json(out/"live-report.json", report)
    gateway, runtime = None, None
    try:
        if args.image:
            runtime = DockerRuntime(args.image, executable=args.docker_executable,
                                    context=args.docker_context, expected_id=args.expected_image_id)
            report["runtime"] = runtime.preflight()
            report["scope"] = "real_27b_sdk_1.0.5_in_actual_docker_image"
            write_json(out/"live-report.json", report)
        ensure_model(config, key)
        if args.recovery_from:
            gateway = RealGateway(config,key,out,args.max_model_calls)
            recovery_test(args.recovery_from.resolve(),out,base,gateway,report,runtime)
            exit_code = finalize_acceptance(report,recovery=True)
            write_json(out/"live-report.json",report)
            print(json.dumps({"event":"recovery_completed" if exit_code == 0 else "recovery_requirements_unmet",
                              "requirements_passed":report["requirements_passed"],"report":str(out/"live-report.json"),
                              "rollback_selected":report["rollback_selected"],
                              "next_round_used_recovered_parent":report["next_round_used_recovered_parent"],
                              "real_verified_model_calls":report["real_verified_model_calls"]}),flush=True)
            return exit_code
        shutil.copytree(base/"public", out/"data")
        (out/"initial").mkdir()
        shutil.copyfile(base/"checkpoint.npz", out/"initial/checkpoint.npz")
        _, validation = load_pair(out/"data/train.jsonl", out/"data/validation.jsonl")
        measured = measurements(out, validation, base.parent,runtime)
        initial = artifact_receipt(out/"initial", "initial")
        measured.measure(out/"initial", initial["artifact"], "initial")
        registry = {"initial":(out/"initial",initial)}
        selected, state, history = "initial", None, []
        gateway = RealGateway(config, key, out, args.max_model_calls)
        report.update(status="running", baseline_macro_f1=measured.by_id["initial"]["metrics"]["macro_f1"],
                      baseline_checkpoint_sha256=info["checkpoint_sha256"],
                      data_sha256={name:digest(out/"data"/name) for name in ("train.jsonl","validation.jsonl")})
        for index in range(1, args.max_rounds+1):
            directory, parent_receipt = registry[selected]
            host = RealModelHost(out, directory/parent_receipt["artifact"]["checkpoint_paths"][0],
                                 index, history, state, measurements=measured, gateway=gateway, runtime=runtime)
            host.frame["parent_artifact"] = parent_receipt
            host.frame["artifacts"] = [value[1] for identifier,value in registry.items() if identifier != selected]
            host.artifact_dirs = {identifier:value[0] for identifier,value in registry.items()}
            host.frame["current_validation"] = {"metrics":measured.by_id[selected]["metrics"],
                "validation_sha256":validation["sha256"], "scope":"authorized_public_validation"}
            run_native_round(host, directory, state, base.parent)
            if host.candidate:
                descriptor = next(f["artifact"] for f in host.frames if f["type"] == "candidate_request")
                identifier = f"candidate-{index}"
                registry[identifier] = (host.candidate, artifact_receipt(host.candidate,identifier,descriptor))
            selected, state = host.record["selected_artifact_id"], host.saved_state
            record = {"round":index, "history_read":len(host.history_read_ids),
                      "experiment":(host.record["configuration"]["experiment"] or {}).get("name"),
                      "action":host.record["configuration"]["action"], "candidate_submitted":host.candidate is not None,
                      "selected_artifact_id":selected, "selected_macro_f1":measured.by_id[selected]["metrics"]["macro_f1"],
                      "model_call_ids":list(host.calls), "finish_acknowledged":True, "record":host.record}
            report["rounds"].append(record)
            append_jsonl(out/"rounds.jsonl", record)
            history.append({"id":f"local-real-history-{index}", **host.record})
            report.update(selected_artifact_id=selected, final_macro_f1=record["selected_macro_f1"],
                          model_calls=gateway.attempts,
                          real_verified_model_calls=len(gateway.verified),
                          complete_history_experiment_decision_records=sum(r["history_read"]>0 and r["candidate_submitted"] for r in report["rounds"]))
            write_json(out/"live-report.json", report)
            print(json.dumps({"event":"round_finished", **{k:record[k] for k in ("round","action","experiment","selected_artifact_id","selected_macro_f1")}},ensure_ascii=False),flush=True)
            if index >= args.rounds and report["complete_history_experiment_decision_records"] >= 3:
                break
        directory, selected_receipt = registry[selected]
        shutil.copytree(directory,out/"selected-model")
        shutil.copytree(state,out/"selected-state")
        write_json(out/"selected-artifact.json", selected_receipt)
        report.update(status="completed", gain=report["final_macro_f1"]-report["baseline_macro_f1"],
                      at_least_three_complete_records=report["complete_history_experiment_decision_records"]>=3,
                      audit=audit(out,runtime), finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__,
                      finished_at=datetime.now(timezone.utc).isoformat())
        if gateway is not None:
            report.update(model_calls=gateway.attempts, real_verified_model_calls=len(gateway.verified))
        write_json(out/"live-report.json", report)
        print(json.dumps({"event":"test_failed", "error_type":type(error).__name__,
                          "report":str(out/"live-report.json")}),flush=True)
        raise SystemExit(2) from None
    exit_code = finalize_acceptance(report,minimum_rounds=args.rounds)
    write_json(out/"live-report.json",report)
    print(json.dumps({"event":"test_completed" if exit_code == 0 else "test_requirements_unmet",
                      "requirements_passed":report["requirements_passed"],"report":str(out/"live-report.json"),
                      "real_verified_model_calls":report["real_verified_model_calls"],
                      "complete_records":report["complete_history_experiment_decision_records"],
                      "baseline_macro_f1":report["baseline_macro_f1"], "final_macro_f1":report["final_macro_f1"]},ensure_ascii=False),flush=True)
    return exit_code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
