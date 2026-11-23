"""v5真实27B和SwanLab多任务验收：每个数据集一个Run，原始回复直接送回SDK。会产生费用。"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from tests.test_submission import OfflineHost
from tests.test_regression import RegressionHost
from scripts.validate_multitask import BASELINES, run_dataset
from scripts.test_real_sdk import RealGateway, MODEL, request_phase
from scripts.test_cloud_suite import DEFAULT_IMAGE, DEFAULT_IMAGE_ID
from scripts.docker_rehearsal import DockerRuntime
from scripts.cloud_experiment import CloudJournal
from scripts.live_platforms import load_env, configuration, ensure_model, termination_handler, write_receipt
from gas_a.sensor_training.io import append_jsonl, digest


LOCAL_PROMPT = (
    "这是本地SDK1.0.4真实模型测试的宿主说明，不是组委会正式宿主。"
    "按本轮task_contract执行：A为气体六分类，Macro-F1越高越好；B/C为剩余寿命回归，RMSE越低越好。"
    "B单位分钟，C单位循环。只用公开train更新梯度，公开validation可评估选版，不获取隐藏标签。"
    "根据历史成败、漂移诊断、允许菜单和真实测量自主决定行动。仅在response_schema允许范围内返回JSON。"
    "证据不足可以保留父模型；不得为测试通过而捏造结果、强行上线或放宽约束。"
)


class RealPeerMixin:
    def write(self, text):
        frame = json.loads(text)
        if frame["type"] != "model_request":
            return super().write(text)
        # 必须在父类离线规划分支之前截获；真实调用失败绝不能退回规则替身。
        if frame["id"] != self.next_sequence:
            raise ValueError("Unexpected real SDK RPC sequence")
        self.next_sequence += 1
        self.frames.append(frame)
        self.messages.append(frame["messages"])
        self.journal.before_call(frame, self.frame)
        # 宿主替身只负责 SDK 协议与持久化；这个分支把请求交给真正的 PyTRiO 模型服务。
        # 原始 response 直接装回 SDK model_response，不能先生成离线决定再替换成“真实调用”日志。
        call_id, response, bound = self.gateway.invoke(frame, self.frame)
        self.journal.after_call(call_id, response, bound)
        self.calls.append(call_id)
        append_jsonl(self.store / "model-calls.jsonl", {
            "phase": request_phase(frame["messages"]), "call_id": call_id, "real_remote_model_call": True,
            "official_contest_model_receipt": False, "raw_messages": frame["messages"],
            "bound_messages": bound, "response": response})
        self.pending.append({"type": "model_response", "id": frame["id"], "call_id": call_id, "response": response})
        return len(text)


class RealClassificationPeer(RealPeerMixin, OfflineHost):
    pass


class RealRegressionPeer(RealPeerMixin, RegressionHost):
    pass


class RealPeerFactory:
    def __init__(self, config, key, journal, max_calls):
        self.config, self.key, self.journal, self.max_calls = config, key, journal, max_calls
        self.gateway = None

    def __call__(self, root, parent, dataset, index, history, state):
        # 每个数据集的正常阶段共享调用预算和网关；每轮用新协议会话，状态/父产物由宿主明确传入。
        if self.gateway is None:
            self.gateway = RealGateway(self.config, self.key, root, max_calls=self.max_calls)
        host = (RealClassificationPeer(root, parent, index, history=history, state=state) if dataset == "A" else
                RealRegressionPeer(root, parent, dataset, index, history=history, state=state))
        host.gateway, host.journal = self.gateway, self.journal
        host.frame.update(model_name=MODEL, prompt=LOCAL_PROMPT, prompt_version="local-abc-real-sdk104-v1")
        return host


def snapshot(out, baseline):
    # 这是本地复现快照，不是云端上传清单。只复制明确代码和官方公开文件，不复制 .env 或隐藏数据。
    paths = [Path(n) for n in ("harness.py", "contest_submission.py", "submission_config.json", "requirements.txt",
             "requirements-platform.txt", "Dockerfile", "Dockerfile.offline", ".dockerignore", ".python-version",
             "vendor/contest_harness-1.0.4-py3-none-any.whl")]
    for folder in ("src", "scripts", "tests"):
        paths.extend(Path(folder).rglob("*.py"))
    hashes = {}
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        target = out / "reproduction" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
        hashes[path.as_posix()] = digest(target)
    for name in ("checkpoint.npz", "baseline-info.json", "TASK.md", "START.md", "TRAINING_GUIDE.md",
                 "public/train.jsonl", "public/validation.jsonl"):
        target = out / "official-baseline" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline / name, target)
    write_receipt(out / "source-sha256.json", hashes)
    return hashes


def preflight_source(runtime, hashes):
    # 镜像 ID 相同仍要核对 /app 的源码字节；本地注释变更后也须重建镜像或使用旧快照。
    # 这让“测的是当前代码”有文件依据，不能仅依赖一个可以被覆盖的 tag 名。
    app = {name: sha for name, sha in hashes.items() if name.startswith("src/") or name in (
        "harness.py", "contest_submission.py", "submission_config.json", "requirements.txt",
        "vendor/contest_harness-1.0.4-py3-none-any.whl")}
    script = "import pathlib,hashlib,json,sys; print(json.dumps({p:'sha256:'+hashlib.sha256((pathlib.Path('/app')/p).read_bytes()).hexdigest() for p in json.loads(sys.argv[1])}))"
    result = runtime._run(["run", "--rm", "--pull=never", "--network", "none", "--read-only", "--entrypoint", "python3",
                           runtime.image, "-c", script, json.dumps(list(app))])
    if json.loads(result.stdout) != app:
        raise ValueError("Current application source differs from tested image")
    return len(app)


def run_one(args, dataset, config):
    out = args.output.resolve() / dataset
    out.mkdir()
    baseline = args.baseline_root.resolve() / BASELINES[dataset]
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                            expected_id=args.expected_image_id, required_tasks=("classification", "regression"))
    probe = runtime.preflight()
    hashes = snapshot(out, baseline)
    probe["application_source_files_matched"] = preflight_source(runtime, hashes)
    write_receipt(out / "image-preflight.json", probe)
    import swanlab
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    journal = CloudJournal(swanlab, config, out, {
        "dataset_key": dataset, "task": "classification" if dataset == "A" else "regression",
        "metric": "macro_f1" if dataset == "A" else "rmse", "model": MODEL, "image_id": runtime.image,
        "maximum_real_calls": args.max_calls, "minimum_rounds": 4, "maximum_rounds": 6,
        "source_sha256": hashes, "public_validation_only": True, "hidden_labels_available": False},
        f"gas-{dataset.lower()}-real-cloud-{stamp}")
    factory = RealPeerFactory(config, os.environ["PYTRIO_API_KEY"], journal, args.max_calls)
    result = {"dataset": dataset, "model": MODEL, "image_id": runtime.image, "status": "starting",
              "real_model_calls": 0, "cloud_verified": False, "official_host_or_hidden_score_verified": False}
    write_receipt(out / "suite.json", result)
    error = None
    finish_state = "crashed"
    # 默认失败收尾，只有全部验收要求达到才改为 success；finally 确保异常也结束自有云端实验。
    try:
        journal.begin()
        result.update(status="running", cloud_url=journal.receipt["url"], cloud_run_path=journal.receipt["run_path"])
        write_receipt(out / "suite.json", result)
        print(json.dumps({"event": "real_cloud_run_started", "dataset": dataset, "url": journal.receipt["url"]}), flush=True)
        report = run_dataset(baseline, out / "normal", dataset, 4, runtime, peer_factory=factory, journal=journal, max_rounds=6)
        calls = factory.gateway.verified
        checks = {"four_or_more_rounds": len(report["rounds"]) >= 4,
                  "three_history_experiments": report["complete_history_experiments"] >= 3,
                  "not_worse_than_public_baseline": report["not_worse_than_public_baseline"],
                  "real_receipts_each_round": all(r["model_call_ids"] for r in report["rounds"]),
                  "no_unknown_outcome_or_retries": len(factory.gateway.attempts) == len(calls),
                  "docker_isolation_verified": runtime.audit()["container_policy_verified"]}
        result.update(status="experiment_completed", checks=checks, requirements_passed=all(checks.values()),
                      real_model_calls=len(calls), provider_reported_total_tokens=sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
                      baseline=report["baseline"], final=report["final"], rounds=len(report["rounds"]),
                      complete_history_experiments=report["complete_history_experiments"])
        journal.emit(texts={"normal/summary": report}, scalars={"suite/requirements_passed": int(all(checks.values()))})
        write_receipt(out / "suite.json", result)
        journal.archive()
        finish_state = "success" if result["requirements_passed"] else "crashed"
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
            cloud = journal.verify_cloud()
            result.update(cloud_verified=True, cloud_state=cloud["state"], cloud_url=cloud["url"])
        except Exception as caught:
            result["cloud_verification_error_type"] = type(caught).__name__
    result["status"] = "completed" if result.get("requirements_passed") and result["cloud_verified"] else "needs_attention"
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_receipt(out / "suite.json", result)
    print(json.dumps({"event": "real_cloud_dataset_finished", **result}, ensure_ascii=False), flush=True)
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--expected-image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    parser.add_argument("--datasets", choices=("A", "B", "C"), nargs="+", default=["A", "B", "C"])
    parser.add_argument("--max-calls", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 16:
        parser.error("max-calls must be 1..16 per dataset")
    load_env(args.env_file)
    config = configuration()
    if any(not os.environ.get(k, "").strip() for k in ("PYTRIO_API_KEY", "SWANLAB_API_KEY")):
        raise ValueError("Host requires PYTRIO_API_KEY and SWANLAB_API_KEY")
    if config["model"] != MODEL or config["base_url"] != "https://pytrio.com":
        raise ValueError("Requires the exact PyTRiO Qwen3.8-27B endpoint/model")
    args.output.resolve().mkdir(parents=True, exist_ok=False)
    # 只列模型确认权限，不发送连通性付费试探；每次真正调用都属于实际SDK决策。
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
