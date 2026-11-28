"""从官方初始基线开始的固定镜像全流程：真实正常/回滚、独立核验、原回复重放。会产生费用。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# 本文件是宿主侧验收工具，不随正式镜像运行；真实 API 密钥只在宿主读取。
# 六个阶段共用固定源码/镜像，任何失败都停止本次验收，不拿旧结果补齐。
sys.path.insert(0, str(ROOT))
import contest_submission
from scripts import test_multitask_cloud as normal
from scripts import test_multitask_recovery_cloud as recovery
from scripts import replay_multitask_cloud as replay
from scripts.results_only_journal import ResultsOnlyJournal
from scripts.verify_closed_loop import verify_recovery
from scripts.verify_multitask_cloud_run import verify_dataset, read, rows
from scripts.live_platforms import load_env, configuration, ensure_model, write_receipt, termination_handler
from gas_a.sensor_training.io import digest


def frozen_source():
    # 记录执行文件、全部 Python 源码及测试的原始字节摘要；文档和 runtime 大文件不参与。
    # 后续每阶段再比较，避免运行中修补逻辑后仍把前后结果称为一次无人干预测试。
    paths = [ROOT / name for name in ("harness.py", "contest_submission.py", "submission_config.json", "requirements.txt")]
    for folder in ("src", "scripts", "tests"):
        paths.extend((ROOT / folder).rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths if "__pycache__" not in p.parts}


def assert_frozen(expected):
    if frozen_source() != expected:
        raise RuntimeError("Execution code changed during fresh test; this run cannot count as uninterrupted acceptance")


def fresh_case_checks(training, restored, key, image, started):
    # 除了“最后分数够不够”，还要检查起点：同一不可变镜像、官方初始权重、空历史和空状态。
    n, r = read(training / "suite.json"), read(restored / "suite.json")
    if n["image_id"] != image or r["image_id"] != image:
        raise ValueError("Normal and recovery must use the same immutable image")
    first = [e["frame"] for e in rows(training / "normal/host-1/protocol.jsonl")]
    start = first[0]
    history = next(f["records"] for f in first if f.get("type") == "history_response")
    if history or start.get("state_path") is not None or start["parent_artifact"]["artifact_id"] != "initial":
        raise ValueError("Fresh training must start with empty history/state and the initial official model")
    if start["parent_checkpoint_sha256"] != digest(training / "official-baseline/checkpoint.npz"):
        raise ValueError("Fresh starting model is not the supplied official baseline")
    ids = []
    # 提供方回执时间必须属于这次启动之后，ID 也不能重复；保存回复重放不在这里冒充新调用。
    for folder, phase in ((training, "normal"), (restored, "recovery")):
        for p in sorted((folder / phase / "provider-calls").glob("*/receipt.json")):
            record = read(p)
            if datetime.fromisoformat(record["started_at"]) < started:
                raise ValueError("Provider receipt predates this fresh execution")
            ids.append(record["call_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate provider ID within fresh execution")
    return {"dataset": key, "fresh_initial_model": True, "initial_history_empty": True,
            "initial_state_empty": True, "same_image_normal_and_recovery": True, "new_provider_call_ids": ids}


def write_summary_cloud(root, config, summary):
    # 第七个 Run 只汇总六阶段的已核验结果，不额外调用模型，也不算新增历史实验。
    import swanlab
    out = root / "cloud-summary"
    out.mkdir()
    journal = ResultsOnlyJournal(swanlab, config, out, {"purpose": "fresh_end_to_end_result_summary_no_new_model_calls",
        "model": normal.MODEL, "image_id": summary["image_id"]}, "fresh-full-complete-results")
    state = "crashed"
    try:
        journal.begin()
        journal.emit({f"normal/selected_{r['dataset']}_{'macro_f1' if r['dataset']=='A' else 'rmse'}":
            r["normal"]["final"]["macro_f1" if r["dataset"] == "A" else "rmse"] for r in summary["datasets"]},
            {"audit/fresh_results": summary})
        journal.archive()
        state = "success" if summary["requirements_passed"] else "crashed"
    finally:
        # 成功和异常都尝试结束自有 Run，再用只读 API 回查，防止正常结束却长期显示“运行中”。
        journal.finish(state)
    cloud = journal.verify_cloud()
    return {"url": cloud["url"], "state": cloud["state"], "verified": cloud["verified"], "new_model_calls": 0}


def run(args):
    """只执行一次；调用方负责把异常也写成明确的失败终态。"""
    load_env(args.env_file)
    config = configuration()
    if any(not os.environ.get(k, "").strip() for k in ("PYTRIO_API_KEY", "SWANLAB_API_KEY")):
        raise ValueError("Host credentials required")
    if config["model"] != normal.MODEL or config["base_url"] != "https://pytrio.com":
        raise ValueError("Exact authorized model and endpoint required")
    root = args.output.resolve()
    # 输出目录必须全新；重跑需要新路径，原始失败证据不会被覆盖。
    root.mkdir(parents=True, exist_ok=False)
    return execute(args, config, root)


def execute(args, config, root):
    for name in ("training", "recovery"):
        (root / name).mkdir()
    source = frozen_source()
    write_receipt(root / "frozen-source.json", source)
    started = datetime.now(timezone.utc)
    summary = {"status": "running", "started_at": started.isoformat(), "image_id": args.expected_image_id,
        "model": normal.MODEL, "fresh_run": True, "reused_old_results": False, "datasets": [],
        "requirements_passed": False, "source_unchanged": True, "cloud_upload_policy": "results_only",
        "maximum_real_calls": 66, "official_host_or_hidden_score_verified": False}
    write_receipt(root / "summary.json", summary)
    try:
        return execute_phases(args, config, root, source, started, summary)
    except BaseException as error:
        # 不吞掉失败后继续训练，不自动重发费用未知的模型请求；也不把旧目录改写成成功。
        summary.update(status="failed", requirements_passed=False, error_type=type(error).__name__,
                       finished_at=datetime.now(timezone.utc).isoformat())
        write_receipt(root / "summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    args = parser.parse_args()
    return run(args)


def execute_phases(args, config, root, source, started, summary):
    # 只列模型核验权限，不做额外付费“连通性试探”；真正调用都服务于实际 SDK 回合。
    ensure_model(config, os.environ["PYTRIO_API_KEY"])
    base = vars(args)
    with termination_handler(), patch.object(normal, "CloudJournal", ResultsOnlyJournal), \
            patch.object(recovery, "CloudJournal", ResultsOnlyJournal):
        # 统一替换为结果归档器：六阶段只上传指标、决策与摘要，完整代码/数据/模型留在本地。
        for key in ("A", "B", "C"):
            assert_frozen(source)
            n_args = SimpleNamespace(**{**base, "output": root / "training", "max_calls": 16})
            # 正常阶段至少 4 轮、至多 6 轮；首轮无历史不计数，必须有 3 次带历史的真实候选实验。
            n = normal.run_one(n_args, key, config)
            if n["status"] != "completed":
                summary.update(status="failed", failed_phase="normal", failed_dataset=key,
                               finished_at=datetime.now(timezone.utc).isoformat())
                write_receipt(root / "summary.json", summary)
                return 1
            assert_frozen(source)
            r_args = SimpleNamespace(**{**base, "output": root / "recovery", "source": root / "training",
                                       "previous_failures": None, "max_calls": 6})
            r = recovery.run_one(r_args, key, config)
            # 回滚源严格取本次 training：预设故障后选历史，再验证下一轮完整父产物继承。
            if r["status"] != "completed":
                summary.update(status="failed", failed_phase="recovery", failed_dataset=key,
                               finished_at=datetime.now(timezone.utc).isoformat())
                write_receipt(root / "summary.json", summary)
                return 1
            assert_frozen(source)
            freshness = fresh_case_checks(root / "training" / key, root / "recovery" / key, key, args.expected_image_id, started)
            checked = {"dataset": key, "normal": verify_dataset(root / "training" / key, key),
                       "recovery": verify_recovery(root / "training" / key, root / "recovery" / key, key),
                       "freshness": freshness}
            summary["datasets"].append(checked)
            # 独立重算保存的逐条预测并校验原始响应/SDK/云端归档，通过后才把该数据集记入总览。
            write_receipt(root / "summary.json", summary)
        argv = ["replay_multitask_cloud.py", "--source", str(root / "training"), "--output", str(root / "replay"),
            "--image", args.image, "--expected-image-id", args.expected_image_id,
            "--docker-executable", args.docker_executable, "--docker-context", args.docker_context]
        with patch.object(sys, "argv", argv):
            # 重放只消费本次正常阶段保存的模型回复；不读密钥、不联网请求新模型、不创建云端 Run。
            replay.main()
        assert_frozen(source)
        replay_result = read(root / "replay/verification.json")
        if {r["dataset"] for r in replay_result["datasets"]} != {"A", "B", "C"} or not all(r["verified"] for r in replay_result["datasets"]):
            raise ValueError("Fresh numerical replay did not verify all three tasks")
        ids = [i for r in summary["datasets"] for i in r["freshness"]["new_provider_call_ids"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Provider IDs were reused across datasets")
        summary.update(status="experiment_verified", requirements_passed=True, real_model_calls=len(ids),
            replay_verified=True, new_real_calls_in_replay=0, source_unchanged=True,
            total_tokens=sum(r[p]["provider_reported_total_tokens"] for r in summary["datasets"] for p in ("normal", "recovery")))
        write_receipt(root / "summary.json", summary)
        summary["cloud_summary"] = write_summary_cloud(root, config, summary)
        # 数值实验通过与云端总览完成是两个状态；总览结束并回读成功后才标记全流程 completed。
        summary.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        write_receipt(root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "real_model_calls": summary["real_model_calls"],
                      "cloud_summary": summary["cloud_summary"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
