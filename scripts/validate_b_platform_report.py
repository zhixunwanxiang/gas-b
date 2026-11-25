"""B 平台报告问题的离线十轮复现：真实训练/预测，模型决定为明确标记的替身。"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_multitask as validation
from scripts.docker_rehearsal import DockerRuntime
from tests.test_regression import RegressionHost
from gas_a.sensor_training.io import digest, write_json

# 取自用户“实验报告B.txt”的逐轮菜单选择；不冒充原始 27B 回复或平台漂移投影。
REPORT_MENU = ("robust_huber", "standard_rate", "head_only", "head_only", "head_only",
               "head_only", "robust_huber", "head_only", "head_only", "standard_rate")


class ReportPeer(RegressionHost):
    receipt_layout = "compact"

    def readline(self):
        # 在 start 真正送往选手进程前精简父测量。原 SDK 本身不做任何修改。
        if self.pending and self.pending[0].get("protocol") == "contest.harness.workspace.v2":
            if self.receipt_layout == "compact":
                score = self.frame["current_validation"]["metrics"]["rmse"]
                self.frame["current_validation"] = {"rmse": score}
        return super().readline()

    def write(self, text):
        frame = json.loads(text)
        request = json.loads(frame["messages"][0]["content"]) if frame["type"] == "model_request" else None
        if request and request["phase"] == "plan":
            self.experiment = REPORT_MENU[(self.frame["round_index"] - 1) % len(REPORT_MENU)]
        result = super().write(text)
        if frame["type"] == "candidate_request" and self.receipt_layout == "compact":
            self.pending[-1]["summary"] = {"metrics": self.pending[-1]["summary"]["metrics"]}
        if request:
            # 修正通用测试替身的分类术语，明确标注离线决定，完整保留数值与 SDK 回执。
            message = self.pending[-1]["response"]["choices"][0]["message"]
            decision = json.loads(message["content"])
            if request["phase"] == "plan":
                decision["hypothesis"] = "Offline report-menu replay: continue the actual B parent and test public RMSE."
                decision["reason"] = "Offline model stub; use the configured report menu and the current allowed plan combinations."
            else:
                decision["reason"] = "Offline model stub; select an eligible candidate with lower public RMSE, otherwise retain the actual parent."
            message["content"] = json.dumps(decision)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--receipt-layout", choices=("explicit", "compact"), default="compact")
    parser.add_argument("--image")
    parser.add_argument("--expected-image-id")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    parser.add_argument("--platform-report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        parser.error("rounds must be 1..10")
    out = args.output.resolve()
    out.mkdir()  # 所有证据写到新目录，不覆盖原平台报告或旧实验。
    tracked = [ROOT / name for name in ("harness.py", "contest_submission.py", "submission_config.json")]
    tracked.extend(sorted((ROOT / "src").rglob("*.py")))
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in tracked}
    write_json(out / "source-sha256.json", source_hashes)
    if args.platform_report:
        write_json(out / "platform-report-source.json", {"path": str(args.platform_report.resolve()),
                   "sha256": digest(args.platform_report), "raw_protocol_available": False})
    runtime = (DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                             expected_id=args.expected_image_id, required_tasks=("regression",)) if args.image else None)
    if runtime:
        write_json(out / "image-preflight.json", runtime.preflight())
    ReportPeer.receipt_layout = args.receipt_layout
    with patch.object(validation, "RegressionHost", ReportPeer):
        report = validation.run_dataset(args.baseline_root / validation.BASELINES["B"], out / "B", "B",
                                        args.rounds, runtime)
    # 回执精简由 ReportPeer 在送帧时执行，覆盖通用宿主默认的布局标签。
    report["validation_receipt_format"] = args.receipt_layout
    write_json(out / "B/report.json", report)
    rows, adopted = [], 0
    previous_id = "initial"
    checks = {"requested_rounds_completed": len(report["rounds"]) == args.rounds,
              "public_rmse_improved": report["final"]["rmse"] < report["baseline"]["rmse"],
              "source_unchanged_during_run": source_hashes == {p.relative_to(ROOT).as_posix(): digest(p) for p in tracked}}
    for row in report["rounds"]:
        cfg = row["record"]["configuration"]
        identifier = row["selected_artifact_id"]
        adopted += identifier != previous_id
        continuation = None
        candidate = out / "B" / f"host-{row['round']}" / "candidate" / "training.json"
        if candidate.exists():
            continuation = json.loads(candidate.read_text(encoding="utf-8"))
            checks[f"round_{row['round']}_continuation"] = all(continuation["checks"].values())
        timing = json.loads((out / "B" / f"host-{row['round']}" / "sdk-timing.json").read_text(encoding="utf-8"))
        checks[f"round_{row['round']}_sdk_order"] = timing["order_valid"]
        rows.append({"round": row["round"], "experiment": (cfg.get("experiment") or {}).get("name"),
                     "selected_artifact_id": identifier, "adopted": identifier != previous_id,
                     "rmse": row["metrics"]["rmse"], "mae": row["metrics"]["mae"],
                     "parent_reference": cfg.get("parent_host_reference"),
                     "candidate_eligible": (cfg.get("candidate_gate") or {}).get("eligible"),
                     "host_eligible": (cfg.get("host_candidate_gate") or {}).get("eligible"),
                     "initial_steps": continuation["initial_steps"] if continuation else None,
                     "final_steps": continuation["final_steps"] if continuation else None,
                     "wall_seconds": timing["wall_total_seconds"]})
        previous_id = identifier
    checks["candidate_actually_adopted"] = adopted > 0
    audit = runtime.audit() if runtime else None
    if runtime:
        checks["docker_isolation_verified"] = audit["container_policy_verified"]
    summary = {"status": "passed" if all(checks.values()) else "requirements_unmet", "checks": checks,
               "dataset": "B", "receipt_layout": args.receipt_layout, "rounds": rows,
               "adoptions": adopted, "baseline": report["baseline"], "final": report["final"],
               "rmse_reduction_percent": 100 * (1 - report["final"]["rmse"] / report["baseline"]["rmse"]),
               "runtime": audit or "native", "real_model_calls": 0, "cloud_writes": 0,
               "official_platform_score_verified": False,
               "boundary": "Fixed official public files; report-menu offline decisions. No platform drift projections, raw platform frames, or hidden labels available."}
    write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rounds", "checks", "runtime")}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
