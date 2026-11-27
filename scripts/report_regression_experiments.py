"""汇总两种 SDK 顺序的验证结果，并回放累计退化阈值。"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_multitask import BASELINES
import numpy as np
from gas_a.regression.data import schema
from gas_a.regression.diagnostics import error_metrics
from gas_a.regression.stage_reference import compare_stage
from gas_a.sensor_training.reference.data import read_dataset


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _resource(folder):
    timing = folder / "sdk-timing.json"
    if not timing.is_file():
        return {}
    # 退出阶段的记录包括保存状态与 finish，资源快照中的阶段和上传数据用于补充。
    log_path = folder / "stderr.log"
    summary = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("[gas-rsi] "):
            event = json.loads(line[len("[gas-rsi] "):])
            if event.get("event") == "round_resources":
                summary = {key: event.get(key) for key in ("cpu_seconds", "wall_seconds", "peak_rss_bytes", "model_call_count")}
                for key in ("input_tokens", "output_tokens"):
                    summary[key] = (event.get("token_totals") or {}).get(key)
    state = folder / "state/resources.json"
    if state.is_file():
        snapshot = read(state)
        summary["candidate_upload_bytes"] = snapshot.get("known_upload_payload_bytes")
        summary["local_training_seconds"] = sum(s["seconds"] for s in snapshot["stages"] if s["stage"] == "local_training")
    return summary


def collect(run, label, *, retrospective_only=False):
    group_rows, round_rows, resources, thresholds = [], [], [], []
    for dataset in ("B", "C"):
        base = run / dataset
        report = read(base / "report.json")
        initial, final = report["baseline"], report["final"]
        if not retrospective_only:
            if initial["validation_sha256"] != final["validation_sha256"]:
                raise ValueError("The order comparison requires a fixed public validation file")
            validation = read_dataset(base / "data/validation.jsonl", "regression", [], schema(dataset))
            cuts = read(base / "host-1/state/regression_monitor_reference.json")["target_cuts"]
            positions = np.searchsorted(cuts, validation["y"], side="right")
            last = max(base.glob("measurement-*"), key=lambda p: int(p.name.split("-")[-1]))
            def segments(directory):
                scope = read(directory / "validation-scope.json")
                if (scope["validation_sha256"] != validation["sha256"] or scope["sample_ids"] != validation["ids"]):
                    raise ValueError("The segment measurements use a different projection")
                lines = (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                error = np.asarray([json.loads(line)["outputs"]["value"] for line in lines]) - validation["y"]
                return {str(i): error_metrics(error[positions == i], dataset) for i in np.unique(positions)}
            initial_groups = {"all": {"all": initial}, **initial["group_metrics"], "rul_segment": segments(base / "measurement-1")}
            final_groups = {"all": {"all": final}, **final["group_metrics"], "rul_segment": segments(last)}
            for axis, groups in final_groups.items():
                for name, metrics in groups.items():
                    start = initial_groups[axis][name]
                    group_rows.append({"order": label, "dataset": dataset, "axis": axis, "group": name,
                        "validation_sha256": final["validation_sha256"], "rows": metrics["rows"],
                        "start_rmse": start.get("rmse"), "final_rmse": metrics["rmse"], "bias": metrics["bias"],
                        "underestimate_count": metrics["underestimate_count"], "overestimate_count": metrics["overestimate_count"],
                        "underestimate_p99": metrics["underestimate_p99"], "overestimate_p99": metrics["overestimate_p99"],
                        "initial_nasa": start.get("nasa_score"), "nasa": metrics.get("nasa_score"), "nasa_share": metrics.get("nasa_share")})
        levels = (.05, .10, .15) if dataset == "B" else (0., .05, .10)
        outcomes = {level: [] for level in levels}
        base_adopted = []
        for row in report["rounds"]:
            cfg = row["record"]["configuration"]
            number = row["round"]
            measured = cfg.get("stage_reference")
            candidate = cfg.get("validation_after")
            origin = measured["start_metrics"] if measured else initial
            if candidate and cfg.get("selection_action") == "adopt_candidate":
                base_adopted.append(number)
                delta = measured["candidate_delta"] if measured else compare_stage(origin, candidate)
                for level in levels:
                    if dataset == "B":
                        losses = [name for name, item in delta.get("device", {}).items()
                            if name != "unknown" and item["rows"] >= 30
                            and item["rmse_delta"] > max(10., origin["group_metrics"]["device"][name]["rmse"] * level)]
                    else:
                        nasa = delta.get("nasa_delta")
                        losses = ["nasa"] if nasa is not None and nasa > origin["nasa_score"] * level else []
                    if losses:
                        outcomes[level].append({"round": number, "groups": losses})
            if not retrospective_only:
                schedule = row["experiment_schedule"]
                metrics = row["metrics"]
                round_rows.append({"order": label, "dataset": dataset, "round": number, **schedule,
                    "selection": cfg.get("selection_action"), "selected_artifact": row["selected_artifact_id"],
                    "stage_start_rmse": origin["rmse"], "parent_rmse": cfg["validation_before"]["rmse"],
                    "candidate_rmse": candidate["rmse"] if candidate else None,
                    "selected_rmse": metrics["rmse"], "stage_start_nasa": origin.get("nasa_score"),
                    "parent_nasa": cfg["validation_before"].get("nasa_score"),
                    "candidate_nasa": candidate.get("nasa_score") if candidate else None, "selected_nasa": metrics.get("nasa_score"),
                    "gate_reasons": "|".join((cfg.get("candidate_gate") or {}).get("reasons", []))})
                resources.append({"order": label, "dataset": dataset, "round": number,
                                  **_resource(base / f"host-{number}")})
        thresholds.append({"order": label, "dataset": dataset, "adopted_rounds": base_adopted,
                           "would_flag": outcomes})
    return group_rows, round_rows, resources, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configured", required=True, type=Path)
    parser.add_argument("--mlp-first", required=True, type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configured = read(args.configured / "registration.json")
    reordered = read(args.mlp_first / "registration.json")
    if (configured["configuration"] != reordered["configuration"]
            or configured["sources_sha256"] != reordered["sources_sha256"]
            or any(configured["inputs"][key]["sha256"] != reordered["inputs"][key]["sha256"] for key in BASELINES)):
        raise ValueError("Order comparison must use identical code, data and configurations")
    args.output.mkdir(parents=True, exist_ok=False)
    runs = [collect(args.configured, "configured"), collect(args.mlp_first, "mlp_first")]
    for i, file in enumerate(("groups.csv", "rounds.csv", "resources.csv")):
        write_table(args.output / file, runs[0][i] + runs[1][i])
    thresholds = runs[0][3] + runs[1][3]
    if args.prior:
        thresholds.extend(collect(args.prior, "prior_run", retrospective_only=True)[3])
    (args.output / "stage-replay.json").write_text(json.dumps({
        "scope": "diagnostic counterfactual over existing adopted rounds; training trajectory is unchanged",
        "thresholds": thresholds,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
