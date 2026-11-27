"""Recompute and chart only one newly registered 30-round run; never fall back to history."""
import argparse
import csv
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
CHINESE = ["乙醇", "乙烯", "氨气", "乙醛", "丙酮", "甲苯"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_export(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sync_metric_summaries(data, output):
    """Derive every reported metric from the same selected-model trajectory."""
    specs = {"A": [("macro_f1", "Macro-F1", "%"), ("accuracy", "Accuracy", "%")],
             "B": [("rmse", "RMSE", "分钟"), ("mae", "MAE", "分钟")],
             "C": [("rmse", "RMSE", "循环"), ("mae", "MAE", "循环"), ("nasa_score", "NASA 非对称评分", "惩罚总和")]}
    summaries, csv_rows, datasets = [], [], []
    for ds, metrics in specs.items():
        dataset = data["datasets"][ds]
        for key, label, unit in metrics:
            values = [row["metrics"][key] for row in dataset["rows"]]
            assert len(values) == 11
            # An overflow or unavailable value is unknown, never a zero improvement.
            if any(value is None or not math.isfinite(value) for value in values):
                continue
            higher = ds == "A"
            gain = ((values[-1] - values[0]) * 100 if higher else
                    100 * (values[0] - values[-1]) / values[0] if values[0] else None)
            summary = {"dataset": ds, "metric": key, "label": label, "display_unit": unit,
                "display_scale": 100 if higher else 1, "initial": values[0], "final": values[-1],
                "mean_rounds_1_to_10": statistics.mean(values[1:]),
                "worst_rounds_1_to_10": min(values[1:]) if higher else max(values[1:]),
                "improvement": gain, "improvement_unit": "百分点" if higher else "%"}
            summaries.append(summary)
            dataset["summary"].update({f"initial_{key}": values[0], f"final_{key}": values[-1],
                f"mean_{key}_rounds_1_to_10": summary["mean_rounds_1_to_10"],
                f"worst_{key}_rounds_1_to_10": summary["worst_rounds_1_to_10"],
                f"{key}_improvement": gain})
            scale = summary["display_scale"]
            csv_rows.append({"项目": ds, "指标": label, "单位": unit, "初始": values[0] * scale,
                "十轮平均": summary["mean_rounds_1_to_10"] * scale,
                "最差轮": summary["worst_rounds_1_to_10"] * scale, "最终": values[-1] * scale,
                "改善": gain, "改善单位": summary["improvement_unit"]})
        datasets.append({"项目": ds, "主指标": dataset["primary"], **dataset["summary"]})
    data["metric_summaries"] = summaries
    data["chart_generator_sha256"] = digest(Path(__file__))
    write(output / "逐轮指标数据.json", data)
    csv_export(output / "十轮汇总.csv", datasets)
    csv_export(output / "各项指标汇总.csv", csv_rows)


def prepare(run, output):
    import numpy as np
    registration = read(run / "registration.json")
    assert registration["historical_results_used"] is False
    assert registration["rounds_per_dataset"] == 10
    code_root = Path(registration.get("code_root", ROOT)).resolve()
    assert all(digest(code_root / p) == h for p, h in registration["sources_sha256"].items())
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    def source(path):
        path = Path(path).resolve()
        assert path.is_relative_to(run.resolve())
        sources[str(path)] = digest(path)
        return path
    source(run / "registration.json")
    data = {"scope": "fresh_native_offline_sdk_numeric_test", "historical_results_used": False,
            "registration": registration, "datasets": {}, "real_model_calls": 0,
            "official_score_verified": False, "formal_stage_and_recovery_evaluation": "not_available_fixed_public_projection"}
    snapshot_manifest = code_root / "source-snapshot.json"
    if snapshot_manifest.exists():
        snapshot = read(snapshot_manifest)
        assert all(digest(code_root / p) == h for p, h in snapshot["files_sha256"].items())
        data["code_snapshot"] = {"manifest": str(snapshot_manifest), "manifest_sha256": digest(snapshot_manifest),
            "created_at": snapshot["created_at"], "origin": snapshot["origin"], "files_sha256": snapshot["files_sha256"]}
    flat, group_rows, class_rows, summary_rows, experiment_rows = [], [], [], [], []
    recalc_count, candidate_count = 0, 0
    for dataset in "ABC":
        directory = run / dataset
        report = read(source(directory / "report.json"))
        completion = read(source(run / f"{dataset}-completion.json"))
        assert completion["fresh_run"] and completion["rounds"] == 10
        assert len(report["rounds"]) == 10
        validation_file = source(directory / "data/validation.jsonl")
        assert digest(validation_file) == registration["inputs"][dataset]["sha256"]["public/validation.jsonl"]
        assert digest(source(directory / "initial/checkpoint.npz")) == registration["inputs"][dataset]["sha256"]["checkpoint.npz"]
        samples = [json.loads(s) for s in validation_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        y = np.array([LABELS.index(s["answer"]) if dataset == "A" else s["answer"] for s in samples])
        group_key = "bearing" if dataset == "B" else "source_subset"
        groups = None if dataset == "A" else np.array([str(s.get("meta", {}).get(group_key, "未标识")) for s in samples])
        def measured(index, expected, *, candidate=False):
            nonlocal recalc_count, candidate_count
            path = directory / f"measurement-{index}"
            prediction_file = source(path / "predictions.jsonl")
            source(path / "metrics.json")
            outputs = [json.loads(line)["outputs"] for line in prediction_file.read_text(encoding="utf-8").splitlines()]
            assert len(outputs) == len(y)
            if dataset == "A":
                assert all([item["label"] for item in row] == LABELS for row in outputs)
                scores = np.array([[item["score"] for item in row] for row in outputs])
                assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
                predicted = scores.argmax(1)
                per_class = {}
                for i, name in enumerate(LABELS):
                    true, chosen = y == i, predicted == i
                    tp = int(np.sum(true & chosen))
                    support, count = int(true.sum()), int(chosen.sum())
                    per_class[name] = {"support": support, "recall": tp / support if support else 0.,
                        "precision": tp / count if count else 0., "f1": 2 * tp / (support + count) if support + count else 0.}
                observed = set(y) | set(predicted)
                metrics = {"macro_f1": float(np.mean([per_class[LABELS[i]]["f1"] for i in observed])),
                           "accuracy": float(np.mean(y == predicted)), "per_class": per_class, "rows": len(y)}
            else:
                assert all(isinstance(o, dict) and set(o) == {"value"} and not isinstance(o["value"], bool) for o in outputs)
                p = np.array([o["value"] for o in outputs], dtype=float)
                assert np.isfinite(p).all() and np.all(p >= 0)
                error = p - y
                metrics = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.abs(error).mean()),
                    "bias": float(error.mean()), "overestimate_fraction": float(np.mean(error > 0)),
                    "p95_absolute_error": float(np.quantile(np.abs(error), .95)),
                    "p99_absolute_error": float(np.quantile(np.abs(error), .99)), "rows": len(y)}
                for direction, mask in (("underestimate", error < 0), ("overestimate", error > 0)):
                    magnitude = np.abs(error[mask])
                    metrics[direction + "_count"] = len(magnitude)
                    metrics[direction + "_p99"] = float(np.quantile(magnitude, .99)) if len(magnitude) else None
                if dataset == "C":
                    with np.errstate(over="ignore"):
                        penalties = np.expm1(np.where(error < 0, -error / 13., error / 10.))
                    total = float(penalties.sum())
                    metrics.update(nasa_score=total if math.isfinite(total) else None,
                        nasa_over=float(penalties[error >= 0].sum()), nasa_under=float(penalties[error < 0].sum()),
                        nasa_top_1pct_share=float(np.sort(penalties)[-math.ceil(len(penalties) * .01):].sum() / total)
                        if math.isfinite(total) and total > 0 else None)
                    for name in ("nasa_over", "nasa_under"):
                        if not math.isfinite(metrics[name]): metrics[name] = None
                grouped = {}
                for group in np.unique(groups):
                    e = error[groups == group]
                    grouped[str(group)] = {"rows": len(e), "rmse": float(np.sqrt(np.mean(e ** 2))),
                        "mae": float(np.abs(e).mean()), "bias": float(e.mean())}
                    if dataset == "C":
                        subtotal = float(penalties[groups == group].sum())
                        grouped[str(group)]["nasa_score"] = subtotal if math.isfinite(subtotal) else None
                metrics.update(groups=grouped, group_source="公开轴承追溯字段" if dataset == "B" else "公开来源子集 FD001–FD004",
                    group_mean_rmse=float(np.mean([g["rmse"] for g in grouped.values()])),
                    group_worst_rmse=max(g["rmse"] for g in grouped.values()))
            for key in ("macro_f1", "accuracy", "rmse", "mae", "nasa_score"):
                if key in expected:
                    if expected[key] is None: assert metrics[key] is None
                    else: assert math.isclose(metrics[key], expected[key], rel_tol=1e-11, abs_tol=1e-9), (dataset, index, key)
            timing = read(source(path / "timing.json"))
            if candidate: candidate_count += len(y)
            else: recalc_count += len(y)
            return metrics, timing
        baseline, _ = measured(1, report["baseline"])
        primary = "macro_f1" if dataset == "A" else "rmse"
        rows = [{"round": 0, "metrics": baseline, "gain_vs_initial": 0., "selected_artifact_id": "initial",
                 "selection_action": "initial", "candidate_gain": None, "selected_round_gain": None}]
        measurement_index = 1
        registry, previous_id = {"initial": directory / "initial/checkpoint.npz"}, "initial"
        for index, item in enumerate(report["rounds"], 1):
            assert item["round"] == index and item["finish_acknowledged"]
            cfg = item["record"]["configuration"]
            host = directory / f"host-{index}"
            frames = [json.loads(line) for line in source(host / "protocol.jsonl").read_text(encoding="utf-8").splitlines()]
            call_ids = [f["frame"]["call_id"] for f in frames if f["frame"].get("type") == "model_response"]
            assert len(call_ids) >= 1 and item["model_call_ids"] == call_ids
            assert any(f["frame"].get("type") == "round_ack" and f["frame"].get("accepted") is True for f in frames)
            assert len(item["history_ids"]) == index - 1
            assert cfg["validation_sha256"].split(":")[-1] == digest(validation_file)
            candidate_metric = None
            candidate_source = None
            if item["candidate_submitted"]:
                child = source(host / "candidate/checkpoint.npz")
                registry[f"candidate-{index}"] = child
                training = read(source(host / "candidate/training.json"))
                assert training["parent_checkpoint_sha256"].split(":")[-1] == digest(registry[previous_id])
                assert training["checkpoint_sha256"].split(":")[-1] == digest(child)
                measurement_index += 1
                expected_candidate = ({"macro_f1": cfg["local_validation_after"]} if dataset == "A" else cfg["validation_after"])
                candidate_values, _ = measured(measurement_index, expected_candidate, candidate=True)
                candidate_metric = candidate_values[primary]
                candidate_source = "independent_native_prediction"
            elif dataset != "A" and isinstance(cfg.get("validation_after"), dict) and not cfg.get("experiment_failure"):
                candidate_metric = cfg["validation_after"].get(primary)
                if candidate_metric is not None:
                    candidate_source = "local_validation_not_uploaded_no_independent_prediction"
            measurement_index += 1
            metrics, timing = measured(measurement_index, item["metrics"])
            before = rows[-1]["metrics"][primary]
            selected = metrics[primary]
            candidate_gain = None if candidate_metric is None else ((candidate_metric - before) * 100 if dataset == "A"
                else (before - candidate_metric) / before * 100)
            selected_gain = (selected - before) * 100 if dataset == "A" else (before - selected) / before * 100
            cumulative = (selected - baseline[primary]) * 100 if dataset == "A" else (baseline[primary] - selected) / baseline[primary] * 100
            events = [json.loads(line[len("[gas-rsi] "):]) for line in source(host / "stderr.log").read_text(encoding="utf-8").splitlines()
                      if line.startswith("[gas-rsi] ")]
            resources = [e for e in events if e.get("event") == "round_resources"]
            assert len(resources) == 1
            resources = resources[0]
            training_seconds = sum(e["seconds"] for e in events if e.get("event") == "stage_ended" and e.get("stage") == "local_training")
            candidate_bytes = sum(p.stat().st_size for p in (host / "candidate").rglob("*") if p.is_file()) if item["candidate_submitted"] else 0
            state_bytes = sum(p.stat().st_size for p in (host / "state").rglob("*") if p.is_file())
            row = {"round": index, "metrics": metrics, "gain_vs_initial": cumulative,
                "candidate_gain": candidate_gain, "selected_round_gain": selected_gain,
                "candidate_primary_metric": candidate_metric,
                "candidate_metric_source": candidate_source,
                "gain_unit": "F1 百分点" if dataset == "A" else "RMSE 降幅 %",
                "selection_action": cfg["selection_action"], "selected_artifact_id": item["selected_artifact_id"],
                "candidate_submitted": item["candidate_submitted"], "experiment": (cfg.get("experiment") or {}).get("name", "observe"),
                "experiment_schedule": item.get("experiment_schedule"), "candidate_gate": cfg.get("candidate_gate"),
                "candidate_upload_skipped": cfg.get("candidate_upload_skipped"),
                "experiment_outcome": cfg.get("experiment_outcome"),
                "controller_decisions": cfg["controller_decisions"], "model_call_count": len(call_ids),
                "wall_seconds": resources["wall_seconds"], "cpu_seconds": resources["cpu_seconds"],
                "local_training_seconds": training_seconds, "prediction_seconds": timing["prediction_process_seconds"],
                "candidate_payload_bytes": candidate_bytes, "state_payload_bytes": state_bytes,
                "token_totals": resources["token_totals"], "peak_rss_bytes": resources["peak_rss_bytes"]}
            assert item["selected_artifact_id"] in registry
            rows.append(row)
            previous_id = item["selected_artifact_id"]
        values = [r["metrics"][primary] for r in rows[1:]]
        stats = {"initial_primary": baseline[primary], "final_primary": values[-1], "final_gain": rows[-1]["gain_vs_initial"],
            "mean_primary_rounds_1_to_10": statistics.mean(values), "std_primary_rounds_1_to_10": statistics.pstdev(values),
            "worst_round_primary": min(values) if dataset == "A" else max(values),
            "adoptions": sum(r["selection_action"] == "adopt_candidate" for r in rows[1:]),
            "retained_parent_rounds": sum(r["selection_action"] == "retain_parent" for r in rows[1:]),
            "rollbacks": sum(r["selection_action"] == "rollback" for r in rows[1:]),
            "local_rejections_without_upload": sum(r["candidate_upload_skipped"] == "local_promotion_gate_rejected" for r in rows[1:]),
            "submitted_candidates": sum(r["candidate_submitted"] for r in rows[1:]),
            "mean_gain_rounds_1_to_10": statistics.mean(r["gain_vs_initial"] for r in rows[1:]),
            "worst_gain_rounds_1_to_10": min(r["gain_vs_initial"] for r in rows[1:]),
            "selected_metric_regressions": sum(r["selected_round_gain"] < -1e-9 for r in rows[1:]),
            "cpu_seconds": sum(r["cpu_seconds"] for r in rows[1:]), "callback_wall_seconds": sum(r["wall_seconds"] for r in rows[1:]),
            "candidate_payload_bytes": sum(r["candidate_payload_bytes"] for r in rows[1:]),
            "state_payload_bytes": sum(r["state_payload_bytes"] for r in rows[1:]),
            "model_calls": sum(r["model_call_count"] for r in rows[1:]), "fresh_worker_wall_seconds": completion["wall_seconds"]}
        data["datasets"][dataset] = {"primary": primary, "rows": rows, "summary": stats}
        summary_rows.append({"项目": dataset, "主指标": primary, **stats})
        for row in rows:
            flat.append({"项目": dataset, "轮次": row["round"], "选中产物": row["selected_artifact_id"],
                **{k: v for k, v in row.items() if k not in ("metrics", "controller_decisions", "round", "selected_artifact_id", "candidate_gate", "experiment_schedule")},
                **{k: v for k, v in row["metrics"].items() if k not in ("per_class", "groups", "group_source")}})
            if row["round"]:
                experiment_rows.append({"项目": dataset, "轮次": row["round"],
                    "请求方案": (row["experiment_schedule"] or {}).get("requested", row["experiment"]),
                    "执行方案": row["experiment"], "替换原因": (row["experiment_schedule"] or {}).get("substitution_reason"),
                    "是否上传": row["candidate_submitted"], "实际动作": row["selection_action"],
                    "候选收益": row["candidate_gain"], "候选测量来源": row["candidate_metric_source"],
                    "实际当轮收益": row["selected_round_gain"],
                    "准入诊断": json.dumps(row["candidate_gate"], ensure_ascii=False),
                    "上传跳过原因": row["candidate_upload_skipped"]})
            for name, values in row["metrics"].get("groups", {}).items():
                group_rows.append({"项目": dataset, "轮次": row["round"], "分组": name, **values})
            for name, values in row["metrics"].get("per_class", {}).items():
                class_rows.append({"项目": dataset, "轮次": row["round"], "类别": name, **values})
    data["validation"] = {"fresh_rounds": 30, "selected_predictions_recalculated": recalc_count,
        "candidate_predictions_recalculated": candidate_count,
        "source_hashes_verified": True, "official_initial_checkpoints_verified": True,
        "acknowledgements_and_model_receipts_verified": True, "parent_checkpoint_chain_verified": True,
        "raw_selected_metrics_reconciled": True, "baseline_round_excluded_from_round_averages": True}
    verification_file = run / "verification.json"
    if verification_file.exists():
        verified = read(source(verification_file))
        assert verified["total_rounds"] == 30 and verified["real_model_calls"] == 0
        data["test_verification"] = {"source": str(verification_file), "tests": verified["tests"],
            "transport": verified["transport"], "total_rounds": 30}
        write(output / "测试与协议验收.json", verified)
    comparison_file = run / "b-default-comparison.json"
    if comparison_file.exists():
        comparison = read(source(comparison_file))
        assert registration.get("b_guard_replay") is True
        assert comparison["fresh_default_rounds"] == 10 and comparison["real_model_calls"] == 0
        assert comparison["first_nine_round_predictions_identical"]
        assert comparison["default_protocol_verified"] and comparison["default_parent_chain_verified"]
        default_root = Path(comparison["default_run_root"]).resolve()
        assert default_root.is_relative_to(ROOT / "runtime")
        for filename, expected in comparison["evidence_sha256"].items():
            path = Path(filename).resolve()
            assert path.is_relative_to(run.resolve()) or path.is_relative_to(default_root)
            assert digest(path) == expected
            sources[str(path)] = expected
        for key in ("rmse", "mae"):
            assert math.isclose(comparison["guard"]["final"][key],
                data["datasets"]["B"]["rows"][-1]["metrics"][key], rel_tol=1e-12)
        data["b_default_comparison"] = comparison
        write(output / "B默认顺序对照.json", comparison)
    data["sources_sha256"] = sources
    write(output / "逐轮指标数据.json", data)
    csv_export(output / "逐轮指标.csv", flat)
    csv_export(output / "分组指标.csv", group_rows)
    csv_export(output / "A分类指标.csv", class_rows)
    csv_export(output / "十轮汇总.csv", summary_rows)
    csv_export(output / "实验与准入.csv", experiment_rows)
    sync_metric_summaries(data, output)
    return data


def render(data, output):
    sync_metric_summaries(data, output)
    import numpy as np
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    from PIL import Image, ImageOps, ImageDraw
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.titlesize": 16,
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#bbc4d1", "axes.labelcolor": "#28374b", "text.color": "#182b42",
        "xtick.color": "#4f6076", "ytick.color": "#4f6076", "grid.color": "#e4e9ef", "grid.alpha": .8,
        "savefig.facecolor": "white", "svg.fonttype": "none"})
    colors = {"A": "#2563eb", "B": "#087f78", "C": "#cb6a22"}
    blue, green, orange, purple = "#2563eb", "#087f78", "#cb6a22", "#8759b4"
    source_colors = {"FD001": blue, "FD002": green, "FD003": purple, "FD004": orange}
    registration = data["registration"]
    branch = registration.get("code_branch", "未记录分支")
    created_date = registration["created_at"][:10]
    sequence_note = " · B 第10轮显式测试新保护候选" if registration.get("b_guard_replay") else ""
    catalog = []
    def rows(ds): return data["datasets"][ds]["rows"]
    def series(ds, metric): return [r["metrics"].get(metric, np.nan) for r in rows(ds)]
    def style(ax, ylabel, rounds=True):
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if rounds:
            ax.set_xlabel("轮次（0 为官方初始模型）")
            ax.set_xticks(range(11))
    def line(ax, values, label=None, color=blue, start=0, **kw):
        ax.plot(range(start, start + len(values)), values, marker="o", markersize=3.8, linewidth=1.9, color=color, label=label, **kw)
    def finish(fig, name, title, note):
        fig.suptitle(title)
        fig.get_layout_engine().set(rect=(0, .055, 1, .945))
        fig.text(.5, .01, f"{branch} · {created_date} · 固定公开集 / 离线控制器 · 实际选中模型 · 非隐藏成绩{sequence_note}", ha="center", fontsize=9, color="#5d6c7d")
        fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=.22)
        fig.savefig(output / f"{name}.svg", bbox_inches="tight", pad_inches=.22)
        plt.close(fig)
        catalog.append({"name": name, "title": title, "note": note})
    fig = plt.figure(figsize=(15, 11), layout="constrained")
    overview_grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, .9))
    axes = np.array([[fig.add_subplot(overview_grid[r, c]) for c in range(3)] for r in range(2)])
    for col, ds in enumerate("ABC"):
        ax = axes[0, col]
        key = data["datasets"][ds]["primary"]
        auxiliary = "accuracy" if ds == "A" else "mae"
        key_label, auxiliary_label = ("Macro-F1", "Accuracy") if ds == "A" else ("RMSE", "MAE")
        values = np.array(series(ds, key)) * (100 if ds == "A" else 1)
        aux_values = np.array(series(ds, auxiliary)) * (100 if ds == "A" else 1)
        line(ax, values, key_label, blue)
        line(ax, aux_values, auxiliary_label, green, linestyle="--")
        ax.set_title(f"{ds} · {'F1 / Accuracy ↑' if ds == 'A' else 'RMSE / MAE ↓'}")
        style(ax, "%" if ds == "A" else "误差 / " + ("分钟" if ds == "B" else "循环"))
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(-.5, 11.25)
        for metric_values in (values, aux_values):
            ax.annotate(f"{metric_values[-1]:.2f}", (10, metric_values[-1]), xytext=(8, 0),
                textcoords="offset points", ha="left", va="center", fontsize=9)
        gains = [r["gain_vs_initial"] for r in rows(ds)]
        aux_gains = ((aux_values - aux_values[0]) if ds == "A" else
                     100 * (aux_values[0] - aux_values) / aux_values[0])
        line(axes[1, col], gains, key_label, blue)
        line(axes[1, col], aux_gains, auxiliary_label, green, linestyle="--")
        axes[1, col].set_title(f"{ds} · 相对初始模型的实际累计提升")
        style(axes[1, col], "F1 / Accuracy 提升 · 百分点" if ds == "A" else "RMSE / MAE 降幅 · %")
        axes[1, col].legend(fontsize=8, loc="best")
    nasa_values = np.array(series("C", "nasa_score"), dtype=float)
    nasa_score_ax = fig.add_subplot(overview_grid[2, :2])
    line(nasa_score_ax, nasa_values / 1e9, "NASA 非对称评分", orange)
    nasa_score_ax.set_title("C · NASA 非对称评分 ↓（越低越好）")
    style(nasa_score_ax, "NASA 评分总和 / 十亿")
    for index in (0, 10):
        nasa_score_ax.annotate(f"{nasa_values[index] / 1e9:.4f}", (index, nasa_values[index] / 1e9),
            xytext=(5 if index == 0 else -5, 9), textcoords="offset points",
            ha="left" if index == 0 else "right", fontsize=9)
    nasa_gain_ax = fig.add_subplot(overview_grid[2, 2])
    nasa_gains = 100 * (nasa_values[0] - nasa_values) / nasa_values[0]
    line(nasa_gain_ax, nasa_gains, color=orange)
    nasa_gain_ax.set_title("C · NASA 相对初始模型的改善")
    style(nasa_gain_ax, "NASA 降幅 / %")
    nasa_gain_ax.annotate(f"{nasa_gains[-1]:.2f}%", (10, nasa_gains[-1]), xytext=(-5, 9),
        textcoords="offset points", ha="right", fontsize=9)
    finish(fig, "01-overview", "新测十轮：F1、Accuracy、RMSE、MAE、NASA 与提升", "只统计各轮实际选中模型；A 用 F1 和 Accuracy 百分点，B/C 用 RMSE、MAE 和 NASA 相对降幅。NASA 单独绘制评分总和及降幅，越低越好；仅比较本次同一公开验证集。")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, ds in zip(axes, "ABC"):
        rr = rows(ds)[1:]
        values = [r["candidate_gain"] if r["candidate_gain"] is not None else np.nan for r in rr]
        adopted = [r["selection_action"] == "adopt_candidate" for r in rr]
        bars = ax.bar(range(1, 11), values, color=[colors[ds] if a else "#c1cad5" for a in adopted])
        for bar, r in zip(bars, rr):
            if (r.get("candidate_metric_source") or "").startswith("local_validation"):
                bar.set_hatch("///")
        line(ax, [r["selected_round_gain"] for r in rr], "实际选中模型当轮收益", color="#26364a", start=1)
        ax.axhline(0, color="#718095", linewidth=.8)
        for r in rr:
            if r["candidate_gain"] is None:
                ax.annotate("无候选", (r["round"], 0), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(ds)
        style(ax, "F1 提升 / 百分点" if ds == "A" else "RMSE 降幅 / %")
        ax.set_xlim(.4, 10.6); ax.set_xticks(range(1, 11)); ax.set_xlabel("轮次")
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor=colors[ds]), Patch(facecolor="#c1cad5")]
        labels += ["彩色柱：已采用", "灰柱：未采用"]
        if any((r.get("candidate_metric_source") or "").startswith("local_validation") for r in rr):
            handles.append(Patch(facecolor="#c1cad5", hatch="///")); labels.append("斜线：本地验证后未上传")
        ax.legend(handles, labels, fontsize=8, loc="best")
    finish(fig, "02-candidate-vs-selected", "候选收益与实际采用收益", "柱为候选相对当轮父模型的收益，折线为实际选中收益。灰柱未采用；斜线柱来自本地验证，未上传且没有额外原生预测复核；实心柱有独立预测。所有累计收益仅取实际选中模型。")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for key, label, color in (("macro_f1", "Macro-F1", blue), ("accuracy", "Accuracy", green)):
        line(axes[0], np.array(series("A", key)) * 100, label, color)
    style(axes[0], "%"); axes[0].legend(); axes[0].set_title("A · 分类主指标")
    recall = np.array([[r["metrics"]["per_class"][name]["recall"] * 100 for r in rows("A")] for name in LABELS])
    im = axes[1].imshow(recall, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    axes[1].set_yticks(range(6), CHINESE); axes[1].set_xticks(range(11)); axes[1].set_xlabel("轮次"); axes[1].set_title("A · 各类别召回率")
    fig.colorbar(im, ax=axes[1], label="召回率 %", shrink=.7)
    for i in range(6):
        for j in range(11): axes[1].text(j, i, f"{recall[i,j]:.0f}", ha="center", va="center", fontsize=7, color="white" if recall[i,j] > 58 else "#182b42")
    x = np.arange(6)
    for offset, row, label, color in ((-.19, rows("A")[0], "初始", "#aebdce"), (.19, rows("A")[-1], "第 10 轮", blue)):
        axes[2].bar(x + offset, [row["metrics"]["per_class"][n]["f1"] * 100 for n in LABELS], width=.36, label=label, color=color)
    axes[2].set_xticks(x, CHINESE, rotation=35); axes[2].set_ylim(0, 103)
    axes[2].legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    axes[2].set_title("A · 各类别 F1 初始与最终", pad=39)
    style(axes[2], "F1 (%)", False)
    finish(fig, "03-a-classification", "A：准确率、Macro-F1 与类别表现", "类别指标从本次独立预测重算；类别召回率热图固定 0–100%。甲苯仅有 5 条验证样本，单类变化应结合支持数解释。")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for key, label, color in (("rmse", "RMSE", blue), ("mae", "MAE", green)):
        line(axes[0,0], series("B", key), label, color)
    axes[0,0].legend(); axes[0,0].set_title("B · 总体误差"); style(axes[0,0], "分钟")
    line(axes[0,1], series("B", "bias"), color=orange); axes[0,1].axhline(0, color="#8995a6", linewidth=1)
    axes[0,1].set_title("B · 平均偏差：负值表示低估"); style(axes[0,1], "预测 − 真实 / 分钟")
    for key, label, color in (("p95_absolute_error", "P95", blue), ("p99_absolute_error", "P99", purple)):
        line(axes[1,0], series("B", key), label, color)
    axes[1,0].legend(); axes[1,0].set_title("B · 尾部绝对误差"); style(axes[1,0], "分钟")
    for key, label, color in (("group_mean_rmse", "三根轴承等权平均", green), ("group_worst_rmse", "最差轴承", orange)):
        line(axes[1,1], series("B", key), label, color)
    axes[1,1].legend(); axes[1,1].set_title("B · 设备均衡诊断"); style(axes[1,1], "设备 RMSE / 分钟")
    finish(fig, "04-b-errors", "B：总体、偏差、尾部与设备表现", "设备等权平均是诊断指标，不替代官方按样本计算的总体 RMSE。")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for key, label, color in (("rmse", "RMSE", blue), ("mae", "MAE", green)):
        line(axes[0,0], series("C", key), label, color)
    axes[0,0].legend(); axes[0,0].set_title("C · 总体误差"); style(axes[0,0], "循环")
    line(axes[0,1], np.array(series("C", "nasa_score")) / 1e9, color=orange)
    axes[0,1].set_title("C · NASA 非对称评分 ↓"); style(axes[0,1], "NASA / 十亿")
    for key, label, color in (("p95_absolute_error", "P95", blue), ("p99_absolute_error", "P99", purple)):
        line(axes[1,0], series("C", key), label, color)
    axes[1,0].legend(); axes[1,0].set_title("C · 尾部绝对误差"); style(axes[1,0], "循环")
    for key, label, color in (("nasa_over", "高估惩罚", purple), ("nasa_under", "低估惩罚", orange)):
        line(axes[1,1], np.array(series("C", key)) / 1e9, label, color)
    axes[1,1].legend(); axes[1,1].set_title("C · 惩罚方向分解"); style(axes[1,1], "NASA / 十亿")
    finish(fig, "05-c-errors", "C：RMSE、MAE、NASA 与尾部误差", "本次按注册配置执行 RMSE、MAE 和 NASA 准入保护；NASA 仍可能在允许幅度内上升。NASA 仅在本次同一批公开投影样本上比较，不与每台发动机末端一条的隐藏测试口径混比。")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for row_index, ds in enumerate("BC"):
        names = list(rows(ds)[0]["metrics"]["groups"])
        for name in names:
            values = [r["metrics"]["groups"][name]["rmse"] for r in rows(ds)]
            count = rows(ds)[0]["metrics"]["groups"][name]["rows"]
            line(axes[row_index,0], values, f"{name} (n={count:,})", color=source_colors.get(name))
        axes[row_index,0].set_title(f"{ds} · {'各轴承' if ds == 'B' else '各来源子集'} RMSE")
        axes[row_index,0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
        axes[row_index,0].set_title(f"{ds} · {'各轴承' if ds == 'B' else '各来源子集'} RMSE", pad=42)
        style(axes[row_index,0], "分钟" if ds == "B" else "循环")
        gains = np.array([[100 * (rows(ds)[0]["metrics"]["groups"][n]["rmse"] - r["metrics"]["groups"][n]["rmse"])
            / rows(ds)[0]["metrics"]["groups"][n]["rmse"] for r in rows(ds)] for n in names])
        limit = max(1., float(np.max(np.abs(gains))))
        image = axes[row_index,1].imshow(gains, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(0, -limit, limit))
        axes[row_index,1].set_yticks(range(len(names)), names); axes[row_index,1].set_xticks(range(11)); axes[row_index,1].set_xlabel("轮次")
        axes[row_index,1].set_title(f"{ds} · 分组相对初始 RMSE 降幅")
        for i in range(len(names)):
            for j in range(11): axes[row_index,1].text(j, i, f"{gains[i,j]:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(gains[i,j]) > .55 * limit else "#182b42")
        fig.colorbar(image, ax=axes[row_index,1], label="降幅 %：正值改善 / 负值退化", shrink=.7)
    finish(fig, "06-group-robustness", "分组表现：收益是否覆盖不同设备与来源子集", "B 用公开 bearing 追溯字段，C 用来源子集 FD001–FD004；这些字段仅用于诊断，不作为模型输入，也不等同于正式比赛阶段。")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for col, ds in enumerate("ABC"):
        rr = rows(ds)[1:]
        line(axes[0,col], [r["wall_seconds"] for r in rr], "回调墙钟", blue, start=1)
        line(axes[0,col], [r["cpu_seconds"] for r in rr], "参与方 CPU", green, start=1)
        axes[0,col].set_title(f"{ds} · 逐轮回调资源"); axes[0,col].legend(fontsize=8); style(axes[0,col], "秒")
        axes[0,col].set_xticks(range(1,11)); axes[0,col].set_xlabel("轮次")
        line(axes[1,col], [r["prediction_seconds"] for r in rr], color=colors[ds], start=1)
        axes[1,col].set_title(f"{ds} · 独立预测（{rows(ds)[0]['metrics']['rows']:,} 条）")
        style(axes[1,col], "预测进程墙钟 / 秒"); axes[1,col].set_xticks(range(1,11)); axes[1,col].set_xlabel("轮次")
    finish(fig, "07-resources", "本次逐轮耗时与独立预测成本", "本机并行测试；回调墙钟包含 RPC/宿主候选测量等待，CPU 仅计参与方进程。预测进程时间包含启动和 JSON。未强制容器 2 CPU/4 GiB，不作为官方效率分；token、峰值内存缺失记未知。")
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.7))
    ax = fig.add_subplot(grid[0,:])
    palette = {"adopt_candidate": "#92c5b1", "retain_parent": "#d7dfe8", "rollback": "#c5abd9"}
    words = {"adopt_candidate": "采用", "retain_parent": "保留", "rollback": "回滚"}
    for i, ds in enumerate("ABC"):
        for r in rows(ds)[1:]:
            j = r["round"] - 1
            ax.add_patch(plt.Rectangle((j-.45, i-.33), .9, .66, facecolor=palette[r["selection_action"]], edgecolor="none"))
            word = "本地拒绝" if r.get("candidate_upload_skipped") == "local_promotion_gate_rejected" else words[r["selection_action"]]
            ax.text(j, i, f"{word}\n{r['model_call_count']} 次回执", ha="center", va="center", fontsize=9)
    ax.set_xlim(-.6,9.6); ax.set_ylim(2.6,-.6); ax.set_xticks(range(10), range(1,11)); ax.set_yticks(range(3), list("ABC")); ax.set_xlabel("轮次")
    ax.set_title("实际选版动作与离线回执数（不代表真实大模型调用）")
    for col, ds in enumerate("ABC"):
        ax = fig.add_subplot(grid[1,col]); rr=rows(ds)[1:]
        candidate=np.array([r["candidate_payload_bytes"] for r in rr])/1024
        state=np.array([r["state_payload_bytes"] for r in rr])/1024
        ax.bar(range(1,11), candidate, label="候选文件", color=colors[ds])
        ax.bar(range(1,11), state, bottom=candidate, label="状态与证据", color="#bdc9d8")
        ax.set_title(ds); style(ax,"KiB / 轮"); ax.set_xticks(range(1,11)); ax.set_xlabel("轮次"); ax.legend(fontsize=8)
    finish(fig,"08-selection-and-upload","选版、回执与上传文件量","载荷仅统计候选和已保存状态的文件字节，不含 RPC/Base64 封装。未采用也已上传的候选仍计入成本；真实模型调用数为 0。")
    fig, axes = plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    for ax,ds in zip(axes,"ABC"):
        s=data["datasets"][ds]["summary"]
        values=[s[k] for k in ("initial_primary","mean_primary_rounds_1_to_10","worst_round_primary","final_primary")]
        values=np.array(values)*(100 if ds=="A" else 1)
        ax.bar(range(4),values,color=["#b8c5d5",colors[ds],"#e4b68e",colors[ds]])
        ax.set_xticks(range(4),["初始","十轮平均","最差轮","最终"])
        for i,value in enumerate(values): ax.text(i,value,f"{value:.2f}",ha="center",va="bottom",fontsize=10)
        ax.set_ylim(0,max(values)*1.16); ax.set_title(f"{ds} · {'Macro-F1 ↑' if ds=='A' else 'RMSE ↓'}")
        style(ax,"F1 (%)" if ds=="A" else "分钟" if ds=="B" else "循环",False)
    finish(fig,"09-average-worst-final","整轮统计：平均、最差轮与最终表现","平均和最差轮仅使用第 1—10 轮的实际选中模型，排除第 0 轮。固定公开投影没有正式阶段切换，最差轮不冒称最差阶段，恢复速度不计算。")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for i, ds in enumerate("BC"):
        for key, label, color in (("underestimate_p99", "低估幅度 P99", blue), ("overestimate_p99", "高估幅度 P99", orange)):
            line(axes[i, 0], series(ds, key), label, color)
        axes[i, 0].set_title(f"{ds} · 分方向极端误差")
        style(axes[i, 0], "分钟" if ds == "B" else "循环"); axes[i, 0].legend(fontsize=9, loc="center right")
        line(axes[i, 1], np.array(series(ds, "overestimate_fraction")) * 100, "高估占比", orange)
        line(axes[i, 1], [100 * r["metrics"]["underestimate_count"] / r["metrics"]["rows"] for r in rows(ds)], "低估占比", blue)
        axes[i, 1].set_title(f"{ds} · 高估与低估的样本占比")
        style(axes[i, 1], "%"); axes[i, 1].set_ylim(0, 100); axes[i, 1].legend(fontsize=9)
    finish(fig, "10-directional-errors", "B/C：高估与低估分别改善了多少", "幅度 P99 在各方向样本内独立计算；负误差为低估，正误差为高估，零误差不计入两类。不能用整体平均偏差代替尾部风险。")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
    names = list(rows("C")[0]["metrics"]["groups"])
    contribution = np.array([[r["metrics"]["groups"][name]["nasa_score"] / r["metrics"]["nasa_score"] * 100
        for r in rows("C")] for name in names])
    axes[0].stackplot(range(11), contribution, labels=names, colors=[source_colors[name] for name in names], alpha=.85)
    axes[0].set_title("C · 各来源子集的 NASA 贡献", pad=39)
    style(axes[0], "NASA 占比 %"); axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=4)
    line(axes[1], np.array(series("C", "nasa_top_1pct_share")) * 100, color=orange)
    axes[1].set_title("C · 惩罚最高的 1% 样本贡献")
    style(axes[1], "NASA 占比 %"); axes[1].set_ylim(0, 100)
    finish(fig, "11-nasa-concentration", "C：NASA 惩罚集中在哪些样本", "每轮分别选取惩罚最高的 ceil(样本数×1%) 条，不保证各轮是相同样本。NASA 总和与集中度需要同时看；来源子集不是比赛阶段。")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), layout="constrained")
    for ax, ds in zip(axes, "ABC"):
        rr = rows(ds)[1:]
        cpu = np.cumsum([r["cpu_seconds"] for r in rr])
        gains = [r["gain_vs_initial"] for r in rr]
        ax.plot(cpu, gains, color=colors[ds], marker="o", markersize=4)
        for i in (0, 4, 9):
            ax.annotate(f"第 {i + 1} 轮", (cpu[i], gains[i]), xytext=(0, 10 if i != 4 else -17),
                textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(f"{ds} · 收益与累计 CPU")
        style(ax, "F1 提升 / 百分点" if ds == "A" else "RMSE 降幅 / %", False)
        ax.set_xlabel("参与方回调累计 CPU / 秒"); ax.margins(x=.15, y=.2)
    finish(fig, "12-gain-vs-cpu", "累计资源投入对应的实际提升", "CPU 来自本次参与方回调账本，不含真实大模型、宿主测量和容器启动成本；只描述此次轨迹。并行本机计时不能推导正式资源分或版本间加速倍数。")
    write(output/"chart-catalog.json",catalog)
    cells=[]
    for i,item in enumerate(catalog):
        with Image.open(output/(item["name"]+".png")) as im:
            thumb=ImageOps.contain(im.convert("RGB"),(900,540))
            canvas=Image.new("RGB",(930,590),"white");canvas.paste(thumb,((930-thumb.width)//2,32))
            ImageDraw.Draw(canvas).text((16,10),item["name"],fill="#26364a");cells.append(canvas)
    contact=Image.new("RGB",(930*3,590*math.ceil(len(cells)/3)),"#e6ebf2")
    for i,im in enumerate(cells):contact.paste(im,((i%3)*930,(i//3)*590))
    contact.save(output/"contact-sheet.jpg",quality=90)
    table=[]
    for metric in data["metric_summaries"]:
        ds = metric["dataset"]; scale = metric["display_scale"]
        gain = "未知" if metric["improvement"] is None else f'{metric["improvement"]:+.3f} {metric["improvement_unit"]}'
        table.append(f'<tr><td>{ds}</td><td>{metric["label"]}（{metric["display_unit"]}）</td>'
            f'<td>{metric["initial"] * scale:,.4f}</td><td>{metric["final"] * scale:,.4f}</td><td>{gain}</td>'
            f'<td>{data["datasets"][ds]["summary"]["adoptions"]}/10</td></tr>')
    gallery="".join(f'<section id="{item["name"]}"><h2>{escape(item["title"])}</h2><img src="{item["name"]}.png" alt="{escape(item["title"])}" loading="lazy"><p>{escape(item["note"])}</p><p><a href="{item["name"]}.png">高清 PNG</a> · <a href="{item["name"]}.svg">矢量 SVG</a></p></section>' for item in catalog)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本次新测 · A/B/C 十轮图表</title><style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#182b42;max-width:1260px;margin:auto;padding:24px}h1,h2{font-weight:500}h2{margin-top:38px}a{color:#2456b2}p{color:#4f6076}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce3ed}section{border-top:1px solid #dce3ed;margin-top:34px}.table{overflow:auto}@media print{section{break-before:page}}</style>'''
    html+=f'<h1>{escape(branch)} · A/B/C 各十轮</h1><p>{escape(created_date)} · 当前工作区源码及配置重新运行 30 轮。第 0 轮为官方初始检查点；全部提升均相对该初始模型，不与历史版本拼接。离线控制器，真实续训及独立逐条预测，非正式隐藏评分。</p>'
    dirty_note = "含尚未提交的工作区改动；实际测试版本由登记中的逐文件 SHA-256 确定。" if registration.get("working_tree_status") else ""
    html+=f'<p>分支：{escape(branch)} · 提交：{escape(registration["code_commit"][:12])}。{dirty_note}测试顺序：{escape(registration.get("regression_order", "原登记顺序"))}。</p>'
    if registration.get("b_guard_replay"):
        html+='<p>B 第 10 轮显式请求 bearing_upward_guard，代替默认顺序中的 bearing_stumps；前九轮不变。默认菜单中保护候选排在第十一项，这条轨迹专门验证新增方案，不能视作默认十轮必然结果。</p>'
    if data.get("code_snapshot"):
        html+=f'<p>源码快照：{escape(data["code_snapshot"]["created_at"])}；共 {len(data["code_snapshot"]["files_sha256"])} 个文件核对摘要。测试期间使用冻结副本。</p>'
    html+='<div class="table"><table><tr><th>项目</th><th>指标（单位）</th><th>初始</th><th>第 10 轮</th><th>实际改善</th><th>候选采用</th></tr>'+''.join(table)+'</table></div>'
    if data.get("b_default_comparison"):
        comparison = data["b_default_comparison"]
        html+='<h2>B 新方案与默认顺序对照</h2><p>另外从官方初始模型重跑 B 默认顺序十轮，独立保存模型历史；两组前九轮的逐条预测完全一致。总计主轨迹 30 轮，加默认 B 对照 10 轮。</p><div class="table"><table><tr><th>指标（分钟）</th><th>默认十轮</th><th>第十轮显式测试保护方案</th><th>相对默认降低</th></tr>'
        for key in ("rmse", "mae"):
            before, after = comparison["default"]["final"][key], comparison["guard"]["final"][key]
            html+=f'<tr><td>{key.upper()}</td><td>{before:,.4f}</td><td>{after:,.4f}</td><td>{100 * (before - after) / before:.3f}%</td></tr>'
        before, after = (comparison[role]["final_groups"]["Bearing1_4"]["mae"] for role in ("default", "guard"))
        html+=f'</table></div><p>Bearing1_4 的 MAE 从 {before:.4f} 增至 {after:.4f} 分钟（增加 {100 * (after / before - 1):.2f}%）；总体收益未覆盖所有设备的所有指标。<a href="B默认顺序对照.json">独立对照与验收记录</a>。</p>'
    html+='<p><a href="各项指标汇总.csv">全部指标汇总 CSV</a> · <a href="逐轮指标.csv">逐轮 CSV</a> · <a href="十轮汇总.csv">任务汇总 CSV</a> · <a href="分组指标.csv">分组 CSV</a> · <a href="A分类指标.csv">分类 CSV</a> · <a href="实验与准入.csv">方案与拒绝原因 CSV</a> · <a href="逐轮指标数据.json">完整指标与来源</a></p>'
    if (output / "测试与协议验收.json").exists():
        verified = read(output / "测试与协议验收.json")
        html+=f'<p>本次测试：项目 {verified["tests"]["project"]["passed"]} 项通过；SDK 离线 {verified["tests"]["sdk104"]["passed"]} 项通过、{verified["tests"]["sdk104"]["skipped"]} 项外部调用跳过；30 轮完成确认。<a href="测试与协议验收.json">测试与协议验收明细</a>。</p>'
    html+='<h2>与客观评分指标的对应关系</h2><div class="table"><table><tr><th>评分项</th><th>本次可查看</th><th>评估范围</th></tr><tr><td>平均性能提升 · 15 分</td><td>逐轮、十轮均值、最终相对初始提升</td><td>固定公开投影；未换算官方得分</td></tr><tr><td>最差阶段 · 5 分</td><td>最差轮、最差公开分组</td><td>诊断代理，缺少正式阶段序列</td></tr><tr><td>恢复速度 · 5 分</td><td>缺失，未计算</td><td>本次没有阶段切换</td></tr><tr><td>稳定性 · 5 分</td><td>30 轮结束确认、选版、主指标变化</td><td>同一条连续训练链，非独立重复试验</td></tr><tr><td>资源效率 · 5 分</td><td>CPU、墙钟、预测耗时、上传文件量</td><td>真实模型 Token 与正式容器计量缺失</td></tr></table></div>'
    html+='<p>token usage 和峰值内存缺失记未知。各项目并行运行，墙钟仅描述本次环境。平均和最差轮排除初始第 0 轮；十轮是连续训练轨迹，不是十次独立随机重复。点击图片链接可查看高清图。</p>'
    html+='<p>图表索引：'+ ' · '.join(f'<a href="#{item["name"]}">{i:02d}</a>' for i,item in enumerate(catalog,1))+'</p>'+gallery+'</html>'
    (output/"index.html").write_text(html,encoding="utf-8")
    notes=f"{branch} · {created_date} · A/B/C 各 10 轮\n基于提交 {registration['code_commit']} 及当时工作区改动。\n{dirty_note}\n全部提升相对本任务官方初始 checkpoint，不是相对 v18/v20。\n12 组图表：逐轮提升、候选与实际采用、A 分类、B 误差、C NASA、分组、耗时、上传、平均/最差/最终、方向误差、NASA 集中度、CPU 与收益。\n打开 index.html 浏览，PNG 为 240 DPI，SVG 可编辑。CSV 使用 UTF-8 BOM。\n实际选中模型指标均从原生逐行预测重算；斜线候选柱来自本地验证，未上传且未另作原生预测复核，详见实验与准入.csv。\n十轮为一条连续训练链，非十次独立随机重复；第 0 轮不计入十轮平均。\n离线控制器不等于真实 27B；本机测试不等于比赛容器或隐藏评测。\n缺失 Token/内存不填零；固定投影不计算恢复速度，不将最差轮当作最差阶段。\n"
    if registration.get("b_guard_replay"):
        notes+="B 第10轮显式测试 bearing_upward_guard，替代默认的 bearing_stumps；这不是默认十轮轨迹。\n"
    if data.get("b_default_comparison"):
        notes+="另有从头运行的 B 默认十轮对照，见 B默认顺序对照.json；主图仍只使用本次主轨迹30轮，不混合两组结果。\n"
    (output/"使用说明.txt").write_text(notes,encoding="utf-8")
    return catalog


def verify_and_package(data, catalog, output, reviewed=False):
    from PIL import Image
    import xml.etree.ElementTree as ET
    assert all(digest(Path(p))==h for p,h in data["sources_sha256"].items())
    code_root = Path(data["registration"].get("code_root", ROOT)).resolve()
    assert all(digest(code_root / p)==h for p,h in data["registration"]["sources_sha256"].items())
    if data.get("code_snapshot"):
        snapshot = data["code_snapshot"]
        assert digest(snapshot["manifest"]) == snapshot["manifest_sha256"]
        assert all(digest(code_root / p) == h for p, h in snapshot["files_sha256"].items())
    dimensions=[]
    for item in catalog:
        with Image.open(output/(item["name"]+".png")) as im:
            assert im.width>=3000 and im.height>=1000
            dimensions.append({"chart":item["name"],"width":im.width,"height":im.height})
        ET.parse(output/(item["name"]+".svg"))
    write(output/"chart-validation.json",{**data["validation"],"chart_count":len(catalog),"png_svg_files_valid":True,
        "source_files_unchanged":True,"dimensions":dimensions,"visual_review_complete":reviewed,"real_model_calls":0,
        "historical_results_used":False,"official_score_verified":False})
    names=[p for p in output.iterdir() if p.is_file() and p.name not in ("file-manifest.json", "contact-sheet.jpg")]
    write(output/"file-manifest.json",[{"file":p.name,"bytes":p.stat().st_size,"sha256":digest(p)} for p in names])
    archive=output.parent/(output.name+".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in [*names,output/"file-manifest.json"]:z.write(p,output.name+"/"+p.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={"charts":len(catalog),"output":str(output),"zip":str(archive),"zip_sha256":digest(archive),
        "zip_bytes":archive.stat().st_size,"source_prediction_rows":data["validation"]["selected_predictions_recalculated"]}
    write(output.parent/(output.name+"-package.json"),receipt)
    print(json.dumps(receipt,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--package-only",action="store_true")
    parser.add_argument("--render-only",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--visual-review-complete",action="store_true")
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve()
    data=read(output/"逐轮指标数据.json") if args.package_only or args.render_only else prepare(run,output)
    if args.prepare_only:
        print(json.dumps(data["validation"],ensure_ascii=False))
        return
    catalog=read(output/"chart-catalog.json") if args.package_only else render(data,output)
    verify_and_package(data,catalog,output,args.visual_review_complete)


if __name__=="__main__":main()
