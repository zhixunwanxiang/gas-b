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


def prepare(run, output):
    import numpy as np
    registration = read(run / "registration.json")
    assert registration["historical_results_used"] is False
    assert registration["rounds_per_dataset"] == 10
    assert all(digest(ROOT / p) == h for p, h in registration["sources_sha256"].items())
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
    flat, group_rows, class_rows, summary_rows = [], [], [], []
    recalc_count, candidate_count = 0, 0
    for dataset in "ABC":
        directory = run / dataset
        report = read(source(directory / "report.json"))
        completion = read(source(run / f"{dataset}-completion.json"))
        assert completion["fresh_run"] and completion["rounds"] == 10
        assert len(report["rounds"]) == 10
        validation_file = source(directory / "data/validation.jsonl")
        assert digest(validation_file) == registration["inputs"][dataset]["sha256"]["public/validation.jsonl"]
        assert digest(directory / "initial/checkpoint.npz") == registration["inputs"][dataset]["sha256"]["checkpoint.npz"]
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
                if dataset == "C":
                    with np.errstate(over="ignore"):
                        penalties = np.expm1(np.where(error < 0, -error / 13., error / 10.))
                    total = float(penalties.sum())
                    metrics.update(nasa_score=total if math.isfinite(total) else None,
                        nasa_over=float(penalties[error >= 0].sum()), nasa_under=float(penalties[error < 0].sum()))
                    for name in ("nasa_over", "nasa_under"):
                        if not math.isfinite(metrics[name]): metrics[name] = None
                grouped = {}
                for group in np.unique(groups):
                    e = error[groups == group]
                    grouped[str(group)] = {"rows": len(e), "rmse": float(np.sqrt(np.mean(e ** 2))),
                        "mae": float(np.abs(e).mean()), "bias": float(e.mean())}
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
            assert len(call_ids) >= 2 and item["model_call_ids"] == call_ids
            assert any(f["frame"].get("type") == "round_ack" and f["frame"].get("accepted") is True for f in frames)
            assert len(item["history_ids"]) == index - 1
            assert cfg["validation_sha256"].split(":")[-1] == digest(validation_file)
            candidate_metric = None
            if item["candidate_submitted"]:
                child = host / "candidate/checkpoint.npz"
                registry[f"candidate-{index}"] = child
                training = read(source(host / "candidate/training.json"))
                assert training["parent_checkpoint_sha256"].split(":")[-1] == digest(registry[previous_id])
                assert training["checkpoint_sha256"].split(":")[-1] == digest(child)
                measurement_index += 1
                expected_candidate = ({"macro_f1": cfg["local_validation_after"]} if dataset == "A" else cfg["validation_after"])
                candidate_values, _ = measured(measurement_index, expected_candidate, candidate=True)
                candidate_metric = candidate_values[primary]
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
                "gain_unit": "F1 百分点" if dataset == "A" else "RMSE 降幅 %",
                "selection_action": cfg["selection_action"], "selected_artifact_id": item["selected_artifact_id"],
                "candidate_submitted": item["candidate_submitted"], "experiment": (cfg.get("experiment") or {}).get("name", "observe"),
                "controller_decisions": cfg["controller_decisions"], "model_call_count": len(call_ids),
                "wall_seconds": resources["wall_seconds"], "cpu_seconds": resources["cpu_seconds"],
                "local_training_seconds": training_seconds, "prediction_seconds": timing["prediction_process_seconds"],
                "candidate_payload_bytes": candidate_bytes, "state_payload_bytes": state_bytes,
                "token_totals": resources["token_totals"], "peak_rss_bytes": resources["peak_rss_bytes"]}
            rows.append(row)
            previous_id = item["selected_artifact_id"]
        values = [r["metrics"][primary] for r in rows[1:]]
        stats = {"initial_primary": baseline[primary], "final_primary": values[-1], "final_gain": rows[-1]["gain_vs_initial"],
            "mean_primary_rounds_1_to_10": statistics.mean(values), "std_primary_rounds_1_to_10": statistics.pstdev(values),
            "worst_round_primary": min(values) if dataset == "A" else max(values),
            "adoptions": sum(r["selection_action"] == "adopt_candidate" for r in rows[1:]),
            "retained_parent_rounds": sum(r["selection_action"] == "retain_parent" for r in rows[1:]),
            "rollbacks": sum(r["selection_action"] == "rollback" for r in rows[1:]),
            "cpu_seconds": sum(r["cpu_seconds"] for r in rows[1:]), "callback_wall_seconds": sum(r["wall_seconds"] for r in rows[1:]),
            "model_calls": sum(r["model_call_count"] for r in rows[1:]), "fresh_worker_wall_seconds": completion["wall_seconds"]}
        data["datasets"][dataset] = {"primary": primary, "rows": rows, "summary": stats}
        summary_rows.append({"项目": dataset, "主指标": primary, **stats})
        for row in rows:
            flat.append({"项目": dataset, "轮次": row["round"], "选中产物": row["selected_artifact_id"],
                **{k: v for k, v in row.items() if k not in ("metrics", "controller_decisions", "round", "selected_artifact_id")},
                **{k: v for k, v in row["metrics"].items() if k not in ("per_class", "groups", "group_source")}})
            for name, values in row["metrics"].get("groups", {}).items():
                group_rows.append({"项目": dataset, "轮次": row["round"], "分组": name, **values})
            for name, values in row["metrics"].get("per_class", {}).items():
                class_rows.append({"项目": dataset, "轮次": row["round"], "类别": name, **values})
    data["validation"] = {"fresh_rounds": 30, "selected_predictions_recalculated": recalc_count,
        "candidate_predictions_recalculated": candidate_count,
        "source_hashes_verified": True, "official_initial_checkpoints_verified": True,
        "acknowledgements_and_model_receipts_verified": True, "parent_checkpoint_chain_verified": True,
        "raw_selected_metrics_reconciled": True, "baseline_round_excluded_from_round_averages": True}
    data["sources_sha256"] = sources
    write(output / "逐轮指标数据.json", data)
    csv_export(output / "逐轮指标.csv", flat)
    csv_export(output / "分组指标.csv", group_rows)
    csv_export(output / "A分类指标.csv", class_rows)
    csv_export(output / "十轮汇总.csv", summary_rows)
    return data


def render(data, output):
    import numpy as np
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import TwoSlopeNorm
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
    bc_version = data["registration"]["code_versions"]["B"]
    version_label = "A 现版 / B、C " + bc_version
    blue, green, orange, purple = "#2563eb", "#087f78", "#cb6a22", "#8759b4"
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
        fig.text(.5, .01, "本次新测 · 固定公开验证集 · 离线控制器 / 真实数值训练与独立预测 · 非官方隐藏成绩", ha="center", fontsize=9, color="#5d6c7d")
        fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=.22)
        fig.savefig(output / f"{name}.svg", bbox_inches="tight", pad_inches=.22)
        plt.close(fig)
        catalog.append({"name": name, "title": title, "note": note})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for col, ds in enumerate("ABC"):
        ax = axes[0, col]
        key = data["datasets"][ds]["primary"]
        values = np.array(series(ds, key)) * (100 if ds == "A" else 1)
        line(ax, values, color=colors[ds])
        ax.axhline(values[0], color="#8995a6", linestyle="--", linewidth=1)
        ax.set_title(f"{ds} · {'Macro-F1 ↑' if ds == 'A' else 'RMSE ↓'}")
        style(ax, "Macro-F1 (%)" if ds == "A" else "RMSE / " + ("分钟" if ds == "B" else "循环"))
        ax.annotate(f"{values[-1]:.4f}" if ds == "A" else f"{values[-1]:.2f}", (10, values[-1]), xytext=(-6, 12), textcoords="offset points", ha="right")
        gains = [r["gain_vs_initial"] for r in rows(ds)]
        line(axes[1, col], gains, color=colors[ds])
        axes[1, col].set_title(f"{ds} · 相对初始模型的实际累计提升")
        style(axes[1, col], "F1 提升 / 百分点" if ds == "A" else "RMSE 降幅 / %")
    finish(fig, "01-overview", "新测十轮：主指标与实际累计提升", "只统计各轮实际选中模型；A 用 F1 百分点，B/C 用 RMSE 相对降幅，不混算任务单位。")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, ds in zip(axes, "ABC"):
        rr = rows(ds)[1:]
        values = [r["candidate_gain"] if r["candidate_gain"] is not None else np.nan for r in rr]
        adopted = [r["selection_action"] == "adopt_candidate" for r in rr]
        ax.bar(range(1, 11), values, color=[colors[ds] if a else "#c1cad5" for a in adopted], label="候选收益：彩色已采用 / 灰色未采用")
        line(ax, [r["selected_round_gain"] for r in rr], "实际选中模型当轮收益", color="#26364a", start=1)
        ax.axhline(0, color="#718095", linewidth=.8)
        for r in rr:
            if r["candidate_gain"] is None:
                ax.annotate("无候选", (r["round"], 0), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(ds)
        style(ax, "F1 提升 / 百分点" if ds == "A" else "RMSE 降幅 / %")
        ax.set_xlim(.4, 10.6); ax.set_xticks(range(1, 11)); ax.set_xlabel("轮次")
        ax.legend(fontsize=8, loc="best")
    finish(fig, "02-candidate-vs-selected", "候选收益与实际采用收益", "正值候选不代表已采用；未采用候选的收益不计入累计提升。观察轮标注无候选，不绘制候选柱。")
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
    axes[2].set_xticks(x, CHINESE, rotation=35); axes[2].set_ylim(0, 103); axes[2].legend(); axes[2].set_title("A · 各类别 F1 初始与最终")
    style(axes[2], "F1 (%)", False)
    finish(fig, "03-a-classification", "A：准确率、Macro-F1 与类别表现", "类别指标从本次独立预测重算；类别召回率热图固定 0–100%，便于跨轮比较。")
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
    axes[0,1].set_title("C · NASA 非对称惩罚总和"); style(axes[0,1], "NASA / 十亿")
    for key, label, color in (("p95_absolute_error", "P95", blue), ("p99_absolute_error", "P99", purple)):
        line(axes[1,0], series("C", key), label, color)
    axes[1,0].legend(); axes[1,0].set_title("C · 尾部绝对误差"); style(axes[1,0], "循环")
    for key, label, color in (("nasa_over", "高估惩罚", purple), ("nasa_under", "低估惩罚", orange)):
        line(axes[1,1], np.array(series("C", key)) / 1e9, label, color)
    axes[1,1].legend(); axes[1,1].set_title("C · 惩罚方向分解"); style(axes[1,1], "NASA / 十亿")
    finish(fig, "05-c-errors", "C：RMSE、MAE、NASA 与尾部误差", ("xiaoyang 同时检查 NASA、高估 P99 和最差工况。" if bc_version == "xiaoyang" else "V18 仅以原 RMSE/MAE 门槛选版；NASA 可能在 RMSE 改善时恶化。") + "曲线保留真实变化；NASA 仅在相同验证样本上比较。")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for row_index, ds in enumerate("BC"):
        names = list(rows(ds)[0]["metrics"]["groups"])
        for name in names:
            values = [r["metrics"]["groups"][name]["rmse"] for r in rows(ds)]
            count = rows(ds)[0]["metrics"]["groups"][name]["rows"]
            line(axes[row_index,0], values, f"{name} (n={count:,})", color=None)
        axes[row_index,0].set_title(f"{ds} · {'各轴承' if ds == 'B' else '各来源子集'} RMSE")
        axes[row_index,0].legend(fontsize=8); style(axes[row_index,0], "分钟" if ds == "B" else "循环")
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
            ax.text(j, i, f"{words[r['selection_action']]}\n{r['model_call_count']} 次调用", ha="center", va="center", fontsize=9)
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
    write(output/"chart-catalog.json",catalog)
    cells=[]
    for i,item in enumerate(catalog):
        with Image.open(output/(item["name"]+".png")) as im:
            thumb=ImageOps.contain(im.convert("RGB"),(900,540))
            canvas=Image.new("RGB",(930,590),"white");canvas.paste(thumb,((930-thumb.width)//2,32))
            ImageDraw.Draw(canvas).text((16,10),item["name"],fill="#26364a");cells.append(canvas)
    contact=Image.new("RGB",(930*3,590*3),"#e6ebf2")
    for i,im in enumerate(cells):contact.paste(im,((i%3)*930,(i//3)*590))
    contact.save(output/"contact-sheet.jpg",quality=90)
    table=[]
    for ds in "ABC":
        s=data["datasets"][ds]["summary"]
        table.append(f"<tr><td>{ds}</td><td>{'Macro-F1 ↑' if ds=='A' else 'RMSE ↓'}</td><td>{s['initial_primary']:.6f}</td><td>{s['final_primary']:.6f}</td><td>{s['final_gain']:+.3f} {'百分点' if ds=='A' else '%'}</td><td>{s['adoptions']}/10</td></tr>")
    gallery="".join(f'<section id="{item["name"]}"><h2>{escape(item["title"])}</h2><img src="{item["name"]}.png" alt="{escape(item["title"])}" loading="lazy"><p>{escape(item["note"])}</p><p><a href="{item["name"]}.png">高清 PNG</a> · <a href="{item["name"]}.svg">矢量 SVG</a></p></section>' for item in catalog)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本次新测 · A/B/C 十轮图表</title><style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#182b42;max-width:1260px;margin:auto;padding:24px}h1,h2{font-weight:500}h2{margin-top:38px}a{color:#2456b2}p{color:#4f6076}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce3ed}section{border-top:1px solid #dce3ed;margin-top:34px}.table{overflow:auto}@media print{section{break-before:page}}</style>'''
    html+='<h1>本次新测：'+escape(version_label)+' · 各十轮</h1><p>2026-11-26 · 全部曲线来自此次重新运行的 30 轮；第 0 轮为官方初始检查点，不使用历史结果拼接。离线控制器，真实续训及独立逐条预测，非正式隐藏评分。</p>'
    html+='<div class="table"><table><tr><th>项目</th><th>主指标</th><th>初始</th><th>第 10 轮</th><th>实际提升</th><th>候选采用</th></tr>'+''.join(table)+'</table></div>'
    html+='<p><a href="逐轮指标.csv">逐轮 CSV</a> · <a href="十轮汇总.csv">汇总 CSV</a> · <a href="分组指标.csv">分组 CSV</a> · <a href="A分类指标.csv">分类 CSV</a> · <a href="逐轮指标数据.json">完整指标与来源</a></p>'
    html+='<p>固定公开验证没有阶段切换；本批无法评价正式最差阶段与恢复速度。token usage 和峰值内存缺失，均记未知。各项目并行运行，墙钟仅描述本次环境，不作算法加速或官方资源得分结论。</p>'+gallery+'</html>'
    (output/"index.html").write_text(html,encoding="utf-8")
    notes="本次新测图表："+version_label+"，各 10 轮。\n所有曲线只使用本次注册运行的独立预测和日志，不使用历史结果。\n打开 index.html 浏览，PNG 为 240 DPI，SVG 可编辑。\nCSV 使用 UTF-8 BOM，可用 Excel 打开。第 0 轮不计入十轮平均。\n离线控制器不等于真实 27B；本机测试不等于比赛容器或隐藏评测。\n候选与实际选中收益分开；缺失 token/内存不填零；固定投影不计算恢复速度。\n"
    (output/"使用说明.txt").write_text(notes,encoding="utf-8")
    return catalog


def verify_and_package(data, catalog, output, reviewed=False):
    from PIL import Image
    import xml.etree.ElementTree as ET
    assert all(digest(Path(p))==h for p,h in data["sources_sha256"].items())
    dimensions=[]
    for item in catalog:
        with Image.open(output/(item["name"]+".png")) as im:
            assert im.width>=3000 and im.height>=1000
            dimensions.append({"chart":item["name"],"width":im.width,"height":im.height})
        ET.parse(output/(item["name"]+".svg"))
    write(output/"chart-validation.json",{**data["validation"],"chart_count":len(catalog),"png_svg_files_valid":True,
        "source_files_unchanged":True,"dimensions":dimensions,"visual_review_complete":reviewed,"real_model_calls":0,
        "historical_results_used":False,"official_score_verified":False})
    names=[p for p in output.iterdir() if p.is_file() and p.name!="file-manifest.json"]
    write(output/"file-manifest.json",[{"file":p.name,"bytes":p.stat().st_size,"sha256":digest(p)} for p in names])
    suffix = "-xiaoyang" if data["registration"]["code_versions"]["B"] == "xiaoyang" else ""
    archive=output.parent/("本次新测-ABC十轮图表"+suffix+"-20261126.zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in [*names,output/"file-manifest.json"]:z.write(p,output.name+"/"+p.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({"charts":len(catalog),"output":str(output),"zip":str(archive),"source_prediction_rows":data["validation"]["selected_predictions_recalculated"]},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--package-only",action="store_true")
    parser.add_argument("--render-only",action="store_true")
    parser.add_argument("--visual-review-complete",action="store_true")
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve()
    data=read(output/"逐轮指标数据.json") if args.package_only or args.render_only else prepare(run,output)
    catalog=read(output/"chart-catalog.json") if args.package_only else render(data,output)
    verify_and_package(data,catalog,output,args.visual_review_complete)


if __name__=="__main__":main()
