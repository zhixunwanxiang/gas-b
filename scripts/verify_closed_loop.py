"""Independent persisted-evidence checks for a completed local closed-loop run."""
# 历史回放的独立复核：从已保存的预测和授权公开标签重算指标，再核对部署/回滚证据。
# 这是本地历史结果复核，不会获得平台隐藏标签或创建真实模型回执。
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from closed_loop_validation import (ROOT, ParentEngine, restore_engine, digest,
                                    Experiment, POLICY, write, train_home)
import numpy as np
from gas_a.model import Model
from gas_a.signals import dat_inputs, reveal_labels
from gas_a.sensor_training.data import load_pair, load_model, evaluate
from gas_a.sensor_training.optimize import train_candidate


def independently_score(y, predicted):
    matrix = [[0] * 6 for _ in range(6)]
    for target, actual in zip(y.tolist(), predicted.tolist()):
        matrix[target][actual] += 1
    f1 = sum(2 * matrix[k][k] / max(1, sum(matrix[k]) + sum(row[k] for row in matrix)) for k in range(6)) / 6
    accuracy = sum(matrix[k][k] for k in range(6)) / len(y)
    return matrix, f1, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    windows = json.loads((run / "history.json").read_text(encoding="utf-8"))
    registry = json.loads((run / "registry.json").read_text(encoding="utf-8"))
    events = [json.loads(s) for s in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    before = json.loads((run / "protected-hashes-before.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, condition, detail=None):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(name)

    check("protected_inputs_and_sources_unchanged", all(digest(p) == h for p,h in before.items()), len(before))
    check("policy_saved_before_first_replay", events[0]["event"] == "policy_locked")
    check("eight_windows_in_natural_order", [w["batch"] for w in windows] == list(range(3,11)))
    all_y, all_pred, all_prior = [], [], []
    previous_state = None
    for w in windows:
        batch = w["batch"]
        directory = run / "windows" / f"batch{batch:02d}"
        commit = json.loads((directory / "prediction_commit.json").read_text(encoding="utf-8"))
        prediction = np.load(directory / "predictions.npy", allow_pickle=False)
        engine = ParentEngine() if previous_state is None else restore_engine(registry[previous_state["version"]], previous_state["state"])
        feature = dat_inputs(Path(summary["dataset"]) / f"batch{batch}.dat", [])
        replay = engine.predict_batch(feature["x"])
        labels = reveal_labels(Path(summary["dataset"]) / f"batch{batch}.dat", feature)["y"]
        check(f"batch{batch}_restored_state_reproduces_predictions", np.array_equal(replay, prediction))
        check(f"batch{batch}_prediction_file_integrity", digest(directory / "predictions.npy") == commit["file_sha256"])
        check(f"batch{batch}_source_integrity", feature["sha256"] == commit["source_sha256"])
        e = [x for x in events if x.get("batch") == batch]
        position = {x["event"]: x["index"] for x in e}
        check(f"batch{batch}_temporal_order", position["drift_monitored_without_labels"] < position["prediction_committed_before_labels"]
              < position["labels_revealed"] < position["diagnosis_completed"])
        if w["candidate"]:
            persisted_candidate = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
            check(f"batch{batch}_candidate_selection_record_consistent", persisted_candidate == w["candidate"])
            check(f"batch{batch}_training_after_reveal", position["labels_revealed"] < position["candidate_trained"] < position["candidate_evaluated"])
            trained = next(x for x in e if x["event"] == "candidate_trained")
            check(f"batch{batch}_no_replay_rows_in_gradient", trained["stream_rows_used_for_gradient"] == 0 and trained["training_rows"] == 445)
        engine.release_labels(labels)
        matrix, f1, accuracy = independently_score(labels, prediction)
        check(f"batch{batch}_independent_metric_reconciliation", matrix == w["operational"]["confusion_matrix"]
              and abs(f1-w["operational"]["macro_f1"]) < 1e-12 and abs(accuracy-w["operational"]["accuracy"]) < 1e-12)
        previous_state = json.loads((directory / "next-state.json").read_text(encoding="utf-8"))
        if batch < 10:
            check(f"batch{batch}_next_batch_uses_selected_version", previous_state["version"] == windows[batch-2]["prediction_version"])
        all_y.append(labels); all_pred.append(prediction)
        old_path = Path(summary["previous_run"]) / "windows" / f"batch{batch:02d}" / "predictions.npz"
        with np.load(old_path, allow_pickle=False) as archive:
            all_prior.append(archive["active"].argmax(1))
    y, predicted, prior = map(np.concatenate, (all_y, all_pred, all_prior))
    matrix, f1, accuracy = independently_score(y, predicted)
    check("pooled_matrix_f1_accuracy_independently_recomputed", matrix == summary["current"]["confusion_matrix"]
          and abs(f1-summary["current"]["macro_f1"]) < 1e-12 and abs(accuracy-summary["current"]["accuracy"]) < 1e-12)
    _, old_f1, old_accuracy = independently_score(y, prior)
    check("previous_pdf_score_independently_reconciled", abs(old_f1-summary["previous_pdf"]["macro_f1"]) < 1e-12
          and abs(old_accuracy-summary["previous_pdf"]["accuracy"]) < 1e-12)
    check("terminal_window_does_not_train", not windows[-1]["candidate"] and any(x["event"] == "terminal_training_skipped" for x in events))
    check("deployment_only_affects_future", all(d["effective_from_batch"] == d["batch"] + 1 for d in summary["deployments"]))
    check("rule_controller_not_mislabeled_as_external_ai", summary["external_ai_calls"] == 0)
    check("natural_rollbacks_separated_from_drills", summary["natural_rollbacks"] == sum(d["reason"] == "natural_performance_rollback" for d in summary["deployments"]))
    for name, result in summary["rollback_drills"].items():
        if isinstance(result, dict):
            check(f"{name}_drill_switch_and_prediction_restoration", result["physical_pointer_switched"] and result["restored_predictions_equal"])
    check("isolated_drills_preserve_operational_pointer", summary["rollback_drills"]["operational_pointer_unchanged"])
    # Exercise the original public router sequencing and feedback validation contracts.
    router_entry = registry["released_retrained"]
    from gas_a.router import FourSensorRouter
    router = FourSensorRouter(router_entry["bundle"], router_entry["config"])
    x = dat_inputs(Path(summary["dataset"]) / "batch3.dat", [])["x"][:4]
    router.predict_batch(x)
    try:
        router.predict_batch(x)
        check("predict_twice_without_feedback_rejected", False)
    except ValueError:
        check("predict_twice_without_feedback_rejected", True)
    try:
        router.release_labels(np.array([0,1,2,6]))
        check("invalid_feedback_rejected", False)
    except ValueError:
        check("invalid_feedback_rejected", True)
    router.release_labels(np.array([0,1,2,3]))
    check("valid_feedback_advances_exactly_one_batch", router.batch_count == 1 and router.waiting is None)
    # Retrain every executed sensor candidate from its actual immutable parent.
    train, validation = load_pair(ROOT / "data/public/train.jsonl", ROOT / "data/public/validation.jsonl")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    limits = dict(cfg["training"]["limits"], checkpoint_reference=evaluate(load_model(ROOT / "models/reference.npz"), validation),
                  checkpoint_reference_sha256="sha256:" + digest(ROOT / "models/reference.npz"))
    by_digest = {"sha256:" + digest(p): p for p in (ROOT / "models").glob("*.npz")}
    for p in (run / "versions").glob("*/*.npz"):
        by_digest["sha256:" + digest(p)] = p
    for w in windows:
        candidate = w["candidate"]
        if not candidate:
            continue
        for component in candidate["components"]:
            destination = run / "verification" / "training-replay" / candidate["version"] / component["role"]
            parent = by_digest[component["parent_sha256"]]
            train_candidate(parent, train, validation, destination, component["parameters"],
                            {"weakest_class": w["diagnosis"]["weakest_class"]}, limits)
            check(f"batch{w['batch']}_{component['role']}_exact_training_replay", "sha256:" + digest(destination / "checkpoint.npz") == component["checkpoint_sha256"])
    home_dest = run / "verification/home-replay"
    home_dest.mkdir(parents=True)
    train_home(Model.load(ROOT / "models/start.npz"), train, cfg["training"]["home"], home_dest)
    for name in ("home_trunk", "home_router"):
        check(name + "_exact_training_replay", digest(home_dest / f"{name}.npz") == digest(Path(router_entry["bundle"]) / f"{name}.npz"))
    check("protected_sources_still_unchanged_after_verification", all(digest(p) == h for p,h in before.items()))
    result = {"passed": all(x["passed"] for x in checks), "check_count": len(checks), "checks": checks,
              "scope": "all eight windows, all fourteen sensor training trials, both home components, persisted state restoration, metrics and two isolated rollback drills"}
    write(run / "verification.json", result)
    print(json.dumps({"passed": result["passed"], "check_count": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
