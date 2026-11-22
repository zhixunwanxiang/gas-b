"""Local tools with explicit baseline/resume, public data, and selected model paths."""
# 显式路径的本地训练/预测/评估命令：由用户指定官方基线或实际已选检查点。
# 这些工具不替代平台的 rnd.history、真实模型调用、产物上传及 finish 回执。
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from contest_harness.predict import serve_predictions

from .continuation import continue_candidate
from .sensor_training.data import evaluate, load_data, load_pair
from .sensor_training.io import digest, write_json
from .frozen_router import load_predictor

ROOT = Path(__file__).resolve().parents[2]


def train_main(argv=None):
    # 官方基线模式核对 baseline-info 摘要；续训模式必须显式给出模型和公开数据路径。
    # 它只做数值实验，不自动完成平台真实模型规划/选版和提交回执。
    parser = argparse.ArgumentParser(description="Continue an explicit official baseline or selected checkpoint on public JSONL.")
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--baseline-dir", type=Path, help="official package containing checkpoint.npz, baseline-info.json, public/")
    origin.add_argument("--from-baseline", type=Path, help="explicit first-round checkpoint file")
    origin.add_argument("--resume", type=Path, help="checkpoint actually selected from a previous round")
    parser.add_argument("--train", type=Path, help="authorized public training JSONL")
    parser.add_argument("--validation", type=Path, help="authorized public validation JSONL")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--experiment", default="frozen_router", help="name from submission_config.json")
    parser.add_argument("--epochs", type=int, help="override epochs within configured maximum")
    args = parser.parse_args(argv)
    expected, evidence = None, []
    if args.baseline_dir:
        base = args.baseline_dir.resolve()
        checkpoint = base / "checkpoint.npz"
        info_path = base / "baseline-info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("dataset_key") != "A" or info.get("task") != "classification":
            parser.error("--baseline-dir must be an official task A classification package")
        expected = info["checkpoint_sha256"]
        if digest(checkpoint).removeprefix("sha256:") != expected.removeprefix("sha256:"):
            parser.error("Official checkpoint differs from baseline-info.json")
        train_path = args.train or base / "public/train.jsonl"
        validation_path = args.validation or base / "public/validation.jsonl"
        evidence = [base / name for name in ("baseline-info.json", "training.json", "history.jsonl", "TASK.md", "TRAINING_GUIDE.md")
                    if (base / name).is_file()]
    else:
        checkpoint = (args.resume or args.from_baseline).resolve()
        if args.train is None or args.validation is None:
            parser.error("--train and --validation are required with --resume or --from-baseline")
        train_path, validation_path = args.train, args.validation
    output = args.output.resolve()
    protected = [ROOT / "models", ROOT / "data", ROOT / "src", ROOT / "vendor", checkpoint.parent,
                 train_path.resolve().parent, validation_path.resolve().parent]
    if any(output.is_relative_to(path.resolve()) for path in protected):
        parser.error("--output must be outside source checkpoint and public data directories")
    config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
    menu = {spec["name"]: spec for spec in config["experiments"]}
    if args.experiment not in menu:
        parser.error("Unknown --experiment; choose from " + ", ".join(menu))
    spec = dict(menu[args.experiment])
    if args.epochs is not None:
        spec["epochs"] = args.epochs
        spec["minimum_epochs"] = min(spec.get("minimum_epochs", 1), args.epochs)
    train, validation = load_pair(train_path, validation_path)
    parent = load_predictor(checkpoint, train["paths"])
    before = evaluate(parent, validation)
    observed = {k: v for k, v in before["per_class"].items() if v["support"]}
    report = continue_candidate(checkpoint, train, validation, output, spec,
                                {"weakest_class": min(observed, key=lambda k: observed[k]["f1"])},
                                config["limits"], expected_parent_sha256=expected)
    sources = {}
    if evidence:
        folder = output / "parent_evidence"
        folder.mkdir()
        for source in evidence:
            shutil.copyfile(source, folder / source.name)
            sources[source.name] = digest(source)
    write_json(output / "local_run.json", {
        "parent_checkpoint_path": str(checkpoint), "parent_checkpoint_sha256": digest(checkpoint),
        "origin": "resume" if args.resume else "baseline", "baseline_evidence_sha256": sources,
        "train_path": str(train_path.resolve()), "validation_path": str(validation_path.resolve()),
        "configuration": spec, "platform_scored": False, "selection_status": "candidate_only",
        "passes_parent_validation_floor": report["validation_after"]["macro_f1"] + 1e-12 >= before["macro_f1"],
    })
    print(json.dumps({"checkpoint": str(output / "checkpoint.npz"),
                      "parent_checkpoint_sha256": report["parent_checkpoint_sha256"],
                      "initial_steps": report["initial_steps"], "final_steps": report["final_steps"],
                      "selection_status": "candidate_only",
                      "local_validation_before": before["macro_f1"],
                      "local_validation_after": report["validation_after"]["macro_f1"]}))
    return 0


def predict_main(argv=None):
    # 加载明确指定的完整产物，通过 SDK 的逐行服务返回结果，日志与协议输出分离。
    parser = argparse.ArgumentParser(description="Predict JSONL from an explicit selected checkpoint; flush every row.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="JSONL requests; defaults to stdin")
    args = parser.parse_args(argv)
    from .submission import predict
    previous = os.environ.get("CONTEST_ARTIFACT_PATH")
    os.environ["CONTEST_ARTIFACT_PATH"] = str(args.checkpoint.resolve())
    try:
        if args.input:
            with args.input.open(encoding="utf-8") as stream:
                serve_predictions(predict, reader=stream, writer=sys.stdout)
        else:
            serve_predictions(predict, reader=sys.stdin, writer=sys.stdout)
    finally:
        if previous is None:
            os.environ.pop("CONTEST_ARTIFACT_PATH", None)
        else:
            os.environ["CONTEST_ARTIFACT_PATH"] = previous
    return 0


def evaluate_main(argv=None):
    # 以调用者明确提供的授权带标签数据测量指定模型，不将该分数标为隐藏测试成绩。
    parser = argparse.ArgumentParser(description="Evaluate any selected task A checkpoint against explicit public validation JSONL.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    args = parser.parse_args(argv)
    model = load_predictor(args.checkpoint)
    data = load_data(args.validation, model.meta["paths"])
    print(json.dumps({"checkpoint_sha256": digest(args.checkpoint), "validation_sha256": data["sha256"],
                      "scope": "local_public_validation", "metrics": evaluate(model, data)}))
    return 0
