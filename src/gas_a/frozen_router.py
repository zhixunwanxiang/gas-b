"""Select routing on authorized public validation; predict without label feedback."""
# 正式无标签路由包：训练阶段用授权公开验证集选择路由，导出后冻结权重和类别对策略。
# 预测时只允许更新由已见特征形成的流状态，既不读取真实标签，也不等待批后反馈。
# 主检查点、路由清单和全部专家组件共同构成一个产物，加载与回滚必须校验完整集合。
from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time

import numpy as np

from .home_training import binary_cutoffs, train_home
from .model import Model as HomeModel
from . import pair_review
from .router import HomeExpert, PAIRS, pair_codes
from .signals import Flatline, correct, home_probabilities
from .sensor_training.data import evaluate, load_model, score_predictions
from .sensor_training.io import CLASSES, digest, write_json
from .sensor_training.reference.data import finite_number
from .sensor_training.optimize import train_candidate

FORMAT = "gas-a.frozen-four-sensor.v1"
MANIFEST = "routing.json"
COMPONENTS = ("four_sensor", "five_sensor", "home_trunk", "home_router")


def _relative(root, name):
    # 路由清单只接受产物内的普通相对路径，拒绝盘符、反斜杠和上级目录引用。
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Bundle paths must be relative artifact paths")
    return Path(root).joinpath(*PurePosixPath(name).parts)


def _new_stream(reference):
    # 为一个新预测流建立空的特征窗口与平线检测器，不保存任何标签或误差反馈。
    return {"drift": (deque(maxlen=128), np.zeros(128), 0),
            "flatline": Flatline(reference.scale[96:104])}


def _expert_scores(models, policy, x, stream):
    # 计算四/五传感器及全输入专家分数；可选特征修正与平线告警仅由已见输入产生。
    reference = models["four_sensor"]
    alarm = stream["flatline"].predict(x[:, 96:104])
    if policy["feature_correction"]:
        adjusted, stream["drift"] = correct(x, reference, stream["drift"])
    else:
        adjusted = x
    four = reference.predict(x)
    five_model = models["five_sensor"]
    five = five_model.predict(x)
    if policy["feature_correction"]:
        four = (four + reference.predict(adjusted)) * .5
        five = (five + five_model.predict(adjusted)) * .5
    home = home_probabilities(HomeExpert(models["home_trunk"], models["home_router"]),
                              adjusted, policy["cutoffs"])
    return four, five, home, alarm


def route_scores(scores, policy):
    # 先融合 home/five 挑战者，再仅对冻结的类别对且无平线告警的样本融合挑战者与 four。
    # 类别对及权重来自导出策略，预测时不根据真实标签重新选择路由。
    four, five, home, alarm = scores
    # 内层权重决定 home/five 占比，外层权重只用于已启用类别对；平线告警样本保留 four。
    challenger = (1 - policy["inner_weight"]) * home + policy["inner_weight"] * five
    codes = pair_codes(four.argmax(1), challenger.argmax(1))
    selected = ~alarm & np.isin(codes, policy["active_pairs"])
    result = four.copy()
    outer = policy["outer_weight"]
    result[selected] = (1 - outer) * four[selected] + outer * challenger[selected]
    return result


class FrozenRouter:
    """Immutable label-selected policy; streaming state contains features only."""
    def __init__(self, models, manifest, component_paths, manifest_path, review=None, review_path=None):
        # 复制冻结策略并创建该进程专属特征流状态，防止外部字典修改改变已加载策略。
        self.models = models
        self.manifest = manifest
        self.component_paths = component_paths
        self.manifest_path = manifest_path
        self.meta = models["four_sensor"].meta
        self.policy = deepcopy(manifest["policy"])
        self.review = review
        self.review_path = review_path
        self._stream = _new_stream(models["four_sensor"])

    def _predict(self, x, stream):
        routed = route_scores(_expert_scores(self.models, self.policy, x, stream), self.policy)
        return routed if self.review is None else self.review.predict(x, routed)

    def predict(self, x):
        # Independent evaluation starts from the same empty state as a new process.
        # 批量独立评估从空状态开始，确保每次公开测量不继承之前评测的特征窗口。
        x = np.asarray(x, dtype=np.float64)
        return self._predict(x, _new_stream(self.models["four_sensor"]))

    def predict_one(self, x):
        # 平台逐行预测沿用本进程特征流状态；它与整批从空状态按顺序处理保持相同语义。
        return self._predict(x, self._stream)


def load_predictor(checkpoint, paths=None, resolve_file=None):
    """Load a plain MLP or its complete frozen bundle; verify every used component."""
    # 先加载主 MLP，再依据标记读取清单；每个组件都需通过路径、摘要、类别顺序和 scaler 一致性检查。
    # resolve_file 由 SDK 场景注入，用于强制所有被引用组件都已在产物声明中出现。
    checkpoint = Path(checkpoint)
    resolve_file = (lambda path: path) if resolve_file is None else resolve_file
    anchor = load_model(resolve_file(checkpoint), paths)
    marker = anchor.meta.get("prediction_bundle")
    if marker is None:
        return anchor
    if not isinstance(marker, dict) or marker.get("format") != FORMAT:
        raise ValueError("Unsupported prediction bundle")
    root = checkpoint.parent
    manifest_path = resolve_file(_relative(root, marker.get("manifest")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 先确认每层容器类型，再读取字段或创建集合，避免坏清单泄漏 AttributeError/TypeError。
    # 格式错误仅拒绝这份产物；调用方可以隔离旧历史，实际父模型则仍须完整通过检查。
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("classes") != CLASSES or not isinstance(manifest.get("components"), dict)
            or set(manifest["components"]) != set(COMPONENTS)):
        raise ValueError("Invalid frozen routing manifest")
    policy = manifest.get("policy")
    if (not isinstance(policy, dict)
            or set(policy) != {"inner_weight", "outer_weight", "active_pairs", "cutoffs", "feature_correction"}
            or type(policy["feature_correction"]) is not bool
            or any(not finite_number(policy[k]) or not 0 <= policy[k] <= 1
                   for k in ("inner_weight", "outer_weight"))
            or not isinstance(policy["active_pairs"], list)
            or any(type(c) is not int or c not in [6*a+b for a,b in PAIRS] for c in policy["active_pairs"])
            or len(set(policy["active_pairs"])) != len(policy["active_pairs"])
            or not isinstance(policy["cutoffs"], dict) or set(policy["cutoffs"]) != {"0", "1"}
            or any(not isinstance(policy["cutoffs"][k], dict)
                   or not finite_number(policy["cutoffs"][k].get("cut"))
                   or not 0 < policy["cutoffs"][k]["cut"] < 1 for k in ("0", "1"))):
        raise ValueError("Invalid frozen routing policy")
    models, component_paths = {}, {}
    # 对所有实际使用的组件逐一验证字节摘要与 scaler，防止部分回滚或混装不同版本。
    for role in COMPONENTS:
        entry = manifest["components"][role]
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError("Invalid routing component: " + role)
        path = resolve_file(_relative(root, entry.get("path")))
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Routing component digest mismatch: {role}")
        if role == "four_sensor" and path.resolve() != checkpoint.resolve():
            raise ValueError("Primary checkpoint must be the routed four-sensor component")
        model = load_model(path, anchor.meta["paths"]) if role in COMPONENTS[:2] else HomeModel.load(path)
        if (model.meta["classes"] != CLASSES or model.meta["paths"] != anchor.meta["paths"]
                or not np.array_equal(model.mean, anchor.mean) or not np.array_equal(model.scale, anchor.scale)):
            raise ValueError("Routing components have incompatible task or standardization")
        models[role], component_paths[role] = model, path
    review, review_path = None, None
    if "pair_review" in manifest or "pair_review_sha256" in marker:
        entry = manifest.get("pair_review")
        if (not isinstance(entry, dict) or entry.get("format") != pair_review.FORMAT
                or entry.get("path") != pair_review.FILENAME or not isinstance(entry.get("sha256"), str)
                or entry["sha256"] != marker.get("pair_review_sha256")):
            raise ValueError("Invalid pair review manifest")
        review_path = resolve_file(_relative(root, entry["path"]))
        if digest(review_path) != entry["sha256"]:
            raise ValueError("Pair review checkpoint digest mismatch")
        review = pair_review.PairReview.load(review_path)
    return FrozenRouter(models, manifest, component_paths, manifest_path, review, review_path)


def checkpoint_paths(checkpoint):
    # 声明完整路由需要的四个检查点；只上传主 MLP 会使平台恢复后无法重建同一预测器。
    model = load_predictor(checkpoint)
    if not isinstance(model, FrozenRouter):
        return [Path(checkpoint).name]
    root = Path(checkpoint).parent
    paths = [model.component_paths[role].relative_to(root).as_posix() for role in COMPONENTS]
    if model.review_path is not None:
        paths.append(model.review_path.relative_to(root).as_posix())
    return paths


def calibrate_policy(models, validation, cutoffs, search, *, controller=None, parent_metrics=None, review=None):
    """Compare a bounded family of class-pair policies using validation labels only."""
    # 在有上限的修正开关/融合权重菜单中，逐个类别对尝试加入路由。
    # 支持数不足或没有严格公开验证增益的类别对不启用；最后还必须通过外层整体晋升门槛。
    weights = search["inner_weights"]
    outer_weights = search.get("outer_weights", [.6])
    minimum_support, minimum_gain = search["minimum_support"], search["minimum_macro_f1_gain"]
    corrections = search["correction_choices"]
    if (not isinstance(weights, list) or not weights or len(weights) > 5
            or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not np.isfinite(w) or not 0 <= w <= 1 for w in weights)
            or not isinstance(outer_weights, list) or not outer_weights or len(outer_weights) > 5
            or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not np.isfinite(w) or not 0 <= w <= 1 for w in outer_weights)
            or type(minimum_support) is not int or minimum_support < 1
            or not isinstance(corrections, list) or not corrections or len(corrections) > 2
            or any(type(v) is not bool for v in corrections)
            or not isinstance(minimum_gain, (int, float)) or not np.isfinite(minimum_gain) or minimum_gain < 0):
        raise ValueError("Invalid bounded routing search")
    if (controller is None) != (parent_metrics is None):
        raise ValueError("Protected routing requires both controller and actual parent metrics")
    from .evolution import promotion_gate
    def gate(metric):
        return promotion_gate(parent_metrics, metric, controller) if controller is not None else {"eligible": True}
    def rank(metric):
        return (gate(metric)["eligible"], metric["macro_f1"])
    # 初始 active_pairs 为空，相当于安全保留 four；只有得到公开验证增益的类别对才逐步开启。
    base = {"inner_weight": .75, "outer_weight": .6, "active_pairs": [], "cutoffs": cutoffs,
            "feature_correction": False}
    y = validation["y"]
    evidence = review.evidence(validation["x"]) if review is not None else None

    def measure(probabilities):
        if evidence is not None:
            probabilities = review.apply(probabilities, evidence)
        return score_predictions(probabilities, y)

    best_policy = deepcopy(base)
    baseline = measure(models["four_sensor"].predict(validation["x"]))
    best = baseline
    trials = []
    for correction in corrections:
        mode = dict(base, feature_correction=correction)
        scores = _expert_scores(models, mode, validation["x"], _new_stream(models["four_sensor"]))
        for weight in weights:
            challenger = (1-weight) * scores[2] + weight * scores[1]
            codes = pair_codes(scores[0].argmax(1), challenger.argmax(1))
            for outer in outer_weights:
                policy = dict(mode, inner_weight=weight, outer_weight=outer, active_pairs=[])
                current = baseline
                for a, b in PAIRS:
                    code = 6*a+b
                    # 支持数按实际会触发这条路由的样本统计；过少样本不据此校准一个新分支。
                    support = int(np.count_nonzero(~scores[3] & (codes == code)))
                    if support < minimum_support:
                        continue
                    proposal = dict(policy, active_pairs=[*policy["active_pairs"], code])
                    metric = measure(route_scores(scores, proposal))
                    # 合格路由优先于不合格高分；已合格时不能为 F1 牺牲类别保护。
                    feasible, current_feasible = gate(metric)["eligible"], gate(current)["eligible"]
                    accepted = ((feasible and not current_feasible) or
                                (feasible == current_feasible and metric["macro_f1"] >
                                 current["macro_f1"] + max(1e-12, minimum_gain)))
                    trials.append({"feature_correction": correction, "inner_weight": weight, "outer_weight": outer,
                                   "pair": [CLASSES[a], CLASSES[b]], "support": support,
                                   "macro_f1": metric["macro_f1"], "gate": gate(metric), "accepted": accepted})
                    if accepted:
                        policy, current = proposal, metric
                if rank(current) > rank(best):
                    best_policy, best = deepcopy(policy), current
    return best_policy, {"scope": "authorized_public_validation", "validation_sha256": validation["sha256"],
                         "validation_rows": len(y), "four_sensor_metrics": baseline, "selected_metrics": best,
                         "parent_metrics": parent_metrics, "promotion_rules": controller, "selected_gate": gate(best),
                         "search": search, "trials": trials}


def _replace_checkpoint(model, target):
    # 在同一目录写新临时检查点后原子替换，避免清单标记写入一半留下损坏模型。
    handle, temporary = tempfile.mkstemp(prefix=".bundle-", suffix=".npz", dir=target.parent)
    os.close(handle)
    Path(temporary).unlink()  # Model.save exclusively creates a new file.
    try:
        model.save(temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _export_review(parent, candidate, fitted=None):
    if fitted is None and parent.review is None:
        return None, None, None
    path = candidate / pair_review.FILENAME
    if fitted is not None:
        review, train_sha = fitted
        review.save(path)
        entry = {"format": pair_review.FORMAT, "path": pair_review.FILENAME,
                 "sha256": digest(path), "train_sha256": train_sha}
    else:
        review, entry = parent.review, parent.manifest["pair_review"]
        shutil.copyfile(parent.review_path, path)
        if digest(path) != entry["sha256"]:
            raise ValueError("Retained pair review differs from the authorized parent")
    return review, path, entry


def _bundle_marker(review_entry):
    marker = {"format": FORMAT, "manifest": MANIFEST}
    if review_entry is not None:
        marker["pair_review_sha256"] = review_entry["sha256"]
    return marker


def retain_bundle(parent, candidate, validation, *, fitted_review=None):
    """Export the continued primary with byte-identical actual-parent auxiliaries."""
    candidate = Path(candidate)
    primary_path = candidate / "checkpoint.npz"
    primary = load_model(primary_path, parent.meta["paths"])
    auxiliary = candidate / "routing"
    auxiliary.mkdir()
    files = {"four_sensor": primary_path}
    retained = {}
    for role in COMPONENTS[1:]:
        source = parent.component_paths[role]
        expected = parent.manifest["components"][role]["sha256"]
        if digest(source) != expected:
            raise ValueError("Retained parent component changed: " + role)
        target = auxiliary / (role + ".npz")
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError("Retained component copy differs: " + role)
        files[role] = target
        retained[role] = {"sha256": expected, "unchanged": True}
    models = {**parent.models, "four_sensor": primary}
    review, review_path, entry = _export_review(parent, candidate, fitted_review)
    measured = evaluate(FrozenRouter(models, parent.manifest, files, candidate / MANIFEST, review, review_path), validation)
    primary.meta["prediction_bundle"] = _bundle_marker(entry)
    _replace_checkpoint(primary, primary_path)
    manifest = deepcopy(parent.manifest)
    manifest["components"] = {role: {"path": path.relative_to(candidate).as_posix(), "sha256": digest(path)}
                              for role, path in files.items()}
    manifest["prediction_parameter_count"] = sum(model.meta["parameter_count"] for model in models.values())
    manifest["prediction_parameter_count"] += sum(len(head["w"]) + 1
        for head in models["home_router"].meta["private_joint_rescue"].values())
    if entry is not None:
        manifest["pair_review"] = entry
        manifest["prediction_parameter_count"] += review.parameter_count
        if fitted_review is None:
            retained["pair_review"] = {"sha256": entry["sha256"], "unchanged": True}
    manifest["calibration"] = {"scope": "authorized_public_validation", "validation_sha256": validation["sha256"],
        "validation_rows": len(validation["y"]), "selected_metrics": measured,
        "policy_source": "actual_parent_policy_with_pair_review" if fitted_review is not None else "retained_actual_parent_policy",
        "retained_components": retained}
    write_json(candidate / MANIFEST, manifest)
    exported = evaluate(load_predictor(primary_path, validation["paths"]), validation)
    if any(abs(exported[key] - measured[key]) > 1e-12 for key in ("macro_f1", "accuracy")):
        raise ValueError("Retained exported bundle differs from epoch predictor")
    return exported, manifest


def build_bundle(parent_path, parent_predictor, train, validation, candidate, spec, diagnosis, limits, started):
    """Continue each expert from its actual parent, then select and freeze routing."""
    # 主组件由前序训练完成，辅助专家分别从真实父路径续训，共享本候选剩余时间预算。
    # 训练完成后校准并冻结路由，保存全部组件摘要，再重载完整包验证导出预测与选版测量一致。
    from .continuation import state_identity
    def remaining():
        # 每个子阶段都扣除前面已经消耗的墙钟时间，不能给每个专家重置一份完整候选预算。
        value = limits["max_seconds_per_candidate"] - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Routed candidate exceeded CPU wall-time budget")
        return value
    routing = spec["routing"]
    anchor_path = candidate / "checkpoint.npz"
    primary = load_model(anchor_path)
    if primary.meta["hidden"] != 32 or primary.meta.get("hidden_layers", 1) != 1:
        raise ValueError("The routed home expert requires a 128-32-6 parent or an explicit compatible migration")
    parents = parent_predictor.component_paths if isinstance(parent_predictor, FrozenRouter) else {}
    five_parent = parents.get("five_sensor", parent_path)
    home_parent = parents.get("home_trunk", parent_path)
    aux_work = Path(tempfile.mkdtemp(prefix="routing-training-", dir=candidate.parent))
    aux = candidate / "routing"
    aux.mkdir()
    five_before = load_model(five_parent, train["paths"])
    five_report = train_candidate(five_parent, train, validation, aux_work / "five_sensor",
                                  routing["five_sensor"], diagnosis,
                                  dict(limits, max_seconds_per_candidate=remaining()))
    shutil.copyfile(aux_work / "five_sensor/checkpoint.npz", aux / "five_sensor.npz")
    remaining()
    home_before = HomeModel.load(home_parent)
    if isinstance(parent_predictor, FrozenRouter):
        old_router = parent_predictor.models["home_router"]
        if any(not np.array_equal(getattr(home_before, group)[k], getattr(old_router, group)[k])
               for group in ("params", "first", "second") for k in home_before.params):
            raise ValueError("Home branches require an explicit migration when their neural states differ")
    home_report = train_home(home_before, train, routing["home"], aux,
                             max_epochs=limits["max_epochs"],
                             deadline=started + limits["max_seconds_per_candidate"])
    # 成对头阈值使用 train 内折外结果，公开 validation 只决定完整专家融合及类别对。
    cutoffs = binary_cutoffs(home_before, train["x"], train["y"])
    cutoffs = {k: {"cut": value["cut"]} for k, value in cutoffs.items()}
    models = {"four_sensor": load_model(anchor_path), "five_sensor": load_model(aux / "five_sensor.npz"),
              "home_trunk": HomeModel.load(aux / "home_trunk.npz"), "home_router": HomeModel.load(aux / "home_router.npz")}
    for role in ("home_trunk", "home_router"):
        model = models[role]
        if (model.meta["steps"] <= home_before.meta["steps"] or not np.array_equal(model.mean, home_before.mean)
                or not np.array_equal(model.scale, home_before.scale)
                or any(not np.array_equal(getattr(model, group)[k], getattr(home_before, group)[k])
                       for group in ("params", "first", "second") for k in ("w1", "b1"))):
            raise ValueError("Home expert did not preserve and continue the actual parent state")
    controller = limits.get("controller_checkpoint_gate")
    review, review_entry = None, None
    if isinstance(parent_predictor, FrozenRouter):
        review, _, review_entry = _export_review(parent_predictor, candidate)
    policy, calibration = calibrate_policy(models, validation, cutoffs, routing["selection"],
                                          controller=controller,
                                          parent_metrics=evaluate(parent_predictor, validation) if controller is not None else None,
                                          review=review)
    remaining()
    anchor = models["four_sensor"]
    # 先把主检查点标记为完整路由入口，再计算各组件摘要和清单，保证平台从主文件能找回整包。
    anchor.meta["prediction_bundle"] = _bundle_marker(review_entry)
    _replace_checkpoint(anchor, anchor_path)
    files = {"four_sensor": anchor_path, **{k: aux / (k + ".npz") for k in COMPONENTS[1:]}}
    parameter_count = sum(model.meta["parameter_count"] for model in models.values())
    parameter_count += sum(len(head["w"]) + 1 for head in models["home_router"].meta["private_joint_rescue"].values())
    manifest = {"format": FORMAT, "classes": CLASSES, "policy": policy, "calibration": calibration,
                "prediction_parameter_count": parameter_count,
                "components": {k: {"path": p.relative_to(candidate).as_posix(), "sha256": digest(p)} for k,p in files.items()}}
    if review_entry is not None:
        manifest["pair_review"] = review_entry
        manifest["prediction_parameter_count"] += review.parameter_count
    write_json(candidate / MANIFEST, manifest)
    write_json(candidate / "routing_training.json", {
        "gradient_and_memory_scope": "authorized_training_rows_only", "route_selection_scope": "authorized_public_validation",
        "five_sensor": five_report, "five_sensor_initial_state": state_identity(five_before),
        "home": {"parent_checkpoint_sha256": digest(home_parent),
                 "router_parent_checkpoint_sha256": digest(parents.get("home_router", parent_path)),
                  "initial_state": state_identity(home_before), "training": home_report,
                 "trunk_final_state": state_identity(models["home_trunk"]),
                 "router_final_state": state_identity(models["home_router"]), "training_rows": len(train["y"])},
    })
    # 必须按最终磁盘文件重载复评，验证组件复制和清单导出没有改变训练时测得的预测。
    predictor = load_predictor(anchor_path, train["paths"])
    measured = evaluate(predictor, validation)
    if abs(measured["macro_f1"] - calibration["selected_metrics"]["macro_f1"]) > 1e-12:
        raise ValueError("Exported router differs from the selected public-validation policy")
    return measured, manifest
