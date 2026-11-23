"""云端只留运行结果与请求摘要；完整源码、样本、权重和原始协议只保存在本地。"""
import base64
import io
import json
import zipfile
from .cloud_experiment import CloudJournal, json_text, sha


def picked(value, keys):
    return {key: value[key] for key in keys if key in value}


def report_summary(value):
    # 大报告只选取成绩、轮次和检查结果，不递归携带源码快照、文件内容或原始请求。
    result = picked(value, ("dataset", "task", "scope", "real_model_calls", "model_call_attempts",
        "selected_artifact_id", "baseline", "final", "fault_metrics", "restored_metrics", "normal_final",
        "checks", "requirements_passed", "complete_history_experiments", "not_worse_than_public_baseline",
        "provider_reported_total_tokens", "official_host_or_hidden_score_verified", "official_platform_score_verified"))
    result["rounds"] = [picked(row, ("round", "history_ids", "candidate_submitted", "model_call_ids",
        "selected_artifact_id", "metrics", "finish_acknowledged", "actual_historical_rollback", "plan"))
        for row in value.get("rounds", [])]
    return result


class ResultsOnlyJournal(CloudJournal):
    """不调用原始源码/模型归档器，仅打包本类已经净化的JSONL事件。"""
    def __init__(self, sdk, config, output, experiment_config, name):
        # init(config=...) 同样属于云端上传，因此在初始化之前就限制字段。
        safe = picked(experiment_config, ("purpose", "dataset_key", "task", "metric", "model", "image_id",
            "maximum_real_calls", "minimum_rounds", "maximum_rounds", "public_validation_only",
            "hidden_labels_available", "source_normal_cloud_url", "source_normal_evidence_sha256"))
        super().__init__(sdk, config, output, safe, name)
        self.receipt.update(upload_policy="results_only", source_uploaded=False,
                            samples_uploaded=False, weights_uploaded=False)
        self._save()

    def emit(self, scalars=None, texts=None):
        # 仅允许这些自有事件类型，未知键直接拒绝；这不是接收任意对象的通用脱敏器。
        safe = {}
        for key, value in (texts or {}).items():
            if key == "suite/configuration":
                safe[key] = picked(value, ("purpose", "dataset_key", "task", "metric", "model", "image_id",
                    "maximum_real_calls", "minimum_rounds", "maximum_rounds", "public_validation_only",
                    "hidden_labels_available", "source_normal_cloud_url", "source_normal_evidence_sha256"))
            elif key in ("normal/summary", "recovery/summary"):
                safe[key] = report_summary(value)
            elif key.startswith(("calls/", "evidence/", "audit/")) or key.endswith((
                    "/round_record", "/monitoring", "/diagnosis", "/epoch_history")):
                safe[key] = value
            else:
                raise ValueError("Cloud text key is outside the results-only whitelist: " + key)
        return super().emit(scalars, safe)

    def before_call(self, frame, start):
        # 原始 messages 保存在宿主本地 provider-calls；云端只留请求摘要与采样选项。
        self.emit({"calls/attempted": len(self.scalars_for("calls/attempted")) + 1}, {
            "calls/request": {"session": self.phase, "dataset": start["dataset_key"],
                "round_index": start["round_index"], "sdk_request_sha256": sha(json_text(frame).encode()),
                "options": picked(frame, ("max_tokens", "temperature", "enable_thinking"))}})

    def after_call(self, call_id, response, request_messages):
        # 不上传绑定提示或源码，只保留模型返回的决定文本、用量、回执号及原响应摘要。
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        self.emit({"calls/verified": len(self.scalars_for("calls/verified")) + 1}, {
            "calls/response": {"session": self.phase, "call_id": call_id, "model": response.get("model"),
                "usage": response.get("usage", {}), "decision_text": content,
                "original_response_sha256": sha(json_text(response).encode())}})

    def after_round(self, host):
        # 宿主已收到 finish 确认并完成选中产物复测，才记录这一轮“已完成”。
        state = host.saved_state
        decision = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        finish, cfg = host.record, host.record["configuration"]
        complete = bool(self.phase == "normal" and finish["history_ids"] and host.candidate and finish["model_call_ids"])
        # 完整历史实验要求非空历史、实际候选和真实模型回执；观察回合、故障回滚和首轮不能凑数。
        entry = {"session": self.phase, "round_index": host.frame["round_index"],
            "history_ids": finish["history_ids"], "model_call_ids": finish["model_call_ids"],
            "candidate_submitted": host.candidate is not None, "finish_acknowledged": True,
            "complete_history_experiment_decision": complete, "plan": decision["plan"],
            "selection": decision["selection"], "selected_artifact_id": finish["selected_artifact_id"],
            "configuration": picked(cfg, ("dataset", "action", "experiment", "parent_checkpoint_sha256",
                "train_sha256", "validation_sha256", "local_validation_before", "local_validation_after",
                "selected_local_macro_f1", "validation_before", "validation_after", "selected_local_rmse",
                "candidate_gate", "host_candidate_gate", "experiment_failure", "controller_prompt_version",
                "strategy_version_after"))}
        self.rounds.append(entry)
        regression = host.frame["task_contract"]["task"] == "regression"
        metric = "rmse" if regression else "macro_f1"
        values = {f"{self.phase}/selected_{metric}": cfg["selected_local_rmse"] if regression else cfg["selected_local_macro_f1"],
                  f"{self.phase}/round_index": host.frame["round_index"],
                  f"{self.phase}/parent_{metric}": cfg["validation_before"]["rmse"] if regression else cfg["local_validation_before"],
                  f"{self.phase}/complete_history_cycle": int(complete),
                  f"{self.phase}/history_records_read": len(finish["history_ids"])}
        if host.candidate_metrics:
            values[f"{self.phase}/candidate_{metric}"] = host.candidate_metrics[metric]
        for key, value in getattr(host, "selected_metrics", {}).items():
            if key in ("mae", "nasa_score", "accuracy") and isinstance(value, (int, float)):
                values[f"{self.phase}/selected_{key}"] = value
        monitor = json.loads((state / "monitoring.json").read_text(encoding="utf-8"))
        values[f"{self.phase}/drift_detected"] = int(monitor["drift"])
        diagnostic = json.loads((state / "diagnosis.json").read_text(encoding="utf-8"))
        safe_monitor = picked(monitor, ("scope", "dataset", "rows", "drift", "changed_features", "thresholds", "top_features", "sensors",
            "reference_policy", "reference_train_sha256", "current_train_sha256", "training_projection_changed"))
        # 只上传统计及诊断聚合，不上传逐样本特征和 answer；完整诊断文件仍保存在本地状态。
        safe_diagnostic = picked(diagnostic, ("scope", "weak_classes", "top_confusions", "drift_sensors", "metrics",
            "rul_segments", "training_target_cuts", "drift_features", "same_scope_previous_macro_f1_delta", "hypothesis", "interpretation", "hidden_feedback_available"))
        self.emit(values, {f"{self.phase}/round_record": entry, f"{self.phase}/monitoring": safe_monitor,
                           f"{self.phase}/diagnosis": safe_diagnostic})
        if host.candidate and (host.candidate / "history.jsonl").is_file():
            epochs = [json.loads(line) for line in (host.candidate / "history.jsonl").read_text(encoding="utf-8").splitlines()]
            self.emit(texts={f"{self.phase}/epoch_history": {"round_index": host.frame["round_index"],
                "epochs": [picked(row, ("epoch", "step", "loss", "validation", "gate")) for row in epochs]}})
        self.receipt.update(completed_rounds=len(self.rounds), complete_history_cycles=sum(
            r["complete_history_experiment_decision"] for r in self.rounds))
        self._save()

    def archive(self):
        # 固定唯一输入为经过本类净化的事件文件；不遍历项目、不读源码/数据/模型目录。
        data = (self.output / "cloud-events.jsonl").read_bytes()
        self._check(data)
        # 以固定事件文件生成单文件 ZIP，再做分块和摘要；父类的完整源码/模型归档器不会被调用。
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("results-only-events.jsonl", data)
        bundle = buffer.getvalue()
        (self.output / "evidence-bundle.zip").write_bytes(bundle)
        size = 384 * 1024
        count = (len(bundle) + size - 1) // size
        self.bundle = {"format": "base64-zip-chunks-v1", "sha256": sha(bundle), "bytes": len(bundle),
            "chunks": count, "files": 1, "results_only": True, "image_archive_included": False}
        self.emit(texts={"evidence/bundle_manifest": self.bundle})
        # verify_cloud 会回读每条 Text、重组这些块并核对 ZIP 摘要，不能只凭上传返回成功就算留档完成。
        for index in range(count):
            block = bundle[index * size:(index + 1) * size]
            self.emit(texts={"evidence/bundle_chunk": {"index": index, "count": count,
                "bundle_sha256": self.bundle["sha256"], "chunk_sha256": sha(block),
                "data_base64": base64.b64encode(block).decode()}})
        self.receipt["evidence_bundle"] = self.bundle
        self._save()
