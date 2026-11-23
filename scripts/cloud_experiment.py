"""SwanLab audit journal for real, host-driven experiments.

Public-cloud Text media store complete JSON records and a reversible evidence
archive; swanlab.save() is not used (it is private-deployment-only).
"""
# 宿主端云端留档：记录真实请求回复、每轮证据、指标和完整归档，然后只读回查内容与终态。
# 一个 CloudJournal 对应一个 SwanLab Run，可包含多个 SDK 回合；完整历史实验次数由每轮实际记录计算。
# API 密钥只在宿主使用；公有云归档使用 Text 分块，不使用仅适用于私有部署的 swanlab.save。
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request
import uuid
import zipfile

from scripts.live_platforms import SWANLAB_HOST, write_receipt, verify_finish


def json_text(value):
    # 固定键序并禁止 NaN，保证本地事件和云端 Text 的字节摘要可直接比较。
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def media_items(payload):
    # 展开 SwanLab 媒体的键、记录步数和文件条目；媒体步数仍使用 index 字段。
    for series in payload.get("list", []):
        for point in series.get("metrics", []):
            for item in point.get("items", []):
                yield series["key"], int(point["index"]), item


class CloudJournal:
    def __init__(self, sdk, config, output, experiment_config, name):
        # 先生成本次独有的 Run ID 并持久化回执，再做任何云端初始化，便于中断后追踪。
        self.sdk, self.config, self.output = sdk, config, Path(output)
        self.experiment_config, self.name = experiment_config, name
        self.run, self.api = None, None
        self.run_id = uuid.uuid4().hex
        self.step, self.texts, self.scalars, self.rounds = 0, {}, {}, []
        self.errors = []
        self.phase = "normal"
        self.bundle = None
        self.secrets = [os.environ[k].encode() for k in ("PYTRIO_API_KEY", "SWANLAB_API_KEY") if os.environ.get(k)]
        self.receipt = {"id": self.run_id, "purpose": "real_model_full_experiment_audit",
                        "name": name, "phase": "created", "cloud_finish_verified": False,
                        "run_path": None, "url": None, "model_calls_retried": False}
        self._save()

    def _save(self):
        write_receipt(self.output / "cloud-receipt.json", self.receipt)

    @classmethod
    def resume_verification(cls, sdk, output):
        """Rebuild only the read-back state; never init, resume, or finish a run."""
        # 仅从既有回执和事件日志重建核验状态，不调用 init/resume/finish，不重新训练或计费。
        obj = cls.__new__(cls)
        obj.sdk, obj.output = sdk, Path(output)
        obj.receipt = json.loads((obj.output / "cloud-receipt.json").read_text(encoding="utf-8"))
        obj.run_id, obj.run = obj.receipt["id"], None
        obj.api = sdk.Api(api_key=os.environ["SWANLAB_API_KEY"], host=SWANLAB_HOST)
        obj.texts, obj.scalars, obj.rounds = {}, {}, []
        obj.bundle = obj.receipt["evidence_bundle"]
        obj.step = 0
        for line in (obj.output / "cloud-events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            obj.step = max(obj.step, event["step"])
            for key, body in event["texts"].items():
                obj.texts[(key, event["step"])] = sha(body.encode())
                if key.endswith("/round_record"):
                    obj.rounds.append(json.loads(body))
            for key, value in event["scalars"].items():
                obj.scalars[(key, event["step"])] = value
        return obj

    def _check(self, data):
        # 上传前查找当前宿主密钥原始字节，阻止把授权头或误拼接的凭证写入云端证据。
        if any(secret in data for secret in self.secrets):
            raise ValueError("Refusing to upload credential-containing evidence")

    def begin(self):
        # 拒绝覆盖已有无关 Run，关闭终端/环境/Git 探针并创建本次私有实验。
        # 只在这里初始化一次，因此正常和回滚各轮都属于同一个 SwanLab Run。
        if self.sdk.has_run():
            raise RuntimeError("An unrelated SwanLab run is already active")
        self.sdk.login(api_key=os.environ["SWANLAB_API_KEY"], host=SWANLAB_HOST, save=False)
        self.api = self.sdk.Api(api_key=os.environ["SWANLAB_API_KEY"], host=SWANLAB_HOST)
        owner = self.config["workspace"] or self.api.username
        if not owner:
            raise RuntimeError("Cannot determine SwanLab workspace")
        self.receipt.update(run_path=f"{owner}/{self.config['project']}/{self.run_id}",
                            url=f"https://swanlab.cn/@{owner}/{self.config['project']}/runs/{self.run_id}",
                            phase="initializing", init_attempted=True)
        self._save()
        settings = self.sdk.Settings(interactive=False, api_host=SWANLAB_HOST, web_host="https://swanlab.cn",
            probe={name: False for name in ("hardware", "runtime", "requirements", "conda", "git", "swanlab", "monitor")},
            terminal={"proxy_type": "none"})
        # 一次总测试仅在这里创建一个云端 Run；每轮通过 log 的不同 step 和 round_record 区分。
        self.run = self.sdk.init(project=self.config["project"], workspace=owner, id=self.run_id,
                                resume="allow", name=self.name, public=False, config=self.experiment_config,
                                mode="online", on_online_error="raise", settings=settings,
                                log_dir=str(self.output / "swanlog"))
        if self.run.id != self.run_id:
            raise RuntimeError("Unexpected SwanLab run identity")
        self.receipt["phase"] = "running"
        self._save()
        self.emit(texts={"suite/configuration": self.experiment_config})

    def emit(self, scalars=None, texts=None):
        # 先写本地 JSONL，再按同一步数上传标量和完整文本，并保存每条文本摘要。
        # 网络日志失败保留本地证据并记错，不丢弃已经返回的真实模型响应；最终回读仍会暴露缺失。
        self.step += 1
        values = dict(scalars or {})
        payloads = {}
        for key, value in (texts or {}).items():
            body = value if isinstance(value, str) else json_text(value)
            self._check(body.encode())
            self.texts[(key, self.step)] = sha(body.encode())
            payloads[key] = body
            values[key] = self.sdk.Text(body)
        for key, value in (scalars or {}).items():
            self.scalars[(key, self.step)] = value
        item = {"step": self.step, "scalars": scalars or {}, "texts": payloads}
        with (self.output / "cloud-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json_text(item) + "\n")
        try:
            self.run.log(values, step=self.step)
        except Exception as error:
            # An audit failure must not discard a real response or interrupt an
            # in-flight contestant RPC. The complete event remains on disk.
            self.errors.append({"step": self.step, "error_type": type(error).__name__})
            self.receipt["log_errors"] = self.errors
            self._save()

    def before_call(self, frame, start):
        # 记录选手通过 SDK 真正发出的模型请求；调用尝试数与已核验响应数分开统计。
        self.emit(scalars={"calls/attempted": len(self.scalars_for("calls/attempted")) + 1},
                  texts={"calls/request": {"session": self.phase, "round_index": start["round_index"],
                         "sdk_model_request": frame}})

    def scalars_for(self, key):
        return {step: value for (name, step), value in self.scalars.items() if name == key}

    def after_call(self, call_id, response, request_messages):
        # 记录实际提供方响应及绑定后的消息，不替换、补写或改造模型选择。
        self.emit(scalars={"calls/verified": len(self.scalars_for("calls/verified")) + 1},
                  texts={"calls/response": {"session": self.phase, "call_id": call_id,
                         "response": response, "bound_messages": request_messages}})

    def after_round(self, host):
        # 调用方已等待镜像 finish 确认并退出成功，才把本轮状态和训练报告计入完成记录。
        # 完整历史实验要求正常阶段、非空历史、真实提交候选及实际模型调用，首轮与观察回合不凑数。
        state = host.saved_state
        record = json.loads((state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        finish = host.record
        # 三次完整记录按真实历史引用+候选提交+模型回执计算，不是三次模型请求或三条页面列表。
        complete = bool(self.phase == "normal" and finish["history_ids"] and host.candidate
                        and finish["model_call_ids"])
        entry = {"session": self.phase, "round_index": host.frame["round_index"],
                 "history_ids": finish["history_ids"], "candidate_submitted": host.candidate is not None,
                 "complete_history_experiment_decision": complete, "finish_acknowledged": True,
                 "record": record, "sdk_finish_record": finish}
        self.rounds.append(entry)
        phase, cfg = self.phase, finish["configuration"]
        regression = host.frame["task_contract"]["task"] == "regression"
        metric = "rmse" if regression else "macro_f1"
        values = {f"{phase}/round_index": host.frame["round_index"],
                  f"{phase}/selected_{metric}": cfg["selected_local_rmse"] if regression else cfg["selected_local_macro_f1"],
                  f"{phase}/parent_{metric}": cfg["validation_before"]["rmse"] if regression else cfg["local_validation_before"],
                  f"{phase}/complete_history_cycle": int(complete),
                  f"{phase}/history_records_read": len(finish["history_ids"]),
                  f"{phase}/strategy_version": cfg["strategy_version_after"]}
        if host.candidate_metrics:
            values[f"{phase}/candidate_{metric}"] = host.candidate_metrics[metric]
        monitoring = json.loads((state / "monitoring.json").read_text(encoding="utf-8"))
        values[f"{phase}/drift_detected"] = int(monitoring["drift"])
        # 新宿主已通过实际predict进程复测选中产物，可直接留档其MAE、NASA等辅助值。
        for key, value in getattr(host, "selected_metrics", {}).items():
            if key in ("mae", "nasa_score", "accuracy") and isinstance(value, (int, float)):
                values[f"{phase}/selected_{key}"] = value
        diagnosis = json.loads((state / "diagnosis.json").read_text(encoding="utf-8"))
        self.emit(values, {f"{phase}/round_record": entry, f"{phase}/monitoring": monitoring,
                           f"{phase}/diagnosis": diagnosis})
        if host.candidate:
            for name in ("training.json", "routing_training.json", "continuation.json"):
                path = host.candidate / name
                if path.is_file():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self.emit(texts={f"{phase}/{name.replace('.', '_')}": {
                        "round_index": host.frame["round_index"], "report": data}})
                    if name == "training.json":
                        for epoch in data.get("epoch_history", []):
                            if "validation" in epoch:
                                self.emit({f"{phase}/trial_{host.frame['round_index']}_main_epoch": epoch["epoch"],
                                           f"{phase}/trial_{host.frame['round_index']}_main_validation_f1": epoch["validation"]["macro_f1"]})
            if regression and (host.candidate / "history.jsonl").is_file():
                epochs = [json.loads(line) for line in (host.candidate / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                self.emit(texts={f"{phase}/epoch_history": {"round_index": host.frame["round_index"], "epochs": epochs}})
        self.receipt.update(completed_rounds=len(self.rounds),
                            complete_history_cycles=sum(r["complete_history_experiment_decision"] for r in self.rounds))
        self._save()

    def archive(self):
        # 只归档指定运行目录、复现源码和公开基线普通文件，排除云端凭证/日志探针与镜像 tar。
        # ZIP 按 384 KiB 原始块转为 base64 Text，记录整包和分块摘要，之后可以完全逆向恢复。
        path = self.output / "evidence-bundle.zip"
        included = []
        roots = ("normal", "recovery", "reproduction", "official-baseline")
        for prefix in roots:
            for item in (self.output / prefix).rglob("*"):
                if item.is_symlink():
                    raise ValueError("Evidence must contain ordinary files")
                if item.is_file() and "__pycache__" not in item.parts:
                    included.append(item)
        included.extend(p for p in self.output.iterdir() if p.is_file() and p.suffix in (".json", ".py", ".md")
                        and p.name != "cloud-receipt.json")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for item in sorted(set(included)):
                data = item.read_bytes()
                self._check(data)
                zipped.writestr(item.relative_to(self.output).as_posix(), data)
        data = path.read_bytes()
        # base64 会扩大体积，因此限制原始块大小；单个 Text 是可还原文件块，不是训练指标。
        chunk_size = 384 * 1024
        count = (len(data) + chunk_size - 1) // chunk_size
        self.bundle = {"format": "base64-zip-chunks-v1", "sha256": sha(data), "bytes": len(data),
                       "chunks": count, "files": len(set(included)), "image_archive_included": False}
        self.emit(texts={"evidence/bundle_manifest": self.bundle})
        for index in range(count):
            block = data[index * chunk_size:(index + 1) * chunk_size]
            self.emit(texts={"evidence/bundle_chunk": {"index": index, "count": count,
                "bundle_sha256": self.bundle["sha256"], "chunk_sha256": sha(block),
                "data_base64": base64.b64encode(block).decode("ascii")}})
        self.receipt["evidence_bundle"] = self.bundle
        self._save()

    def finish(self, state):
        # 只结束本次 ID 的 Run；即使初始化中断，也不能误关其他活动实验。
        # finish 返回后另用 API 读取终态，未确认的结果标为 needs_attention，而不是默认完成。
        self.receipt.update(phase="finishing", desired_finish_state=state)
        self._save()
        owned = self.run
        # 初始化可能创建了 SDK 活动对象但还没返回给 self.run；只在 ID 一致时接管收尾。
        if owned is None and self.sdk.has_run():
            candidate = self.sdk.get_run()
            owned = candidate if candidate.id == self.run_id else None
        try:
            if owned is not None and owned.id == self.run_id:
                self.receipt["finish_attempted"] = True
                self._save()
                owned.finish(state=state, error="Experiment suite did not meet all requirements" if state == "crashed" else None,
                             async_log_timeout=60)
                self.receipt["finish_returned"] = True
        except BaseException as error:
            self.receipt["finish_error_type"] = type(error).__name__
        if self.api and self.receipt.get("run_path"):
            self.receipt["cloud_state"] = verify_finish(self.api, self.receipt["run_path"], state)
            self.receipt["cloud_finish_verified"] = self.receipt["cloud_state"]["status"] == "verified"
        self.receipt["phase"] = "finished" if self.receipt["cloud_finish_verified"] else "needs_attention"
        self._save()

    def verify_cloud(self):
        # 等媒体索引完整后逐条下载校验文本，再重组完整 ZIP，并核对关键标量的步数和值。
        # 索引延迟只触发只读重查；它不能重新调用模型、重发训练或伪造缺失数据。
        if not self.receipt["cloud_finish_verified"]:
            raise RuntimeError("Cloud terminal state is not verified")
        media_keys = {key for key, _ in self.texts}
        # 云端已结束不代表媒体索引立即完整；只读等待索引，不重新提交日志、模型调用或 finish。
        for attempt in range(6):
            remote = self.api.run(self.receipt["run_path"])
            payload = remote.medias(keys=sorted(media_keys), all=True)
            available = {(key, step) for key, step, _ in media_items(payload)}
            if set(self.texts) <= available:
                break
            if attempt == 5:
                raise RuntimeError("Cloud media index has not exposed all experiment records yet")
            time.sleep(min(2 * (attempt + 1), 8))
        received, chunks = {}, {}
        for key, step, item in media_items(payload):
            if (key, step) not in self.texts:
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError("Cloud media URL is unavailable")
            with urllib.request.urlopen(url, timeout=45) as response:
                body = response.read()
            if sha(body) != self.texts[(key, step)]:
                raise ValueError("Cloud text differs from local experiment evidence")
            received[(key, step)] = sha(body)
            if key == "evidence/bundle_chunk":
                value = json.loads(body)
                block = base64.b64decode(value["data_base64"], validate=True)
                assert sha(block) == value["chunk_sha256"]
                chunks[value["index"]] = block
        if set(received) != set(self.texts):
            raise RuntimeError("Some cloud experiment records are missing")
        restored = b"".join(chunks[i] for i in range(self.bundle["chunks"]))
        assert sha(restored) == self.bundle["sha256"]
        restore_path = self.output / "cloud-downloaded-evidence.zip"
        restore_path.write_bytes(restored)
        # 完整 Text 已逐条核对；关键标量另比对步骤和值，避免页面曲线来自错误轮次或分数。
        # 从实际事件选择指标：A用F1，B/C用RMSE，兼容只有正常阶段以及只读恢复核验。
        scalar_keys = sorted({key for key, _ in self.scalars if key == "normal/complete_history_cycle"
                              or key.startswith(("normal/selected_", "recovery/selected_"))})
        actual = remote.metrics(keys=scalar_keys, all=True)
        verified_values = 0
        for series in actual.get("list", []):
            key = series["key"]
            expected = self.scalars_for(key)
            # all=True uses the full CSV schema (step/value); sampled API
            # responses use index/data. Both are documented SwanLab formats.
            actual_points = {int(p["step"] if "step" in p else p["index"]):
                             (p["value"] if "value" in p else p["data"])
                             for p in series.get("metrics", [])}
            if set(actual_points) != set(expected):
                raise ValueError("Cloud scalar steps differ from actual experiment steps")
            for step, value in expected.items():
                if not math.isclose(actual_points[step], value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Cloud score differs from platform-simulation measurement")
                verified_values += 1
        if verified_values != sum(len(self.scalars_for(k)) for k in scalar_keys):
            raise ValueError("Required cloud metric columns are missing")
        result = {"verified": True, "run_path": self.receipt["run_path"], "url": remote.url,
                  "state": remote.state, "text_records_downloaded_and_hash_checked": len(received),
                  "score_points_checked": verified_values, "evidence_zip_sha256": sha(restored),
                  "complete_history_cycles": sum(r["complete_history_experiment_decision"] for r in self.rounds),
                  "evidence_restored_from_cloud": True}
        write_receipt(self.output / "cloud-verification.json", result)
        self.receipt.update(content_verified=True, url=remote.url)
        self._save()
        return result
