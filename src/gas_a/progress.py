"""轻量阶段日志，仅写 stderr，不向 stdout 插入伪造的协议或心跳帧。"""
from contextlib import contextmanager
from contextvars import ContextVar
import atexit
import json
import sys
import time
import traceback

_resources = ContextVar("round_resources", default=None)
_predictions = {"samples": 0, "wall_seconds": 0.0, "cpu_seconds": 0.0}


def _peak_memory():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return None


class ResourceLedger:
    def __init__(self, **fields):
        self.fields = fields
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.stages, self.calls, self.transfers = [], [], []

    def snapshot(self, cutoff):
        known = [call["usage"] for call in self.calls if call["usage"] is not None]
        complete = len(known) == len(self.calls)
        totals = {key: sum(usage[key] for usage in known) for key in ("input_tokens", "output_tokens")}
        return {"format": "gas-rsi.resources.v1", **self.fields, "cutoff": cutoff,
                "scope": "participant_process_only; RPC wall time includes waiting; stages may overlap",
                "wall_seconds": time.monotonic() - self.started,
                "cpu_seconds": time.process_time() - self.cpu_started,
                "peak_rss_bytes": _peak_memory(), "stages": list(self.stages),
                "model_calls": list(self.calls), "usage_complete": complete,
                "known_token_subtotal": totals,
                "token_totals": totals if complete else None,
                "missing_usage_calls": len(self.calls) - len(known)}


def resource_snapshot():
    ledger = _resources.get()
    if ledger is None:
        return {}
    result = ledger.snapshot("before_save_state")
    result["transfers"] = list(ledger.transfers)
    result["known_upload_payload_bytes"] = sum(t["payload_bytes"] for t in ledger.transfers)
    result["upload_byte_scope"] = "file payloads; excludes protocol framing; acknowledgements recorded separately"
    return result


def record_transfer(kind, directory):
    item = {"kind": kind, "payload_bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            "acknowledged": False}
    ledger = _resources.get()
    if ledger is not None:
        ledger.transfers.append(item)
    return item


@contextmanager
def resource_scope(**fields):
    ledger = ResourceLedger(**fields)
    token = _resources.set(ledger)
    try:
        yield ledger
    finally:
        summary = ledger.snapshot("callback_exit")
        # stdout 是协议，stderr 也可能经小管道读取；逐阶段明细已在状态中，退出只输出紧凑总计。
        progress("round_resources", **{key: summary[key] for key in (
            "cutoff", "wall_seconds", "cpu_seconds", "peak_rss_bytes", "usage_complete", "token_totals")},
            model_call_count=len(ledger.calls), **fields)
        _resources.reset(token)


def record_model_receipt(receipt):
    ledger = _resources.get()
    if ledger is None:
        return
    raw = receipt.response.get("usage") if isinstance(receipt.response, dict) else None
    usage = None
    if isinstance(raw, dict):
        values = (raw.get("prompt_tokens", raw.get("input_tokens")),
                  raw.get("completion_tokens", raw.get("output_tokens")))
        if all(type(value) is int and value >= 0 for value in values):
            usage = dict(zip(("input_tokens", "output_tokens"), values))
    ledger.calls.append({"call_id": receipt.call_id, "usage": usage})


def save_resources(directory):
    ledger = _resources.get()
    if ledger is not None:
        from .sensor_training.io import write_json
        # 本次 save_state/finish 尚未发生，完整耗时在 callback_exit 的 stderr 摘要中。
        write_json(directory / "resources.json", resource_snapshot())


@contextmanager
def prediction_resource():
    wall, cpu = time.monotonic(), time.process_time()
    try:
        yield
    finally:
        _predictions["samples"] += 1
        _predictions["wall_seconds"] += time.monotonic() - wall
        _predictions["cpu_seconds"] += time.process_time() - cpu


def _prediction_summary():
    if _predictions["samples"]:
        progress("prediction_resources", **_predictions, peak_rss_bytes=_peak_memory(),
                 scope="predict_callback_only; excludes SDK JSON and process startup")


atexit.register(_prediction_summary)


def progress(event, **fields):
    # 调用方只传阶段名、数据集、轮次等摘要，禁止写入样本、密钥和完整模型请求。
    print("[gas-rsi] " + json.dumps({"event": event, **fields}, ensure_ascii=False),
          file=sys.stderr, flush=True)


def training_failure(error, **fields):
    frames = traceback.extract_tb(error.__traceback__)
    origin = frames[-1] if frames else None
    progress("train_failed", **fields, error_type=type(error).__name__, message=str(error)[:1000],
             source=origin.filename if origin else None, line=origin.lineno if origin else None)


@contextmanager
def stage(name, **fields):
    started = time.monotonic()
    cpu_started = time.process_time()
    progress("stage_started", stage=name, **fields)
    completed = False
    try:
        yield
        completed = True
    except Exception as error:
        progress("stage_failed", stage=name, **fields,
                 error_type=type(error).__name__, message=str(error)[:1000])
        raise
    finally:
        # 仅记录耗时，不捕获或重试 RPC 异常；SDK 结果未知时仍按原逻辑终止。
        record = {"stage": name, "completed": completed, "seconds": round(time.monotonic() - started, 6),
                  "cpu_seconds": round(time.process_time() - cpu_started, 6), **fields}
        ledger = _resources.get()
        if ledger is not None:
            ledger.stages.append(record)
        progress("stage_ended", **record)
