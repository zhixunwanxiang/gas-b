"""轻量阶段日志，仅写 stderr，不向 stdout 插入伪造的协议或心跳帧。"""
from contextlib import contextmanager
import json
import sys
import time
import traceback


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
        progress("stage_ended", stage=name, completed=completed,
                 seconds=round(time.monotonic() - started, 6), **fields)
