"""Check PyTRiO's exact configured model and record one real call in SwanLab.

This is a connectivity check, independent of the competition host and training.
Use --check-config to inspect credential presence without making network calls.
"""
# 宿主连通性和云端生命周期工具：限制端点/模型，安全加载本地凭证，核验自建实验的结束状态。
# 连通性通过只证明服务可访问，不证明工业模型训练完成、正式参赛资格或隐藏集得分。
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
ENV_NAMES = frozenset(("PYTRIO_API_KEY", "PYTRIO_BASE_URL", "PYTRIO_MODEL",
                       "SWANLAB_API_KEY", "GAS_A_SWANLAB_PROJECT", "GAS_A_SWANLAB_WORKSPACE"))
SWANLAB_HOST = "https://api.swanlab.cn"
TERMINAL_STATES = {"success": "FINISHED", "crashed": "CRASHED", "aborted": "ABORTED"}


class PlatformCheckError(RuntimeError):
    """Safe CLI message constructed locally, without third-party error bodies."""


def write_receipt(path, receipt):
    """Replace the receipt atomically, preserving the last copy on write failure."""
    # 同目录临时文件写入、flush/fsync 后原子替换，写失败时保留上一份有效回执。
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".receipt-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def termination_handler():
    """Let graceful termination reach finally; forced process kills cannot do so."""
    # 把可处理终止转为 Python 退出异常，使 finally 能执行；强制杀进程或断电不能靠此保证收尾。
    previous = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    try:
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGTERM", "SIGBREAK"):
                signum = getattr(signal, name, None)
                if signum is not None:
                    previous[signum] = signal.getsignal(signum)
                    signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def check_run(api, path):
    """Read one run, never resume, finish or modify it."""
    # 校验 workspace/project/run_id，并每次新建 API 实体读取状态，避免 SDK 元数据缓存。
    parts = path.strip("/").split("/")
    if (len(parts) != 3 or any(not part or part in (".", "..")
                              or any(c in part for c in "\\?#%") for part in parts)):
        raise ValueError("Run path must be workspace/project/run_id")
    path = "/".join(parts)
    remote = api.run(path)  # New object on every read: SDK entities cache metadata.
    state = remote.state
    if state not in ("RUNNING", "OFFLINE", *TERMINAL_STATES.values()):
        raise RuntimeError("Cloud run state is unavailable or unrecognized")
    return {"run_path": path, "state": state,
            "terminal": state in TERMINAL_STATES.values()}


def verify_finish(api, path, desired_state):
    """Bounded, read-only polling; never retry a model call or a finish request."""
    # 仅有限次只读查询预期终态，不重发 finish、不重试模型，也不改变无关实验。
    verification = {"status": "unverified", "expected_state": TERMINAL_STATES[desired_state]}
    for attempt in range(3):
        try:
            observed = check_run(api, path)
            verification.update(observed)
            verification.pop("error_type", None)
            verification["status"] = ("verified" if observed["state"] == verification["expected_state"]
                                      else "state_mismatch")
            if observed["terminal"]:
                break
        except Exception as error:
            # API error bodies can include credentials. Persist the type only.
            verification.update(status="unverified", error_type=type(error).__name__)
        if attempt < 2:
            time.sleep(.5)
    return verification


def run_check(swanlab, config, output):
    # 这条路径只测连通性，使用独有 Run ID 记录一次真实调用并确保收尾；不表示工业模型训练验收。
    run_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc)
    receipt_path = output / "receipt.json"
    receipt = {"format": "gas-a.platform-connectivity.v2", "purpose": "connectivity_only",
               "requested_model": config["model"], "run_id": run_id,
               "created_at_utc": created.isoformat(),
               "run_name": "qwen27b-connectivity-" + created.strftime("%Y%m%dT%H%M%SZ") + "-" + run_id[:8],
               "project": config["project"], "workspace": config["workspace"],
               "phase": "preflight", "outcome": "pending", "model_call_status": "not_started",
               "model_call_attempted": False, "real_model_call": False,
               "swanlab_url": None, "swanlab_finish_attempted": False,
               "swanlab_finish_returned": False, "swanlab_finish_completed": False,
               "cloud_verification": {"status": "not_started"}}
    # Persist identity before any network operation, even if init never returns.
    write_receipt(receipt_path, receipt)
    run, api, failure, init_attempted, model_started = None, None, None, False, False
    finish_state = "crashed"
    with termination_handler():
        try:
            # Never end an unrelated run in an interactive Python process.
            if swanlab.has_run():
                raise RuntimeError("An unrelated SwanLab run is already active")
            ensure_model(config, os.environ["PYTRIO_API_KEY"])
            receipt["phase"] = "cloud_authentication"
            write_receipt(receipt_path, receipt)
            swanlab.login(api_key=os.environ["SWANLAB_API_KEY"], host=SWANLAB_HOST, save=False)
            api = swanlab.Api(api_key=os.environ["SWANLAB_API_KEY"], host=SWANLAB_HOST)
            owner = config["workspace"] or api.username
            if not owner:
                raise RuntimeError("Cannot determine SwanLab workspace")
            path = f"{owner}/{config['project']}/{run_id}"
            receipt.update(run_path=path,
                           swanlab_url=f"https://swanlab.cn/@{owner}/{config['project']}/runs/{run_id}",
                           phase="initializing")
            write_receipt(receipt_path, receipt)
            settings = swanlab.Settings(
                interactive=False, api_host=SWANLAB_HOST, web_host="https://swanlab.cn",
                probe={name: False for name in ("hardware", "runtime", "requirements", "conda", "git", "swanlab", "monitor")},
                terminal={"proxy_type": "none"},
            )
            init_attempted = True
            receipt["swanlab_init_attempted"] = True
            write_receipt(receipt_path, receipt)
            run = swanlab.init(project=config["project"], workspace=owner,
                               id=run_id, resume="allow", name=receipt["run_name"],
                               public=False, settings=settings,
                               config={**config, "purpose": "connectivity_only", "max_tokens": 256},
                               mode="online", on_online_error="raise", log_dir=str(output / "swanlog"))
            if run.id != run_id:
                raise RuntimeError("SwanLab returned an unexpected run identity")
            receipt.update(phase="model_call", model_call_attempted=True, model_call_status="outcome_unknown")
            write_receipt(receipt_path, receipt)
            model_started = True
            result = call_model(config, os.environ["PYTRIO_API_KEY"])
            receipt.update(real_model_call=True, model_call_status="verified_response", pytrio=result,
                           phase="logging")
            # Preserve the real model response before logging or cleanup can fail.
            write_receipt(receipt_path, receipt)
            run.log({"call/success": 1, "call/latency_seconds": result["latency_seconds"],
                     **{"usage/" + key: value for key, value in result["usage"].items()
                        if isinstance(value, (int, float))}})
            finish_state = "success"
        except BaseException as error:
            failure = error
            if not model_started:
                receipt.update(model_call_attempted=False, model_call_status="not_started")
            finish_state = "aborted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "crashed"
            receipt["failure"] = {"phase": receipt["phase"], "error_type": type(error).__name__}
        finally:
            receipt.update(phase="finishing", desired_finish_state=finish_state)
            # An init failure may leave the SDK's registered Run accessible.
            # Fetch it only for this newly generated identity.
            if init_attempted:
                try:
                    owned = run if run is not None else (swanlab.get_run() if swanlab.has_run() else None)
                    if owned is not None and owned.id == run_id:
                        receipt["swanlab_finish_attempted"] = True
                        try:
                            write_receipt(receipt_path, receipt)
                        except Exception as error:
                            receipt["receipt_write_error_type"] = type(error).__name__
                        owned.finish(state=finish_state,
                                     error="Connectivity check failed; see local receipt." if finish_state == "crashed" else None)
                        receipt["swanlab_finish_returned"] = True
                    else:
                        receipt["cleanup_error_type"] = "OwnedRunUnavailable"
                except BaseException as error:
                    receipt["cleanup_error_type"] = type(error).__name__
                    if failure is None and isinstance(error, (KeyboardInterrupt, SystemExit)):
                        failure = error
                # finish() can log an upload failure and still return normally.
                try:
                    receipt["cloud_verification"] = verify_finish(api, receipt["run_path"], finish_state)
                except BaseException as error:
                    receipt["cloud_verification"] = {"status": "unverified", "error_type": type(error).__name__}
                    if failure is None and isinstance(error, (KeyboardInterrupt, SystemExit)):
                        failure = error
                receipt["swanlab_finish_completed"] = receipt["cloud_verification"]["status"] == "verified"
            receipt["phase"] = ("finished" if receipt["swanlab_finish_completed"] else
                                "needs_attention" if init_attempted else "failed")
            receipt["outcome"] = ("aborted" if isinstance(failure, (KeyboardInterrupt, SystemExit)) else
                                  "failed" if failure is not None else
                                  "succeeded" if receipt["swanlab_finish_completed"] else "needs_attention")
            # This final disk write must not suppress the original interruption.
            try:
                write_receipt(receipt_path, receipt)
            except Exception:
                if failure is None:
                    raise PlatformCheckError(f"Cannot save final receipt at {receipt_path}; cloud cleanup was attempted") from None
                print("Cannot update final receipt; cloud cleanup was attempted.", file=sys.stderr)
    if failure is not None:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise failure
        raise PlatformCheckError(f"Platform check failed ({type(failure).__name__}); see {receipt_path}") from None
    if not receipt["swanlab_finish_completed"]:
        raise PlatformCheckError(f"Cloud completion not confirmed; do not repeat the model call. Inspect {receipt_path}")
    return receipt


def load_env(path):
    """Read simple KEY=value settings without executing or expanding their values."""
    # 只按白名单读取 .env 的字面值，不执行 shell 展开，也不把凭证值打印到日志。
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in ENV_NAMES:
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("Unclosed quoted value for " + name)
            value = value[1:-1]
        os.environ.setdefault(name, value)


def configuration():
    # 组装宿主允许的端点、模型和云端项目标识，具体凭证仍留在宿主环境。
    return {"base_url": os.environ.get("PYTRIO_BASE_URL", "https://pytrio.com").rstrip("/"),
            "model": os.environ.get("PYTRIO_MODEL", "Qwen/Qwen3.8-27B"),
            "project": os.environ.get("GAS_A_SWANLAB_PROJECT", "gas-a"),
            "workspace": os.environ.get("GAS_A_SWANLAB_WORKSPACE") or None}


def request_json(url, key, body=None):
    # 统一 HTTP JSON 请求，不自动重试；调用方保存结果状态，避免不明计费结果被悄悄重发。
    headers = {"Authorization": "Bearer " + key, "X-API-Key": key,
               "Content-Type": "application/json"}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)
    # Do not forward credentials through HTTP redirects to another endpoint.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Server error bodies may contain request headers; do not echo them.
        raise RuntimeError(f"PyTRiO HTTP {error.code}; check credentials, model access and quota") from None
    except urllib.error.URLError:
        raise RuntimeError("Cannot connect to PyTRiO; no automatic retry was performed") from None


def ensure_model(config, key):
    # 发送付费请求前确认允许端点和精确模型可用，不悄悄换用其他模型。
    for offset in range(0, 10000, 100):
        payload = request_json(config["base_url"] + f"/api/models?limit=100&offset={offset}", key)
        items = payload.get("items")
        if not isinstance(items, list):
            raise RuntimeError("PyTRiO returned an unrecognized model-list response")
        if any(isinstance(item, dict) and item.get("model_id") == config["model"] for item in items):
            return
        if len(items) < 100:
            break
    raise RuntimeError("Configured model is not available to this account: " + config["model"])


def call_model(config, key):
    started = time.monotonic()
    payload = request_json(config["base_url"] + "/api/openai/v1/chat/completions", key, {
        "model": config["model"],
        "messages": [{"role": "user", "content": "Connectivity check. Reply with PYTRIO_QWEN27B_OK only. /no_think"}],
        "temperature": 0, "max_tokens": 256, "stream": False,
    })
    if payload.get("model") != config["model"]:
        raise RuntimeError("Returned model identifier differs from the requested model; not verified")
    choices = payload.get("choices")
    text = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices else None
    if not isinstance(text, str) or not text.strip() or not payload.get("id"):
        raise RuntimeError("PyTRiO did not return both text and a request ID")
    usage = payload.get("usage")
    return {"id": payload["id"], "model": payload["model"], "text": text,
            "usage": usage if isinstance(usage, dict) else {},
            "latency_seconds": time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Report configuration presence without networking")
    mode.add_argument("--check-run", metavar="WORKSPACE/PROJECT/RUN_ID",
                      help="Read an existing cloud run's state without calling the model or changing it")
    parser.add_argument("--output", type=Path, help="Directory for the local call receipt and SwanLab logs")
    args = parser.parse_args(argv)
    load_env(args.env_file)
    config = configuration()
    if args.check_run:
        if not os.environ.get("SWANLAB_API_KEY", "").strip():
            raise ValueError("Set SWANLAB_API_KEY for the read-only cloud check")
        import swanlab
        api = swanlab.Api(api_key=os.environ["SWANLAB_API_KEY"], host=SWANLAB_HOST)
        print(json.dumps({**check_run(api, args.check_run), "model_calls": 0, "state_changes": 0},
                         ensure_ascii=False, indent=2))
        return 0
    missing = [name for name in ("PYTRIO_API_KEY", "SWANLAB_API_KEY") if not os.environ.get(name, "").strip()]
    if config["base_url"] != "https://pytrio.com":
        raise ValueError("This check sends credentials only to https://pytrio.com")
    if not config["model"].startswith("Qwen/") or not config["model"].endswith("-27B"):
        raise ValueError("PYTRIO_MODEL must identify the requested Qwen 27B model")
    installed = importlib.util.find_spec("swanlab") is not None
    if args.check_config:
        print(json.dumps({**config, "missing_credentials": missing, "swanlab_installed": installed,
                          "network_calls": 0}, ensure_ascii=False, indent=2))
        return 2 if missing or not installed else 0
    if missing:
        raise ValueError("Fill these settings in .env: " + ", ".join(missing))
    if not installed:
        raise ValueError("Install optional dependencies: .venv/Scripts/python.exe -m pip install -r requirements-platform.txt")
    import swanlab
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "runtime" / f"live-platforms-{stamp}-{uuid.uuid4().hex[:8]}").resolve()
    # Never overwrite a previous run's receipt or use its identity accidentally.
    output.mkdir(parents=True, exist_ok=False)
    receipt = run_check(swanlab, config, output)
    print(json.dumps({"receipt": str(output / "receipt.json"), **receipt}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        raise SystemExit(main())
    except PlatformCheckError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
    except Exception as error:
        # Third-party exceptions may echo request headers. Keep CLI errors safe.
        print(f"Platform check failed ({type(error).__name__}); inspect the local receipt or --check-config.", file=sys.stderr)
        raise SystemExit(2)
