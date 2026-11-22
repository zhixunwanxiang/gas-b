"""共享文件工具：配置继承、SHA-256 摘要、原子 JSON 写入、JSONL 历史与运行锁。"""
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_config(path, _parents=()):
    """解析相对路径的 JSON 配置继承，递归合并嵌套对象并避免共享合并字典。"""
    path = Path(path).resolve()
    if path in _parents or len(_parents) >= 8:
        raise ValueError("Configuration inheritance cycle or excessive depth")
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    if not isinstance(parent, str) or not parent:
        raise ValueError("extends must name a configuration file")
    base = read_config(path.parent / parent, (*_parents, path))

    def merge(old, new):
        result = dict(old)
        for key, value in new.items():
            result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        return result
    return merge(base, config)


def write_json(path, obj):
    """通过原子替换写入 JSON，避免活动模型指针出现写入不完整的内容。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # 临时文件与目标位于同一目录，通过 os.replace 一次切换，读者无需读取半成品。
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def write_jsonl(path, rows):
    """持运行锁时原子替换短事务日志，避免切版记录被中断为半行JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def history(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _process_alive(pid):
    """只查询进程，不能在Windows上用os.kill(pid, 0)模拟存在性检测。"""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # 无效PID为已退出，其余情况按仍存活处理。
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def writer_active(root):
    """只检查实际系统锁及旧格式活跃PID，不修改锁文件内容。"""
    path = Path(root) / ".run.lock"
    if not path.exists():
        return False
    try:
        stream = path.open("r+b")
    except OSError:
        return True
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        content = stream.read().decode("utf-8").strip()
        return content.isdecimal() and int(content) > 0 and _process_alive(int(content))
    except (OSError, UnicodeError):
        return True
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


@contextmanager
def lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    file = root / ".run.lock"
    # 持久的锁文件只是元数据，互斥由操作系统管理；进程异常退出会自动释放。
    fd = os.open(str(file), os.O_CREAT | os.O_RDWR)
    stream = os.fdopen(fd, "r+b")
    try:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError(f"Another process owns {file}; retry after the current writer exits") from exc
    try:
        stream.seek(0)
        old = stream.read().decode("utf-8").strip()
        if old.isdecimal() and int(old) > 0 and _process_alive(int(old)):
            raise RuntimeError("A legacy writer still owns the run lock; do not replace its live process")
        stream.seek(0)
        stream.write(json.dumps({"format": "gas-a.os-lock.v1", "pid": os.getpid()}).encode())
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

