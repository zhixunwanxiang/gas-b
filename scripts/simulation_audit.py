"""Native-only simulation adapter: audited I/O and a /previous stand-in.

Not part of the submission image. This is application auditing, not OS isolation.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "predict"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    allowed = [args.model_root.resolve(), args.workspace.resolve()]
    allowed.extend(p.resolve() for p in (args.state_root, args.data_root) if p)
    # A reproduction package can sit under the supplied dataset/provenance
    # parent. Permit its application code without permitting data/models/runtime.
    allowed.extend(root / name for name in ("src", "scripts", "tests", "vendor", ".venv", ".python", "__pycache__",
                                            "harness.py", "contest_submission.py", "submission_config.json"))
    forbidden = [root / "data", root / "models", root / "runtime", args.dataset_root.resolve()]
    forbidden = [p.resolve() for p in forbidden]
    workspace = args.workspace.resolve()
    null_device = Path(os.devnull).resolve()
    def credential_file(path):
        parts = [part.casefold() for part in path.parts]
        return any(part == ".env" or part.startswith(".env.")
                   or part in (".netrc", "_netrc", ".aws", ".ssh", ".swanlab") for part in parts)
    audit = args.audit.open("x", encoding="utf-8")
    def log(value):
        audit.write(json.dumps(value, ensure_ascii=False) + "\n")
        audit.flush()
    log({"type": "policy", "command": args.command, "allowed_read_roots": list(map(str, allowed)),
         "forbidden_dataset_roots": list(map(str, forbidden)), "writable_root": str(workspace),
         "network_allowed": False, "os_sandbox": False})
    def hook(event, values):
        if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
            command = str(values[1]) if event == "subprocess.Popen" else None
            metadata_query = (sys.platform == "win32" and command is not None
                              and command.lower().endswith((' /c "ver"', ' /c "command /c ver"', ' /c "cmd /c ver"')))
            log({"type": "denied", "event": event, "command": command,
                 "category": "blocked_os_version_query" if metadata_query else "blocked_external_operation"})
            raise PermissionError("Native rehearsal forbids networking and subprocesses")
        paths, writing = [], True
        if event == "open":
            path, mode, flags = values
            writing = bool((isinstance(mode, str) and any(c in mode for c in "wax+"))
                           or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            paths = [path]
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"):
            paths = [values[0]]
        elif event == "os.rename":
            paths = [values[0], values[1]]
        for value in paths:
            if not isinstance(value, (str, bytes, os.PathLike)):
                continue
            original_path = Path(os.fsdecode(value)).absolute()
            # Resolving an absolute spelling of Windows' relative `nul`
            # changes its device semantics in pathlib on Python 3.12.
            path = Path(os.fsdecode(value)).resolve()
            credential = credential_file(original_path) or credential_file(path)
            permitted = not credential and (path == null_device or (path.is_relative_to(workspace) if writing else
                         not any(path.is_relative_to(p) for p in forbidden)
                         or any(path.is_relative_to(p) for p in allowed)))
            log({"type": "file", "event": event, "write": writing, "path": str(path), "allowed": permitted,
                 "credential_path_denied": credential})
            if not permitted:
                raise PermissionError("Access outside rehearsal mounts: " + str(path))
    sys.addaudithook(hook)
    os.environ["CONTEST_WORK"] = str(workspace)
    if args.command == "train":
        from contest_harness.round import ParentView
        original = ParentView.__init__
        def init(self, frame):
            original(self, frame)
            self.root = args.model_root.resolve()
        ParentView.__init__ = init
    sys.path.insert(0, str(root))
    sys.argv = [str(root / "harness.py"), args.command]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
