"""Host-side Docker runner; credentials never enter contestant containers."""
import json
import os
from pathlib import Path
import subprocess
import uuid


def child_environment():
    result = {k: v for k, v in os.environ.items()
              if not k.startswith(("PYTRIO_", "SWANLAB_", "CONTEST_")) and k != "OPENAI_API_KEY"}
    result.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return result


class DockerRuntime:
    def __init__(self, image, *, executable="docker", context=None, expected_id=None):
        self.cli = [executable, *(["--context", context] if context else [])]
        self.requested_image, self.image, self.expected_id = image, None, expected_id
        self.launches = []
        self.probe = None

    def _run(self, args):
        return subprocess.run([*self.cli, *args], capture_output=True, text=True, encoding="utf-8",
                              env=child_environment(), timeout=60, check=True)

    def preflight(self):
        info = json.loads(self._run(["image", "inspect", self.requested_image]).stdout)[0]
        if info["Os"] != "linux" or info["Architecture"] != "amd64":
            raise ValueError("The rehearsal requires a Linux amd64 image")
        if self.expected_id and info["Id"] != self.expected_id:
            raise ValueError("Image ID differs from the delivered image")
        if info["Config"].get("User") != "65534:65534":
            raise ValueError("Image is not configured for the expected non-root user")
        self.image = info["Id"]  # Every launch uses this immutable ID and --pull=never.
        probe = """
import errno,importlib.metadata as m,json,os
assert os.geteuid()==65534
mount=next(line.split()[3].split(',') for line in open('/proc/mounts') if line.split()[1]=='/')
assert 'ro' in mount
status=dict(line.strip().split(':',1) for line in open('/proc/self/status') if ':' in line)
assert int(status['CapEff'].strip(),16)==0 and status['NoNewPrivs'].strip()=='1'
assert open('/sys/fs/cgroup/memory.max').read().strip()=='4294967296'
assert open('/sys/fs/cgroup/cpu.max').read().strip()=='200000 100000'
assert set(os.listdir('/sys/class/net'))=={'lo'}
assert not any(name.startswith(('PYTRIO_','SWANLAB_')) or name=='OPENAI_API_KEY' for name in os.environ)
assert not any(os.path.exists(p) for p in ('/app/.env','/app/data','/app/models','/app/runtime'))
try:
 open('/app/write-probe','w').close()
except OSError as error:
 assert error.errno in (errno.EACCES,errno.EROFS)
else:
 raise AssertionError('Code directory is writable')
assert m.version('contest-harness')=='1.0.4' and m.version('numpy')=='1.26.4'
print(json.dumps({'verified':True,'uid':os.geteuid(),'readonly_root':True,'network':'none',
 'cpus':2,'memory_bytes':4294967296,'sdk':'1.0.4','numpy':'1.26.4','credentials_absent':True}))
"""
        argv, _ = self._base("probe")
        result = subprocess.run([*argv, "--entrypoint", "python3", self.image, "-c", probe],
                                capture_output=True, text=True, encoding="utf-8",
                                env=child_environment(), timeout=60, check=True)
        self.probe = json.loads(result.stdout)
        description = json.loads(self._run(["run", "--rm", "--pull=never", "--network", "none",
                                           "--read-only", self.image, "describe"]).stdout)
        if description.get("sdk_version") != "1.0.4" or description.get("tasks") != ["classification"]:
            raise ValueError("Unexpected image describe contract")
        return {"requested_image": self.requested_image, "image_id": self.image,
                "describe": description, "runtime_probe": self.probe}

    def _base(self, kind):
        if self.image is None:
            raise RuntimeError("Image preflight must complete before launching a container")
        name = "gas-real-" + kind + "-" + uuid.uuid4().hex
        return [*self.cli, "run", "--rm", "-i", "--name", name,
                "--label", "gas-rsi-real-rehearsal=true", "--pull=never", "--network", "none",
                "--read-only", "--cpus", "2", "--memory", "4g", "--pids-limit", "128",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777"], name

    def _record(self, path, kind, argv, mounts):
        item = {"command": kind, "image_id": self.image, "argv": argv, "mounts": mounts,
                "policy": {"readonly_root": True, "network": "none", "cpus": 2, "memory": "4g",
                           "credentials_passed": False}}
        self.launches.append(item)
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def train(self, directory, data, state, log):
        argv, name = self._base("train")
        mounts = {"/previous": str(Path(directory).resolve()), "/data": str(Path(data).resolve())}
        if state:
            mounts["/state"] = str(Path(state).resolve())
        argv += ["--tmpfs", "/workspace:rw,nosuid,nodev,size=4g,mode=1777",
                 "--tmpfs", "/output:rw,nosuid,nodev,size=1g,mode=1777"]
        for target, source in mounts.items():
            argv += ["--mount", f"type=bind,source={source},target={target},readonly"]
        argv += [self.image, "train"]
        self._record(log, "train", argv, mounts)
        return argv, name

    def predict(self, directory, primary, output):
        relative = Path(primary).relative_to(Path(directory)).as_posix()
        argv, _ = self._base("predict")
        mounts = {"/model": str(Path(directory).resolve())}
        argv += ["--mount", f"type=bind,source={mounts['/model']},target=/model,readonly",
                 "-e", "CONTEST_ARTIFACT_PATH=/model/" + relative, self.image, "predict"]
        self._record(Path(output) / "container.jsonl", "predict", argv, mounts)
        return argv

    def cleanup_prediction(self, argv):
        self.cleanup(argv[argv.index("--name") + 1])

    def cleanup(self, name):
        if not name.startswith("gas-real-"):
            raise ValueError("Refusing to remove a container not owned by this rehearsal")
        subprocess.run([*self.cli, "rm", "-f", name], env=child_environment(),
                       capture_output=True, timeout=30, check=False)

    def audit(self):
        policies = all(item["image_id"] == self.image
                       and set(item["mounts"]) <= ({"/previous", "/data", "/state"}
                            if item["command"] == "train" else {"/model"}) for item in self.launches)
        return {"mode": "docker_enforced_mounts_and_network", "os_level_container_isolation": True,
                "container_policy_verified": bool(self.probe and self.probe.get("verified") and policies and self.launches),
                "train_containers": sum(item["command"] == "train" for item in self.launches),
                "prediction_containers": sum(item["command"] == "predict" for item in self.launches),
                "file_access_instrumented": False, "network_attempts_instrumented": False,
                "note": "Docker restrictions and permitted mounts verified; individual file/socket calls were not traced."}
