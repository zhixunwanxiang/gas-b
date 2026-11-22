"""Probe the native audit policy without accessing actual private datasets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SimulationAuditTest(unittest.TestCase):
    def probe(self, body, *, credential_name=None):
        with tempfile.TemporaryDirectory(prefix="gas-audit-probe-") as tmp:
            root = Path(tmp)
            for name in ("model", "work", "dataset", "module"):
                (root / name).mkdir()
            (root / "dataset/probe.txt").write_text("synthetic access probe", encoding="utf-8")
            if credential_name:
                probe_data = root / "module" / credential_name
                probe_data.parent.mkdir(parents=True, exist_ok=True)
                probe_data.write_text("SYNTHETIC_SENTINEL_NOT_A_CREDENTIAL", encoding="utf-8")
            else:
                probe_data = root / "dataset/probe.txt"
            source = "TASKS = ('classification',)\ndef train(rnd): pass\ndef predict(sample, model_dir):\n"
            source += "\n".join("    " + line for line in body.splitlines())
            source += "\n    return [{'label': str(i), 'score': .5 if i == 0 else .1} for i in range(6)]\n"
            (root / "module/audit_probe.py").write_text(source, encoding="utf-8")
            env = os.environ.copy()
            env.update(PYTHONPATH=str(root / "module"), CONTEST_SUBMISSION="audit_probe",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PROBE_DATA=str(probe_data))
            command = [sys.executable, "-B", str(ROOT / "scripts/simulation_audit.py"), "predict",
                       "--model-root", str(root / "model"), "--workspace", str(root / "work"),
                       "--audit", str(root / "access.jsonl"), "--dataset-root", str(root / "dataset")]
            result = subprocess.run(command, cwd=ROOT, env=env,
                                    input=json.dumps({"task": "classification", "inputs": {"features": [0.] * 128}}) + "\n",
                                    capture_output=True, text=True, encoding="utf-8", timeout=15)
            audit = [json.loads(line) for line in (root / "access.jsonl").read_text(encoding="utf-8").splitlines()]
            return result, audit

    def test_null_device_is_allowed_without_opening_dataset(self):
        result, audit = self.probe("import os\nwith open(os.devnull, 'w') as sink:\n    sink.write('discarded log')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(item.get("allowed") is False for item in audit))
        self.assertEqual(len(json.loads(result.stdout)["outputs"]), 6)

    def test_dataset_read_outside_declared_mount_is_denied(self):
        result, audit = self.probe("import os\nwith open(os.environ['PROBE_DATA']) as stream:\n    stream.read()")
        self.assertNotEqual(result.returncode, 0)
        denied = [item for item in audit if item.get("allowed") is False]
        self.assertEqual(len(denied), 1)
        self.assertTrue(denied[0]["path"].endswith("probe.txt"))
        self.assertFalse(denied[0]["write"])

    @unittest.skipUnless(sys.platform == "win32", "Windows version helper only")
    def test_blocked_windows_version_query_is_distinct_from_data_access(self):
        result, audit = self.probe('import subprocess\ntry:\n    subprocess.run(\'cmd /c "ver"\')\nexcept PermissionError:\n    pass')
        self.assertEqual(result.returncode, 0, result.stderr)
        queries = [item for item in audit if item.get("event") == "subprocess.Popen"]
        self.assertTrue(queries)
        self.assertTrue(all(item["category"] == "blocked_os_version_query" for item in queries))
        self.assertFalse(any(item.get("allowed") is False for item in audit))

    def test_credential_paths_are_denied_even_outside_dataset_roots(self):
        for name in (".env", ".env.local", ".netrc", ".aws/credentials", ".ssh/id_rsa", ".swanlab/config"):
            with self.subTest(name=name):
                result, audit = self.probe("import os\nwith open(os.environ['PROBE_DATA']) as stream:\n    stream.read()",
                                           credential_name=name)
                self.assertNotEqual(result.returncode, 0)
                denied = [item for item in audit if item.get("allowed") is False]
                self.assertEqual(len(denied), 1)
                self.assertTrue(denied[0]["credential_path_denied"])


if __name__ == "__main__":
    unittest.main()
