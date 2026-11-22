"""Acceptance status uses actual outcomes, including an observe-only SDK session.

All remote calls are mocked. Synthetic replies stay in temporary directories.
"""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import test_real_sdk as live
from gas_a.sensor_training.io import digest

ROOT = Path(__file__).resolve().parents[1]


class SdkAcceptanceTest(unittest.TestCase):
    def report(self):
        return {"rounds": [{"finish_acknowledged": True, "model_call_ids": [f"offline-{i}"]} for i in range(4)],
                "audit": {"denied_accesses": [], "network_attempts": []},
                "baseline_macro_f1": .7, "final_macro_f1": .8,
                "complete_history_experiment_decision_records": 3}

    def test_success_requires_all_normal_requirements(self):
        report = self.report()
        self.assertEqual(live.finalize_acceptance(report), 0)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["requirements_passed"])
        self.assertEqual(report["acceptance_scope"], "local_authorized_public_validation_only")

    def test_missing_experiments_bad_scores_ack_and_audit_cannot_pass(self):
        cases = [dict(complete_history_experiment_decision_records=0), dict(final_macro_f1=.6),
                 dict(final_macro_f1=float("nan")), dict(final_macro_f1=True),
                 dict(final_macro_f1=10**1000), dict(baseline_macro_f1=None), dict(audit={}),
                 dict(audit={"denied_accesses": ["blocked"], "network_attempts": []})]
        no_ack = self.report()["rounds"]
        no_ack[-1]["finish_acknowledged"] = False
        no_call = deepcopy(self.report()["rounds"])
        no_call[0]["model_call_ids"] = []
        cases += [dict(rounds=no_ack), dict(rounds=no_call), dict(rounds=self.report()["rounds"][:3])]
        for changes in cases:
            with self.subTest(changes=changes):
                report = {**self.report(), **changes}
                self.assertEqual(live.finalize_acceptance(report), 1)
                self.assertEqual(report["status"], "requirements_unmet")
                self.assertFalse(report["requirements_passed"])

    def test_recovery_has_separate_rollback_parent_and_baseline_checks(self):
        report = {**self.report(), "rounds": self.report()["rounds"][:2], "rollback_selected": True,
                  "next_round_used_recovered_parent": True, "recovered_macro_f1": .8}
        self.assertEqual(live.finalize_acceptance(report, recovery=True), 0)
        report["next_round_used_recovered_parent"] = False
        self.assertEqual(live.finalize_acceptance(report, recovery=True), 1)
        report["next_round_used_recovered_parent"] = True
        report["recovered_macro_f1"] = .6
        self.assertEqual(live.finalize_acceptance(report, recovery=True), 1)

    def test_four_observe_only_rounds_return_unmet_instead_of_success(self):
        with tempfile.TemporaryDirectory(prefix="sdk-observe-acceptance-") as temporary:
            root = Path(temporary)
            base = root / "baseline"
            base.mkdir()
            shutil.copyfile(ROOT / "models/reference.npz", base / "checkpoint.npz")
            shutil.copytree(ROOT / "data/public", base / "public")
            (base / "baseline-info.json").write_text(json.dumps({"dataset_key": "A",
                "checkpoint_sha256": digest(base / "checkpoint.npz")}), encoding="utf-8")
            count = 0
            def synthetic_http(url, key, body):
                nonlocal count
                count += 1
                request = json.loads(next(m["content"] for m in body["messages"] if m["role"] == "user"))
                parent = request["parent_artifact_id"]
                if request["phase"] == "plan":
                    reply = {"action": "observe", "experiment": "none", "target_artifact_id": parent,
                             "hypothesis": "Synthetic observe-only control", "reason": "Do not run experiments"}
                else:
                    reply = {"selected_artifact_id": parent, "reason": "Synthetic control retains the parent"}
                return {"id": f"SYNTHETIC-OFFLINE-{count}", "model": live.MODEL,
                        "choices": [{"message": {"content": json.dumps(reply)}}], "usage": {"total_tokens": 1}}
            args = ["test_real_sdk.py", "--baseline-dir", str(base), "--output", str(root / "out"),
                    "--rounds", "4", "--max-rounds", "4", "--max-model-calls", "8"]
            with patch.object(sys, "argv", args), patch.object(live, "load_env"), \
                    patch.object(live, "ensure_model"), patch.object(live, "configuration", return_value={
                        "model": live.MODEL, "base_url": "https://pytrio.com"}), \
                    patch.object(live, "request_json", side_effect=synthetic_http), \
                    patch.dict(os.environ, {"PYTRIO_API_KEY": "SYNTHETIC-NOT-A-KEY"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = live.main()
            report = json.loads((root / "out/live-report.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertEqual(report["status"], "requirements_unmet")
            self.assertFalse(report["requirements_passed"])
            self.assertEqual(report["complete_history_experiment_decision_records"], 0)
            self.assertEqual(len(report["rounds"]), 4)
            self.assertEqual(count, 8)


if __name__ == "__main__":
    unittest.main()
