"""Classified rejection budgets and fresh historical selection through SDK 1.0.4."""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import contest_submission
from contest_harness.round import Round
from gas_a import submission as classification
from gas_a.evolution import (advance_policy, available_experiments, eligible_history,
    experiment_outcome, fingerprint, initial_policy, validate_attempts)
from gas_a.regression import submission as regression
from gas_a.sensor_training.reference.model import Model
from gas_a.sensor_training.io import digest
from tests.test_submission import ROOT, OfflineHost, writes_only_in
from tests.test_regression import make_fixture, RegressionHost


class PolicyHistorySelectionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="policy-history-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def peer(self, dataset, name):
        root = self.root / (dataset + "-" + name)
        if dataset == "A":
            shutil.copytree(ROOT / "data/public", root / "data")
            (root / "previous").mkdir()
            parent = root / "previous/checkpoint.npz"
            shutil.copyfile(ROOT / "models/reference.npz", parent)
            return OfflineHost(root, parent, experiment="frozen_router")
        return RegressionHost(root, make_fixture(root, dataset), dataset)

    def run_peer(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = host.root / ("work-" + str(rnd.round_index))
        with patch.dict(os.environ, {"CONTEST_WORK": str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertFalse(host.pending)
        record = json.loads((host.saved_state / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["model_call_ids"], host.calls)
        return record

    def historical_peer(self, dataset, name):
        host = self.peer(dataset, name)
        historical = host.root / "historical"
        historical.mkdir()
        shutil.copyfile(host.parent, historical / "checkpoint.npz")
        parent = Model.load(host.parent)
        if dataset == "A":
            parent.params["w2"][:] = 0
            parent.params["b2"][:] = 0
        else:
            parent.params["b2"] += 100
        host.parent.unlink()
        parent.save(host.parent)
        host.frame["parent_checkpoint_sha256"] = digest(host.parent)
        host.frame["parent_artifact"]["files"][0].update(sha256=digest(host.parent), size_bytes=host.parent.stat().st_size)
        host.history = [{"id": "past", "selected_artifact_id": "history-good"}]
        host.add_artifact("history-good", historical)
        return host, historical

    def force_selection(self, host, identifier, *, reject_candidate=False, bad_plan=False):
        original = host.write
        def inject(text):
            result = original(text)
            frame = json.loads(text)
            if frame["type"] == "candidate_request" and reject_candidate:
                host.pending[-1]["summary"]["scope"] = "hidden"
            if frame["type"] == "model_request":
                request = json.loads(frame["messages"][0]["content"])
                if request["phase"] == "select":
                    content = json.dumps({"selected_artifact_id": identifier, "reason": "Select the current measured historical model."})
                    host.pending[-1]["response"]["choices"][0]["message"]["content"] = content
                elif bad_plan:
                    host.pending[-1]["response"]["choices"][0]["message"]["content"] = "{invalid"
            return result
        host.write = inject

    def test_only_performance_and_local_failures_consume_rejection_budget(self):
        cases = [
            ("performance_rejected", False, {"eligible": False}, {"eligible": True}, "model", True),
            ("measurement_unverified", False, {"eligible": True}, {"eligible": False, "reason": "unknown_validation_scope"}, "model", False),
            ("controller_guard", False, {"eligible": True}, {"eligible": True}, "program_guard", False),
            ("not_selected", False, {"eligible": True}, {"eligible": True}, "model", False),
            ("local_training_failed", True, None, None, "model", True),
        ]
        for outcome, failed, local_gate, host_gate, source, should_block in cases:
            for dataset in "ABC":
                with self.subTest(dataset=dataset, outcome=outcome):
                    menu = {"x": {"name": "x"}}
                    identity = fingerprint("parent", "train", "validation", menu["x"])
                    policy = initial_policy()
                    for index in (1, 2):
                        common = dict(local_gate=local_gate, host_gate=host_gate, selection_source=source)
                        if dataset == "A":
                            policy = advance_policy(policy, selected="parent", parent_id="parent",
                                candidate_id=None if failed else f"candidate-{index}", plan={"experiment": "x"},
                                attempt_key=identity, gain=.05, validation_sha256="validation", selected_f1=.7,
                                experiment_failed=failed, **common)
                        else:
                            policy = regression._advance(policy, "parent", "parent", None if failed else f"candidate-{index}",
                                {"experiment": "x"}, identity, 1., failed, **common)
                    validate_attempts(policy["attempts"])
                    self.assertEqual({a["outcome"] for a in policy["attempts"]}, {outcome})
                    allowed, blocked = available_experiments(policy, menu,
                        SimpleNamespace(parent=SimpleNamespace(checkpoint_sha256="parent")),
                        {"sha256": "train"}, {"sha256": "validation"}, {"max_repeat_rejections": 2})
                    self.assertEqual("x" in blocked, should_block)
                    self.assertEqual("x" in allowed, not should_block)

    def test_legacy_unknown_rejections_are_not_invented_performance_failures(self):
        menu = {"x": {"name": "x"}}
        identity = fingerprint("parent", "train", "validation", menu["x"])
        policy = initial_policy()
        for status, blocked_expected in (("measured", False), ("local_training_failed", True)):
            policy["attempts"] = [{"fingerprint": identity, "adopted": False, "status": status}] * 2
            validate_attempts(policy["attempts"])
            _, blocked = available_experiments(policy, menu,
                SimpleNamespace(parent=SimpleNamespace(checkpoint_sha256="parent")),
                {"sha256": "train"}, {"sha256": "validation"}, {"max_repeat_rejections": 2})
            self.assertEqual("x" in blocked, blocked_expected)
        for fields in ({"outcome": []}, {"outcome": "unexpected"},
                       {"outcome": "controller_guard", "counts_toward_rejection_limit": True}):
            with self.assertRaises(ValueError):
                validate_attempts([{"fingerprint": identity, "adopted": False, **fields}])

    def test_repeated_controller_or_measurement_failures_leave_experiment_available(self):
        for dataset in "ABC":
            for mode in ("controller", "measurement"):
                first = self.peer(dataset, mode)
                root, parent, state, history = first.root, first.parent, None, []
                for index in (1, 2, 3):
                    host = first if index == 1 else (OfflineHost(root, parent, index, history, state, experiment="frozen_router")
                        if dataset == "A" else RegressionHost(root, parent, dataset, index, history=history, state=state))
                    if index == 3:
                        host.action = "observe"
                    original = host.write
                    def inject(text):
                        result = original(text)
                        frame = json.loads(text)
                        if index <= 2 and mode == "measurement" and frame["type"] == "candidate_request":
                            host.pending[-1]["summary"]["scope"] = "hidden"
                        if index <= 2 and mode == "controller" and frame["type"] == "model_request":
                            if json.loads(frame["messages"][0]["content"])["phase"] == "select":
                                host.pending[-1]["response"]["choices"][0]["message"]["content"] = "{invalid"
                        return result
                    host.write = inject
                    record = self.run_peer(host)
                    state = host.saved_state
                    if index <= 2:
                        outcome = record["configuration"]["experiment_outcome"]
                        self.assertEqual(outcome["outcome"], "controller_guard" if mode == "controller" else "measurement_unverified")
                        self.assertFalse(outcome["counts_toward_rejection_limit"])
                        self.assertEqual(host.record["selected_artifact_id"], host.frame["parent_artifact"]["artifact_id"])
                    else:
                        request = json.loads(host.messages[0][0]["content"])
                        self.assertIn("frozen_router" if dataset == "A" else "low_rate", request["experiments"])
                    history.append({"id": f"history-{index}", **host.record})

    def test_observe_or_experiment_can_select_fresh_history_and_continue_it_next_round(self):
        for dataset in "ABC":
            for action in ("observe", "experiment"):
                host, historical = self.historical_peer(dataset, action)
                host.action = action
                parent_sha = digest(host.parent)
                self.force_selection(host, "history-good", reject_candidate=action == "experiment")
                record = self.run_peer(host)
                cfg = record["configuration"]
                self.assertEqual(record["plan"]["action"], action)
                self.assertEqual(cfg["action"], action)
                self.assertEqual(cfg["selection_action"], "rollback")
                self.assertEqual(host.record["selected_artifact_id"], "history-good")
                self.assertEqual(digest(host.parent), parent_sha)
                request = json.loads(host.messages[-1][0]["content"])
                self.assertTrue(request["historical_admission"]["history-good"]["eligible"])
                self.assertIn("history-good", request["response_schema"]["selected_artifact_id"])
                if action == "experiment":
                    self.assertIsNotNone(host.candidate)
                    self.assertEqual(cfg["parent_checkpoint_sha256"], parent_sha)
                    self.assertFalse(cfg["host_candidate_gate"]["eligible"])
                else:
                    self.assertIsNone(host.candidate)
                next_host = (OfflineHost(host.root, historical / "checkpoint.npz", 2,
                    history=[{"id": "previous", **host.record}], state=host.saved_state) if dataset == "A" else
                    RegressionHost(host.root, historical / "checkpoint.npz", dataset, 2,
                        history=[{"id": "previous", **host.record}], state=host.saved_state))
                next_host.frame["parent_artifact"]["artifact_id"] = "history-good"
                next_host.action = "observe"
                following = self.run_peer(next_host)
                self.assertEqual(following["configuration"]["parent_checkpoint_sha256"], digest(historical / "checkpoint.npz"))
                self.assertEqual(next_host.record["selected_artifact_id"], "history-good")

    def test_stale_unmeasured_unauthorized_or_failed_history_cannot_enter_choices(self):
        fresh = {"reevaluated_this_round": True, "validation_sha256": "current", "gate": {"eligible": True}}
        for changes in ({"validation_sha256": "old"}, {"reevaluated_this_round": False}, {"gate": {"eligible": False}}):
            self.assertEqual(eligible_history({"old": {**fresh, **changes}}, "current", True, {"old"}), [])
        self.assertEqual(eligible_history({"old": fresh}, "current", True, set()), [])
        self.assertEqual(eligible_history({"old": fresh}, "current", False, {"old"}), [])
        for dataset in "ABC":
            host, _ = self.historical_peer(dataset, "stale")
            host.action = "observe"
            module = classification if dataset == "A" else regression
            original = module._restore_history
            def stale(*args, **kwargs):
                result = original(*args, **kwargs)
                result["history-good"]["validation_sha256"] = "old-validation"
                return result
            self.force_selection(host, "history-good")
            with patch.object(module, "_restore_history", side_effect=stale):
                record = self.run_peer(host)
            self.assertEqual(host.record["selected_artifact_id"], host.frame["parent_artifact"]["artifact_id"])
            self.assertEqual(record["selection_calls"]["decision_source"], "program_guard")

    def test_planning_guard_still_locks_selection_to_actual_parent(self):
        for dataset in "ABC":
            host, _ = self.historical_peer(dataset, "planning-guard")
            self.force_selection(host, "history-good", bad_plan=True)
            record = self.run_peer(host)
            self.assertEqual(host.record["selected_artifact_id"], host.frame["parent_artifact"]["artifact_id"])
            self.assertIsNone(host.candidate)
            self.assertEqual(record["planning_calls"]["decision_source"], "program_guard")
            self.assertEqual(record["selection_calls"]["decision_source"], "program_guard")
            self.assertEqual(json.loads(host.messages[-1][0]["content"])["response_schema"]["selected_artifact_id"],
                             [host.frame["parent_artifact"]["artifact_id"]])

    def test_explicit_parent_scope_conflict_still_blocks_historical_selection(self):
        for dataset in "ABC":
            host, _ = self.historical_peer(dataset, "parent-scope-conflict")
            host.action = "observe"
            host.frame["current_validation"] = {"macro_f1" if dataset == "A" else "rmse": 0., "scope": "hidden"}
            self.force_selection(host, "history-good")
            record = self.run_peer(host)
            self.assertEqual(host.record["selected_artifact_id"], host.frame["parent_artifact"]["artifact_id"])
            self.assertEqual(record["selection_calls"]["decision_source"], "program_guard")
            self.assertIsNone(host.candidate)


if __name__ == "__main__":
    unittest.main()
