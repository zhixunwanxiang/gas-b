"""Offline checks for credential handling and exact-model verification."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import live_platforms as live


class LivePlatformsTest(unittest.TestCase):
    def test_env_is_literal_and_process_settings_take_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PYTRIO_MODEL": "process-model"}, clear=True):
            env = Path(directory) / ".env"
            env.write_text('PYTRIO_API_KEY="literal-$(echo-test)"\nPYTRIO_MODEL=file-model\nUNRELATED=value\n', encoding="utf-8")
            live.load_env(env)
            self.assertEqual(os.environ["PYTRIO_API_KEY"], "literal-$(echo-test)")
            self.assertEqual(os.environ["PYTRIO_MODEL"], "process-model")
            self.assertNotIn("UNRELATED", os.environ)

    def test_config_check_never_sends_requests_or_prints_keys(self):
        with patch.dict(os.environ, {"PYTRIO_API_KEY": "private-test-key"}, clear=True), patch.object(live, "request_json") as request:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = live.main(["--env-file", "does-not-exist.env", "--check-config"])
            self.assertEqual(code, 2)
            self.assertNotIn("private-test-key", output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["missing_credentials"], ["SWANLAB_API_KEY"])
            request.assert_not_called()

    def test_missing_credentials_stop_before_networking(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(live, "request_json") as request:
            with self.assertRaisesRegex(ValueError, "PYTRIO_API_KEY, SWANLAB_API_KEY"):
                live.main(["--env-file", "does-not-exist.env"])
            request.assert_not_called()

    def test_missing_exact_model_does_not_fall_back_to_four_b(self):
        with patch.object(live, "request_json", return_value={"items": [{"model_id": "Qwen/Qwen3-4B"}]}) as request:
            with self.assertRaisesRegex(RuntimeError, "not available"):
                live.ensure_model(live.configuration(), "fake-test-key")
            self.assertEqual(request.call_count, 1)

    def test_different_response_model_is_not_accepted(self):
        payload = {"id": "response-1", "model": "Qwen/Qwen3-4B", "choices": [{"message": {"content": "ok"}}]}
        with patch.object(live, "request_json", return_value=payload):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                live.call_model(live.configuration(), "fake-test-key")

    def test_provider_receipt_requires_text_and_request_id(self):
        config = live.configuration()
        payload = {"model": config["model"], "choices": [{"message": {"content": "ok"}}]}
        with patch.object(live, "request_json", return_value=payload):
            with self.assertRaisesRegex(RuntimeError, "request ID"):
                live.call_model(config, "fake-test-key")


class FakeSwanlab:
    """No network, workers or real model calls; expose realistic lifecycle failures."""
    def __init__(self):
        self.active = None
        self.cloud_state = "RUNNING"
        self.init_error = None
        self.finish_error = None
        self.silent_finish_failure = False
        self.api = SimpleNamespace(username="test-user", run=Mock(side_effect=self.read_run))
        self.Api = Mock(return_value=self.api)
        self.login = Mock()
        self.Settings = Mock()
        self.init = Mock(side_effect=self.initialize)
        self.has_run = Mock(side_effect=lambda: self.active is not None)
        self.get_run = Mock(side_effect=lambda: self.active)

    def read_run(self, path):
        return SimpleNamespace(state=self.cloud_state)

    def initialize(self, **kwargs):
        self.active = SimpleNamespace(id=kwargs["id"], log=Mock(), finish=Mock(side_effect=self.finish_run))
        if self.init_error is not None:
            raise self.init_error
        return self.active

    def finish_run(self, state, error=None):
        if self.finish_error is not None:
            raise self.finish_error
        if not self.silent_finish_failure:
            self.cloud_state = live.TERMINAL_STATES[state]


class PlatformLifecycleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="platform-lifecycle-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "run"
        self.output.mkdir()
        self.swanlab = FakeSwanlab()
        self.config = {"base_url": "https://pytrio.com", "model": "Qwen/Qwen3.8-27B",
                       "project": "gas-b", "workspace": None}
        self.receipt_result = {"id": "mock-provider-response", "model": self.config["model"],
                               "text": "OK", "usage": {"total_tokens": 5}, "latency_seconds": .1}
        self.enterContext(patch.dict(os.environ, {"PYTRIO_API_KEY": "private-pytrio-test-key",
                                                "SWANLAB_API_KEY": "private-swanlab-test-key"}, clear=True))
        self.enterContext(patch.object(live, "request_json", side_effect=AssertionError("Network is forbidden")))
        self.ensure_model = self.enterContext(patch.object(live, "ensure_model"))
        self.call_model = self.enterContext(patch.object(live, "call_model", return_value=self.receipt_result))
        self.enterContext(patch.object(live.time, "sleep"))

    def read_receipt(self):
        text = (self.output / "receipt.json").read_text(encoding="utf-8")
        self.assertNotIn("private-pytrio-test-key", text)
        self.assertNotIn("private-swanlab-test-key", text)
        return json.loads(text)

    def run_check(self):
        return live.run_check(self.swanlab, self.config, self.output)

    def test_identity_is_saved_before_preflight_and_init(self):
        def before_preflight(*args):
            saved = self.read_receipt()
            self.assertEqual(saved["model_call_status"], "not_started")
            self.assertTrue(saved["run_id"])
        self.ensure_model.side_effect = before_preflight
        original = self.swanlab.init.side_effect
        def before_init(**kwargs):
            saved = self.read_receipt()
            self.assertEqual(saved["run_id"], kwargs["id"])
            self.assertTrue(saved["swanlab_init_attempted"])
            self.assertEqual(saved["run_path"], "test-user/gas-b/" + kwargs["id"])
            return original(**kwargs)
        self.swanlab.init.side_effect = before_init
        result = self.run_check()
        self.assertTrue(result["swanlab_finish_completed"])
        self.assertEqual(result["outcome"], "succeeded")
        self.assertEqual(result["cloud_verification"]["state"], "FINISHED")
        self.assertIn("/gas-b/runs/", result["swanlab_url"])
        self.call_model.assert_called_once()

    def test_partial_init_failure_closes_its_registered_run(self):
        self.swanlab.init_error = RuntimeError("private-swanlab-test-key")
        with self.assertRaisesRegex(RuntimeError, "Platform check failed"):
            self.run_check()
        saved = self.read_receipt()
        self.assertEqual(saved["failure"]["phase"], "initializing")
        self.assertEqual(saved["model_call_status"], "not_started")
        self.assertTrue(saved["swanlab_finish_completed"])
        self.assertEqual(saved["cloud_verification"]["state"], "CRASHED")
        self.swanlab.active.finish.assert_called_once()
        self.call_model.assert_not_called()

    def test_preflight_failure_has_receipt_without_creating_cloud_run(self):
        self.ensure_model.side_effect = RuntimeError("Model unavailable")
        with self.assertRaises(RuntimeError):
            self.run_check()
        saved = self.read_receipt()
        self.assertEqual(saved["failure"]["phase"], "preflight")
        self.assertEqual(saved["phase"], "failed")
        self.swanlab.init.assert_not_called()
        self.swanlab.Api.assert_not_called()
        self.call_model.assert_not_called()

    def test_init_failure_without_owned_handle_is_reported_not_claimed_closed(self):
        self.swanlab.init.side_effect = RuntimeError("init failed before registering a handle")
        with self.assertRaises(RuntimeError):
            self.run_check()
        saved = self.read_receipt()
        self.assertEqual(saved["cleanup_error_type"], "OwnedRunUnavailable")
        self.assertFalse(saved["swanlab_finish_completed"])
        self.assertEqual(saved["phase"], "needs_attention")
        self.assertTrue(saved["run_id"])
        self.call_model.assert_not_called()

    def test_model_timeout_is_not_retried_and_retains_unknown_outcome(self):
        self.call_model.side_effect = TimeoutError("private-pytrio-test-key")
        with self.assertRaises(RuntimeError):
            self.run_check()
        saved = self.read_receipt()
        self.assertTrue(saved["model_call_attempted"])
        self.assertEqual(saved["model_call_status"], "outcome_unknown")
        self.assertEqual(saved["cloud_verification"]["state"], "CRASHED")
        self.call_model.assert_called_once()

    def test_keyboard_interrupt_closes_as_aborted_and_propagates(self):
        self.call_model.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_check()
        saved = self.read_receipt()
        self.assertEqual(saved["outcome"], "aborted")
        self.assertEqual(saved["cloud_verification"]["state"], "ABORTED")
        self.call_model.assert_called_once()

    def test_finish_returning_without_cloud_completion_is_not_success(self):
        self.swanlab.silent_finish_failure = True
        with self.assertRaisesRegex(RuntimeError, "Cloud completion not confirmed"):
            self.run_check()
        saved = self.read_receipt()
        self.assertTrue(saved["swanlab_finish_returned"])
        self.assertFalse(saved["swanlab_finish_completed"])
        self.assertEqual(saved["phase"], "needs_attention")
        self.assertEqual(saved["pytrio"]["id"], "mock-provider-response")
        self.call_model.assert_called_once()
        self.swanlab.active.finish.assert_called_once()
        self.assertEqual(self.swanlab.api.run.call_count, 3)

    def test_finish_exception_preserves_model_receipt(self):
        self.swanlab.finish_error = RuntimeError("private-swanlab-test-key")
        with self.assertRaisesRegex(RuntimeError, "Cloud completion not confirmed"):
            self.run_check()
        saved = self.read_receipt()
        self.assertTrue(saved["real_model_call"])
        self.assertEqual(saved["cleanup_error_type"], "RuntimeError")
        self.assertFalse(saved["swanlab_finish_completed"])
        self.call_model.assert_called_once()

    def test_model_receipt_and_finish_attempt_are_saved_before_cleanup(self):
        original = self.swanlab.finish_run
        def check_saved_before_finish(*args, **kwargs):
            saved = self.read_receipt()
            self.assertEqual(saved["phase"], "finishing")
            self.assertTrue(saved["swanlab_finish_attempted"])
            self.assertEqual(saved["pytrio"]["id"], "mock-provider-response")
            self.assertFalse(saved["swanlab_finish_completed"])
            return original(*args, **kwargs)
        self.swanlab.finish_run = check_saved_before_finish
        self.run_check()

    def test_cloud_lookup_failure_does_not_claim_completion(self):
        self.swanlab.api.run.side_effect = RuntimeError("private-swanlab-test-key")
        with self.assertRaisesRegex(RuntimeError, "Cloud completion not confirmed"):
            self.run_check()
        saved = self.read_receipt()
        self.assertEqual(saved["cloud_verification"]["status"], "unverified")
        self.assertFalse(saved["swanlab_finish_completed"])

    def test_eventual_cloud_completion_uses_fresh_reads(self):
        self.swanlab.api.run.side_effect = [SimpleNamespace(state="RUNNING"), SimpleNamespace(state="FINISHED")]
        result = self.run_check()
        self.assertTrue(result["swanlab_finish_completed"])
        self.assertEqual(self.swanlab.api.run.call_count, 2)
        self.call_model.assert_called_once()

    def test_unexpected_terminal_state_is_not_success(self):
        self.swanlab.silent_finish_failure = True
        self.swanlab.cloud_state = "CRASHED"
        with self.assertRaisesRegex(RuntimeError, "Cloud completion not confirmed"):
            self.run_check()
        saved = self.read_receipt()
        self.assertEqual(saved["cloud_verification"]["state"], "CRASHED")
        self.assertFalse(saved["swanlab_finish_completed"])

    def test_unrelated_run_is_never_finished(self):
        foreign = SimpleNamespace(id="another-experiment", finish=Mock())
        self.swanlab.active = foreign
        with self.assertRaises(RuntimeError):
            self.run_check()
        foreign.finish.assert_not_called()
        self.swanlab.init.assert_not_called()
        self.ensure_model.assert_not_called()

    def test_receipt_write_failure_after_init_does_not_skip_cleanup(self):
        original = live.write_receipt
        failed_once = False
        def fail_one_write(path, receipt):
            nonlocal failed_once
            if receipt["phase"] == "model_call" and not failed_once:
                failed_once = True
                raise OSError("Injected disk failure")
            return original(path, receipt)
        with patch.object(live, "write_receipt", side_effect=fail_one_write), self.assertRaises(RuntimeError):
            self.run_check()
        self.swanlab.active.finish.assert_called_once()
        self.assertEqual(self.read_receipt()["cloud_verification"]["state"], "CRASHED")
        self.assertFalse(self.read_receipt()["model_call_attempted"])
        self.call_model.assert_not_called()

    def test_receipt_disk_error_during_cleanup_does_not_skip_finish(self):
        original = live.write_receipt
        def fail_cleanup_write(path, receipt):
            if receipt["phase"] == "finishing":
                raise OSError("Injected disk failure")
            return original(path, receipt)
        with patch.object(live, "write_receipt", side_effect=fail_cleanup_write):
            result = self.run_check()
        self.swanlab.active.finish.assert_called_once()
        self.assertEqual(result["receipt_write_error_type"], "OSError")
        self.assertTrue(self.read_receipt()["swanlab_finish_completed"])

    def test_check_run_needs_only_swanlab_key_and_never_calls_model(self):
        with patch.dict(os.environ, {"SWANLAB_API_KEY": "private-swanlab-test-key"}, clear=True), \
                patch.dict(sys.modules, {"swanlab": self.swanlab}), contextlib.redirect_stdout(io.StringIO()) as output:
            code = live.main(["--env-file", "missing.env", "--check-run", "test-user/gas-b/example"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["state"], "RUNNING")
        self.call_model.assert_not_called()
        self.ensure_model.assert_not_called()
        self.swanlab.init.assert_not_called()

    def test_existing_output_is_not_overwritten(self):
        with patch.dict(sys.modules, {"swanlab": self.swanlab}), \
                patch.object(live.importlib.util, "find_spec", return_value=object()), \
                self.assertRaises(FileExistsError):
            live.main(["--env-file", "missing.env", "--output", str(self.output)])
        self.ensure_model.assert_not_called()

    def test_main_success_saves_receipt_in_a_new_directory(self):
        target = self.output.parent / "new-check"
        with patch.dict(sys.modules, {"swanlab": self.swanlab}), \
                patch.object(live.importlib.util, "find_spec", return_value=object()), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = live.main(["--env-file", "missing.env", "--output", str(target)])
        self.assertEqual(code, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["outcome"], "succeeded")
        self.assertTrue((target / "receipt.json").is_file())
        self.call_model.assert_called_once()

    def test_atomic_receipt_failure_preserves_old_copy(self):
        path = self.output / "receipt.json"
        live.write_receipt(path, {"original": True})
        with patch.object(live.os, "replace", side_effect=OSError("Injected replace failure")), self.assertRaises(OSError):
            live.write_receipt(path, {"original": False})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"original": True})
        self.assertEqual(list(self.output.glob(".receipt-*.tmp")), [])

    def test_graceful_termination_reaches_cleanup_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        def terminate(*args):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.call_model.side_effect = terminate
        with self.assertRaises(SystemExit):
            self.run_check()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(self.read_receipt()["cloud_verification"]["state"], "ABORTED")


if __name__ == "__main__":
    unittest.main()
