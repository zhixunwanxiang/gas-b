"""Resource accounting, deterministic rejection identity and asymmetric gradients."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
import contest_submission  # Initialize the shared package path for standalone test runs.

from gas_a.progress import resource_scope, stage, record_model_receipt, save_resources
from gas_a.evolution import REPEAT_POLICY, available_experiments, fingerprint, experiment_evidence


class ItemTuningTest(unittest.TestCase):
    def test_unknown_token_usage_is_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            with resource_scope(dataset='B', round_index=1) as ledger:
                with stage('local_training'):
                    sum(i * i for i in range(1000))
                record_model_receipt(SimpleNamespace(call_id='known', response={
                    'usage': {'prompt_tokens': 25, 'completion_tokens': 7}}))
                record_model_receipt(SimpleNamespace(call_id='unknown', response={}))
                save_resources(Path(directory))
                report = json.loads((Path(directory) / 'resources.json').read_text())
                self.assertIsNone(report['token_totals'])
                self.assertEqual(report['known_token_subtotal'], {'input_tokens': 25, 'output_tokens': 7})
                self.assertEqual(report['missing_usage_calls'], 1)
                self.assertGreaterEqual(report['cpu_seconds'], 0)
                self.assertEqual(report['stages'][0]['stage'], 'local_training')
                self.assertEqual(report['cutoff'], 'before_save_state')

    def test_failed_stage_remains_failed_and_exception_propagates(self):
        with redirect_stderr(io.StringIO()), resource_scope() as ledger:
            with self.assertRaisesRegex(RuntimeError, 'unknown RPC outcome'):
                with stage('model_rpc'):
                    raise RuntimeError('unknown RPC outcome')
            self.assertFalse(ledger.stages[0]['completed'])

    def test_only_identical_completed_performance_failure_is_skipped(self):
        menu = {'test': {'name': 'test'}}
        execution = {'repeat_policy': REPEAT_POLICY}
        rnd = SimpleNamespace(parent=SimpleNamespace(checkpoint_sha256='parent'))
        train, validation = {'sha256': 'train'}, {'sha256': 'validation'}
        rules = {'max_repeat_rejections': 2}
        identity = fingerprint('parent', 'train', 'validation', menu['test'], execution)
        for outcome in ('performance_rejected', 'local_training_failed', 'measurement_unverified', 'controller_guard', 'not_selected'):
            policy = {'attempts': [{'fingerprint': identity, 'adopted': False, 'outcome': outcome}]}
            allowed, blocked = available_experiments(policy, menu, rnd, train, validation, rules, execution)
            self.assertEqual(bool(blocked), outcome == 'performance_rejected')
            evidence = experiment_evidence([], policy['attempts'], menu, 'parent', 'train', 'validation', execution, rules, blocked=blocked)
            if blocked:
                self.assertEqual(evidence['test']['remaining_rejection_budget'], 0)
            allowed, _ = available_experiments(policy, menu, rnd, train, {'sha256': 'new-stage'}, rules, execution)
            self.assertIn('test', allowed)
            new_parent = SimpleNamespace(parent=SimpleNamespace(checkpoint_sha256='new-parent'))
            allowed, _ = available_experiments(policy, menu, new_parent, train, validation, rules, execution)
            self.assertIn('test', allowed)
