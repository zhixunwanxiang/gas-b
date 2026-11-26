"""Resource accounting, deterministic rejection identity and asymmetric gradients."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
import numpy as np

from tests.test_submission import ROOT
from tests.test_regression import make_fixture
from gas_a.progress import resource_scope, stage, record_model_receipt, save_resources
from gas_a.evolution import REPEAT_POLICY, available_experiments, fingerprint, experiment_evidence
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.training import fit_epoch
from gas_a.regression.submission import dataset_config, validate_config


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

    def test_underestimation_gradient_and_default_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = make_fixture(root, 'B')
            train, _ = load_pair(root / 'data/train.jsonl', root / 'data/validation.jsonl', 'B')
            model = load_model(path, 'B')
            model.meta['batch_size'] = len(train['x'])
            original = deepcopy(model)
            spec = dict(name='test', loss='mse', learning_rate=.001, epochs=1, freeze_features=True)
            default, explicit = deepcopy(model), deepcopy(model)
            fit_epoch(default, train, spec, time.monotonic() + 10)
            fit_epoch(explicit, train, {**spec, 'underestimate_weight': 1.0}, time.monotonic() + 10)
            for group in ('params', 'first', 'second'):
                for name in model.params:
                    np.testing.assert_array_equal(getattr(default, group)[name], getattr(explicit, group)[name])
            x = model.normalized(train['x'])
            hidden, output = model.forward(x)
            error = output[:, 0] - (train['y'] - model.meta['target_mean']) / model.meta['target_scale']
            self.assertTrue(np.any(error < 0) and np.any(error > 0))
            delta = (2 * error * np.where(error < 0, 1.25, 1) / len(x))[:, None]
            gradients = {'w2': hidden.T @ delta, 'b2': delta.sum(axis=0)}
            norm = np.sqrt(sum(np.sum(g * g) for g in gradients.values()))
            fit_epoch(model, train, {**spec, 'underestimate_weight': 1.25}, time.monotonic() + 10)
            for name in gradients:
                expected = .9 * original.first[name] + .1 * gradients[name] * min(1, 5 / max(norm, 1e-12))
                np.testing.assert_allclose(model.first[name], expected, rtol=1e-13, atol=1e-14)
            np.testing.assert_array_equal(model.params['w1'], original.params['w1'])
            np.testing.assert_array_equal(model.first['w1'], original.first['w1'])
            self.assertGreater(model.meta['steps'], original.meta['steps'])

    def test_weight_is_bounded_and_b_only(self):
        config = json.loads((ROOT / 'submission_config.json').read_text(encoding='utf-8'))
        b, c = dataset_config(config, 'B'), dataset_config(config, 'C')
        self.assertIn('underestimate_mild', {spec['name'] for spec in b['experiments']})
        self.assertNotIn('underestimate_mild', {spec['name'] for spec in c['experiments']})
        for weight in (True, float('nan'), .5, 1.51):
            bad = deepcopy(b)
            bad['experiments'][-1]['underestimate_weight'] = weight
            with self.assertRaises(ValueError):
                validate_config(bad, bad['limits'])
