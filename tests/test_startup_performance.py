"""启动/输入优化的合同回归，不使用 Docker、模型服务或云端。"""
import contextlib
import io
import json
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from gas_a.regression.data import BEARING_FEATURES, input_values, schema
from gas_a.sensor_training.reference.data import flatten
from gas_a.progress import stage
from gas_a import submission


class StartupPerformanceTest(unittest.TestCase):
    def test_harness_is_the_unmodified_sdk_template(self):
        template = ROOT / 'vendor/contest-harness-sdk-1.0.4/template/harness.py'
        self.assertEqual((ROOT / 'harness.py').read_bytes(), template.read_bytes())

    def test_sdk_describe_does_not_import_numpy_or_business_modules(self):
        # 阻止重模块导入，确保轻量 describe 是结构保证，而非依赖本机快慢的秒数断言。
        code = """
import importlib.abc, runpy, sys
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith(('numpy.', 'gas_a')):
            raise AssertionError('describe imported heavy module: ' + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[1] + '/harness.py', 'describe']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        r = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code, str(ROOT)],
                           capture_output=True, text=True, encoding='utf-8', timeout=10)
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertEqual(result['tasks'], ['classification', 'regression'])
        self.assertEqual(result['sdk_version'], '1.0.4')
        self.assertEqual(len(r.stdout.splitlines()), 1)

    def test_describe_replies_with_stdin_still_open(self):
        p = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'harness.py'), 'describe'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding='utf-8')
        lines = queue.Queue()
        reader = threading.Thread(target=lambda: lines.put(p.stdout.readline()), daemon=True)
        reader.start()
        try:
            response = json.loads(lines.get(timeout=10))
            self.assertFalse(p.stdin.closed)
            self.assertEqual(response['protocol'], 'contest.harness.workspace.v2')
            self.assertEqual(p.wait(timeout=10), 0)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)
            reader.join(timeout=2)
            for stream in (p.stdin, p.stdout, p.stderr):
                stream.close()

    def test_fast_regression_inputs_match_official_flatten_with_shuffled_keys(self):
        rng = random.Random(20261124)
        for _ in range(100):
            names = list(BEARING_FEATURES)
            rng.shuffle(names)
            b = {'features': {channel: {name: rng.uniform(-1e6, 1e6) for name in names}
                              for channel in ('vertical', 'horizontal')}}
            c = {'settings': [rng.randrange(-10000, 10000) for _ in range(3)],
                 'sensors': [rng.uniform(-1e6, 1e6) for _ in range(21)]}
            for dataset, inputs in (('B', b), ('C', c)):
                with self.subTest(dataset=dataset):
                    self.assertEqual(input_values(inputs, dataset), (dataset, *flatten(inputs)))
                    self.assertEqual(input_values(inputs), (dataset, *flatten(inputs)))

    def test_fast_inputs_keep_scalar_and_extra_field_rejection(self):
        for bad in (True, float('nan'), float('inf'), '1', None, [], {}, 10**1000):
            inputs = {'sensors': [0.0] * 21, 'settings': [0.0] * 3}
            inputs['sensors'][0] = bad
            with self.subTest(value_type=type(bad).__name__), self.assertRaises(ValueError):
                input_values(inputs, 'C')
        for dataset in ('B', 'C'):
            inputs = ({'features': {c: dict.fromkeys(BEARING_FEATURES, 0) for c in ('horizontal', 'vertical')}}
                      if dataset == 'B' else {'sensors': [0] * 21, 'settings': [0] * 3})
            inputs['answer'] = 1
            with self.assertRaises(ValueError):
                input_values(inputs, dataset)

    def test_schema_result_mutation_does_not_poison_cached_schema(self):
        original = schema('C')
        schema('C')[0] = 'wrong'
        self.assertEqual(schema('C'), original)

    def test_readonly_model_mount_reuses_resolved_file_identity(self):
        sample = SimpleNamespace(context={'artifact_id': 'version-1'})
        info = SimpleNamespace(st_mtime_ns=1, st_ctime_ns=2, st_size=3)
        with patch.dict(submission._READONLY_CHECKPOINTS, {}, clear=True), \
                patch.object(submission, '_selected_checkpoint', return_value=Path('/model/checkpoint.npz')) as resolve, \
                patch.object(Path, 'stat', return_value=info) as stat_file, \
                patch.object(submission.os, 'statvfs', return_value=SimpleNamespace(f_flag=1), create=True):
            first = submission._prediction_checkpoint(sample, '/model')
            second = submission._prediction_checkpoint(sample, '/model')
        self.assertEqual(first, second)
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(stat_file.call_count, 1)

    def test_writable_model_still_checks_identity_for_each_sample(self):
        sample = SimpleNamespace(context={'artifact_id': 'version-1'})
        values = [SimpleNamespace(st_mtime_ns=v, st_ctime_ns=v, st_size=3) for v in (1, 2)]
        with patch.dict(submission._READONLY_CHECKPOINTS, {}, clear=True), \
                patch.object(submission, '_selected_checkpoint', return_value=Path('/model/checkpoint.npz')) as resolve, \
                patch.object(Path, 'stat', side_effect=values), \
                patch.object(submission.os, 'statvfs', return_value=SimpleNamespace(f_flag=0), create=True):
            first = submission._prediction_checkpoint(sample, '/model')
            second = submission._prediction_checkpoint(sample, '/model')
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(resolve.call_count, 2)

    def test_changed_artifact_context_does_not_reuse_readonly_selection(self):
        sample = SimpleNamespace(context={'artifact_id': 'version-1'})
        info = SimpleNamespace(st_mtime_ns=1, st_ctime_ns=2, st_size=3)
        with patch.dict(submission._READONLY_CHECKPOINTS, {}, clear=True), \
                patch.object(submission, '_selected_checkpoint', return_value=Path('/model/checkpoint.npz')) as resolve, \
                patch.object(Path, 'stat', return_value=info), \
                patch.object(submission.os, 'statvfs', return_value=SimpleNamespace(f_flag=1), create=True):
            submission._prediction_checkpoint(sample, '/model')
            sample.context = {'artifact_id': 'version-2'}
            submission._prediction_checkpoint(sample, '/model')
        self.assertEqual(resolve.call_count, 2)

    def test_stage_logging_preserves_interruption_and_keeps_stdout_clean(self):
        out, err = io.StringIO(), io.StringIO()
        interrupted = KeyboardInterrupt()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(KeyboardInterrupt) as raised:
                with stage('model_rpc', phase='plan', attempt=1):
                    raise interrupted
        self.assertIs(raised.exception, interrupted)
        self.assertEqual(out.getvalue(), '')
        records = [json.loads(line.removeprefix('[gas-rsi] ')) for line in err.getvalue().splitlines()]
        self.assertEqual([r['event'] for r in records], ['stage_started', 'stage_ended'])
        self.assertFalse(records[-1]['completed'])


if __name__ == '__main__':
    unittest.main()
