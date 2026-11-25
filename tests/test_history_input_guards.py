"""坏历史提示、数值及路由结构应被隔离；不允许替换损坏的实际父模型。"""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import contest_submission
from contest_harness.round import Round
from gas_a.frozen_router import FORMAT, COMPONENTS, load_predictor
from gas_a.sensor_training.io import CLASSES, digest
from gas_a.sensor_training.reference.model import Model
from tests.test_submission import ROOT, OfflineHost, writes_only_in
from tests.test_regression import RegressionHost, make_fixture


def rewrite_metadata(source, target, value):
    """直接构造损坏文件，避免用会校验输入的 save() 掩盖加载器缺口。"""
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    buffer = io.BytesIO()
    np.save(buffer, np.array(json.dumps(value)), allow_pickle=False)
    entries['metadata.npy'] = buffer.getvalue()
    with zipfile.ZipFile(target, 'w') as archive:
        for name, body in entries.items():
            archive.writestr(name, body)


class HistoryInputGuardsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='history-input-guards-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, name, dataset):
        root = self.root / name
        if dataset == 'A':
            shutil.copytree(ROOT / 'data/public', root / 'data')
            (root / 'previous').mkdir()
            parent = root / 'previous/checkpoint.npz'
            shutil.copyfile(ROOT / 'models/reference.npz', parent)
        else:
            parent = make_fixture(root, dataset)
        return root, parent

    def host(self, root, parent, dataset):
        host = OfflineHost(root, parent) if dataset == 'A' else RegressionHost(root, parent, dataset)
        host.action = 'observe'
        return host

    def run_host(self, host):
        rnd = Round.start(host, host)
        rnd.parent.root = host.parent.parent
        work = host.root / 'work'
        with patch.dict(os.environ, {'CONTEST_WORK': str(work)}), writes_only_in(work, host.store), \
                contextlib.redirect_stderr(io.StringIO()):
            contest_submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertFalse(host.pending)
        self.assertEqual(host.record['model_call_ids'], host.calls)
        return json.loads((host.saved_state / 'decisions.jsonl').read_text(encoding='utf-8').splitlines()[-1])

    def add_history(self, host, identifier, folder):
        # SDK 只负责恢复文件和校验摘要，不要求测试宿主先成功加载一份坏模型。
        with patch.object(host, 'checkpoint_paths', return_value=['checkpoint.npz']):
            host.add_artifact(identifier, folder)

    def test_invalid_optional_history_id_does_not_hide_an_authorized_rollback(self):
        for dataset in 'ABC':
            for index, hint in enumerate(([], {}, True, 0, '', ' ', None)):
                with self.subTest(dataset=dataset, hint=hint):
                    root, parent = self.fixture(dataset + str(index), dataset)
                    good = root / 'good'
                    good.mkdir()
                    shutil.copyfile(parent, good / 'checkpoint.npz')
                    model = Model.load(parent)
                    if dataset == 'A':
                        model.params['w2'][:] = 0
                        model.params['b2'][:] = 0
                    else:
                        model.params['b2'] += 100
                    parent.unlink()
                    model.save(parent)
                    original = digest(parent)
                    host = self.host(root, parent, dataset)
                    host.action = 'rollback'
                    host.history = [{'id': 'past', 'record': {'selected_artifact_id': hint}}]
                    self.add_history(host, 'good', good)
                    self.run_host(host)
                    self.assertEqual(host.record['selected_artifact_id'], 'good')
                    self.assertEqual(host.record['history_ids'], ['past'])
                    self.assertEqual(digest(parent), original)

    def test_oversized_checkpoint_scalars_are_format_errors_and_bad_history_is_skipped(self):
        for dataset in 'ABC':
            for field in ('learning_rate', 'target_mean', 'target_scale'):
                with self.subTest(dataset=dataset, field=field):
                    root, parent = self.fixture(dataset + field, dataset)
                    old = root / 'old'
                    old.mkdir()
                    meta = deepcopy(Model.load(parent).meta)
                    meta[field] = 10 ** 500
                    rewrite_metadata(parent, old / 'checkpoint.npz', meta)
                    with self.assertRaisesRegex(ValueError, field):
                        Model.load(old / 'checkpoint.npz')
                    host = self.host(root, parent, dataset)
                    self.add_history(host, 'bad-old', old)
                    record = self.run_host(host)
                    rejected = record['historical_validation']['bad-old']
                    self.assertFalse(rejected['gate']['eligible'])
                    self.assertEqual(rejected['error_type'], 'ValueError')
                    self.assertEqual(host.record['selected_artifact_id'], 'parent-1')

    def test_same_numeric_error_in_actual_parent_is_not_replaced_by_history(self):
        for dataset in 'ABC':
            root, good = self.fixture(dataset, dataset)
            broken = root / 'broken'
            broken.mkdir()
            meta = deepcopy(Model.load(good).meta)
            meta['target_scale'] = 10 ** 500
            parent = broken / 'checkpoint.npz'
            rewrite_metadata(good, parent, meta)
            host = self.host(root, parent, dataset)
            self.add_history(host, 'good', good.parent)
            with self.assertRaisesRegex(ValueError, 'target_scale'):
                self.run_host(host)
            self.assertEqual(host.calls, [])
            self.assertIsNone(host.record)

    def bundle(self, folder):
        folder.mkdir()
        parent = ROOT / 'models/reference.npz'
        model = Model.load(parent)
        model.meta['prediction_bundle'] = {'format': FORMAT, 'manifest': 'routing.json'}
        model.save(folder / 'checkpoint.npz')
        return {'format': FORMAT, 'classes': CLASSES,
                'components': {role: {} for role in COMPONENTS},
                'policy': {'inner_weight': .75, 'outer_weight': .6, 'active_pairs': [],
                           'cutoffs': {'0': {'cut': .5}, '1': {'cut': .5}}, 'feature_correction': False}}

    def invalid_manifests(self, manifest):
        yield 'null_manifest', None
        yield 'list_manifest', []
        for key, value in (('components', None), ('components', list(COMPONENTS)), ('policy', None), ('policy', [])):
            item = deepcopy(manifest)
            item[key] = value
            yield key + str(value), item
        for key, value in (('active_pairs', [[]]), ('active_pairs', [True]), ('inner_weight', 10**500),
                           ('cutoffs', None), ('cutoffs', [])):
            item = deepcopy(manifest)
            item['policy'][key] = value
            yield key + type(value).__name__, item
        for value in (None, {}, 'wrong', True, 10**500, float('nan'), 0, 1):
            item = deepcopy(manifest)
            item['policy']['cutoffs']['0']['cut'] = value
            yield 'cut' + type(value).__name__ + str(value)[:8], item
        item = deepcopy(manifest)
        item['components']['four_sensor'] = None
        yield 'component_null', item

    def test_invalid_bundle_shapes_raise_local_format_errors(self):
        folder = self.root / 'bundle'
        manifest = self.bundle(folder)
        for name, value in self.invalid_manifests(manifest):
            with self.subTest(case=name):
                (folder / 'routing.json').write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_predictor(folder / 'checkpoint.npz')

    def test_bad_route_history_is_skipped_and_error_is_recorded(self):
        for index, fault in enumerate(('null_manifest', 'null_policy', 'bad_cutoff')):
            root, parent = self.fixture(str(index), 'A')
            old = root / 'old'
            manifest = self.bundle(old)
            if fault == 'null_manifest':
                manifest = None
            elif fault == 'null_policy':
                manifest['policy'] = None
            else:
                manifest['policy']['cutoffs']['0']['cut'] = 'wrong'
            (old / 'routing.json').write_text(json.dumps(manifest), encoding='utf-8')
            host = self.host(root, parent, 'A')
            self.add_history(host, 'bad-route', old)
            record = self.run_host(host)
            rejected = record['historical_validation']['bad-route']
            self.assertFalse(rejected['gate']['eligible'])
            self.assertEqual(rejected['error_type'], 'ValueError')
            self.assertEqual(host.record['selected_artifact_id'], 'parent-1')


if __name__ == '__main__':
    unittest.main()
