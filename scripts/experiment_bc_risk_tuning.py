"""Fixed single-factor B/C risk experiments on authorized public projections.

Development validation is reused for comparison; this is not a hidden-test score.
No metadata enters prediction. Prototype options remain offline until accepted.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gas_a.regression import bearing_guard as bg, bounded_residual as cr
from gas_a.regression.data import load_model, load_pair
from gas_a.regression.evaluation import evaluate
from gas_a.regression.model import transformed
from gas_a.sensor_training.io import digest, write_json

B_OPTIONS = {
    'current': {},
    'upward_cutoff_half': {'upward_cutoff': 115.},
    'no_upward': {'upward_cutoff': 0.},
    'signed_downward': {'downward_floor': .25},
    'half_upward_signed_downward': {'upward_cutoff': 115., 'downward_floor': .25},
}
C_OPTIONS = {
    'current': {},
    'high_shrink_half': {'high_shrink': .10},
    'no_high_shrink': {'high_shrink': 0.},
    'train_tail_weight_1_5': {'tail_weight': 1.5},
    'train_tail_weight_2': {'tail_weight': 2.},
}
B_EXTRA = {
    'short_overestimate_1_5': {'short_overestimate_weight': 1.5},
    'short_overestimate_2': {'short_overestimate_weight': 2.},
    'short_overestimate_2_half_upward': {'short_overestimate_weight': 2., 'upward_cutoff': 115.},
}
C_EXTRA = {
    'absolute_mix_0_25': {'absolute_mix': .25},
    'tail_1_5_absolute_mix_0_25': {'tail_weight': 1.5, 'absolute_mix': .25},
}
C_DIRECTIONAL = {
    'upward_cap_55': {'upward_cap': 55.},
    'upward_cap_70': {'upward_cap': 70.},
    'tail_1_5_downward_cap_55': {'tail_weight': 1.5, 'downward_cap': 55.},
}


def fit_weighted_b(model, train, boost):
    head = deepcopy(model.meta['bounded_bearing'])
    anchor = model.predict_anchor(train['x'])
    features = transformed(train['x'], 'bearing_energy', {})
    x = (np.column_stack([features, anchor])-head['mean'])/head['scale']
    y = (train['y']-anchor)/bg.TARGET_SCALE
    rng = np.random.RandomState(42)
    weights, biases = [], []
    for fan_in, fan_out in zip(bg.WIDTHS, bg.WIDTHS[1:]):
        radius = np.sqrt(6/(fan_in+fan_out))
        weights.append(rng.uniform(-radius, radius, (fan_in, fan_out)))
        biases.append(rng.uniform(-radius, radius, fan_out))
    params = weights+biases
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    order = np.arange(len(x))
    best_loss, stale, steps = np.inf, 0, 0
    for epoch in range(1, bg.MAX_EPOCHS+1):
        rng.shuffle(order)
        total_loss = 0.
        for start in range(0, len(order), 256):
            rows = order[start:start+256]
            values = bg._activations(x[rows], weights, biases)
            error = values[-1]-y[rows, None]
            weight = np.where((train['y'][rows, None] <= bg.LOW_CUTOFF) & (error > 0), boost, 1.)
            total_loss += .5*np.sum(weight*error**2)+.02*sum(np.dot(w.ravel(), w.ravel()) for w in weights)
            error *= weight
            dw, db = [None]*len(weights), [None]*len(weights)
            for layer in reversed(range(len(weights))):
                dw[layer] = (values[layer].T@error+.04*weights[layer])/len(rows)
                db[layer] = error.sum(axis=0)/len(rows)
                if layer:
                    error = error@weights[layer].T
                    error[values[layer] <= 0] = 0
            steps += 1
            rate = .0015*np.sqrt(1-.999**steps)/(1-.9**steps)
            for p, m, v, grad in zip(params, first, second, dw+db):
                m *= .9
                m += .1*grad
                v *= .999
                v += .001*grad**2
                p -= rate*m/(np.sqrt(v)+1e-8)
        loss = total_loss/len(x)
        stale = stale+1 if loss > best_loss-1e-4 else 0
        best_loss = min(best_loss, loss)
        if stale > 10:
            break
    head.update(weights=[v.tolist() for v in weights], biases=[v.tolist() for v in biases], epochs=epoch, steps=steps)
    child = deepcopy(model)
    child.meta['bounded_bearing'] = head
    child.validate()
    return child


def b_prediction(model, x, options):
    head = model.meta['bounded_bearing']
    mean, scale, support, weights, biases = model._bounded_parameters
    anchor = model.predict_anchor(x)
    features = transformed(x, 'bearing_energy', {})
    inputs = (np.column_stack([features, anchor]) - mean) / scale
    residual = bg._activations(inputs, weights, biases)[-1][:, 0] * bg.TARGET_SCALE
    delta = bg.MAX_CORRECTION * np.tanh(residual / bg.MAX_CORRECTION)
    cutoff = options.get('upward_cutoff', bg.LOW_CUTOFF)
    upward_weight = np.clip((cutoff-anchor)/max(cutoff/2, 1), 0, 1)
    downward_weight = np.clip((anchor-bg.LOW_CUTOFF)/(bg.HIGH_CUTOFF-bg.LOW_CUTOFF),
                              options.get('downward_floor', 0.), 1.)
    correction = np.maximum(delta, 0)*upward_weight + np.minimum(
        np.minimum(delta, 0)*downward_weight,
        -np.clip(.20*(anchor-bg.HIGH_CUTOFF), 0, bg.HIGH_REDUCTION_LIMIT))
    return np.maximum(0, anchor + head['strength'] * bg.support_weight(features, support)*correction)


def c_predict(x, anchor, head, high_shrink=.20, upward_cap=40., downward_cap=40.):
    stats, weights, biases = cr.deserialize(head)
    raw = cr._forward(cr._inputs(x, anchor, stats), weights, biases)*head['target_scale']
    cap = np.where(raw >= 0, upward_cap, downward_cap)
    delta = cap*np.tanh(raw/cap)
    lower, upper = head['lower'], head['upper']
    upward = np.clip((upper-anchor)/(upper-lower), 0, 1)
    adjusted = anchor + np.minimum(delta, 0) + upward*np.maximum(delta, 0)
    adjusted -= high_shrink*np.maximum(adjusted-lower, 0)
    return np.maximum(0, anchor+head['strength']*(adjusted-anchor))


def fit_weighted_c(x, target, anchor, train_sha, seed, tail_weight, absolute_mix=0.):
    """Same initialization, steps and inference as the current C head; weighted train MSE only."""
    started = time.monotonic()
    stats = cr._base_statistics(x)
    values = np.column_stack([cr._condition_features(x, stats), anchor])
    stats.update(mean=values.mean(axis=0), scale=np.maximum(values.std(axis=0), .005))
    z = np.clip((values-stats['mean'])/stats['scale'], -10, 10).astype(np.float32)
    residual = ((target-anchor)/84).astype(np.float32)[:, None]
    cutoff = np.quantile(target, .90)
    sample_weight = np.where(target >= cutoff, tail_weight, 1.).astype(np.float32)
    sample_weight /= sample_weight.mean()
    rng_init = np.random.default_rng(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(cr.WIDTHS, cr.WIDTHS[1:]):
        radius = 1/math.sqrt(fan_in)
        weights.append(rng_init.uniform(-radius, radius, (fan_in, fan_out)).astype(np.float32))
        biases.append(rng_init.uniform(-radius, radius, fan_out).astype(np.float32))
    weights[-1].fill(0)
    biases[-1].fill(0)
    params = weights+biases
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(seed)
    step = 0
    lr = .0015
    for _ in range(cr.EPOCHS):
        order = rng.permutation(len(z))
        for ii in np.array_split(order, (len(order)+511)//512):
            if time.monotonic()-started > 30:
                raise TimeoutError('Weighted C fit exceeded 30 seconds')
            activations, slopes = [z[ii]], []
            for weight, bias in zip(weights[:-1], biases[:-1]):
                value = activations[-1]@weight+bias
                sigmoid = 1/(1+np.exp(-np.clip(value, -75, 75)))
                activations.append(value*sigmoid)
                slopes.append(sigmoid*(1+value*(1-sigmoid)))
            error = activations[-1]@weights[-1]+biases[-1]-residual[ii]
            delta = (2*error+absolute_mix*error/np.sqrt(error**2+.01))/len(ii)
            delta *= sample_weight[ii, None]
            dw, db = [None]*len(weights), [None]*len(weights)
            for j in reversed(range(len(weights))):
                dw[j] = activations[j].T@delta
                db[j] = delta.sum(axis=0)
                if j:
                    delta = (delta@weights[j].T)*slopes[j-1]
            step += 1
            for p, m, v, grad in zip(params, first, second, dw+db):
                m *= .9
                m += .1*grad
                v *= .999
                v += .001*grad**2
                p *= 1-lr*.005
                p -= (lr/(1-.9**step))*m/(np.sqrt(v/(1-.999**step))+1e-8)
    lower, upper = np.quantile(target, [.90, .97]).tolist()
    return {'format': cr.METHOD, 'train_sha256': train_sha, 'strength': 1., 'steps': step,
            'statistics': {key: value.tolist() for key, value in stats.items()},
            'weights': [w.tolist() for w in weights], 'biases': [b.tolist() for b in biases],
            'lower': lower, 'upper': max(lower+1, upper), 'target_scale': 84.}


def cmapss_final_cycles(root, train, validation):
    """Read one endpoint per public test engine, disjoint from both projections."""
    if any(set(data['metadata']['source_split']) != {'train'} for data in (train, validation)):
        raise ValueError('C-MAPSS original test check requires train-only projections')
    arrays, targets, sources, engines, hashes = [], [], [], [], {}
    for number in range(1, 5):
        source = f'FD{number:03d}'
        test_path, rul_path = (root / f'{prefix}_{source}.txt' for prefix in ('test', 'RUL'))
        hashes.update({p.name: digest(p) for p in (test_path, rul_path)})
        table = np.loadtxt(test_path, ndmin=2)
        labels = np.loadtxt(rul_path, ndmin=1)
        if table.shape[1] != 26 or not np.isfinite(table).all() or not np.isfinite(labels).all():
            raise ValueError(f'Invalid public C-MAPSS test data for {source}')
        units = np.unique(table[:, 0])
        if not np.array_equal(units, np.arange(1, len(labels)+1)):
            raise ValueError(f'Unexpected public C-MAPSS engine IDs for {source}')
        endpoints = np.array([block[np.argmax(block[:, 1])] for unit in units
                              for block in [table[table[:, 0] == unit]]])
        arrays.append(np.column_stack([endpoints[:, 5:], endpoints[:, 2:5]]))
        targets.append(labels)
        engines.extend(units.astype(int).tolist())
        sources.extend([source]*len(units))
    data = {'x': np.vstack(arrays), 'y': np.concatenate(targets), 'metadata': {
        'source_subset': sources, 'source_split': ['test']*len(sources), 'unit': engines}}
    return data, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices='BC', required=True)
    parser.add_argument('--cmapss-test-root', type=Path, help='Also evaluate independent C-MAPSS final cycles')
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--extended', action='store_true', help='Fixed follow-up train-loss options; new registration/output')
    variants.add_argument('--directional', action='store_true', help='Fixed C directional cap comparisons')
    args = parser.parse_args()
    if args.dataset != 'C' and (args.directional or args.cmapss_test_root):
        parser.error('C-MAPSS test and directional options require C')
    args.output.mkdir(parents=True, exist_ok=False)
    task = args.dataset
    base = args.baseline_root / ('cpu-b-rul-20261110-v1' if task == 'B' else 'cpu-c-guide-20261110-v1')
    train, validation = load_pair(base/'public/train.jsonl', base/'public/validation.jsonl', task)
    run = args.parent_run/task
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    number = int(report['selected_artifact_id'].split('-')[-1])
    checkpoint = run/f'host-{number}/candidate/checkpoint.npz'
    parent = load_model(checkpoint, task)
    options = B_OPTIONS if task == 'B' else C_OPTIONS
    if args.extended:
        options = {'current': {}, **(B_EXTRA if task == 'B' else C_EXTRA)}
    if args.directional:
        options = {'current': {}, **C_DIRECTIONAL}
    registration = {'dataset': task, 'scope': 'development_public_single_factor_prototypes',
        'parent_sha256': digest(checkpoint), 'script_sha256': digest(__file__),
        'train_sha256': train['sha256'], 'validation_sha256': validation['sha256'],
        'fixed_options': options, 'real_model_calls': 0, 'official_score_verified': False}
    test = None
    if args.cmapss_test_root:
        test, hashes = cmapss_final_cycles(args.cmapss_test_root, train, validation)
        registration['additional_public_test'] = {'rows': len(test['y']), 'sha256_by_file': hashes}
    write_json(args.output/'registration.json', registration)
    (args.output/'prototype-source.py').write_bytes(Path(__file__).read_bytes())
    results = {}
    for name, opts in options.items():
        started = time.monotonic()
        if task == 'B':
            model = fit_weighted_b(parent, train, opts['short_overestimate_weight']) if 'short_overestimate_weight' in opts else parent
            prediction = b_prediction(model, validation['x'], opts)
        else:
            head = parent.meta['bounded_cmapss']
            if 'tail_weight' in opts or 'absolute_mix' in opts:
                head = fit_weighted_c(train['x'], train['y'], parent.predict_anchor(train['x']),
                                      train['sha256'], parent.meta['seed'], opts.get('tail_weight', 1.), opts.get('absolute_mix', 0.))
                write_json(args.output/f'{name}-head.json', head)
            predictor_options = {'high_shrink': opts.get('high_shrink', .20),
                                 'upward_cap': opts.get('upward_cap', 40.), 'downward_cap': opts.get('downward_cap', 40.)}
            prediction = c_predict(validation['x'], parent.predict_anchor(validation['x']), head, **predictor_options)
        if name == 'current':
            np.testing.assert_allclose(prediction, parent.predict(validation['x']), atol=1e-9, rtol=1e-12)
        np.save(args.output/f'{name}-predictions.npy', prediction, allow_pickle=False)
        metrics = evaluate(None, validation, task, prediction=prediction)
        results[name] = {'options': opts, 'metrics': metrics, 'seconds': time.monotonic()-started}
        if test is not None:
            test_values = c_predict(test['x'], parent.predict_anchor(test['x']), head, **predictor_options)
            test_metrics = evaluate(None, test, task, prediction=test_values)
            test_metrics['scope'] = 'independent_public_cmapss_test_final_cycles'
            results[name]['independent_public_test'] = test_metrics
        write_json(args.output/'results.json', results)
        axis = 'device' if task == 'B' else 'source'
        print(json.dumps({'experiment': name, 'rmse': metrics['rmse'], 'mae': metrics['mae'],
            'nasa': metrics.get('nasa_score'), 'groups': {k: {m: v[m] for m in
            (('rmse', 'mae') if task == 'B' else ('rmse', 'mae', 'nasa_score'))}
            for k, v in metrics['group_metrics'][axis].items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
