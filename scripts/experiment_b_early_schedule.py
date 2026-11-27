"""Paired fresh B ten-round schedules with identical source and official inputs."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from scripts.validate_multitask import run_dataset
from gas_a.sensor_training.io import digest, write_json

SEQUENCES = {
    'guard_late': ['low_rate', 'standard_rate', 'head_only', 'robust_huber', 'bearing_balanced',
                   'bearing_log_ridge', 'bearing_balanced_half', 'bearing_rbf', 'bearing_mlp', 'bearing_upward_guard'],
    'guard_early': ['bearing_balanced', 'bearing_log_ridge', 'bearing_balanced_half', 'bearing_rbf', 'low_rate',
                    'bearing_upward_guard', 'standard_rate', 'head_only', 'robust_huber', 'bearing_stumps'],
    'ridge_start': ['bearing_log_ridge', 'bearing_balanced', 'bearing_balanced_half', 'bearing_rbf', 'low_rate',
                    'bearing_upward_guard', 'standard_rate', 'head_only', 'robust_huber', 'bearing_stumps'],
    'mlp_start': ['bearing_mlp', 'bearing_balanced', 'bearing_balanced_half', 'bearing_rbf', 'low_rate',
                  'bearing_upward_guard', 'standard_rate', 'head_only', 'robust_huber', 'bearing_stumps'],
    'stumps_start': ['bearing_stumps', 'bearing_balanced', 'bearing_balanced_half', 'bearing_rbf', 'low_rate',
                     'bearing_upward_guard', 'standard_rate', 'head_only', 'robust_huber', 'bearing_mlp'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', choices=(*SEQUENCES, 'configured_recommendation'), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [*sorted((ROOT/'src').rglob('*.py')), ROOT/'submission_config.json',
             ROOT/'scripts/validate_multitask.py', Path(__file__).resolve()]
    versions = {str(p.relative_to(ROOT)): digest(p) for p in files}
    base = args.baseline_root/'cpu-b-rul-20261110-v1'
    inputs = {name: digest(base/name) for name in ['checkpoint.npz', 'public/train.jsonl', 'public/validation.jsonl']}
    sequence = SEQUENCES.get(args.sequence)
    write_json(args.output/'registration.json', {'source_hashes': versions, 'input_hashes': inputs,
        'sequence': sequence, 'follow_menu_priority': sequence is None, 'initial_state': 'official_checkpoint_and_empty_state',
        'controller': 'explicit_offline_stub', 'real_model_calls': 0, 'official_score_verified': False})
    report = run_dataset(base, args.output/'B', 'B', 10, None, max_rounds=10,
                         compact_validation=True, aggregate_parent=True,
                         experiment_sequence=sequence, follow_menu_priority=sequence is None,
                         exchange_options={'round_timeout': 120})
    assert all(digest(ROOT/name) == value for name, value in versions.items())
    assert len(report['rounds']) == 10 and all(r['finish_acknowledged'] for r in report['rounds'])
    baseline = report['baseline']['rmse']
    result = {'final': report['final'], 'mean_rmse': statistics.mean(r['metrics']['rmse'] for r in report['rounds']),
        'mean_mae': statistics.mean(r['metrics']['mae'] for r in report['rounds']),
        'first_adoption_round': next((r['round'] for r in report['rounds'] if r['record']['configuration'].get('experiment_outcome', {}).get('outcome') == 'adopted'), None),
        'all_ten_rounds_acknowledged': True, 'source_unchanged': True,
        'selected_rounds': [(r['round'], r['selected_artifact_id']) for r in report['rounds']]}
    result['mean_rmse_gain_percent'] = (1-result['mean_rmse']/baseline)*100
    for metric in ('rmse', 'mae'):
        worst = max(r['metrics'][metric] for r in report['rounds'])
        result[f'worst_{metric}'] = worst
        result[f'worst_{metric}_rounds'] = [r['round'] for r in report['rounds'] if r['metrics'][metric] == worst]
    write_json(args.output/'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'final'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
