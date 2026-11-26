"""Ten-round public SDK acceptance for v20; the controller is an explicit offline stub."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.validate_multitask import run_dataset, BASELINES
from scripts.docker_rehearsal import DockerRuntime
from gas_a.sensor_training.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image')
    parser.add_argument('--expected-image-id')
    parser.add_argument('--docker-executable', default='docker')
    args = parser.parse_args()
    config = json.loads((ROOT / 'submission_config.json').read_text(encoding='utf-8'))['regression']
    extras = config.get('dataset_experiments', {})
    if (not any(s['name'] == 'underestimate_mild' for s in extras.get('B', []))
            or not any(s['name'] == 'fine_rate' for s in extras.get('C', []))):
        parser.error('This historical v20 driver does not apply to restored V18 B/C; use scripts/validate_multitask.py.')
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    previous = {r['dataset']: r['final'] for r in reference['datasets']}
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context='desktop-linux',
                            expected_id=args.expected_image_id,
                            required_tasks=('classification', 'regression')) if args.image else None
    if runtime:
        write_json(args.output / 'image-preflight.json', runtime.preflight())
    sequences = {
        'A': ['frozen_router', 'frozen_router', 'conservative_head', 'frozen_router', 'conservative_head',
              'four_sensor', 'conservative_head', 'five_sensor', 'conservative_head', 'frozen_router'],
        'B': ['robust_huber', 'standard_rate', 'underestimate_mild', 'underestimate_mild', 'head_only',
              'standard_rate', 'underestimate_mild', 'low_rate', 'robust_huber', 'underestimate_mild'],
        'C': ['robust_huber', 'standard_rate', 'head_only', 'fine_rate', 'fine_rate',
              'low_rate', 'head_only', 'fine_rate', 'robust_huber', 'fine_rate'],
    }
    reports = [run_dataset(args.baseline_root / BASELINES[key], args.output / key, key, 10, runtime,
                           compact_validation=True, aggregate_parent=True, experiment_sequence=sequences[key])
               for key in ('A', 'B', 'C')]
    comparison = {r['dataset']: {'metric': 'macro_f1' if r['dataset'] == 'A' else 'rmse',
                  'v19': previous[r['dataset']]['macro_f1' if r['dataset'] == 'A' else 'rmse'],
                  'v20': r['final']['macro_f1' if r['dataset'] == 'A' else 'rmse']} for r in reports}
    checks = {'all_30_rounds_completed': all(len(r['rounds']) == 10 for r in reports),
              'all_finish_acknowledged': all(row['finish_acknowledged'] for r in reports for row in r['rounds']),
              'all_sdk_model_receipts_present': all(row['model_call_ids'] for r in reports for row in r['rounds']),
              'a_no_regression_vs_v19': comparison['A']['v20'] >= comparison['A']['v19'] - 1e-9,
              'b_better_than_v19': comparison['B']['v20'] < comparison['B']['v19'] - .0005 * comparison['B']['v19'],
              'c_no_regression_vs_v19': comparison['C']['v20'] <= comparison['C']['v19'] + 1e-9,
              'b_mild_loss_executed': any(row['record']['configuration'].get('experiment', {}).get('name') == 'underestimate_mild'
                                        for row in reports[1]['rounds'] if row['record']['configuration'].get('experiment'))}
    # 只有实际 SDK 保存的状态文件才算资源记录成功。
    saved = list(args.output.glob('*/host-*/state/resources.json'))
    checks['all_resource_ledgers_saved'] = len(saved) == 30
    audit = runtime.audit() if runtime else None
    if runtime:
        checks['container_isolation_verified'] = audit['container_policy_verified']
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
               'scope': 'fixed_initial_public_data_sdk_acceptance_with_offline_controller',
               'real_model_calls': 0, 'official_platform_score_verified': False,
               'comparison_vs_v19': comparison, 'runtime': audit or 'native',
               'datasets': [{'dataset': r['dataset'], 'baseline': r['baseline'], 'final': r['final'],
                             'completed_rounds': len(r['rounds']), 'selected_artifact_id': r['selected_artifact_id']}
                            for r in reports]}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
