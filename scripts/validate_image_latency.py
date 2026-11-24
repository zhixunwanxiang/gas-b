"""禁网镜像延迟核验：保持 stdin 打开测首帧；可核对既有公开预测，全程不调用模型。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.docker_rehearsal import DockerRuntime, child_environment

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--expected-image-id', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--recorded-run', type=Path, help='完整历史证据根目录；仅消费公开输入和已保存输出')
parser.add_argument('--docker-executable', default='docker')
parser.add_argument('--docker-context', default='desktop-linux')
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                        expected_id=args.expected_image_id, required_tasks=('classification', 'regression'))
results = {'image': args.image, 'new_real_model_calls': 0, 'probes': []}
results['preflight'] = runtime.preflight()


def save():
    (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

def stream_probe(kind, argv, requests):
    started = time.perf_counter()
    name = argv[argv.index('--name') + 1]
    with (out / (kind + '-stderr.log')).open('w', encoding='utf-8') as error:
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error,
                             text=True, encoding='utf-8', env=child_environment())
        lines = queue.Queue()
        def read():
            for line in p.stdout:
                lines.put(line)
            lines.put(None)
        t = threading.Thread(target=read, daemon=True)
        t.start()
        latencies, outputs = [], []
        try:
            for value in requests:
                tick = time.perf_counter()
                if value is not None:
                    p.stdin.write(json.dumps(value) + '\n')
                    p.stdin.flush()
                line = lines.get(timeout=30)
                assert line is not None, 'Process exited without response'
                latencies.append(time.perf_counter() - tick)
                outputs.append(json.loads(line))
            # 请求下一条和 EOF 前必须已收到本条输出。
            p.stdin.close()
            assert p.wait(timeout=15) == 0
            t.join(timeout=2)
            assert lines.get(timeout=2) is None, 'Unexpected extra protocol output'
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=10)
            for s in (p.stdin, p.stdout):
                s.close()
            runtime.cleanup(name)
    row = {'kind': kind, 'first_response_seconds': latencies[0],
           'max_following_response_seconds': max(latencies[1:], default=0),
           'total_seconds': time.perf_counter() - started, 'outputs': len(outputs),
           'response_before_eof': True}
    results['probes'].append(row)
    (out / (kind + '-outputs.json')).write_text(json.dumps(outputs, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(row), flush=True)
    save()

for kind, command in [('default', []), ('describe', ['describe'])]:
    argv, _ = runtime._base(kind)
    stream_probe(kind, [*argv, runtime.image, *command], [None])
argv, _ = runtime._base('canonical')
stream_probe('canonical-describe', [*argv, '--entrypoint', 'python3', runtime.image, '/app/harness.py', 'describe'], [None])

if args.recorded_run is None:
    results.update(verified=True, prediction_tested=False)
    save()
    raise SystemExit(0)
run = args.recorded_run.resolve() / 'training'
for dataset in 'ABC':
    source = run / dataset / 'normal'
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    selected = report['selected_artifact_id']
    directory = (source / 'initial' if selected == 'initial' else source / ('host-' + selected.split('-')[-1]) / 'candidate')
    if selected == 'initial':
        descriptor = {'checkpoint_paths': ['checkpoint.npz'], 'code_paths': [],
                      'predict_argv': ['python3', '/app/harness.py', 'predict']}
    else:
        protocol = directory.parent / 'protocol.jsonl'
        frames = [json.loads(line)['frame'] for line in protocol.read_text(encoding='utf-8').splitlines() if line.strip()]
        descriptor = next(frame['artifact'] for frame in frames if frame.get('type') == 'candidate_request')
    rows = [json.loads(line) for line in (source / 'data/validation.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    task = 'classification' if dataset == 'A' else 'regression'
    context = {'protocol': 'contest.predict-context.v1', 'dataset_key': dataset, 'task': task,
               'artifact_id': selected, 'artifact': descriptor}
    requests = [{'task': task, 'inputs': row['inputs']} for row in rows]
    argv = runtime.predict(directory, directory / descriptor['checkpoint_paths'][0], out, context=context)
    stream_probe(dataset + '-interactive', argv, requests[:10])
    argv = runtime.predict(directory, directory / descriptor['checkpoint_paths'][0], out, context=context)
    started = time.perf_counter()
    try:
        result = subprocess.run(argv, input=''.join(json.dumps(r) + '\n' for r in requests),
                                text=True, encoding='utf-8', env=child_environment(), capture_output=True, timeout=150)
        elapsed = time.perf_counter() - started
        assert result.returncode == 0, result.stderr[-2000:]
        # observe/rollback 轮不一定提交候选，不能假定每轮恰有两次测量。
        # 现有验收器每轮最后都会测选中版本，因此编号最大的测量就是最终选中产物。
        original = max(source.glob('measurement-*'), key=lambda p: int(p.name.split('-')[-1])) / 'predictions.jsonl'
        expected = original.read_text(encoding='utf-8')
        assert result.stdout == expected, 'Prediction bytes differ from original selected model'
        row = {'kind': dataset + '-full-public-validation', 'rows': len(requests), 'seconds': elapsed,
               'same_predictions_as_original_real_test': True,
               'prediction_sha256': hashlib.sha256(result.stdout.encode()).hexdigest()}
        results['probes'].append(row)
        print(json.dumps(row), flush=True)
        save()
    finally:
        runtime.cleanup_prediction(argv)

results.update(verified=True, prediction_tested=True, source_run=str(args.recorded_run.resolve()))
save()
