"""Build auditable multi-round chart exports without training or modifying release images.

Prepare with the bundled Python runtime; render with a Python that has Matplotlib.
"""
import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {'v19': ROOT / 'runtime/first-four-v19-20261126/container-10',
         'v20': ROOT / 'runtime/item-tuning-v20-20261126/container-10'}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
CN = ['乙醇', '乙烯', '氨气', '乙醛', '丙酮', '甲苯']
CLASSES = ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene']
EXPERIMENTS = {'frozen_router': '路由', 'conservative_head': '冻结头', 'four_sensor': '4传感器',
               'five_sensor': '5传感器', 'standard_rate': '标准', 'low_rate': '低率',
               'robust_huber': 'Huber', 'head_only': '冻结头', 'fine_rate': '微调',
               'underestimate_mild': '低估加权'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def xlsx_cells(path):
    """Read original XLSX values only; no Excel execution or modification."""
    import posixpath
    with zipfile.ZipFile(path) as book:
        shared = []
        if 'xl/sharedStrings.xml' in book.namelist():
            shared = [''.join(si.itertext()) for si in ET.fromstring(book.read('xl/sharedStrings.xml'))]
        links = {node.attrib['Id']: node.attrib['Target'] for node in ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(book.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            identifier = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = links[identifier]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            cells = {}
            for cell in ET.fromstring(book.read(target)).findall('.//s:sheetData/s:row/s:c', NS):
                kind = cell.attrib.get('t')
                if kind == 'inlineStr':
                    value = ''.join(n.text or '' for n in cell.findall('.//s:t', NS))
                else:
                    node = cell.find('s:v', NS)
                    value = '' if node is None else node.text or ''
                    if kind == 's':
                        value = shared[int(value)]
                cells[cell.attrib['r']] = value
            yield sheet.attrib['name'], cells


def display_metrics(text):
    result = {}
    for line in text.splitlines():
        match = re.search(r'[：:]\s*([-+\d.,]+)', line)
        if not match:
            continue
        value = float(match.group(1).replace(',', ''))
        for word, key in [('Macro F1', 'macro_f1'), ('Accuracy', 'accuracy'), ('RMSE', 'rmse'),
                          ('MAE', 'mae'), ('NASA', 'nasa_score')]:
            if word in line:
                result[key] = value
                break
    return result


def load_official(path, kind):
    output = {}
    objective = None
    for sheet, cells in xlsx_cells(path):
        if sheet == '总成绩说明' and kind == 'formal':
            objective = cells['A1']
        if sheet not in ('项目A', '项目B', '项目C'):
            continue
        dataset = sheet[-1]
        rows = []
        addresses = sorted((key for key in cells if re.fullmatch(r'D\d+', key)), key=lambda k: int(k[1:]))
        for address in addresses:
            narrative = cells[address]
            marker = re.search(r'配置\s*\n\s*(?=\{)', narrative)
            if marker is None:
                continue
            config, _ = json.JSONDecoder().raw_decode(narrative[marker.end():])
            number = int(address[1:])
            label = cells[f'A{number}']
            round_index = int(re.search(r'\d+', label).group())
            shown = display_metrics(cells[f'B{number}'])
            adopted = config['selection_action'] == 'adopt_candidate'
            before = config.get('local_validation_before') if dataset == 'A' else config.get('validation_before')
            candidate = config.get('local_validation_after') if dataset == 'A' else config.get('validation_after')
            if dataset == 'A':
                metric = 'macro_f1'
                selected = {**shown, 'macro_f1': config['selected_local_macro_f1']}
                gain = None if candidate is None else candidate - before
            else:
                metric = 'rmse'
                selected = dict(candidate if adopted else before)
                selected['rmse'] = config['selected_local_rmse']
                gain = None if candidate is None else before['rmse'] - candidate['rmse']
            assert abs(selected[metric] - shown[metric]) < .000001, (kind, dataset, round_index)
            gate = config.get('candidate_gate')
            if gate is not None and gain is not None:
                assert abs(gain - gate['macro_f1_gain' if dataset == 'A' else 'rmse_reduction']) < 1e-9
            stage = re.search(r'[ABC]-stage-(\d+)', label)
            rows.append({'round': round_index, 'stage': stage.group(1) if stage else None,
                'selected': selected, 'parent': before, 'candidate': candidate, 'candidate_gain': gain,
                'adopted': adopted, 'action': config['action'], 'selection_action': config['selection_action'],
                'experiment': (config.get('experiment') or {}).get('name'), 'gate': gate,
                'source_sheet': sheet, 'source_row': number})
        assert [row['round'] for row in rows] == list(range(1, 11)), (kind, sheet)
        output[dataset] = rows
    assert set(output) == {'A', 'B', 'C'}
    return output, objective


def event_log(path, event):
    items = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('[gas-rsi] '):
            value = json.loads(line[len('[gas-rsi] '):])
            if value.get('event') == event:
                items.append(value)
    assert len(items) == 1, (str(path), event, len(items))
    return items[0]


def prepare(output, report_root):
    sources = []
    def source(path):
        path = Path(path)
        sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    local = {}
    for version, directory in LOCAL.items():
        local[version] = {}
        for dataset in ('A', 'B', 'C'):
            path = directory / dataset / 'report.json'
            source(path)
            report = read(path)
            metric = 'macro_f1' if dataset == 'A' else 'rmse'
            baseline = report['baseline']
            rows, previous, previous_id, measurement = [], baseline, 'initial', 1
            for row in report['rounds']:
                index, cfg = row['round'], row['record']['configuration']
                chosen = row['metrics']
                before = cfg['local_validation_before'] if dataset == 'A' else cfg['validation_before']['rmse']
                assert abs(before - previous[metric]) < 1e-9
                after = cfg.get('local_validation_after') if dataset == 'A' else (cfg.get('validation_after') or {}).get('rmse')
                gain = None if after is None else ((after - before) if dataset == 'A' else (before - after))
                if cfg.get('candidate_gate') is not None:
                    assert abs(gain - cfg['candidate_gate']['macro_f1_gain' if dataset == 'A' else 'rmse_reduction']) < 1e-9
                adopted = cfg['selection_action'] == 'adopt_candidate'
                if not adopted and row['selected_artifact_id'] == previous_id:
                    assert abs(chosen[metric] - previous[metric]) < 1e-9
                assert chosen[metric] >= previous[metric] - 1e-9 if dataset == 'A' else chosen[metric] <= previous[metric] + 1e-9
                value = {'round': index, 'selected': chosen, 'parent_primary': before,
                         'candidate_primary': after, 'candidate_gain': gain,
                         'gain_vs_initial': ((chosen[metric] - baseline[metric]) * 100 if dataset == 'A'
                            else (baseline[metric] - chosen[metric]) / baseline[metric] * 100),
                         'gain_unit': 'percentage_points' if dataset == 'A' else 'percent_RMSE_reduction',
                         'candidate_submitted': row['candidate_submitted'], 'adopted': adopted,
                         'selection_action': cfg['selection_action'], 'gate': cfg.get('candidate_gate'),
                         'experiment': (cfg.get('experiment') or {}).get('name'),
                         'validation_sha256': cfg['validation_sha256']}
                measurement += 1 + int(row['candidate_submitted'])
                if version == 'v20':
                    host = directory / dataset / f'host-{index}'
                    resource_path = host / 'state/resources.json'
                    source(resource_path)
                    ledger = read(resource_path)
                    terminal = event_log(host / 'stderr.log', 'round_resources')
                    source(host / 'stderr.log')
                    assert ledger['round_index'] == index and ledger['dataset'] == dataset
                    assert terminal['token_totals'] is None and not terminal['usage_complete']
                    stage_cpu = defaultdict(float)
                    for stage in ledger['stages']:
                        name = stage['stage']
                        group = ('读取数据' if name == 'load_public_data' else
                                 '训练' if name == 'local_training' else
                                 '历史读取与复评' if name in ('history_rpc', 'restore_and_evaluate_history') else
                                 '父模型评估与核验' if name in ('evaluate_parent', 'validate_parent_measurement') else
                                 '模型请求' if name == 'model_rpc' else
                                 '候选上传与测量' if name == 'submit_candidate_rpc' else '其他')
                        stage_cpu[group] += stage['cpu_seconds']
                    remainder = terminal['cpu_seconds'] - sum(stage_cpu.values())
                    assert remainder > -.005, (dataset, index, remainder)
                    stage_cpu['其他'] += max(0, remainder)
                    value['resources'] = {'cpu_seconds': terminal['cpu_seconds'], 'wall_seconds': terminal['wall_seconds'],
                        'peak_rss_mib': terminal['peak_rss_bytes'] / 1024 ** 2,
                        'model_calls': len(ledger['model_calls']), 'token_totals': None,
                        'stage_cpu_seconds': dict(stage_cpu), 'scope': 'participant_train_callback; excludes container startup'}
                    prediction = directory / dataset / f'measurement-{measurement}'
                    source(prediction / 'stderr.log')
                    source(prediction / 'timing.json')
                    perf = event_log(prediction / 'stderr.log', 'prediction_resources')
                    timing = read(prediction / 'timing.json')
                    assert perf['samples'] == chosen['rows'] == timing['rows']
                    value['prediction'] = {'cpu_ms_per_sample': perf['cpu_seconds'] / perf['samples'] * 1000,
                        'callback_samples_per_second': perf['samples'] / perf['wall_seconds'],
                        'process_seconds': timing['prediction_process_seconds'],
                        'scope': 'callback excludes SDK JSON/startup; process_seconds includes container launch/protocol'}
                rows.append(value)
                previous, previous_id = chosen, row['selected_artifact_id']
            assert len(rows) == 10
            assert len({row['validation_sha256'] for row in rows}) == 1
            local[version][dataset] = {'baseline': baseline, 'rounds': rows, 'final': report['final']}
    for dataset in ('A', 'B', 'C'):
        assert local['v19'][dataset]['rounds'][0]['validation_sha256'] == local['v20'][dataset]['rounds'][0]['validation_sha256']
    official = {}
    objective = None
    for kind, name in [('formal', '正式测评实验报告.xlsx'), ('trial', '实验报告记录.xlsx')]:
        path = report_root / name
        source(path)
        official[kind], score_text = load_official(path, kind)
        objective = score_text or objective
    categories = [('平均性能提升', 15), ('最差阶段表现', 5), ('恢复速度', 5), ('稳定性', 5), ('资源效率', 5)]
    scores = []
    for label, maximum in categories:
        match = re.search(re.escape(label) + r'\s*([\d.]+)\s*/\s*(\d+)', objective)
        assert match and int(match.group(2)) == maximum
        scores.append({'name': label, 'score': float(match.group(1)), 'maximum': maximum})
    total = float(re.search(r'客观成绩\s*·\s*([\d.]+)', objective).group(1))
    assert abs(total - sum(item['score'] for item in scores)) < .000002
    data = {'local': local, 'official': official, 'objective': {'total': total, 'maximum': 35, 'components': scores},
        'definitions': {'local': '固定初始公开投影，离线控制器，真实工业模型训练与预测。0为初始父模型。',
            'formal': '原始Excel报告，阶段会变化。阶段间绝对值变化不能当作同口径提升。',
            'candidate_gain': 'A为候选F1减当轮父F1；B/C为当轮父RMSE减候选RMSE。正数表示改善，未必通过全部门槛。',
            'deployed_gain': '只反映实际选中版本；未采用候选不能计入上线收益。',
            'resources': 'v20实际镜像回调观测；非多次重复基准，token未知不记0，不能据此换算正式资源分。'},
        'sources': sources,
        'checks': {'local_rounds': 60, 'official_report_rounds': 60, 'resource_ledgers': 30,
                   'fixed_validation_scope_checked': True, 'parent_and_selected_metrics_reconciled': True,
                   'candidate_gains_recalculated': True, 'original_xlsx_read_only': True}}
    output.mkdir(parents=True, exist_ok=True)
    write(output / '逐轮指标数据.json', data)
    print(json.dumps(data['checks'], ensure_ascii=False))


def render(output):
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    from PIL import Image, ImageOps, ImageDraw
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 11,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#334155', 'text.color': '#18273d', 'xtick.color': '#475569',
        'ytick.color': '#475569', 'axes.edgecolor': '#cbd5e1', 'axes.titleweight': 'normal',
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'grid.color': '#e6ebf1', 'grid.linewidth': .7, 'svg.fonttype': 'path'})
    data = read(output / '逐轮指标数据.json')
    local, official = data['local'], data['official']
    files = []
    versions = {'v19': '#94a3b8', 'v20': '#2265cf'}
    tasks = {'A': '#2265cf', 'B': '#008c82', 'C': '#d7832c'}
    rounds = np.arange(11)
    def series(version, dataset, metric):
        d = local[version][dataset]
        return np.array([d['baseline'][metric]] + [row['selected'][metric] for row in d['rounds']])
    def style(ax, xlabel='SDK 回合（0 为初始模型）', zero=False):
        ax.set_xlabel(xlabel)
        ax.set_xticks(range(11) if '0 为' in xlabel else range(1, 11))
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        if zero:
            ax.axhline(0, color='#94a3b8', linewidth=.9)
    def pair(ax, dataset, metric, transform=None, precision=3):
        arrays = []
        for version in ('v19', 'v20'):
            y = series(version, dataset, metric)
            if transform:
                y = transform(y)
            arrays.append(y)
            ax.plot(rounds, y, color=versions[version], linestyle='--' if version == 'v19' else '-',
                    marker='s' if version == 'v19' else 'o', markersize=6 if version == 'v19' else 3.5,
                    markerfacecolor='white' if version == 'v19' else versions[version], linewidth=1.8,
                    label=version, zorder=3 if version == 'v19' else 4)
        same = np.allclose(arrays[0], arrays[1], atol=1e-10, rtol=0)
        for i, y in enumerate(arrays):
            if i == 0 and same:
                continue
            offset = (8, (8 if y[-1] > arrays[1 - i][-1] else -10) if not same else 0)
            ax.annotate(f'{y[-1]:.{precision}f}', (10, y[-1]), xytext=offset,
                        textcoords='offset points', fontsize=10, va='center')
        if same:
            ax.text(.05, .86, '两版曲线重合', transform=ax.transAxes, color='#64748b', fontsize=10)
        ax.set_xlim(-.2, 11.65)
        ax.margins(y=.16)
        style(ax)
        return arrays
    def save(fig, name, title, note, headline):
        fig.text(.06, .978, title, ha='left', va='top', fontsize=19)
        fig.text(.06, .930, headline, va='top', fontsize=11, color='#52627a')
        fig.text(.06, .022, note, fontsize=10, color='#52627a')
        fig.tight_layout(rect=(.035, .065, .98, .885), h_pad=2, w_pad=2)
        fig.canvas.draw()
        fig.savefig(output / f'{name}.png', dpi=240)
        fig.savefig(output / f'{name}.svg')
        plt.close(fig)
        files.append({'name': name, 'title': title, 'note': note, 'headline': headline})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for j, dataset in enumerate(('A', 'B', 'C')):
        metric = 'macro_f1' if dataset == 'A' else 'rmse'
        pair(axes[0, j], dataset, metric, precision=4 if dataset == 'A' else 3)
        axes[0, j].set_title({'A': 'A 分类 Macro-F1（越高越好）', 'B': 'B 轴承 RMSE（越低越好）', 'C': 'C 发动机 RMSE（越低越好）'}[dataset], loc='left')
        axes[0, j].set_ylabel({'A': 'Macro-F1', 'B': '分钟', 'C': '循环'}[dataset])
        base = local['v20'][dataset]['baseline'][metric]
        transform = (lambda y, b=base: (y - b) * 100) if dataset == 'A' else (lambda y, b=base: (b - y) / b * 100)
        pair(axes[1, j], dataset, metric, transform, 2)
        axes[1, j].set_title(f'{dataset} 相对初始模型的累计提升', loc='left')
        axes[1, j].set_ylabel('F1 增加（百分点）' if dataset == 'A' else 'RMSE 降低（%）')
    axes[0, 0].legend(frameon=False, loc='lower right')
    save(fig, '01-performance-overview', '十轮性能与累计提升',
         '固定初始公开验证集；曲线表示每轮实际选中模型。A 使用百分点，B/C 使用相对 RMSE 降幅。',
         'v19 / v20 镜像验证对照。离线控制器，工业模型真实训练与预测；不代表正式隐藏集成绩。')

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10))
    for ax, dataset in zip(axes, ('A', 'B', 'C')):
        records = local['v20'][dataset]['rounds']
        scale = 100 if dataset == 'A' else 1
        for row in records:
            x, gain = row['round'], row['candidate_gain']
            if gain is None:
                ax.plot(x, 0, marker='x', color='#64748b', markersize=8)
                ax.annotate('观察', (x, 0), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=10)
                continue
            color = '#008c82' if row['adopted'] else '#d18b31' if gain > 0 else '#c65b62'
            ax.bar(x, gain * scale, width=.62, color=color)
            ax.annotate(f'{gain * scale:+.3f}' if dataset != 'B' else f'{gain:+.2f}', (x, gain * scale),
                        xytext=(0, 5 if gain >= 0 else -5), textcoords='offset points', ha='center',
                        va='bottom' if gain >= 0 else 'top', fontsize=10)
        ax.set_title(f'{dataset} 每轮候选相对当轮父模型的增益', loc='left')
        ax.set_ylabel('F1 增加（百分点）' if dataset == 'A' else f'RMSE 减少（{"分钟" if dataset == "B" else "循环"}）')
        ax.margins(y=.28)
        style(ax, 'SDK 回合', True)
    axes[0].legend(handles=[Patch(color='#008c82', label='候选采用'), Patch(color='#d18b31', label='正增益但未采用'),
                            Patch(color='#c65b62', label='负增益、未采用')], frameon=False, ncol=3, loc='upper right')
    save(fig, '02-candidate-gains', 'v20 每轮实验到底有没有收益',
         '正数表示候选优于当轮父模型；正增益仍须通过最小提升、类别召回或 MAE 等完整门槛。观察表示未训练。',
         '候选实验结果与实际采用情况分开记录，未采用候选不计入上线提升。')

    fig, axes = plt.subplots(3, 1, figsize=(13.8, 11.2), gridspec_kw={'height_ratios': [1, 1.35, 1.35]})
    states = [local['v20']['A']['baseline']] + [r['selected'] for r in local['v20']['A']['rounds']]
    pair(axes[0], 'A', 'accuracy', precision=4)
    axes[0].set_title('全局 Accuracy（准确率）', loc='left')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(frameon=False, loc='lower right')
    for ax, metric, label in zip(axes[1:], ('f1', 'recall'), ('逐类 F1', '逐类召回率')):
        values = np.array([[state['per_class'][name][metric] for state in states] for name in CLASSES])
        image = ax.imshow(values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        ax.set_yticks(range(6), [f'{cn}（n={states[0]["per_class"][name]["support"]}）' for cn, name in zip(CN, CLASSES)])
        ax.set_xticks(range(11), ['初始'] + [str(i) for i in range(1, 11)])
        ax.set_xlabel('SDK 回合')
        ax.set_title(label, loc='left')
        for i in range(6):
            for j in range(11):
                ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', fontsize=10,
                        color='white' if values[i, j] > .68 else '#18273d')
        fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    save(fig, '03-a-class-quality', 'A 各气体类别的十轮表现',
         '固定公开验证集。甲苯只有 5 条样本；类别保护按当轮父模型执行，不能据此保证累计召回不下降。',
         '本次 v19 与 v20 的 A 逐轮选中指标一致；此图展示 v20 的真实完整预测器结果。')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8))
    for ax, metric, title, unit, transform in zip(axes.flat,
        ('rmse', 'mae', 'bias', 'overestimate_fraction'),
        ('RMSE（越低越好）', 'MAE（越低越好）', '平均偏差（预测 − 真实）', '高估样本比例'),
        ('分钟', '分钟', '分钟；负数为低估', '%'), (None, None, None, lambda y: y * 100)):
        pair(ax, 'B', metric, transform, 2)
        ax.set_title(title, loc='left')
        ax.set_ylabel(unit)
        if metric == 'bias':
            ax.axhline(0, color='#94a3b8', linewidth=.8)
    axes[0, 0].legend(frameon=False)
    save(fig, '04-b-regression-quality', 'B 回归误差与高低估',
         'v20 最终 RMSE 比 v19 降低约 0.386%；MAE 增加约 0.850%，仍在原 2% 保护范围内。',
         '第 3 轮采用低估权重 1.25 的新方案；总体 RMSE 下降不意味着所有寿命段都改善。')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    metrics = [('rmse', 'RMSE', '循环', None), ('mae', 'MAE', '循环', None),
               ('nasa_score', 'NASA 非对称评分', '十亿；越低越好', lambda y: y / 1e9),
               ('bias', '平均偏差（预测 − 真实）', '循环', None),
               ('overestimate_fraction', '高估样本比例', '%', lambda y: y * 100)]
    for ax, (metric, label, unit, transform) in zip(axes.flat, metrics):
        pair(ax, 'C', metric, transform, 3 if metric == 'nasa_score' else 2)
        ax.set_title(label, loc='left')
        ax.set_ylabel(unit)
    initial_nasa = local['v20']['C']['baseline']['nasa_score']
    pair(axes[1, 2], 'C', 'nasa_score', lambda y: (initial_nasa - y) / initial_nasa * 100, 2)
    axes[1, 2].set_title('NASA 评分相对初始降低', loc='left')
    axes[1, 2].set_ylabel('%')
    axes[0, 0].legend(frameon=False)
    save(fig, '05-c-regression-quality', 'C 回归误差与非对称惩罚',
         'NASA 为逐样本惩罚之和。本图样本数固定为 23,049，可在同一公开范围内比较；不同阶段不可直接比较总分。',
         '两版曲线一致；第 4 轮 fine_rate 的收益保留。未过门槛的偏置和残差修正未进入镜像菜单。')

    status_colors = ['#c8e7df', '#f0cdd0', '#e5eaf0', '#f3e2c5', '#dcd4ef']
    fig, ax = plt.subplots(figsize=(15, 6.8))
    matrix, texts, labels = [], [], []
    for dataset in ('A', 'B', 'C'):
        for version in ('v19', 'v20'):
            records = local[version][dataset]['rounds']
            matrix.append([0 if r['adopted'] else 2 if not r['candidate_submitted'] else
                           3 if (r['gate'] or {}).get('eligible') else 1 for r in records])
            texts.append([('采用' if r['adopted'] else '观察' if not r['candidate_submitted'] else '保留父版') +
                          ('\n' + EXPERIMENTS.get(r['experiment'], r['experiment'] or '') if r['experiment'] else '') for r in records])
            labels.append(f'{dataset}  {version}   采用 {sum(r["adopted"] for r in records)} / 训练 {sum(r["candidate_submitted"] for r in records)}')
    ax.imshow(matrix, cmap=ListedColormap(status_colors), norm=BoundaryNorm(np.arange(-.5, 5), 5), aspect='auto')
    for i, line in enumerate(texts):
        for j, text in enumerate(line):
            ax.text(j, i, text, ha='center', va='center', fontsize=10)
    ax.set_yticks(range(6), labels)
    ax.set_xticks(range(10), range(1, 11))
    ax.set_xlabel('SDK 回合')
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(color=status_colors[i], label=label) for i, label in enumerate(
        ('采用候选', '未达门槛，保留父版', '观察，未训练', '合格未采用'))],
        frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(.5, -.12))
    save(fig, '06-adoption-timeline', '每轮用了什么方案，是否真正采用',
         '两版控制器均为离线替身，序列预先指定。可复核数值闭环，不能据此预测真实 27B 控制器的决策。',
         'v20 训练 27 个候选，v19 训练 30 个候选；少做 3 次训练不等于总资源成本下降 10%。')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, metric, label in zip(axes, ('cpu_seconds', 'wall_seconds', 'peak_rss_mib'),
                               ('训练回调 CPU（秒）', '训练回调墙钟（秒）', '进程峰值内存（MiB）')):
        for dataset in ('A', 'B', 'C'):
            records = local['v20'][dataset]['rounds']
            ax.plot(range(1, 11), [r['resources'][metric] for r in records], marker='o', markersize=4,
                    color=tasks[dataset], label=dataset)
        ax.set_title(label, loc='left')
        ax.set_ylabel('MiB' if metric == 'peak_rss_mib' else '秒')
        style(ax, 'SDK 回合')
    axes[0].legend(frameon=False)
    save(fig, '07-round-resources', 'v20 十轮资源观测',
         '范围为选手训练回调，未含容器启动。墙钟包含 RPC 等待；内存为进程峰值。单次观测，不是重复性能基准。',
         '30 轮共 60 次离线模型请求。宿主未返回 token usage，因此 token 成本未知，未画成零。')

    groups = ['读取数据', '父模型评估与核验', '历史读取与复评', '训练', '模型请求', '候选上传与测量', '其他']
    colors = ['#b6cde8', '#5b8db8', '#9b8fc0', '#008c82', '#d5b46d', '#dc9680', '#cbd5e1']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10.5))
    for ax, dataset in zip(axes, ('A', 'B', 'C')):
        bottom = np.zeros(10)
        for group, color in zip(groups, colors):
            values = np.array([r['resources']['stage_cpu_seconds'].get(group, 0) for r in local['v20'][dataset]['rounds']])
            ax.bar(range(1, 11), values, bottom=bottom, width=.65, color=color, label=group)
            bottom += values
        ax.set_title(f'{dataset} 每轮 CPU 用在哪里', loc='left')
        ax.set_ylabel('CPU 秒')
        style(ax, 'SDK 回合')
    axes[0].legend(frameon=False, ncol=4, loc='upper left', fontsize=10)
    save(fig, '08-cpu-breakdown', 'v20 阶段 CPU 构成',
         '阶段明细来自保存前快照；保存、结束和未分阶段时间归入其他。图中是 CPU，不是 RPC 等待墙钟。',
         '按每轮真实资源记录求和；不得将阶段总计与回调总计重复相加。')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, key, title, ylabel in zip(axes,
        ('cpu_ms_per_sample', 'callback_samples_per_second', 'process_seconds'),
        ('每样本预测回调 CPU', '预测回调吞吐', '完整预测进程耗时'),
        ('毫秒 / 样本（对数坐标）', '样本 / 秒（对数坐标）', '秒')):
        for dataset in ('A', 'B', 'C'):
            ax.plot(range(1, 11), [r['prediction'][key] for r in local['v20'][dataset]['rounds']],
                    marker='o', markersize=4, color=tasks[dataset], label=dataset)
        if key != 'process_seconds':
            ax.set_yscale('log')
            if key == 'cpu_ms_per_sample':
                ax.set_yticks([.03, .05, .1, .3, .5, 1])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            else:
                ax.set_yticks([1000, 3000, 10000, 30000])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        ax.set_title(title, loc='left')
        ax.set_ylabel(ylabel)
        style(ax, 'SDK 回合')
    axes[0].legend(frameon=False)
    save(fig, '09-prediction-resources', 'v20 每轮选中模型的独立预测',
         '回调口径不含 SDK JSON 和启动。完整进程含容器启动及协议，A/B/C 样本数不同，不能直接当作模型速度排名。',
         'A 1,244 条，B 1,679 条，C 23,049 条。前两图使用对数坐标，仅展示本次实际观测。')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, dataset in zip(axes, ('A', 'B', 'C')):
        metric = 'macro_f1' if dataset == 'A' else 'rmse'
        formal_rows, trial_rows = official['formal'][dataset], official['trial'][dataset]
        fy = np.array([r['selected'][metric] for r in formal_rows])
        ty = np.array([r['selected'][metric] for r in trial_rows])
        same = np.allclose(fy, ty, rtol=0, atol=1e-9)
        stage_groups = defaultdict(list)
        for i, row in enumerate(formal_rows):
            stage_groups[row['stage']].append(i)
        for indices in stage_groups.values():
            ax.plot([i + 1 for i in indices], fy[indices], color='#2265cf', marker='o', linewidth=1.8)
        if not same:
            ax.scatter(range(1, 11), ty, marker='s', facecolors='none', edgecolors='#c56e32', s=45,
                       label='试运行（阶段标签未提供）', zorder=4)
        ax.plot([], [], color='#2265cf', marker='o', label='正式 / 试运行一致' if same else '正式报告')
        for i in range(1, 10):
            if formal_rows[i]['stage'] != formal_rows[i - 1]['stage']:
                ax.axvline(i + .5, color='#cbd5e1', linestyle=':', linewidth=1)
        ax.set_title(f'{dataset} 每轮实际选中模型指标', loc='left')
        ax.set_ylabel('Macro-F1' if dataset == 'A' else 'RMSE（分钟）' if dataset == 'B' else 'RMSE（循环）')
        ax.set_xlabel('轮次 / 正式报告阶段')
        ax.set_xticks(range(1, 11), [f'{i + 1}\nS{r["stage"]}' for i, r in enumerate(formal_rows)])
        ax.grid(axis='y')
        ax.legend(frameon=False, loc='best')
        ax.margins(y=.2)
    save(fig, '10-official-stage-metrics', '原始正式与试运行报告：按阶段看指标',
         '阶段标记取自正式报告；正式曲线只连接同阶段点。跨阶段绝对值变化不代表训练提升，不能与固定投影本地曲线混算。',
         '直接只读解析两份原始 Excel 的实际选中指标，并用配置中的完整精度值核对。')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, dataset in zip(axes, ('A', 'B', 'C')):
        scale = 100 if dataset == 'A' else 1
        for kind, offset, color, label in [('formal', -.17, '#2265cf', '正式'), ('trial', .17, '#d29050', '试运行')]:
            for row in official[kind][dataset]:
                x, gain = row['round'] + offset, row['candidate_gain']
                if gain is None:
                    ax.plot(x, 0, marker='x', color=color, markersize=7)
                else:
                    ax.bar(x, gain * scale, width=.3, color=color, alpha=.9 if row['adopted'] else .33,
                           hatch=None if row['adopted'] else '//')
        ax.set_title(f'{dataset} 候选相对当轮父模型的收益', loc='left')
        ax.set_ylabel('F1 增加（百分点）' if dataset == 'A' else 'RMSE 减少（分钟）' if dataset == 'B' else 'RMSE 减少（循环）')
        style(ax, '报告轮次', True)
    axes[0].legend(handles=[Patch(color='#2265cf', label='正式'), Patch(color='#d29050', label='试运行')],
                   frameon=False, ncol=2)
    save(fig, '11-official-candidate-gains', '原始报告：每轮实验的真实收益与采用',
         '实色为采用候选，斜线为未采用，×为无候选。每根柱只比较当轮同一验证范围；不同阶段的柱值不累加。',
         '正数表示候选指标改善，是否采用还取决于完整准入门槛；两份报告的实验与采用时点可能不同。')

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    scores = data['objective']['components']
    positions = np.arange(len(scores))
    maximum = np.array([r['maximum'] for r in scores])
    earned = np.array([r['score'] for r in scores])
    ax.barh(positions, maximum, color='#e8edf3', height=.52, label='该项满分')
    ax.barh(positions, earned, color='#2265cf', height=.52, label='当前正式得分')
    ax.set_yticks(positions, [r['name'] for r in scores])
    ax.invert_yaxis()
    for i, row in enumerate(scores):
        ax.text(row['maximum'] + .2, i, f'{row["score"]:.6f} / {row["maximum"]}  （{row["score"] / row["maximum"]:.1%}）',
                va='center', fontsize=11)
    ax.set_xlim(0, 20.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel('分')
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    save(fig, '12-objective-score', f'当前正式客观成绩 {data["objective"]["total"]:.6f} / 35',
         '本图来自正式报告。v20 尚未取得新的正式客观分；分项显示值相加与总分存在末位舍入差。',
         '平均性能、最差阶段与资源效率仍有较大缺口。不能将本地指标改善直接换算为正式得分。')

    # One browsable collection, with the exact inputs and definitions kept beside it.
    rows = []
    for dataset in ('A', 'B', 'C'):
        metric = 'macro_f1' if dataset == 'A' else 'rmse'
        old, new = local['v19'][dataset]['final'][metric], local['v20'][dataset]['final'][metric]
        gain = (new - old) * 100 if dataset == 'A' else (old - new) / old * 100
        rows.append(f'<tr><td>{dataset}</td><td>{"Macro-F1" if dataset == "A" else "RMSE"}</td>'
                    f'<td>{old:.6f}</td><td>{new:.6f}</td><td>{gain:+.3f}{" 个百分点" if dataset == "A" else "%"}</td></tr>')
    gallery = '\n'.join(f'<section id="{item["name"]}"><h2>{escape(item["title"])}</h2>'
        f'<p>{escape(item["headline"])}</p><img src="{item["name"]}.svg" alt="{escape(item["title"])}" loading="lazy">'
        f'<p>{escape(item["note"])}</p><p><a href="{item["name"]}.png">PNG 高清图</a>　'
        f'<a href="{item["name"]}.svg">SVG 矢量图</a></p></section>' for item in files)
    links = ' · '.join(f'<a href="#{item["name"]}">{i + 1:02d}</a>' for i, item in enumerate(files))
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>工业模型多轮指标图表</title><style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#18273d;max-width:1300px;margin:auto;padding:32px 20px}h1{font-weight:500}h2{font-weight:500;margin-top:48px}p{color:#52627a}a{color:#2265cf}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #e5eaf0}section{border-top:1px solid #e5eaf0;margin-top:34px}@media print{a{color:inherit}section{break-before:page}img{max-height:78vh}}</style>'
    html += '<h1>工业模型多轮指标图表</h1><p>2026-11-26。图 01—09：固定初始公开投影的 v19/v20 本地镜像验证。图 10—12：原始正式与试运行报告。两类数据分别展示。</p>'
    html += '<table><thead><tr><th>任务</th><th>主指标</th><th>v19 第十轮</th><th>v20 第十轮</th><th>v20 相对 v19</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    html += f'<p>图表索引：{links}。<a href="逐轮指标数据.json">逐轮完整数据与来源</a></p>' + gallery
    html += '<section><h2>数据口径</h2><p>本地十轮使用离线控制器，工业模型真实训练与预测；不包含正式隐藏成绩，也不是十组独立随机实验。资源为本次回调观测，未返回 token 用量时保留未知。</p>'
    html += '<p>正式报告阶段会切换，只连接同阶段点；候选收益只与当轮父模型比较。未采用候选不计入上线收益。没有将不同任务单位混为同一个平均提升率。</p>'
    html += '<p>来源：正式测评实验报告.xlsx、实验报告记录.xlsx，以及 v19/v20 镜像十轮 report、resources、预测计时。完整绝对路径与 SHA-256 记录在逐轮指标数据中。</p></section></html>'
    (output / 'index.html').write_text(html, encoding='utf-8')
    write(output / 'chart-catalog.json', files)
    # Contact sheet for visual verification; the export itself contains the full-size figures.
    thumbs = []
    for i, item in enumerate(files):
        with Image.open(output / f'{item["name"]}.png') as im:
            thumb = ImageOps.contain(im.convert('RGB'), (1000, 650))
            canvas = Image.new('RGB', (1040, 700), 'white')
            canvas.paste(thumb, ((1040 - thumb.width) // 2, 30))
            ImageDraw.Draw(canvas).text((20, 8), f'{i + 1:02d} {item["name"]}', fill='#18273d')
            thumbs.append(canvas)
    sheet = Image.new('RGB', (2080, 700 * math.ceil(len(thumbs) / 2)), '#e5eaf0')
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 2) * 1040, (i // 2) * 700))
    sheet.save(output / 'contact-sheet.jpg', quality=90)
    write(output / 'chart-validation.json', {**data['checks'], 'figure_count': len(files), 'formats': ['PNG 240 DPI', 'SVG'],
        'all_figures_saved': all((output / f'{i["name"]}.{ext}').stat().st_size > 1000 for i in files for ext in ('png', 'svg')),
        'matplotlib_version': matplotlib.__version__, 'formal_score_remeasured': False, 'real_model_calls': 0})
    print(json.dumps({'figures': len(files), 'output': str(output), 'contact_sheet': str(output / 'contact-sheet.jpg')}, ensure_ascii=False))


def package(output, reviewed=False):
    import struct
    catalog = read(output / 'chart-catalog.json')
    data = read(output / '逐轮指标数据.json')
    for item in data['sources']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    for version, directory in LOCAL.items():
        for dataset in ('A', 'B', 'C'):
            original = read(directory / dataset / 'report.json')
            prepared = data['local'][version][dataset]
            assert original['baseline'] == prepared['baseline'] and original['final'] == prepared['final']
            for before, after in zip(original['rounds'], prepared['rounds']):
                assert before['round'] == after['round'] and before['metrics'] == after['selected']
                assert before['candidate_submitted'] == after['candidate_submitted']
    dimensions = []
    for item in catalog:
        png = output / (item['name'] + '.png')
        with png.open('rb') as stream:
            header = stream.read(24)
        assert header[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', header[16:24])
        assert width >= 3000 and height >= 1300
        ET.parse(output / (item['name'] + '.svg'))
        dimensions.append({'figure': item['name'], 'width': width, 'height': height})
    validation = read(output / 'chart-validation.json')
    validation.update(original_source_hashes_unchanged=True, selected_metrics_independently_reconciled=True,
                      png_svg_files_checked=True, image_dimensions=dimensions, visual_review_complete=reviewed)
    write(output / 'chart-validation.json', validation)
    notes = ['工业模型多轮指标图表', '日期：2026-11-26', '',
             '共 12 组图表。每组提供 240 DPI PNG 和 SVG 矢量图。打开 index.html 可浏览全部图表。',
             '图 01—09 是固定初始公开投影的本地镜像验证；图 10—12 来自原始正式/试运行报告。',
             'A 使用 F1 百分点，B/C 使用 RMSE 相对降幅，不混算单位。0 为初始模型。',
             '候选收益与最终采用分开；阶段变化不计算累计提升。',
             '本地为离线控制器的真实数值闭环，不能代表正式隐藏评分；token 未返回即为未知。',
             '逐轮指标数据.json 保留全部精度值、口径和来源 SHA-256。原始 Excel 未修改。', '', '图表目录：']
    notes.extend(f'{i + 1:02d}. {item["title"]}' for i, item in enumerate(catalog))
    (output / '使用说明.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    names = [f'{item["name"]}.{ext}' for item in catalog for ext in ('png', 'svg')]
    names += ['index.html', '逐轮指标数据.json', 'chart-catalog.json', 'chart-validation.json', '使用说明.txt']
    manifest = [{'file': name, 'bytes': (output / name).stat().st_size,
                 'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()} for name in names]
    write(output / 'file-manifest.json', manifest)
    names.append('file-manifest.json')
    target = output.parent / '多轮指标图表-v20-20261126.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            archive.write(output / name, output.name + '/' + name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for item in manifest:
            assert hashlib.sha256(archive.read(output.name + '/' + item['file'])).hexdigest() == item['sha256']
    receipt = {'archive': str(target), 'bytes': target.stat().st_size,
               'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'files': len(names),
               'figure_count': len(catalog), 'verified': True, 'visual_review_complete': reviewed}
    write(output.parent / 'multiround-charts-v20-20261126-package.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/multiround-charts-v20-20261126')
    parser.add_argument('--report-root', type=Path, default=Path('D:/liyue/工业模型赛题研究'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--pack-only', action='store_true')
    parser.add_argument('--visual-review-complete', action='store_true')
    args = parser.parse_args()
    if args.pack_only:
        package(args.output.resolve(), args.visual_review_complete)
        return
    if not args.render_only:
        prepare(args.output.resolve(), args.report_root)
    if not args.prepare_only:
        render(args.output.resolve())


if __name__ == '__main__':
    main()
