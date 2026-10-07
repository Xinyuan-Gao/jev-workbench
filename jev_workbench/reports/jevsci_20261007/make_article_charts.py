"""Generate the Chinese article figures from the frozen final analysis.

Run:
    python3 jev_workbench/reports/jevsci_20261007/make_article_charts.py

Reads only jev_workbench/scientific/artifacts/final_analysis.json (produced by
`scientific.analyze`). No network, no model calls, no recomputation of the
official statistics: every plotted number is copied from that file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # /Users/gxy/投稿管理
SCI = ROOT / 'jev_workbench' / 'scientific'
ANALYSIS = HERE / 'final_analysis.json'
OUT = HERE / 'assets'
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path('/Library/Fonts/Arial Unicode.ttf')
if not FONT.exists():
    cands = [f.fname for f in font_manager.fontManager.ttflist
             if f.name in ('Arial Unicode MS', 'Songti SC', 'Heiti TC', 'PingFang SC')]
    assert cands, 'local Chinese font unavailable'
    FONT = Path(cands[0])
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(),
    'axes.unicode_minus': False, 'font.size': 12, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'figure.dpi': 150, 'savefig.dpi': 190,
    'axes.spines.top': False, 'axes.spines.right': False,
})

MODEL_NAMES = {
    'local:majority': '多数类',
    'local:lr': 'TF-IDF + LR',
    'local:linear_svc': 'TF-IDF + LinearSVC',
    'local:xgboost': 'TF-IDF + XGBoost',
    'local:lightgbm': 'TF-IDF + LightGBM',
    'jev:jev-1.13.0': 'JEV (jev-1.13.0)',
    'gemini:gemini-3.8-flash': 'Gemini (gemini-3.8-flash)',
}
DATASET_NAMES = {'massive': '中文意图（MASSIVE）', 't2': '问题—片段相关性（T2Ranking）'}


def color(name: str) -> str:
    if name.startswith('local:majority'):
        return '#B6BFC6'
    if name.startswith('local:'):
        return '#7895A5'
    if name.startswith('jev:'):
        return '#846AA4'
    return '#C07D72'


def label(name: str) -> str:
    return MODEL_NAMES.get(name, name)


def save(fig, name):
    for fmt in ('png',):
        fig.savefig(OUT / f'{name}.{fmt}', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', OUT / f'{name}.png')


def analysis(path=None):
    target = Path(path) if path else ANALYSIS
    raw = target.read_bytes()
    data = json.loads(raw)
    assert data.get('complete') is True, 'final analysis is not complete'
    return data, hashlib.sha256(raw).hexdigest()


def order(names):
    """Stable article order: majority first, then supervised locals, then remote."""
    pref = ['local:majority', 'local:lr', 'local:linear_svc', 'local:xgboost',
            'local:lightgbm']
    out = [n for n in pref if n in names]
    out += [n for n in names if n not in out and n.startswith('jev:')]
    out += [n for n in names if n not in out and n.startswith('gemini:')]
    out += [n for n in names if n not in out]
    return out


def fig_data_split(data):
    spec = [('massive', '中文意图 MASSIVE', [('train', 10684, 10684), ('validation', 1958, 1958), ('test', 558, 558)]),
            ('t2', '问题—片段 T2Ranking', [('train', 4030, 1200), ('validation', 671, 200), ('test', 674, 200)])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (alias, title, rows) in zip(axes, spec):
        x = np.arange(len(rows))
        rec = [r[1] for r in rows]
        grp = [r[2] for r in rows]
        ax.bar(x - 0.2, rec, width=0.38, color='#7895A5', label='记录数')
        ax.bar(x + 0.2, grp, width=0.38, color='#C0A6D8', label='独立单元数')
        for i, (v1, v2) in enumerate(zip(rec, grp)):
            ax.text(i - 0.2, v1, f'{v1:,}', ha='center', va='bottom', fontsize=9)
            ax.text(i + 0.2, v2, f'{v2:,}', ha='center', va='bottom', fontsize=9)
        ax.set_yscale('log')
        ax.set_xticks(x, [r[0] for r in rows])
        ax.set_title(title)
        ax.set_ylabel('数量（对数轴）')
        ax.grid(axis='y', alpha=.15)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('两份公开标注数据的划分（对数轴）', fontsize=14)
    save(fig, 'c01_data_split')


def _interval_plot(data, alias, metric, title, xlabel, name):
    ds = data['datasets'][alias]
    names = order(ds['models'])
    fig, ax = plt.subplots(figsize=(8.6, max(3.6, .62 * len(names) + 1.6)))
    for i, m in enumerate(names):
        block = ds['models'][m]['interval'][metric]
        est, (lo, hi) = block['estimate'], block['ci95']
        c = color(m)
        ax.plot([lo * 100, hi * 100], [i, i], color=c, lw=2.4, solid_capstyle='round')
        ax.scatter(est * 100, i, color=c, s=46, zorder=3)
        ax.text(101.5, i, f'{est * 100:.1f}%', va='center', fontsize=10)
    ax.set_yticks(range(len(names)), [label(m) for m in names])
    ax.invert_yaxis()
    ax.set_xlim(-2, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis='x', alpha=.15)
    save(fig, name)


def fig_accuracy(data):
    _interval_plot(data, 'massive', 'accuracy',
                   '中文意图 558 条：全提交正确率与 95% 区间',
                   '全部计划记录中选对的比例（%）', 'c02_accuracy_massive')
    _interval_plot(data, 't2', 'accuracy',
                   '问题—片段 674 对：全提交正确率与 95% 区间',
                   '全部计划记录中选对的比例（%）', 'c03_accuracy_t2')


def fig_macro_f1(data):
    _interval_plot(data, 'massive', 'macro_f1',
                   '中文意图：全提交 Macro-F1 与 95% 区间（探索性）',
                   'Macro-F1（%，60 类固定标签集，无支持类计 0）', 'c04_macro_f1_massive')
    _interval_plot(data, 't2', 'macro_f1',
                   '问题—片段：全提交 Macro-F1 与 95% 区间（探索性）',
                   'Macro-F1（%，相关/不相关两类）', 'c05_macro_f1_t2')


def fig_valid(data):
    ok = False
    for alias, ds in data['datasets'].items():
        for m in ds['models']:
            if ds['models'][m]['score']['failed']:
                ok = True
    if not ok:
        print('no failed outputs in this matrix; skip valid/failure figure')
        return
    fig, axes = plt.subplots(1, len(data['datasets']), figsize=(13.5, 5.0), squeeze=False,
                             constrained_layout=True)
    for ax, (alias, ds) in zip(axes[0], sorted(data['datasets'].items())):
        names = order(ds['models'])
        n = ds['test_records']
        for i, m in enumerate(names):
            s = ds['models'][m]['score']
            left = s['valid'] / n * 100
            ax.barh(i, left, height=.6, color=color(m), alpha=.85)
            for status, count in s['status_counts'].items():
                if status == 'valid' or not count:
                    continue
                ax.barh(i, count / n * 100, left=left, height=.6, color='#D83931', hatch='///')
                left += count / n * 100
        ax.set_yticks(range(len(names)),
                      [f'{label(m)}\n{ds["models"][m]["score"]["valid"]} 有效 / '
                       f'{ds["models"][m]["score"]["failed"]} 失败' for m in names], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('占全部计划记录的百分比（%）')
        ax.set_title(DATASET_NAMES.get(alias, alias))
    fig.suptitle('有效输出与失败（斜线为未取得允许类别）', fontsize=14)
    save(fig, 'c06_valid_failure')


def fig_rag_confusion(data):
    ds = data['datasets']['t2']
    best_local = 'local:' + data['analysis_policy']['datasets']['t2']['best_local_family']
    names = [n for n in order(ds['models']) if n.startswith(('jev:', 'gemini:'))] + [best_local]
    labels = ds['labels']
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), squeeze=False,
                             constrained_layout=True)
    for j, (ax, m) in enumerate(zip(axes[0], names)):
        mat = np.array(ds['models'][m]['score']['confusion_matrix'])[:, :len(labels)]
        ax.imshow(mat, cmap='Blues', vmin=0, vmax=max(1, mat.max()))
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, str(mat[r, c]), ha='center', va='center', fontsize=13,
                        color='white' if mat[r, c] > mat.max() * .55 else '#233E50')
        ax.set_xticks(range(len(labels)), labels, fontsize=10)
        ax.set_yticks(range(len(labels)))
        ax.set_xlabel('预测')
        if j == 0:
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_ylabel('标准答案')
        else:
            ax.set_yticklabels([])
        ax.set_title(label(m), fontsize=12)
    fig.suptitle('问题—片段相关性的混淆计数（674 对，200 个独立问题）', fontsize=14)
    save(fig, 'c07_rag_confusion')


def fig_intent_recall(data):
    ds = data['datasets']['massive']
    pick = ['local:lr', 'jev:jev-1.13.0', 'gemini:gemini-3.8-flash']
    pick = [m for m in pick if m in ds['models']]
    per = {m: ds['models'][m]['score']['per_class'] for m in pick}
    common = [l for l in ds['labels'] if per[pick[0]][l]['support'] >= 3]
    common.sort(key=lambda l: -(per[pick[-1]][l]['recall'] + per[pick[0]][l]['recall']))
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    x = np.arange(len(common))
    w = .8 / len(pick)
    for j, m in enumerate(pick):
        vals = [per[m][l]['recall'] * 100 for l in common]
        ax.bar(x + (j - (len(pick) - 1) / 2) * w, vals, width=w, color=color(m), label=label(m))
    ax.set_xticks(x, common, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('召回率（%）')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=.15)
    ax.legend(frameon=False, ncol=3)
    ax.set_title(f'中文意图：有 3 条以上测试支持的类别召回率（共 {len(common)} 类）')
    save(fig, 'c08_intent_recall')


def fig_latency(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), constrained_layout=True)
    for j, (ax, (alias, ds)) in enumerate(zip(axes, sorted(data['datasets'].items()))):
        names = order(ds['models'])
        for i, m in enumerate(names):
            op = ds['models'][m].get('primary_operational') or ds['models'][m]['operational']
            latency = op.get('valid_latency') or op.get('logical_latency')
            mean, p95 = latency.get('mean_ms'), latency.get('p95_ms')
            c = color(m)
            if mean is not None:
                ax.scatter(mean, i, color=c, s=48, marker='o', zorder=3)
            if p95 is not None:
                ax.scatter(p95, i, color=c, s=52, marker='^', zorder=3)
            if mean is not None and p95 is not None:
                ax.plot([mean, p95], [i, i], color=c, lw=1.6, alpha=.7)
        ax.set_yticks(range(len(names)))
        if j == 0:
            ax.set_yticklabels([label(m) for m in names], fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlabel('毫秒（对数轴）：实心圆均值、三角 p95')
        ax.set_title(DATASET_NAMES.get(alias, alias))
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('有效输出的单条延迟（本地含特征变换与预测、不含训练；远程含网络与重试等待）', fontsize=12)
    save(fig, 'c09_latency')


def fig_difference(data):
    rows = []
    for alias, ds in data['datasets'].items():
        for comp in ds.get('confirmatory_comparisons', []):
            b = comp['bootstrap']['accuracy_difference']
            rows.append({
                'title': f"{DATASET_NAMES.get(alias, alias)}\nJEV − {label(comp['b'])}",
                'est': b['estimate'], 'lo': b['ci95'][0], 'hi': b['ci95'][1],
                'p': comp.get('accuracy_test', {}).get('p_value'),
                'holm': comp.get('holm_p_value'),
                'reject': comp.get('reject_equal_accuracy_at_alpha'),
            })
    if not rows:
        print('no confirmatory comparisons; skip difference figure')
        return
    fig, ax = plt.subplots(figsize=(9.4, max(3.2, 1.15 * len(rows) + 1.8)))
    for i, r in enumerate(rows):
        inside = (r['lo'] <= 0 <= r['hi'])
        c = '#8F959E' if inside else ('#2EA121' if r['est'] > 0 else '#D83931')
        ax.plot([r['lo'] * 100, r['hi'] * 100], [i, i], color=c, lw=2.6, solid_capstyle='round')
        ax.scatter(r['est'] * 100, i, color=c, s=56, zorder=3)
        p = r['holm']
        txt = f"{r['est'] * 100:+.1f}pp" + (f"  Holm p={p:.3f}" if p is not None else '')
        ax.text(r['hi'] * 100 + 0.9, i, txt, va='center', fontsize=9.5)
    ax.axvline(0, color='#333333', lw=1, ls='--')
    ax.set_yticks(range(len(rows)), [r['title'] for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('全提交正确率差值（百分点，正值表示 JEV 更高）')
    ax.set_title('JEV 与对照的成对差值区间（95%，按独立单元重采样）')
    ax.grid(axis='x', alpha=.15)
    save(fig, 'c10_difference')


def fig_stability(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (alias, ds) in zip(axes, sorted(data['datasets'].items())):
        names = order(ds['models'])
        i = 0
        ticks, ticklabels = [], []
        for m in names:
            rounds = ds['models'][m]['stability'].get('per_round', {})
            for repeat, s in sorted(rounds.items(), key=lambda kv: int(kv[0])):
                ax.scatter(s['accuracy'] * 100, i, color=color(m), s=42)
                ticks.append(i)
                ticklabels.append(f"{label(m)} · 轮 {int(repeat) + 1}")
                i += 1
        ax.set_yticks(ticks, ticklabels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-2, 102)
        ax.set_xlabel('全提交正确率（%）')
        ax.set_title(DATASET_NAMES.get(alias, alias))
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('每一轮实际调用的成绩（重复轮次是描述性波动，不增加独立样本量）', fontsize=12)
    save(fig, 'c11_stability')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis', type=Path, default=None,
                    help='override final_analysis.json (schema dry-run only)')
    args = ap.parse_args(argv)
    data, sha = analysis(args.analysis)
    print('analysis sha256', sha)
    fig_data_split(data)
    fig_accuracy(data)
    fig_macro_f1(data)
    fig_valid(data)
    fig_rag_confusion(data)
    fig_intent_recall(data)
    fig_latency(data)
    fig_difference(data)
    fig_stability(data)
    (OUT / 'article_figures_manifest.json').write_text(json.dumps({
        'source_analysis': str(ANALYSIS), 'analysis_sha256': sha,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figures': sorted(p.name for p in OUT.glob('*.png')),
    }, ensure_ascii=False, indent=2))
    print('done')


if __name__ == '__main__':
    main()
