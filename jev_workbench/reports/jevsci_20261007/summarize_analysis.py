"""Print a compact fact sheet from the frozen final analysis.

Only reads the analysis JSON; every number printed is copied from it, so the
article prose can be written against one authoritative source.

Run:
    python3 jev_workbench/reports/jevsci_20261007/summarize_analysis.py \
        [--analysis path/to/final_analysis.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'final_analysis.json'

MODEL_ORDER = ['local:majority', 'local:lr', 'local:linear_svc', 'local:xgboost',
               'local:lightgbm', 'jev:jev-1.13.0', 'gemini:gemini-3.8-flash']
NAME = {
    'local:majority': '多数类',
    'local:lr': 'TF-IDF + LR',
    'local:linear_svc': 'TF-IDF + LinearSVC',
    'local:xgboost': 'TF-IDF + XGBoost',
    'local:lightgbm': 'TF-IDF + LightGBM',
    'jev:jev-1.13.0': 'JEV (jev-1.13.0)',
    'gemini:gemini-3.8-flash': 'Gemini (gemini-3.8-flash)',
}
DATASET = {'massive': '中文意图 MASSIVE', 't2': '问题—片段 T2Ranking'}


def pct(x):
    return 'n/a' if x is None else f'{x * 100:.2f}%'


def pp(x):
    return 'n/a' if x is None else f'{x * 100:+.2f}pp'


def order(models):
    names = [m for m in MODEL_ORDER if m in models]
    return names + [m for m in models if m not in names]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', type=Path, default=DEFAULT)
    args = ap.parse_args(argv)
    data = json.loads(args.analysis.read_text())
    print(f"analysis: {args.analysis}")
    print(f"complete: {data.get('complete')}  primary_repeat: {data.get('primary_repeat')}  "
          f"holm family: {data.get('holm_family_size')}")
    for alias, ds in data['datasets'].items():
        print('\n' + '=' * 78)
        print(f"{alias}  ({DATASET.get(alias, alias)})")
        print(f"  test records: {ds['test_records']}   independent groups: {ds['independent_groups']}")
        print(f"  labels: {len(ds['labels'])}   dataset_id: {ds.get('dataset_id')}")
        print('  ' + '-' * 74)
        print(f"  {'model':28} {'acc':>8} {'95% CI':>18} {'macroF1':>9} {'valid':>7} {'fail':>5} {'rate':>7}")
        for m in order(ds['models']):
            blk = ds['models'][m]
            s, iv = blk['score'], blk['interval']
            acc = iv['accuracy']
            f1 = blk['supported_label_macro_f1']
            print(f"  {NAME.get(m, m):28} {pct(s['accuracy']):>8} "
                  f"{'[' + pct(acc['ci95'][0]) + ',' + pct(acc['ci95'][1]) + ']':>18} "
                  f"{pct(s['macro_f1']):>9} {s['valid']:>7} {s['failed']:>5} "
                  f"{pct(s['valid_rate']):>7}")
            print(f"  {'':28} supported-label Macro-F1={pct(f1)}  "
                  f"status={ {k: v for k, v in s['status_counts'].items() if v} }  "
                  f"rare_strata={iv.get('rare_strata')}")
        print('\n  confirmatory comparisons (primary metric: all-planned accuracy):')
        for comp in ds.get('confirmatory_comparisons', []):
            b = comp['bootstrap']
            d, m1 = b['accuracy_difference'], b['macro_f1_difference']
            test = comp.get('accuracy_test', {})
            print(f"    JEV - {NAME.get(comp['b'], comp['b'])}: "
                  f"{pp(d['estimate'])} CI [{pp(d['ci95'][0])},{pp(d['ci95'][1])}]  "
                  f"Holm p={test.get('p_value') if 'p_value' in test else comp.get('holm_p_value')}  "
                  f"holm={comp.get('holm_p_value')}  "
                  f"reject={comp.get('reject_equal_accuracy_at_alpha')}  "
                  f"test={test.get('test') or test.get('method')}  "
                  f"macroF1 diff={pp(m1['estimate'])} CI [{pp(m1['ci95'][0])},{pp(m1['ci95'][1])}]")
        print('\n  per-model operational scopes (primary repeat):')
        for m in order(ds['models']):
            op = ds['models'][m].get('primary_operational') or ds['models'][m]['operational']
            def scope(key):
                v = op.get(key) or {}
                return (f"n={v.get('count')} mean={v.get('mean_ms')} med={v.get('median_ms')} "
                        f"p95={v.get('p95_ms')}")
            print(f"    {NAME.get(m, m):28} valid: {scope('valid_latency')}")
            print(f"    {'':28} logical: {scope('logical_latency')}  firstHTTP: {scope('first_http_attempt_latency')}")
        print('\n  per-class support (labels with support):')
        sup = [(l, ds['models'][order(ds['models'])[0]]['score']['per_class'][l]['support']) for l in ds['labels']]
        print('    n_labels_with_support=', sum(1 for _, s in sup if s > 0),
              ' zero_support=', [l for l, s in sup if s == 0],
              ' support_lt_3=', [f'{l}:{s}' for l, s in sup if 0 < s < 3])
        for m in [m for m in order(ds['models']) if m.startswith(('jev:', 'gemini:', 'local:lr', 'local:linear_svc'))]:
            pc = ds['models'][m]['score']['per_class']
            worst = sorted([(l, pc[l]['recall'], pc[l]['support']) for l in ds['labels'] if pc[l]['support'] > 0],
                           key=lambda t: t[1])[:5]
            print(f"    {NAME.get(m, m):28} worst recalls: " +
                  ', '.join(f'{l} {r * 100:.0f}% (n={s})' for l, r, s in worst))
        for m in [m for m in order(ds['models']) if m.startswith(('jev:', 'gemini:'))]:
            blk = ds['models'][m]
            print(f"\n    {NAME.get(m, m)} failures={len(blk.get('failures') or [])} "
                  f"misclassifications={len(blk.get('misclassifications') or [])}")
            for ex in (blk.get('failures') or [])[:5]:
                print('      FAIL', json.dumps(ex, ensure_ascii=False)[:260])
            for ex in (blk.get('misclassifications') or [])[:6]:
                print('      ERR ', json.dumps(ex, ensure_ascii=False)[:260])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
