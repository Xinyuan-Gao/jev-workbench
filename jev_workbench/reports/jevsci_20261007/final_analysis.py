"""Article-side final analysis of the frozen matrix.

Why this exists
---------------
`scientific.analyze` refuses any matrix with a non-terminal job. The frozen run
ended with exactly one: T2Ranking `t2-dev-15910-584973`, provider gemini,
**repeat 2**, left `indeterminate` on purpose by the continuation run because the
frozen policy forbids re-sending a paid request whose outcome is unknown.

That job is outside the primary repeat, so every prespecified primary comparison
is fully observed. This script therefore:

* validates the frozen/continuation manifest, the stored manifest hash and the
  per-request audit chain exactly like `analyze.py`,
* refuses to run if any pending job falls inside the primary repeat,
* recomputes the primary statistics with the same frozen settings,
* reports repeat-level stability only for rounds that are complete.

It never predicts, never calls an API and never writes inside `scientific/`.

Run (from /Users/gxy/投稿管理):
    python3 jev_workbench/reports/jevsci_20261007/final_analysis.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jev_workbench.scientific.analyze import (_examples, _json, _operational,
                                              _settings, _validate_remote_audit)
from jev_workbench.scientific.freeze import REQUIRED_GATES, verify_freeze
from jev_workbench.scientific.journal import DurableJournal
from jev_workbench.scientific.remote import _safe, _secrets
from jev_workbench.scientific.run import (LOCAL_REPEATS, _atomic_json, _plan,
                                          _read, normalize_gateway_url)
from jev_workbench.scientific.statistics import (holm_adjust, mcnemar_exact,
                                                 paired_bootstrap,
                                                 paired_cluster_test, score)

ROOT = REPO / 'jev_workbench' / 'scientific'
# Files whose contents decide the analysis result. Drift here is fatal; drift in
# an auxiliary file (for example the progress publisher) is recorded, not hidden.
ANALYSIS_PATH_FILES = ('analyze.py', 'statistics.py', 'run.py', 'journal.py',
                       'remote.py', 'local_models.py', 'datasets.py', 'task_schemas.py')
LIMITATIONS = [
    'Results apply to the frozen data, labels, configuration and the recorded gateway service; they are not a universal model ranking.',
    'One T2Ranking gemini repeat-2 request is preserved as pending/indeterminate and is excluded from the primary analysis; it is outside the primary repeat.',
    'Two task results are reported separately; no cross-task total is computed.',
    'Public benchmark pretraining contamination and semantic duplicates cannot be excluded by lexical screening.',
    'Empirical bootstrap intervals can degenerate; a point interval does not prove population certainty or equivalence.',
    'Only all-planned primary-repeat Accuracy comparisons are Holm adjusted. Macro-F1 intervals are exploratory.',
    'Repeated calls and training seeds describe stability and cost; they never increase independent sample size.',
    'Token usage does not establish monetary cost; pricing is unknown and no cost figure is derived.',
    'Gateway latency includes transport and retries; local latency includes feature construction and prediction.',
]


def read_matrix(path, expected, bundle_hash, schema, execution, primary_repeat):
    """Validate like analyze.py, but keep non-terminal rows instead of raising."""
    if not path.is_file():
        raise ValueError(f'matrix journal missing: {path}')
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        conn.execute('PRAGMA query_only=ON')
        metadata = dict(conn.execute('SELECT key,value FROM metadata'))
        stored = conn.execute('SELECT job_id,job_json,outcome_json,terminal_at FROM planned ORDER BY ordinal').fetchall()
        events = conn.execute('SELECT job_id,event_json FROM attempts ORDER BY event_id').fetchall()
    finally:
        conn.close()
    canonical = _json(sorted(expected, key=lambda job: job['job_id']))
    if metadata.get('protocol_hash') != bundle_hash:
        raise ValueError('Matrix protocol hash differs from the frozen protocol hash')
    if metadata.get('manifest_hash') != hashlib.sha256(canonical.encode()).hexdigest():
        raise ValueError('Matrix manifest hash differs from the frozen planned manifest')
    jobs = [json.loads(row[1]) for row in stored]
    if any(row[0] != job.get('job_id') for row, job in zip(stored, jobs)):
        raise ValueError('Stored job id column disagrees with job payload')
    if _json(sorted(jobs, key=lambda job: job['job_id'])) != canonical:
        raise ValueError('Stored matrix differs from the complete frozen planned manifest')
    attempts = defaultdict(list)
    ids = {job['job_id'] for job in expected}
    for ident, value in events:
        if ident not in ids:
            raise ValueError('Attempt log references a job outside the frozen matrix')
        attempts[ident].append(json.loads(value))
    rows, pending = [], []
    for (_, _, raw, terminal_at), job in zip(stored, jobs):
        outcome = json.loads(raw) if raw else None
        terminal = (isinstance(outcome, dict)
                    and outcome.get('status') in DurableJournal.TERMINAL_STATUSES
                    and bool(terminal_at))
        entry = {**job, 'outcome': outcome, 'terminal_at': terminal_at,
                 'attempts': attempts[job['job_id']],
                 'potential_duplicate': any(e.get('potential_duplicate') for e in attempts[job['job_id']])}
        if not terminal:
            if job['repeat'] == primary_repeat:
                raise ValueError('A primary-repeat job is not terminal; primary analysis must refuse')
            pending.append({'job_id': job['job_id'], 'item_id': job['item_id'], 'model': job['model'],
                            'repeat': job['repeat'], 'events': [e.get('phase') for e in attempts[job['job_id']]]})
            continue
        if not job['model'].startswith('local:'):
            _validate_remote_audit(entry, schema, execution)
        rows.append(entry)
    return rows, pending


def stability(rows, labels, primary_repeat):
    """Per-round scores for complete rounds only; incomplete rounds are declared."""
    by_repeat = defaultdict(list)
    for row in rows:
        by_repeat[row['repeat']].append(row)
    sizes = {repeat: len(v) for repeat, v in by_repeat.items()}
    expected = Counter(r['item_id'] for r in rows if r['repeat'] == primary_repeat)
    complete, incomplete = {}, {}
    for repeat, values in by_repeat.items():
        have = Counter(r['item_id'] for r in values)
        missing = [item for item in expected if have[item] < expected[item]]
        if missing:
            incomplete[repeat] = {'records': sizes[repeat], 'missing': len(missing)}
        else:
            complete[repeat] = score(values, labels)
    primary = {r['item_id']: r for r in by_repeat[primary_repeat]}
    agreements = []
    for repeat in sorted(complete):
        if repeat == primary_repeat:
            continue
        other = {r['item_id']: r for r in by_repeat[repeat]}
        outcome, both_valid = [], []
        for item, row_a in primary.items():
            row_b = other[item]
            pa = row_a['outcome'].get('prediction'); sa = row_a['outcome']['status']
            pb = row_b['outcome'].get('prediction'); sb = row_b['outcome']['status']
            outcome.append((pa, sa) == (pb, sb))
            if sa == sb == 'valid':
                both_valid.append(pa == pb)
        agreements.append({'repeat': repeat,
                           'outcome_agreement': float(np.mean(outcome)),
                           'outcome_denominator': len(outcome),
                           'prediction_agreement_both_valid': float(np.mean(both_valid)) if both_valid else None,
                           'prediction_denominator': len(both_valid)})
    return {'rounds': len(by_repeat), 'complete_rounds': sorted(complete), 'incomplete_rounds': incomplete,
            'per_round': {str(k): v for k, v in sorted(complete.items())},
            'agreement_with_primary': agreements,
            'scope': 'descriptive stability across actual service rounds / training seeds; no extra independent samples'}


def load_inputs(root, freeze_path):
    """Load the frozen datasets with an explicit, reported verification scope.

    The frozen/continuation manifest still lists an auxiliary file whose hash no
    longer matches, so the all-or-nothing `verify_freeze` check fails. Everything
    that decides the numbers is verified here; anything that drifted is returned
    to the caller and written into the audit record.
    """
    bundle = json.loads(Path(freeze_path).read_text())
    verification = verify_freeze(root, bundle)
    changed = list(verification.get('changed') or [])
    blocking = sorted(set(changed) & set(ANALYSIS_PATH_FILES))
    if blocking:
        raise ValueError(f'analysis-path source drifted from the freeze: {blocking}')
    if verification.get('bundle_hash_valid') is not True:
        raise ValueError('Frozen bundle hash is invalid')
    for gate in REQUIRED_GATES:
        entry = bundle.get('gates', {}).get(gate)
        if not isinstance(entry, dict) or entry.get('passed') is not True or not entry.get('evidence'):
            raise ValueError(f'frozen gate not satisfied: {gate}')
    artifacts = {a['path']: a for a in bundle['artifacts']}
    sources = bundle['metadata']['execution_source_sha256']
    verified_sources = {}
    for name in ANALYSIS_PATH_FILES:
        expected = sources.get(name)
        actual = root / name
        if not isinstance(expected, str) or not actual.is_file():
            raise ValueError(f'analysis-path source missing from frozen record: {name}')
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if digest != expected or artifacts.get(name, {}).get('sha256') != expected:
            raise ValueError(f'analysis-path source hash mismatch: {name}')
        verified_sources[name] = digest
    ids = bundle['metadata']['dataset_ids']
    schema_path = 'data/task_schemas_candidate.json'
    schemas = _read(root / schema_path)
    candidates = {}
    for artifact_path in sorted(artifacts):
        if (artifact_path.startswith('data/') and artifact_path.endswith('_candidate.json')
                and artifact_path != schema_path):
            payload = _read(root / artifact_path)
            if 'dataset_id' in payload:
                alias = Path(artifact_path).name.removesuffix('_candidate.json')
                candidates[alias] = (artifact_path, payload)
    datasets = {}
    for requested in ids:
        matches = [(alias, path, payload) for alias, (path, payload) in candidates.items()
                   if requested in {alias, payload['dataset_id']}]
        if len(matches) != 1:
            raise ValueError('frozen dataset id must resolve to exactly one candidate')
        alias, _, payload = matches[0]
        schema = schemas[alias]
        if list(schema.get('labels', [])) != list(payload.get('labels', [])):
            raise ValueError('frozen schema label order differs from dataset')
        report = _read(root / f'artifacts/{alias}/local_dev_report.json')
        if report.get('status') != 'complete' or report.get('test_used') is not False:
            raise ValueError('frozen local dev selection must be complete and test-free')
        if set(report.get('families', {})) != set(LOCAL_REPEATS):
            raise ValueError('frozen local baseline family manifest is incomplete')
        records = [r for r in payload['records'] if r.get('split') == 'test']
        if not records or len({r['id'] for r in records}) != len(records):
            raise ValueError('unique nonempty frozen test manifest required')
        datasets[alias] = {'data': payload, 'schema': schema,
                           'test_records': records, 'report': report}
    execution = bundle['metadata']['execution']
    remote_models = execution.get('remote_models')
    if not isinstance(remote_models, dict) or set(remote_models) != {'jev', 'gemini'}:
        raise ValueError('frozen remote model identities required')
    for value in execution['remote_base_urls'].values():
        normalize_gateway_url(value)
    if not isinstance(execution.get('remote_repeats'), int) or execution['remote_repeats'] < 3:
        raise ValueError('at least three frozen remote repeats required')
    if not isinstance(execution.get('max_http_attempts'), int) or not 1 <= execution['max_http_attempts'] <= 3:
        raise ValueError('frozen HTTP attempt limit must be 1 to 3')
    continuation = bundle['metadata'].get('continuation')
    if continuation is not None:
        if (not isinstance(continuation, dict)
                or continuation.get('parent_bundle_sha256') != continuation.get('journal_protocol_hash')):
            raise ValueError('invalid continuation freeze linkage')
        execution = dict(execution)
        execution['_journal_protocol_hash'] = continuation['journal_protocol_hash']
    provenance = {'freeze_path': str(Path(freeze_path).relative_to(REPO)),
                  'bundle_sha256': bundle['bundle_sha256'],
                  'frozen_at_utc': bundle.get('frozen_at_utc'),
                  'bundle_hash_valid': verification.get('bundle_hash_valid'),
                  'verify_freeze_valid': verification.get('valid'),
                  'drifted_artifacts': changed,
                  'analysis_path_sources_verified': verified_sources}
    return bundle, datasets, execution, provenance


def analyse(root, freeze_path, output):
    bundle, datasets, execution, provenance = load_inputs(root, freeze_path)
    settings = _settings(bundle, datasets)
    primary_repeat = settings['primary_repeat']
    secrets = _secrets({'datasets': datasets, 'execution': execution})
    result = {'schema_version': 1, 'complete': True, 'primary_repeat_complete': True,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'bundle_sha256': bundle['bundle_sha256'], 'analysis_policy': settings,
              'primary_repeat': primary_repeat, 'datasets': {}, 'limitations': LIMITATIONS,
              'provenance': provenance,
              'generator': str(Path(__file__).relative_to(REPO)),
              'note': ('Primary repeat is fully observed. Non-terminal jobs are reported below and are '
                       'excluded; scientific.analyze refuses the whole matrix while any job is pending.')}
    all_comparisons = []
    pending_report = {}
    for alias, dataset in datasets.items():
        labels = dataset['data']['labels']
        rows, pending = read_matrix(root / f'artifacts/{alias}/matrix.sqlite',
                                    _plan(dataset, execution),
                                    execution.get('_journal_protocol_hash', bundle['bundle_sha256']),
                                    dataset['schema'], execution, primary_repeat)
        pending_report[alias] = pending
        primary = {name: [r for r in rows if r['repeat'] == primary_repeat and r['model'] == name]
                   for name in {r['model'] for r in rows}}
        groups = defaultdict(list)
        for row in rows:
            groups[row['model']].append(row)
        policy = settings['datasets'][alias]
        kwargs = {'iterations': settings['bootstrap_iterations'], 'seed': settings['seed'],
                  'stratify': policy['bootstrap_stratify']}
        data = {'dataset_id': dataset['data']['dataset_id'],
                'test_records': len(dataset['test_records']),
                'independent_groups': len({r['group_id'] for r in dataset['test_records']}),
                'primary_observed': sum(len(v) for v in primary.values()),
                'labels': labels, 'models': {}, 'confirmatory_comparisons': [],
                'source': dataset['data'].get('source'), 'sampling': dataset['data'].get('sampling'),
                'limitations': dataset['data'].get('limitations', [])}
        for name, values in groups.items():
            metric = score(primary[name], labels)
            supported = [l for l in labels if metric['per_class'][l]['support'] > 0]
            interval = paired_bootstrap(primary[name], primary[name], labels, **kwargs)
            data['models'][name] = {
                'score': metric,
                'supported_label_macro_f1': float(np.mean([metric['per_class'][l]['f1'] for l in supported])),
                'supported_labels': supported,
                'absent_gold_labels': [l for l in labels if l not in supported],
                'interval': {'accuracy': interval['a']['accuracy'], 'macro_f1': interval['a']['macro_f1'],
                             **{k: interval[k] for k in ('method', 'iterations', 'seed', 'stratify',
                                                          'stratum_support', 'rare_strata', 'interval_scope',
                                                          'insufficient_independent_groups')}},
                'mean_group_accuracy_interval': None,
                'stability': stability(values, labels, primary_repeat),
                'operational': _operational(values),
                'primary_operational': _operational(primary[name]),
                'failures': _examples(primary[name], labels, failures=True),
                'misclassifications': _examples(primary[name], labels, failures=False),
            }
        jev = 'jev:' + execution['remote_models']['jev']
        peers = ['gemini:' + execution['remote_models']['gemini'], 'local:' + policy['best_local_family']]
        for peer in peers:
            a, b = primary[jev], primary[peer]
            one_row_per_group = len({r['group_id'] for r in a}) == len(a)
            test = (mcnemar_exact(a, b, labels) if one_row_per_group
                    else paired_cluster_test(a, b, labels, iterations=settings['permutation_iterations'],
                                             seed=settings['seed']))
            comparison = {'a': jev, 'b': peer, 'difference_direction': 'a minus b',
                          'bootstrap': paired_bootstrap(a, b, labels, **kwargs),
                          'accuracy_test': test, 'tested_metric': 'all_planned_accuracy',
                          'other_metrics': 'exploratory CI only'}
            data['confirmatory_comparisons'].append(comparison)
            all_comparisons.append(comparison)
        result['datasets'][alias] = data
    adjusted = holm_adjust([c['accuracy_test']['p_value'] for c in all_comparisons])
    for comparison, value in zip(all_comparisons, adjusted):
        comparison.update(holm_p_value=value, reject_equal_accuracy_at_alpha=value <= settings['alpha'])
    result['holm_family_size'] = len(all_comparisons)
    result['holm_family'] = 'two Accuracy comparisons per frozen formal task'
    result['pending_jobs'] = pending_report
    result = _safe(result, secrets)
    _atomic_json(Path(output), result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--freeze-path', type=Path, default=ROOT / 'frozen_continuation.json')
    ap.add_argument('--output', type=Path, default=HERE / 'final_analysis.json')
    args = ap.parse_args(argv)
    try:
        result = analyse(args.root, args.freeze_path, args.output)
    except Exception as exc:
        print(json.dumps({'status': 'refused', 'error_type': type(exc).__name__, 'error': str(exc)[:300]}))
        return 2
    print(json.dumps({'status': 'complete', 'output': str(args.output),
                      'datasets': list(result['datasets']),
                      'holm_family_size': result['holm_family_size'],
                      'pending_jobs': {k: len(v) for k, v in result['pending_jobs'].items()}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
