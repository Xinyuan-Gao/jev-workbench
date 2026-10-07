"""Final analysis of a complete, frozen matrix; never predicts or calls an API.

Reproduce: python -m scientific.analyze --root /path/to/scientific --output report.json
The frozen metadata.analysis supplies primary_repeat=0, bootstrap_iterations,
permutation_iterations, seed, alpha, and datasets[alias] containing
bootstrap_stratify (gold_label/null) and best_local_family. Existing candidate
configuration names statistical_seed and best_local_by_validation are accepted.
The latter uses <alias>_bootstrap_strata, never a hardcoded dataset size.

Only all-planned Accuracy is tested: two comparisons per frozen dataset share
one Holm family. F1 intervals are exploratory. mean_group_accuracy is a point
estimate; it does not inherit the passage-weighted Accuracy interval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .journal import DurableJournal, _json
from .remote import _safe, _secrets, build_payload, parse_response
from .run import ROOT, LOCAL_REPEATS, _atomic_json, _frozen_inputs, _plan
from .statistics import score, paired_bootstrap, mcnemar_exact, paired_cluster_test, holm_adjust, repeat_stability


LIMITATIONS = [
    'Results apply to the frozen data, labels, configuration and gateway service; they do not establish a universal model ranking.',
    'Public benchmark pretraining contamination and semantic duplicates cannot be excluded by lexical screening.',
    'Empirical bootstrap intervals can degenerate with perfect outputs or rare strata; a point interval does not prove population certainty or equivalence.',
    'Only all-planned primary-repeat Accuracy comparisons are tested and Holm adjusted. Macro-F1 intervals are exploratory.',
    'mean_group_accuracy is an equal-group point estimate, without a separately computed interval; passage-weighted Accuracy intervals cannot be assigned to it.',
    'Repeated calls and training seeds describe stability and operational cost; they never increase independent test sample size.',
    'Reported token usage does not establish monetary cost; missing usage or pricing remains unknown.',
    'Local prediction latency includes feature construction, prediction and optional probability computation; gateway latency includes transport and retries.',
    'Exact McNemar assumes independent one-row groups. Group sign permutation assumes independent groups and exchangeability under its null.',
    'Best local comparison is fixed using validation scores, never selected from the formal test outputs.',
]


def _settings(bundle, datasets):
    config = bundle.get('metadata', {}).get('analysis')
    if not isinstance(config, dict) or config.get('primary_repeat') != 0:
        raise ValueError('Frozen analysis policy with primary_repeat=0 is required')
    seed = config.get('seed', config.get('statistical_seed'))
    if 'seed' in config and 'statistical_seed' in config and config['seed'] != config['statistical_seed']:
        raise ValueError('Conflicting frozen statistical seeds')
    iterations = config.get('bootstrap_iterations')
    permutations = config.get('permutation_iterations', iterations)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (iterations, permutations)) or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('Frozen positive iteration counts and integer analysis seed required')
    alpha = config.get('alpha', .05)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError('Invalid frozen significance level')
    policies = {}
    for alias, dataset in datasets.items():
        if isinstance(config.get('datasets'), dict):
            entry = config['datasets'].get(alias)
            if not isinstance(entry, dict) or 'bootstrap_stratify' not in entry:
                raise ValueError('Every dataset needs a frozen analysis policy')
            family = entry.get('best_local_family')
            stratify = entry['bootstrap_stratify']
        else:
            entry = config.get('best_local_by_validation', {}).get(alias, {})
            family = entry.get('model', '').removeprefix('local:')
            stratify = config.get(alias + '_bootstrap_strata')
        if stratify not in (None, 'gold_label'):
            raise ValueError('Unsupported frozen bootstrap stratification')
        families = dataset['report']['families']
        order = list(LOCAL_REPEATS)
        best = min(order, key=lambda name: (-families[name]['selection']['validation_macro_f1'], order.index(name)))
        if family != best:
            raise ValueError('Frozen best local family differs from validation selection')
        policies[alias] = {'best_local_family': family, 'bootstrap_stratify': stratify}
    return {'primary_repeat': 0, 'bootstrap_iterations': iterations, 'permutation_iterations': permutations,
            'seed': seed, 'alpha': alpha, 'datasets': policies}


def _validate_remote_audit(row, schema, execution):
    """A terminal row is final only when its request/response chain is auditable."""
    provider, model = row['model'].split(':', 1)
    expected = build_payload(provider, model, row['input'], schema)
    base = execution.get('remote_base_urls', {}).get(provider)
    if not isinstance(base, str) or not base:
        raise ValueError('Frozen HTTP endpoint mapping is required for final audit')
    endpoint = base.rstrip('/')
    if not endpoint.endswith('/chat/completions'):
        endpoint += '/chat/completions'
    shared = {'attempt_id', 'attempt', 'execution_id', 'provider', 'endpoint', 'request_model',
              'timeout_seconds', 'started_at', 'request_body', 'request_headers'}
    finished_fields = {'finished_at', 'outcome', 'duration_ms', 'http_status', 'raw_body',
                       'response_headers', 'usage', 'response_model', 'retry_reason',
                       'will_retry', 'wait_seconds', 'error'}
    jobs = {event.get('execution_id') for event in row['attempts']
            if event.get('kind') == 'job' and event.get('phase') == 'started'}
    starts = {}; finishes = {}; final = None; counts = Counter()
    previous = None
    for event in row['attempts']:
        if event.get('kind') in {'job', 'recovery', 'duplicate_retry_authorization'}:
            continue
        phase = event.get('phase')
        if phase not in {'started', 'finished'} or not shared.issubset(event):
            raise ValueError('HTTP audit requires complete started/finished attempt schema')
        ident = event['attempt_id']
        if (not isinstance(ident, str) or not ident or event['execution_id'] not in jobs
                or event['provider'] != provider or event['request_model'] != model
                or event['endpoint'] != endpoint or not _number(event['timeout_seconds'])
                or event['timeout_seconds'] <= 0 or not isinstance(event['request_headers'], dict)):
            raise ValueError('HTTP audit request identity differs from frozen execution')
        try:
            payload = json.loads(event['request_body'])
            datetime.fromisoformat(event['started_at'])
        except (ValueError, TypeError):
            raise ValueError('HTTP audit request/timestamp is malformed') from None
        if _json(payload) != _json(expected):
            raise ValueError('HTTP request payload differs from frozen input/model/schema')
        if phase == 'started':
            if ident in starts or previous is not None and previous not in finishes:
                raise ValueError('HTTP audit has duplicate or unpaired attempts')
            counts[event['execution_id']] += 1
            if (type(event['attempt']) is not int or event['attempt'] != counts[event['execution_id']]
                    or event['attempt'] > execution['max_http_attempts']):
                raise ValueError('HTTP attempt number violates frozen retry limit')
            starts[ident] = event; previous = ident
            continue
        if ident not in starts or ident in finishes or not finished_fields.issubset(event):
            raise ValueError('HTTP audit requires paired complete finished attempt')
        if any(event[key] != starts[ident][key] for key in shared):
            raise ValueError('HTTP started/finished request evidence mismatch')
        try:
            if datetime.fromisoformat(event['finished_at']) < datetime.fromisoformat(event['started_at']):
                raise ValueError('Negative duration')
        except (ValueError, TypeError):
            raise ValueError('HTTP audit finished timestamp is malformed') from None
        if (not _number(event['duration_ms']) or not _number(event['wait_seconds'])
                or type(event['will_retry']) is not bool or not isinstance(event['raw_body'], str)
                or not isinstance(event['response_headers'], dict)
                or event['usage'] is not None and not isinstance(event['usage'], dict)
                or event['http_status'] is not None and
                (type(event['http_status']) is not int or not 100 <= event['http_status'] <= 599)):
            raise ValueError('HTTP audit response metadata is malformed')
        finishes[ident] = event
        if event['will_retry']:
            if event['outcome'] is not None or not event['retry_reason']:
                raise ValueError('Retrying HTTP audit cannot declare terminal outcome')
            continue
        outcome = event['outcome']
        if not isinstance(outcome, dict) or outcome.get('status') not in DurableJournal.TERMINAL_STATUSES:
            raise ValueError('Finished HTTP audit requires terminal outcome')
        status = event['http_status']
        if status is None:
            if outcome['status'] not in {'network_error', 'transport_error'} or not event['error']:
                raise ValueError('Null HTTP status needs recorded transport failure')
        elif 200 <= status <= 299:
            try:
                raw = json.loads(event['raw_body'])
            except (ValueError, TypeError):
                raw = None
            parsed = parse_response(provider, raw, schema['labels']) if isinstance(raw, dict) else {
                'status': 'malformed_response', 'prediction': None, 'confidence': None, 'confidence_status': 'missing'}
            for key in ('status', 'prediction', 'confidence', 'confidence_status'):
                if outcome.get(key) != parsed.get(key):
                    raise ValueError('HTTP raw response differs from recorded terminal outcome')
        elif outcome['status'] != 'http_error':
            raise ValueError('HTTP error status differs from recorded outcome')
        final = event
    if not starts or set(starts) != set(finishes) or final is None or final is not finishes[previous]:
        raise ValueError('Complete remote HTTP audit evidence is missing/unpaired')
    for key, value in final['outcome'].items():
        if _safe(row['outcome'].get(key), _secrets(row['outcome'])) != _safe(value, _secrets(final['outcome'])):
            raise ValueError('Logical terminal outcome differs from final HTTP outcome')
    if row['outcome'].get('recovered_from_attempt', final['attempt_id']) != final['attempt_id']:
        raise ValueError('Recovered terminal references a different HTTP attempt')


def _read_matrix(path, expected, bundle_hash, schema, execution):
    """Snapshot SQLite read-only, without running constructor/recovery writes."""
    if not path.is_file():
        raise ValueError('Complete matrix journal is missing')
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        metadata = dict(conn.execute('SELECT key,value FROM metadata'))
        stored = conn.execute('SELECT job_id,job_json,outcome_json,terminal_at FROM planned ORDER BY ordinal').fetchall()
        events = conn.execute('SELECT job_id,event_json FROM attempts ORDER BY event_id').fetchall()
    finally:
        conn.close()
    canonical = _json(sorted(expected, key=lambda job: job['job_id']))
    if metadata.get('protocol_hash') != bundle_hash or metadata.get('manifest_hash') != hashlib.sha256(canonical.encode()).hexdigest():
        raise ValueError('Matrix manifest or frozen protocol hash mismatch')
    jobs = [json.loads(row[1]) for row in stored]
    if any(row[0] != job.get('job_id') for row, job in zip(stored, jobs)) or _json(sorted(jobs, key=lambda job: job['job_id'])) != canonical:
        raise ValueError('Stored matrix differs from the complete frozen planned manifest')
    attempts = defaultdict(list)
    ids = {job['job_id'] for job in expected}
    for ident, value in events:
        if ident not in ids:
            raise ValueError('Attempt log references a job outside frozen matrix')
        attempts[ident].append(json.loads(value))
    result = []
    for (_, _, raw, terminal_at), job in zip(stored, jobs):
        outcome = json.loads(raw) if raw else None
        if not isinstance(outcome, dict) or outcome.get('status') not in DurableJournal.TERMINAL_STATUSES or not terminal_at:
            raise ValueError('All planned matrix combinations must be terminal before final analysis; pending remains')
        row = {**job, 'outcome': outcome, 'terminal_at': terminal_at,
               'attempts': attempts[job['job_id']],
               'potential_duplicate': any(event.get('potential_duplicate') for event in attempts[job['job_id']])}
        if not job['model'].startswith('local:'):
            _validate_remote_audit(row, schema, execution)
        result.append(row)
    return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _distribution(values):
    values = [float(value) for value in values if _number(value)]
    if not values:
        return {'count': 0, 'mean_ms': None, 'median_ms': None, 'p90_ms': None, 'p95_ms': None}
    return {'count': len(values), 'mean_ms': float(np.mean(values)), 'median_ms': float(np.median(values)),
            'p90_ms': float(np.quantile(values, .9)), 'p95_ms': float(np.quantile(values, .95))}


def _operational(rows):
    latencies = []; successful = []; failed = []; first = []; attempts = []; unfinished = 0
    for row in rows:
        outcome = row['outcome']
        latency = outcome.get('transport', {}).get('total_latency_ms', outcome.get('latency_ms'))
        if _number(latency):
            latencies.append(latency)
            (successful if outcome['status'] == 'valid' else failed).append(latency)
        starts = [event for event in row['attempts'] if event.get('phase') == 'started' and event.get('kind') != 'job']
        finished = [event for event in row['attempts'] if event.get('phase') == 'finished' and event.get('kind') != 'job']
        finish_ids = [event.get('attempt_id') for event in finished]
        if len(set(finish_ids)) != len(finish_ids):
            raise ValueError('Duplicate finished attempt IDs would double-count usage')
        finish_by_id = {event.get('attempt_id'): event for event in finished}
        unfinished += sum(event.get('attempt_id') not in finish_by_id for event in starts)
        if starts and starts[0].get('attempt_id') in finish_by_id:
            first.append(finish_by_id[starts[0]['attempt_id']].get('duration_ms'))
        attempts.extend(finished)
    usage_sums = Counter(); missing_usage = 0; reported = 0
    for event in attempts:
        usage = event.get('usage')
        if not isinstance(usage, dict):
            missing_usage += 1; continue
        reported += 1
        for key, value in usage.items():
            if _number(value):
                usage_sums[key] += value
    return {'logical_records': len(rows), 'repeat_indices': sorted({row['repeat'] for row in rows}),
            'logical_latency': _distribution(latencies), 'valid_latency': _distribution(successful),
            'failure_latency': _distribution(failed), 'first_http_attempt_latency': _distribution(first),
            'all_http_attempt_latency': _distribution([event.get('duration_ms') for event in attempts]),
            'latency_missing_logical_records': len(rows) - len(latencies),
            'finished_http_attempts': len(attempts), 'unfinished_http_attempts': unfinished,
            'attempts_with_retry': sum(bool(event.get('will_retry')) for event in attempts),
            'logged_backoff_seconds': sum(event.get('wait_seconds', 0) for event in attempts if _number(event.get('wait_seconds', 0))),
            'potential_duplicate_jobs': sum(row['potential_duplicate'] for row in rows),
            'usage': {**dict(usage_sums), 'reported_attempts': reported, 'missing_usage_attempts': missing_usage,
                      'scope': 'sums over all recorded HTTP attempts and repeats; missing usage is not zero'},
            'cost': None, 'cost_status': 'not_applicable' if all(row['model'].startswith('local:') for row in rows) else 'unknown'}


def _examples(rows, labels, *, failures):
    result = []
    for row in rows:
        outcome = row['outcome']
        valid = outcome['status'] == 'valid' and outcome.get('prediction') in labels
        if (failures and not valid) or (not failures and valid and outcome['prediction'] != row['gold_label']):
            result.append({'job_id': row['job_id'], 'item_id': row['item_id'], 'group_id': row['group_id'],
                           'repeat': row['repeat'], 'input': row['input'], 'gold_label': row['gold_label'],
                           'outcome': outcome})
    return result


def analyze_experiment(root=ROOT, *, output_path=None, freeze_path=None):
    """Reject every incomplete task before producing any final model comparison."""
    root = Path(root).resolve()
    bundle, datasets, execution = _frozen_inputs(root, freeze_path)
    sources = bundle.get('metadata', {}).get('execution_source_sha256', {})
    if sources.get('analyze.py') != hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest():
        raise ValueError('Actual analyze.py source must be included in the frozen source mapping')
    settings = _settings(bundle, datasets)
    # Validate every task before computing/writing any partial "final" analysis.
    journal_protocol_hash = execution.get('_journal_protocol_hash', bundle['bundle_sha256'])
    matrices = {alias: _read_matrix(root / f'artifacts/{alias}/matrix.sqlite', _plan(dataset, execution), journal_protocol_hash, dataset['schema'], execution)
                for alias, dataset in datasets.items()}
    result = {'schema_version': 1, 'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'bundle_sha256': bundle['bundle_sha256'], 'analysis_policy': settings, 'primary_repeat': 0,
              'datasets': {}, 'limitations': list(LIMITATIONS)}
    all_comparisons = []
    for alias, dataset in datasets.items():
        labels = dataset['data']['labels']; rows = matrices[alias]; groups = defaultdict(list)
        for row in rows:
            groups[row['model']].append(row)
        primary = {name: [row for row in values if row['repeat'] == 0] for name, values in groups.items()}
        policy = settings['datasets'][alias]
        kwargs = {'iterations': settings['bootstrap_iterations'], 'seed': settings['seed'], 'stratify': policy['bootstrap_stratify']}
        data = {'dataset_id': dataset['data']['dataset_id'], 'test_records': len(dataset['test_records']),
                'independent_groups': len({row['group_id'] for row in dataset['test_records']}),
                'labels': labels, 'models': {}, 'confirmatory_comparisons': [],
                'source': dataset['data'].get('source'), 'sampling': dataset['data'].get('sampling'),
                'limitations': dataset['data'].get('limitations', [])}
        for name, values in groups.items():
            metric = score(primary[name], labels)
            supported = [label for label in labels if metric['per_class'][label]['support'] > 0]
            interval = paired_bootstrap(primary[name], primary[name], labels, **kwargs)
            data['models'][name] = {'score': metric,
                'supported_label_macro_f1': float(np.mean([metric['per_class'][label]['f1'] for label in supported])),
                'supported_labels': supported, 'absent_gold_labels': [label for label in labels if label not in supported],
                'interval': {'accuracy': interval['a']['accuracy'], 'macro_f1': interval['a']['macro_f1'],
                             **{key: interval[key] for key in ('method', 'iterations', 'seed', 'stratify', 'stratum_support', 'rare_strata', 'interval_scope', 'insufficient_independent_groups')}},
                'mean_group_accuracy_interval': None, 'stability': repeat_stability(values, labels),
                'operational': _operational(values), 'primary_operational': _operational(primary[name]),
                'failures': _examples(values, labels, failures=True), 'misclassifications': _examples(values, labels, failures=False)}
        jev = 'jev:' + execution['remote_models']['jev']
        peers = ['gemini:' + execution['remote_models']['gemini'], 'local:' + policy['best_local_family']]
        for peer in peers:
            a, b = primary[jev], primary[peer]
            one_row_per_group = len({row['group_id'] for row in a}) == len(a)
            test = mcnemar_exact(a, b, labels) if one_row_per_group else paired_cluster_test(
                a, b, labels, iterations=settings['permutation_iterations'], seed=settings['seed'])
            comparison = {'a': jev, 'b': peer, 'difference_direction': 'a minus b',
                          'bootstrap': paired_bootstrap(a, b, labels, **kwargs), 'accuracy_test': test,
                          'tested_metric': 'all_planned_accuracy', 'other_metrics': 'exploratory CI only'}
            data['confirmatory_comparisons'].append(comparison)
            all_comparisons.append(comparison)
        result['datasets'][alias] = data
    adjusted = holm_adjust([comparison['accuracy_test']['p_value'] for comparison in all_comparisons])
    for comparison, value in zip(all_comparisons, adjusted):
        comparison.update(holm_p_value=value, reject_equal_accuracy_at_alpha=value <= settings['alpha'])
    result.update(holm_family_size=len(all_comparisons), holm_family='two Accuracy comparisons per all frozen formal tasks')
    result = _safe(result, _secrets(result))
    if output_path is not None:
        target = Path(output_path).resolve()
        selected_freeze = Path(freeze_path) if freeze_path is not None else root / 'frozen.json'
        if not selected_freeze.is_absolute():
            selected_freeze = root / selected_freeze
        protected = {selected_freeze.resolve()} | {(root / artifact['path']).resolve() for artifact in bundle['artifacts']}
        protected.update(root / f'artifacts/{alias}/matrix.sqlite' for alias in datasets)
        if target in protected:
            raise ValueError('Analysis output cannot overwrite frozen inputs or journals')
        _atomic_json(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze-path', type=Path,
                        help='Explicit frozen/continuation manifest; defaults to frozen.json')
    args = parser.parse_args(argv)
    try:
        result = analyze_experiment(args.root, output_path=args.output or args.root / 'artifacts/final_analysis.json',
                                    freeze_path=args.freeze_path)
    except Exception as exc:
        print(json.dumps({'status': 'analysis_refused', 'error_type': type(exc).__name__}))
        return 2
    print(json.dumps({'status': 'complete', 'bundle_sha256': result['bundle_sha256'],
                      'datasets': list(result['datasets']), 'holm_family_size': result['holm_family_size']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
