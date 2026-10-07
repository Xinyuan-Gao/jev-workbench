"""Synthetic-only figure contract tests; never load real predictions or services."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def figures_module():
    name = 'jev_workbench.scientific.report_figures'
    assert importlib.util.find_spec(name) is not None, 'The final-analysis figure pipeline is not implemented'
    return importlib.import_module(name)


def synthetic_complete_fixture():
    metric = {'complete': True, 'provisional': False, 'planned': 4, 'terminal': 4, 'pending': 0,
              'independent_groups': 2, 'valid': 3, 'failed': 1, 'valid_rate': .75, 'accuracy': .75,
              'macro_f1': 5/6, 'mean_group_accuracy': .75,
              'labels': ['relevant', 'irrelevant'], 'status_counts': {'valid': 3, 'refusal': 1},
              'confusion_columns': ['relevant', 'irrelevant', 'NO_VALID_OUTPUT'],
              'confusion_matrix': [[1, 0, 1], [0, 2, 0]],
              'macro_policy': 'fixed supplied label set; unsupported classes score zero',
              'valid_only': {'denominator': 3, 'accuracy': 1., 'macro_f1': 1.}}
    distribution = lambda n: {'count': n, 'mean_ms': 20. if n else None, 'median_ms': 18. if n else None,
                               'p90_ms': 28. if n else None, 'p95_ms': 29. if n else None}
    operational = {'logical_records': 8, 'repeat_indices': [0, 1], 'logical_latency': distribution(7),
                   'valid_latency': distribution(5), 'failure_latency': distribution(2),
                   'first_http_attempt_latency': distribution(8), 'all_http_attempt_latency': distribution(9),
                   'latency_missing_logical_records': 1, 'finished_http_attempts': 9,
                   'unfinished_http_attempts': 0, 'attempts_with_retry': 1}
    model = {'score': metric, 'interval': {'accuracy': {'estimate': .75, 'ci95': [.5, 1.]},
                                        'macro_f1': {'estimate': 5/6, 'ci95': [.4, 1.]},
                                        'method': 'paired percentile cluster bootstrap', 'iterations': 100,
                                        'seed': 7, 'stratify': None, 'interval_scope': 'synthetic empirical',
                                        'rare_strata': [], 'insufficient_independent_groups': False},
             'stability': {'rounds': 2, 'unique_items': 4, 'independent_groups': 2, 'assessable': True,
                           'primary_repeat': 0, 'per_round': {'0': metric, '1': copy.deepcopy(metric)},
                           'outcome_agreement_with_primary': [{'repeat': 1, 'rate': .75, 'denominator': 4}],
                           'prediction_agreement_conditional_on_both_valid': [{'repeat': 1, 'rate': 1., 'denominator': 2}],
                           'all_rounds_correct_rate': .5,
                           'inference_scope': 'descriptive; repeats do not add independent samples'},
             'operational': operational, 'failures': [], 'misclassifications': []}
    return {'schema_version': 1, 'synthetic_fixture': True, 'complete': True, 'bundle_sha256': 'b'*64,
            'primary_repeat': 0, 'analysis_policy': {'primary_repeat': 0, 'datasets': {'rag': {'best_local_family': 'lr', 'bootstrap_stratify': None}}},
            'datasets': {'rag': {'dataset_id': 'SYNTHETIC_TEST_ONLY', 'test_records': 4, 'independent_groups': 2,
                                 'labels': ['relevant', 'irrelevant'], 'models': {
                                     'jev:synthetic': copy.deepcopy(model), 'gemini:synthetic': copy.deepcopy(model),
                                     'local:lr': copy.deepcopy(model)}, 'limitations': ['synthetic fixture']}},
            'limitations': ['synthetic test data; no formal model results']}


def write_fixture(tmp_path, value):
    p = tmp_path / 'final_analysis.json'
    p.write_text(json.dumps(value), encoding='utf-8')
    return p


@pytest.mark.parametrize('bad', ['global', 'model', 'pending', 'round_pending', 'bundle', 'counts', 'confusion', 'interval', 'macro', 'latency', 'top_pending'])
def test_incomplete_or_inconsistent_analysis_is_rejected_before_any_files(tmp_path, bad):
    module = figures_module()
    value = synthetic_complete_fixture()
    score = value['datasets']['rag']['models']['jev:synthetic']['score']
    if bad == 'global': value['complete'] = False
    elif bad == 'model': score['complete'] = False
    elif bad == 'pending': score['pending'] = 1
    elif bad == 'round_pending': value['datasets']['rag']['models']['jev:synthetic']['stability']['per_round']['1']['pending'] = 1
    elif bad == 'bundle': value['bundle_sha256'] = 'missing'
    elif bad == 'counts': score['valid'] = 4
    elif bad == 'confusion': score['confusion_columns'] = ['relevant', 'irrelevant', 'removed_failure']
    elif bad == 'interval': value['datasets']['rag']['models']['jev:synthetic']['interval']['accuracy']['estimate'] = .9
    elif bad == 'macro': score['macro_f1'] = .1
    elif bad == 'latency': value['datasets']['rag']['models']['jev:synthetic']['operational']['latency_missing_logical_records'] = 0
    elif bad == 'top_pending': value['pending'] = 1
    output = tmp_path / 'figures'
    with pytest.raises(ValueError): module.export_report(write_fixture(tmp_path, value), output, allow_synthetic=True)
    assert not output.exists()


def test_synthetic_data_requires_explicit_test_mode(tmp_path):
    module = figures_module()
    with pytest.raises(ValueError, match='synthetic'):
        module.export_report(write_fixture(tmp_path, synthetic_complete_fixture()), tmp_path/'figures')


def test_complete_fixture_exports_traceable_figures_and_preserves_failure_column(tmp_path):
    module = figures_module()
    source = write_fixture(tmp_path, synthetic_complete_fixture())
    output = tmp_path / 'synthetic_figures'
    manifest = module.export_report(source, output, allow_synthetic=True)
    saved = json.loads((output/'assets_manifest.json').read_text())
    assert saved == manifest
    assert manifest['purpose'] == 'synthetic_contract_test'
    assert manifest['bundle_sha256'] == 'b'*64
    assert manifest['analysis_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(manifest['figures']) >= 8
    for figure in manifest['figures']:
        assert figure['title'] and figure['caption'] and figure['denominators'] and figure['limitations']
        assert figure['analysis_sha256'] == manifest['analysis_sha256']
        assert figure['bundle_sha256'] == manifest['bundle_sha256']
        assert (output/figure['png']).read_bytes().startswith(b'\x89PNG')
        assert '<svg' in (output/figure['svg']).read_text()
        assert json.loads((output/figure['manifest']).read_text()) == figure
    cms = [f for f in manifest['figures'] if f['kind'] == 'confusion']
    assert len(cms) == 3
    for f in cms:
        assert f['data']['columns'][-1] == 'NO_VALID_OUTPUT'
        assert sum(row[-1] for row in f['data']['matrix']) == 1
    latency = [f for f in manifest['figures'] if f['kind'] == 'latency']
    for f in latency:
        assert f['data']['latency_missing_logical_records'] == 1
        assert f['denominators']['logical_records'] == 8
    rag_accuracy = next(f for f in manifest['figures'] if f['kind'] == 'accuracy')
    assert 'pair-weighted' in rag_accuracy['caption']
    assert 'query' in rag_accuracy['caption']
    f1 = next(f for f in manifest['figures'] if f['kind'] == 'macro_f1')
    assert 'exploratory' in f1['caption']
