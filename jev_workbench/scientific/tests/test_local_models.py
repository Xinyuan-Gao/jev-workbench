"""Contract tests: dev selection, train-only features, and genuine seed fits."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def local():
    return importlib.import_module('scientific.local_models')


def rows(split, count=24, rag=False):
    output=[]
    for i in range(count):
        positive = i % 2 == 0
        text = ('篮球体育比赛' if positive else '金融股票投资') + str(i)
        item = {'query': '篮球体育规则' if positive else '金融股票行情', 'passage': text} if rag else {'text': text}
        output.append({'id':f'{split}-{i}', 'group_id':f'{split}-g-{i}', 'split':split,
                       'input':item, 'label':'sports' if positive else 'finance'})
    return output


def test_predeclared_grids_cover_twelve_unique_configs_and_fixed_seed_policy(local):
    assert set(local.CANDIDATE_GRIDS) == {'lr', 'linear_svc', 'xgboost', 'lightgbm'}
    for configs in local.CANDIDATE_GRIDS.values():
        assert len(configs) >= 12
        assert len({json.dumps(config, sort_keys=True) for config in configs}) == len(configs)
    assert len(local.DEFAULT_SEEDS) == 5 and len(set(local.DEFAULT_SEEDS)) == 5


def test_features_fit_train_only_and_unknown_dev_does_not_change_vocab(local):
    train = rows('train')
    features = local.InputFeatures(seed=1)
    features.fit(train)
    before = dict(features.vectorizer.vocabulary_)
    val = rows('validation', 2); val[0]['input']['text'] = 'ΩUNSEEN_SENTINELΩ'
    features.transform(val)
    assert features.vectorizer.vocabulary_ == before
    assert not any('Ω' in token for token in before)
    with pytest.raises(ValueError):
        local.InputFeatures().fit(rows('test'))


def test_rag_has_separate_embedding_interactions_and_train_only_svd(local):
    features = local.InputFeatures(dense=True, svd_components=8, seed=1).fit(rows('train', rag=True))
    val = rows('validation', 2, rag=True)
    first = features.transform(val)
    swapped = [{**r, 'input': {'query': r['input']['passage'], 'passage': r['input']['query']}} for r in val]
    second = features.transform(swapped)
    assert first.shape[1] == 4 * features.embedding_dimensions + 3
    assert features.feature_names[-3:] == ['cosine', 'token_jaccard', 'query_token_coverage']
    assert not np.allclose(first, second)
    components = features.svd.components_.copy()
    features.transform([{**val[0], 'input': {'query':'ΩUNSEEN', 'passage':'ΩUNSEEN'}}])
    assert np.array_equal(components, features.svd.components_)
    assert all(record_id.startswith('train-') for record_id in features.fit_record_ids)


def test_tuning_scores_all_candidates_and_deterministic_models_run_once(local, tmp_path):
    path = tmp_path / 'dev.json'
    result = local.train_local_models(rows('train'), rows('validation', 10), ['finance', 'sports'],
                                     families=['majority', 'lr', 'linear_svc'], output_path=path)
    report = result['report']
    for family in ['lr', 'linear_svc']:
        candidates = report['families'][family]['candidates']
        assert len(candidates) >= 12
        assert all(candidate['status'] == 'valid' for candidate in candidates)
        selected = report['families'][family]['selection']
        assert selected['validation_macro_f1'] == max(candidate['validation_macro_f1'] for candidate in candidates)
        assert len(result['models'][family]) == 1
        assert report['families'][family]['repeat_policy'] == 'deterministic_primary_only'
    assert json.loads(path.read_text())['selection_split'] == 'validation'
    assert report['versions']['scikit-learn']
    assert all(candidate['model_fit_ms'] >= 0 for candidate in report['families']['lr']['candidates'])


def test_test_labels_cannot_influence_fit_or_prediction(local):
    train = rows('train'); val = rows('validation', 8)
    with pytest.raises(ValueError):
        local.train_local_models(train + rows('test', 2), val, ['finance', 'sports'], families=['lr'])
    with pytest.raises(ValueError):
        local.train_local_models(train, rows('test', 2), ['finance', 'sports'], families=['lr'])
    model = local.train_local_models(train, val, ['finance', 'sports'], families=['lr'])['models']['lr'][0]
    class BlindRecord(dict):
        def __getitem__(self, key):
            if key in {'label', 'gold_label'}:
                raise AssertionError('Prediction read test gold')
            return super().__getitem__(key)
    tests = [BlindRecord(r) for r in rows('test', 4)]
    predictions = model.predict(tests)
    assert len(predictions) == 4
    assert all(result['status'] == 'valid' for result in predictions)
    assert all(result['latency_ms'] >= 0 for result in predictions)


def test_tree_models_train_five_real_seeds_and_log_all_candidates(local):
    result = local.train_local_models(rows('train', 30), rows('validation', 8), ['finance', 'sports'],
                                     families=['xgboost', 'lightgbm'])
    for family in ['xgboost', 'lightgbm']:
        models = result['models'][family]
        repeats = result['report']['families'][family]['repeats']
        assert len(models) == len(repeats) == 5
        assert len({id(model.estimator) for model in models}) == 5
        assert len({id(model.features.svd) for model in models}) == 5
        assert [model.seed for model in models] == list(local.DEFAULT_SEEDS)
        assert all(repeat['fit_performed'] and repeat['model_fit_ms'] >= 0 for repeat in repeats)
        assert len(result['report']['families'][family]['candidates']) >= 12
        assert all(row['prediction'] in {'finance', 'sports'} for row in models[0].predict(rows('test', 2)))


def test_tie_break_uses_prespecified_complexity_order(local):
    candidates = [{'candidate_index': 2, 'validation_macro_f1': 0.8, 'status':'valid'},
                  {'candidate_index': 0, 'validation_macro_f1': 0.8, 'status':'valid'},
                  {'candidate_index': 1, 'validation_macro_f1': 0.7, 'status':'valid'}]
    assert local.select_candidate(candidates)['candidate_index'] == 0


def test_candidate_report_is_saved_before_next_candidate(local, tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    original = local._estimator
    observed = []
    def estimator(family, params, seed, classes):
        assert path.exists(), 'No initial report persisted'
        current = json.loads(path.read_text())
        observed.append(len(current['families'][family]['candidates']))
        return original(family, params, seed, classes)
    monkeypatch.setattr(local, '_estimator', estimator)
    result = local.train_local_models(rows('train'), rows('validation', 8), ['finance', 'sports'],
                                     families=['lr'], output_path=path)
    assert observed == list(range(12))
    assert result['report']['status'] == 'complete'


def test_report_has_fixed_label_support_and_model_roundtrips(local, tmp_path):
    import joblib
    validation = rows('validation', 8)
    validation = [record for record in validation if record['label'] == 'sports']
    result = local.train_local_models(rows('train'), validation, ['finance', 'sports'], families=['lr'])
    assert result['report']['validation_label_support'] == {'finance':0,'sports':4}
    model = result['models']['lr'][0]
    target = tmp_path / 'model.joblib'; joblib.dump(model,target)
    restored = joblib.load(target)
    assert [r['prediction'] for r in restored.predict(validation)] == [r['prediction'] for r in model.predict(validation)]


def test_all_failed_candidates_are_preserved_in_progress_report(local, tmp_path, monkeypatch):
    path = tmp_path / 'failure.json'
    def broken(*args):
        raise RuntimeError('local estimator unavailable')
    monkeypatch.setattr(local, '_estimator', broken)
    with pytest.raises(RuntimeError):
        local.train_local_models(rows('train'), rows('validation', 8), ['finance','sports'],families=['lr'],output_path=path)
    report = json.loads(path.read_text())
    assert report['status'] == 'failed'
    assert len(report['families']['lr']['candidates']) == 12
    assert all(row['status']=='failed' for row in report['families']['lr']['candidates'])


def test_multiclass_tree_objective_is_explicit_in_saved_parameters(local):
    assert local._estimator('xgboost',local.CANDIDATE_GRIDS['xgboost'][0],1,60).get_params()['objective']=='multi:softprob'
    assert local._estimator('lightgbm',local.CANDIDATE_GRIDS['lightgbm'][0],1,60).get_params()['objective']=='multiclass'


def test_failed_stochastic_repeat_is_preserved(local, tmp_path, monkeypatch):
    path=tmp_path/'repeat-failure.json'
    original=local._estimator
    calls=[]
    def estimator(family,params,seed,classes):
        calls.append(seed)
        if seed != local.DEFAULT_SEEDS[0]:
            raise RuntimeError('repeat fit failure')
        return original(family,params,seed,classes)
    monkeypatch.setattr(local,'_estimator',estimator)
    with pytest.raises(RuntimeError):
        local.train_local_models(rows('train',30),rows('validation',8),['finance','sports'],families=['xgboost'],output_path=path)
    report=json.loads(path.read_text())
    assert report['status']=='failed'
    assert report['families']['xgboost']['repeats'][-1]['status']=='failed'
    assert report['families']['xgboost']['repeats'][-1]['seed']==local.DEFAULT_SEEDS[1]


def test_sparse_rag_features_and_real_lr_svc_training(local):
    from scipy import sparse
    train=rows('train',30,rag=True); validation=rows('validation',8,rag=True)
    features=local.InputFeatures(dense=False).fit(train)
    matrix=features.transform(validation)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape==(8,4*features.embedding_dimensions+3)
    assert np.isfinite(matrix.data).all()
    result=local.train_local_models(train,validation,['finance','sports'],families=['lr','linear_svc'])
    for family in ['lr','linear_svc']:
        assert len(result['report']['families'][family]['candidates'])==12
        assert all(row['status']=='valid' for row in result['models'][family][0].predict(validation))


def test_transform_error_is_persisted_as_failed_not_running(local,tmp_path,monkeypatch):
    path=tmp_path/'transform-failure.json'
    def broken_transform(self,records):
        raise TypeError('transform failure fixture')
    monkeypatch.setattr(local.InputFeatures,'transform',broken_transform)
    with pytest.raises(TypeError):
        local.train_local_models(rows('train'),rows('validation',8),['finance','sports'],families=['lr'],output_path=path)
    report=json.loads(path.read_text())
    assert report['status']=='failed' and report['error_type']=='TypeError'
    assert report['families']['lr']['status']=='failed'
