"""Synthetic journals exercise final analysis without predictions or real APIs."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_run import fixture_bundle
from scientific.freeze import freeze_bundle
from scientific.journal import DurableJournal
from scientific.run import _frozen_inputs, _plan
from scientific.remote import build_payload, parse_response


def module():
    try:
        return importlib.import_module('scientific.analyze')
    except ImportError:
        pytest.fail('Final analysis module is not implemented')


def complete_fixture(root, *, second=False, leave_pending=False, absent_label=False):
    fixture_bundle(root)
    bundle=json.loads((root/'frozen.json').read_text())
    paths=[artifact['path'] for artifact in bundle['artifacts']]
    metadata=bundle['metadata']
    source=Path(__file__).resolve().parents[1]/'analyze.py'
    shutil.copyfile(source,root/'analyze.py')
    paths.append('analyze.py')
    metadata['execution_source_sha256']['analyze.py']=hashlib.sha256(source.read_bytes()).hexdigest()
    schemas=json.loads((root/'data/task_schemas_candidate.json').read_text())
    datasets=['massive']
    if absent_label:
        path=root/'data/massive_candidate.json'
        data=json.loads(path.read_text());data['labels'].append('unseen')
        path.write_text(json.dumps(data))
        schemas['massive']['labels'].append('unseen')
        schemas['massive']['definitions']['unseen']='没有实际测试支持的类别'
        path=root/'artifacts/massive/local_dev_report.json'
        report=json.loads(path.read_text());report['labels']=data['labels']
        path.write_text(json.dumps(report))
    if second:
        source=json.loads((root/'data/massive_candidate.json').read_text())
        source['dataset_id']='fixture-rag'
        for row in source['records']:
            row['input']={'query':'共同问题','passage':row['input']['text']}
            if row['split']=='test':row['group_id']='one-query'
        (root/'data/rag_candidate.json').write_text(json.dumps(source))
        destination=root/'artifacts/rag';destination.mkdir()
        for name in ['local_models.joblib','local_dev_report.json']:
            shutil.copyfile(root/f'artifacts/massive/{name}',destination/name)
        schemas['rag']={**schemas['massive'],'task':'relevance'}
        paths.extend(['data/rag_candidate.json','artifacts/rag/local_models.joblib','artifacts/rag/local_dev_report.json'])
        datasets.append('rag')
    (root/'data/task_schemas_candidate.json').write_text(json.dumps(schemas))
    for alias in datasets:
        path=root/f'artifacts/{alias}/local_dev_report.json'
        report=json.loads(path.read_text())
        for family,data in report['families'].items():
            data['selection']['validation_macro_f1']=.9 if family=='lr' else .2
        path.write_text(json.dumps(report))
    metadata['dataset_ids']=datasets
    metadata['execution']['remote_base_urls']={provider:'https://unused.invalid/v1' for provider in ['jev','gemini']}
    metadata['analysis']={'primary_repeat':0,'bootstrap_iterations':100,'permutation_iterations':100,
        'seed':42,'alpha':.05,'datasets':{alias:{'bootstrap_stratify':'gold_label' if alias=='massive' else None,
                                              'best_local_family':'lr'} for alias in datasets}}
    (root/'frozen.json').unlink()
    bundle=freeze_bundle(root,paths,bundle['gates'],metadata)
    _,loaded,execution=_frozen_inputs(root)
    for alias,dataset in loaded.items():
        journal=DurableJournal(root/f'artifacts/{alias}/matrix.sqlite',_plan(dataset,execution),bundle['bundle_sha256'])
        for index,job in enumerate(journal.pending_jobs()):
            if leave_pending and index==0:continue
            refusal=job['model'].startswith('gemini:') and job['item_id']=='test-0'
            outcome={'status':'refusal' if refusal else 'valid','prediction':None if refusal else job['gold_label'],
                     'confidence':None,'confidence_status':'missing','latency_ms':5}
            execution_id=job['job_id']+'-e'
            journal.append_attempt(job['job_id'],{'phase':'started','kind':'job',
                'attempt_id':execution_id,'execution_id':execution_id,'started_at':'2026-10-07T00:00:00+00:00'})
            if not job['model'].startswith('local:'):
                provider,model=job['model'].split(':',1)
                payload=build_payload(provider,model,job['input'],dataset['schema'])
                response={'refusal':'blocked'} if refusal else {'prediction':job['gold_label']}
                outcome=parse_response(provider,response,dataset['data']['labels'])
                started={'phase':'started','attempt_id':job['job_id']+'-a','attempt':1,
                    'execution_id':execution_id,'provider':provider,'endpoint':'https://unused.invalid/v1/chat/completions',
                    'request_model':model,'timeout_seconds':30,'started_at':'2026-10-07T00:00:00+00:00',
                    'request_body':json.dumps(payload,ensure_ascii=False),'request_headers':{'content-type':'application/json'}}
                journal.append_attempt(job['job_id'],started)
                journal.append_attempt(job['job_id'],{**started,'phase':'finished','finished_at':'2026-10-07T00:00:00.005+00:00',
                    'duration_ms':5,'http_status':200,'raw_body':json.dumps(response),'response_headers':{},
                    'response_model':None,'retry_reason':None,'error':None,'wait_seconds':0,'will_retry':False,
                    'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3},'outcome':outcome})
                outcome={**outcome,'transport':{'total_latency_ms':5}}
            journal.record_terminal(job['job_id'],outcome)
    return root


def test_complete_analysis_keeps_refusals_in_denominator_and_direction(tmp_path):
    root=complete_fixture(tmp_path/'complete')
    result=module().analyze_experiment(root)
    data=result['datasets']['massive'];models=data['models']
    assert result['complete'] and result['primary_repeat']==0
    assert models['gemini:gemini-fixed']['score']['planned']==2
    assert models['gemini:gemini-fixed']['score']['accuracy']==.5
    assert models['gemini:gemini-fixed']['score']['valid_only']['accuracy']==1
    assert len(models['gemini:gemini-fixed']['failures'])==3
    pair=data['confirmatory_comparisons'][0]
    assert pair['a']=='jev:jev-fixed' and pair['b']=='gemini:gemini-fixed'
    assert pair['bootstrap']['accuracy_difference']['estimate']==.5
    assert pair['accuracy_test']['a_only_correct']==1
    assert result['holm_family_size']==2
    assert models['gemini:gemini-fixed']['operational']['usage']['total_tokens']==18


def test_incomplete_journal_refuses_final_output_and_is_readonly(tmp_path):
    root=complete_fixture(tmp_path/'pending',leave_pending=True)
    path=root/'artifacts/massive/matrix.sqlite';before=path.read_bytes()
    with pytest.raises(ValueError,match='terminal|complete|pending'):
        module().analyze_experiment(root,output_path=root/'final.json')
    assert not (root/'final.json').exists()
    assert path.read_bytes()==before


def test_missing_planned_repeat_or_changed_gold_refuses_analysis(tmp_path):
    root=complete_fixture(tmp_path/'missing')
    path=root/'artifacts/massive/matrix.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM planned WHERE job_id=(SELECT job_id FROM planned LIMIT 1)")
    with pytest.raises(ValueError,match='manifest|matrix'):
        module().analyze_experiment(root)


def test_group_inference_and_holm_include_all_tasks(tmp_path):
    root=complete_fixture(tmp_path/'two',second=True)
    result=module().analyze_experiment(root)
    comparisons=[c for d in result['datasets'].values() for c in d['confirmatory_comparisons']]
    assert len(comparisons)==result['holm_family_size']==4
    from scientific.statistics import holm_adjust
    assert [c['holm_p_value'] for c in comparisons]==pytest.approx(holm_adjust([c['accuracy_test']['p_value'] for c in comparisons]))
    rag=result['datasets']['rag']
    assert rag['test_records']==2 and rag['independent_groups']==1
    assert rag['confirmatory_comparisons'][0]['accuracy_test']['method']=='exact group sign permutation'
    assert rag['models']['jev:jev-fixed']['interval']['insufficient_independent_groups']
    assert rag['models']['jev:jev-fixed']['stability']['unique_items']==2


def test_changed_frozen_data_and_unfrozen_root_are_rejected(tmp_path):
    root=complete_fixture(tmp_path/'changed')
    (root/'data/massive_candidate.json').write_text('{}')
    with pytest.raises(ValueError,match='hash|artifact'):
        module().analyze_experiment(root)
    with pytest.raises(ValueError,match='frozen'):
        module().analyze_experiment(tmp_path/'unfrozen')


def test_configured_best_local_cannot_be_selected_on_test(tmp_path):
    root=complete_fixture(tmp_path/'bad-selection')
    bundle=json.loads((root/'frozen.json').read_text())
    bundle['metadata']['analysis']['datasets']['massive']['best_local_family']='lightgbm'
    payload={k:v for k,v in bundle.items() if k!='bundle_sha256'}
    bundle['bundle_sha256']=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'frozen.json').write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='validation|best'):
        module().analyze_experiment(root)


def test_written_report_is_json_and_preserves_nonranking_limitations(tmp_path):
    root=complete_fixture(tmp_path/'write')
    target=root/'artifacts/final_analysis.json'
    result=module().analyze_experiment(root,output_path=target)
    assert json.loads(target.read_text())['bundle_sha256']==result['bundle_sha256']
    assert result['limitations'] and 'ranking' not in result
    assert result['datasets']['massive']['models']['local:lr']['stability']['assessable'] is False


def test_absent_labels_keep_fixed_macro_and_supported_macro_separate(tmp_path):
    root=complete_fixture(tmp_path/'absent',absent_label=True)
    model=module().analyze_experiment(root)['datasets']['massive']['models']['local:lr']
    assert model['score']['macro_f1']==pytest.approx(2/3)
    assert model['supported_label_macro_f1']==1
    assert model['absent_gold_labels']==['unseen']
    assert model['mean_group_accuracy_interval'] is None


def test_original_frozen_candidate_analysis_keys_are_supported(tmp_path):
    root=complete_fixture(tmp_path/'legacy')
    bundle=json.loads((root/'frozen.json').read_text())
    bundle['metadata']['analysis']={'primary_repeat':0,'bootstrap_iterations':100,'statistical_seed':42,
        'best_local_by_validation':{'massive':{'model':'local:lr'}},'massive_bootstrap_strata':'gold_label'}
    payload={k:v for k,v in bundle.items() if k!='bundle_sha256'}
    bundle['bundle_sha256']=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'frozen.json').write_text(json.dumps(bundle))
    with sqlite3.connect(root/'artifacts/massive/matrix.sqlite') as conn:
        conn.execute("UPDATE metadata SET value=? WHERE key='protocol_hash'",(bundle['bundle_sha256'],))
    result=module().analyze_experiment(root)
    assert result['analysis_policy']['datasets']['massive']['bootstrap_stratify']=='gold_label'


def test_output_cannot_overwrite_journal_or_frozen_inputs(tmp_path):
    root=complete_fixture(tmp_path/'protected')
    for path in [root/'frozen.json',root/'artifacts/massive/matrix.sqlite',root/'data/massive_candidate.json']:
        before=path.read_bytes()
        with pytest.raises(ValueError,match='overwrite'):
            module().analyze_experiment(root,output_path=path)
        assert path.read_bytes()==before


def test_missing_paid_response_audit_cannot_be_called_final(tmp_path):
    root=complete_fixture(tmp_path/'missing-http-log')
    path=root/'artifacts/massive/matrix.sqlite'
    with sqlite3.connect(path) as conn:
        ident=conn.execute("SELECT job_id FROM planned WHERE json_extract(job_json,'$.model') LIKE 'jev:%' LIMIT 1").fetchone()[0]
        conn.execute("DELETE FROM attempts WHERE job_id=? AND json_extract(event_json,'$.phase')='finished'",(ident,))
    with pytest.raises(ValueError,match='attempt|HTTP|audit'):
        module().analyze_experiment(root)


def mutate_http(root, transform):
    with sqlite3.connect(root/'artifacts/massive/matrix.sqlite') as conn:
        ident=conn.execute("SELECT job_id FROM planned WHERE json_extract(job_json,'$.model') LIKE 'jev:%' LIMIT 1").fetchone()[0]
        events=conn.execute('SELECT event_id,event_json FROM attempts WHERE job_id=? ORDER BY event_id',(ident,)).fetchall()
        for event_id,raw in events:
            event=json.loads(raw)
            if event.get('kind')=='job':continue
            transform(event)
            conn.execute('UPDATE attempts SET event_json=? WHERE event_id=?',(json.dumps(event),event_id))
    return ident


def test_analyzer_source_cannot_be_omitted_or_differ_from_running_module(tmp_path):
    for mode in ['omitted','changed']:
        root=complete_fixture(tmp_path/mode)
        bundle=json.loads((root/'frozen.json').read_text())
        paths=[a['path'] for a in bundle['artifacts']]
        if mode=='omitted':
            paths.remove('analyze.py');bundle['metadata']['execution_source_sha256'].pop('analyze.py')
        else:
            (root/'analyze.py').write_text('# unrelated source')
            bundle['metadata']['execution_source_sha256']['analyze.py']=hashlib.sha256((root/'analyze.py').read_bytes()).hexdigest()
        (root/'frozen.json').unlink()
        freeze_bundle(root,paths,bundle['gates'],bundle['metadata'])
        with pytest.raises(ValueError,match='source|analy'):
            module().analyze_experiment(root)


@pytest.mark.parametrize('key',['request_body','raw_body','http_status','outcome'])
def test_remote_http_evidence_requires_complete_schema(tmp_path,key):
    root=complete_fixture(tmp_path/key)
    mutate_http(root,lambda event:event.pop(key,None) if event['phase']=='finished' else None)
    with pytest.raises(ValueError,match='HTTP|audit|attempt'):
        module().analyze_experiment(root)


def test_journal_request_must_match_frozen_input_model_and_schema(tmp_path):
    root=complete_fixture(tmp_path/'leak')
    def leak(event):
        payload=json.loads(event['request_body']);payload['gold_label']='yes'
        event['request_body']=json.dumps(payload)
    mutate_http(root,leak)
    with pytest.raises(ValueError,match='request|payload|HTTP'):
        module().analyze_experiment(root)


def test_terminal_prediction_must_match_finished_and_raw_response(tmp_path):
    root=complete_fixture(tmp_path/'mismatch')
    mutate_http(root,lambda event:event.update(raw_body=json.dumps({'prediction':'wrong'})) if event['phase']=='finished' else None)
    with pytest.raises(ValueError,match='outcome|response|HTTP'):
        module().analyze_experiment(root)


def test_network_failure_with_null_http_status_is_complete_evidence(tmp_path):
    root=complete_fixture(tmp_path/'network')
    outcome={'status':'network_error','prediction':None,'confidence':None,'confidence_status':'missing',
             'raw_output':'','error':'URLError: offline'}
    def offline(event):
        if event['phase']=='finished':
            event.update(http_status=None,raw_body='',usage=None,outcome=outcome,error=outcome['error'],retry_reason='network_error')
    ident=mutate_http(root,offline)
    with sqlite3.connect(root/'artifacts/massive/matrix.sqlite') as conn:
        conn.execute('UPDATE planned SET outcome_json=? WHERE job_id=?',(json.dumps(outcome),ident))
    result=module().analyze_experiment(root)
    assert result['complete']
    assert result['datasets']['massive']['models']['jev:jev-fixed']['score']['accuracy']==.5


def test_analysis_redacts_secret_echo_without_reading_credentials(tmp_path):
    root=complete_fixture(tmp_path/'secret')
    with sqlite3.connect(root/'artifacts/massive/matrix.sqlite') as conn:
        ident,raw=conn.execute("SELECT job_id,outcome_json FROM planned WHERE json_extract(job_json,'$.model')='local:lr' LIMIT 1").fetchone()
        outcome=json.loads(raw);outcome.update(status='failed',prediction=None,raw_output={'api_key':'synthetic-ordinary-value','echo':'synthetic-ordinary-value'})
        conn.execute('UPDATE planned SET outcome_json=? WHERE job_id=?',(json.dumps(outcome),ident))
    result=module().analyze_experiment(root)
    assert 'synthetic-ordinary-value' not in json.dumps(result)


def test_changed_logical_terminal_cannot_overrule_paid_response(tmp_path):
    root=complete_fixture(tmp_path/'terminal-change')
    with sqlite3.connect(root/'artifacts/massive/matrix.sqlite') as conn:
        ident,raw=conn.execute("SELECT job_id,outcome_json FROM planned WHERE json_extract(job_json,'$.model') LIKE 'gemini:%' AND json_extract(outcome_json,'$.status')='refusal' LIMIT 1").fetchone()
        outcome=json.loads(raw);outcome.update(status='valid',prediction='yes')
        conn.execute('UPDATE planned SET outcome_json=? WHERE job_id=?',(json.dumps(outcome),ident))
    with pytest.raises(ValueError,match='terminal|outcome'):
        module().analyze_experiment(root)


def test_paired_retry_with_nonterminal_first_response_is_allowed(tmp_path):
    root=complete_fixture(tmp_path/'retry-complete')
    path=root/'artifacts/massive/matrix.sqlite'
    with sqlite3.connect(path) as conn:
        ident=conn.execute("SELECT job_id FROM planned WHERE json_extract(job_json,'$.model') LIKE 'jev:%' LIMIT 1").fetchone()[0]
        events=[json.loads(row[0]) for row in conn.execute('SELECT event_json FROM attempts WHERE job_id=? ORDER BY event_id',(ident,))]
        job,start,finish=events
        retry={**finish,'will_retry':True,'outcome':None,'http_status':429,'retry_reason':'http_429',
               'raw_body':'rate limited','usage':None,'wait_seconds':1}
        start2={**start,'attempt_id':start['attempt_id']+'-retry','attempt':2}
        finish2={**finish,**start2,'phase':'finished'}
        conn.execute('DELETE FROM attempts WHERE job_id=?',(ident,))
        for event in [job,start,retry,start2,finish2]:
            conn.execute('INSERT INTO attempts(job_id,event_json,recorded_at) VALUES (?,?,?)',(ident,json.dumps(event),'2026-10-07T00:00:00+00:00'))
    result=module().analyze_experiment(root)
    operations=result['datasets']['massive']['models']['jev:jev-fixed']['operational']
    assert operations['attempts_with_retry']==1 and operations['finished_http_attempts']==7
    assert operations['usage']['missing_usage_attempts']==1


def test_recovered_outcome_retains_complete_final_attempt_evidence(tmp_path):
    root=complete_fixture(tmp_path/'recover')
    path=root/'artifacts/massive/matrix.sqlite'
    with sqlite3.connect(path) as conn:
        ident=conn.execute("SELECT job_id FROM planned WHERE json_extract(job_json,'$.model') LIKE 'jev:%' LIMIT 1").fetchone()[0]
        conn.execute('UPDATE planned SET outcome_json=NULL,terminal_at=NULL WHERE job_id=?',(ident,))
    bundle,datasets,execution=_frozen_inputs(root)
    DurableJournal(path,_plan(datasets['massive'],execution),bundle['bundle_sha256'])
    assert module().analyze_experiment(root)['complete']


def test_actual_audited_client_events_work_without_network(tmp_path,monkeypatch):
    from scientific.remote import AuditedHTTPClient
    from scientific import remote
    root=complete_fixture(tmp_path/'actual-client')
    bundle,datasets,execution=_frozen_inputs(root)
    dataset=datasets['massive'];path=root/'artifacts/massive/matrix.sqlite'
    with sqlite3.connect(path) as conn:
        ident,job_raw=conn.execute("SELECT job_id,job_json FROM planned WHERE json_extract(job_json,'$.model') LIKE 'jev:%' LIMIT 1").fetchone()
        conn.execute('DELETE FROM attempts WHERE job_id=?',(ident,))
        conn.execute('UPDATE planned SET outcome_json=NULL,terminal_at=NULL WHERE job_id=?',(ident,))
    job=json.loads(job_raw)
    class Response:
        status=200
        headers={'content-type':'application/json'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps({'prediction':job['gold_label']}).encode()
    monkeypatch.setattr(remote.urllib.request,'urlopen',lambda *args,**kwargs:Response())
    journal=DurableJournal(path,_plan(dataset,execution),bundle['bundle_sha256'])
    execution_id='actual-offline-client'
    journal.append_attempt(ident,{'kind':'job','phase':'started','attempt_id':execution_id,'execution_id':execution_id})
    response=AuditedHTTPClient('https://unused.invalid/v1','offline-placeholder','jev').send(
        build_payload('jev','jev-fixed',job['input'],dataset['schema']),execution_id=execution_id,
        on_attempt=lambda event:journal.append_attempt(ident,event))
    outcome={**response['outcome'],'transport':{key:value for key,value in response.items() if key!='outcome'}}
    journal.record_terminal(ident,outcome)
    assert module().analyze_experiment(root)['complete']
