"""Offline CLI execution contracts; fake models/clients never call a gateway."""
import importlib
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import sys

import joblib
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scientific.freeze import REQUIRED_GATES, freeze_bundle
from scientific.journal import DurableJournal, execute_jobs

SOURCE_FILES=('__init__.py','run.py','remote.py','journal.py','statistics.py','datasets.py','local_models.py','freeze.py','task_schemas.py')


class FakeModel:
    calls=[]
    def __init__(self,family,seed):
        self.family,self.seed=family,seed
    def predict(self,records):
        assert all(set(record)=={'input'} for record in records)
        self.calls.extend((self.family,record['input']) for record in records)
        return [{'status':'valid','prediction':'yes','confidence':None,'confidence_status':'missing'} for _ in records]


class FakeClient:
    calls=[]
    def __init__(self,base_url,api_key,provider,**kwargs):
        self.provider=provider
    def send(self,payload,*,on_attempt,**kwargs):
        attempt_id=str(len(self.calls)+1)
        on_attempt({'phase':'started','attempt_id':attempt_id,'provider':self.provider})
        self.calls.append((self.provider,payload))
        outcome={'status':'refusal','prediction':None} if len(self.calls)==1 else {'status':'valid','prediction':'yes','confidence':None}
        on_attempt({'phase':'finished','attempt_id':attempt_id,'provider':self.provider,
                    'will_retry':False,'outcome':outcome})
        return {'outcome':outcome,'attempts':[],'metadata':{'cost':None,'cost_status':'unknown'},'response':{},'total_latency_ms':1}


@pytest.fixture
def run():
    FakeModel.calls=[]; FakeClient.calls=[]
    return importlib.import_module('scientific.run')


def fixture_bundle(root, *, gates=None, execution=None):
    root.mkdir(parents=True,exist_ok=True)
    labels=['yes','no']
    data={'dataset_id':'fake-official-dataset','labels':labels,'records':[
        {'id':'test-0','group_id':'query-0','split':'test','label':'yes','input':{'text':'safe 0'}},
        {'id':'test-1','group_id':'query-1','split':'test','label':'no','input':{'text':'safe 1'}},
        {'id':'train-0','group_id':'train-0','split':'train','label':'yes','input':{'text':'TRAIN_NOT_FOR_TEST'}}]}
    schemas={'massive':{'task':'classification','labels':labels,'definitions':{'yes':'符合','no':'不符合'},'instructions':'选择类别'}}
    families=['majority','lr','linear_svc','xgboost','lightgbm']
    models={family:[FakeModel(family,seed) for seed in range(5 if family in {'xgboost','lightgbm'} else 1)] for family in families}
    report={'status':'complete','test_used':False,'labels':labels,'families':{family:{
        'selection':{'params':{},'validation_macro_f1':.5},'repeats':[{'fit_performed':True} for _ in values]} for family,values in models.items()}}
    artifacts=['data/massive_candidate.json','data/task_schemas_candidate.json','artifacts/massive/local_models.joblib','artifacts/massive/local_dev_report.json']
    for path,value in [(artifacts[0],data),(artifacts[1],schemas),(artifacts[3],report)]:
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value))
    joblib.dump(models,root/artifacts[2])
    selected={gate:{'passed':True,'evidence':'offline fixture'} for gate in REQUIRED_GATES}
    metadata={'dataset_ids':['massive'],'execution':{
        'remote_models':{'jev':'jev-fixed','gemini':'gemini-fixed'},'remote_repeats':3,'order_seed':17,
        'remote_base_urls':{'jev':'https://unused.invalid','gemini':'https://unused.invalid'},
        'max_http_attempts':3,'allow_indeterminate_retry':True,'max_indeterminate_retries_per_job':1,
        'budget':{'unknown_cost_policy':'bounded_calls','max_total_http_attempts':100}}}
    metadata['execution_source_sha256']={}
    for source in SOURCE_FILES:
        actual=Path(__file__).resolve().parents[1]/source
        shutil.copyfile(actual,root/source)
        metadata['execution_source_sha256'][source]=hashlib.sha256(actual.read_bytes()).hexdigest()
        artifacts.append(source)
    if execution:metadata['execution'].update(execution)
    if gates:selected.update(gates)
    freeze_bundle(root,artifacts,selected,metadata)
    return root


def test_freeze_required_before_any_model_load_or_prediction(run,tmp_path,monkeypatch):
    root=tmp_path/'unfrozen';root.mkdir()
    monkeypatch.setattr(run.joblib,'load',lambda _:pytest.fail('Loaded before freeze verification'))
    with pytest.raises(ValueError,match='frozen'):
        run.run_experiment(root,local=True)
    assert FakeModel.calls==[] and FakeClient.calls==[]


def test_false_gate_or_changed_artifact_blocks_all_execution(run,tmp_path):
    root=fixture_bundle(tmp_path/'changed')
    (root/'data/massive_candidate.json').write_text('{}')
    with pytest.raises(ValueError):run.run_experiment(root,local=True)
    root=fixture_bundle(tmp_path/'false-gate')
    bundle=json.loads((root/'frozen.json').read_text());bundle['gates']['executable_budget']['passed']=False
    (root/'frozen.json').write_text(json.dumps(bundle))
    with pytest.raises(ValueError):run.run_experiment(root,local=True)
    assert not FakeModel.calls and not FakeClient.calls


def test_local_only_preserves_remote_planned_denominator_and_resume(run,tmp_path):
    root=fixture_bundle(tmp_path/'local')
    result=run.run_experiment(root,local=True)
    summary=result['datasets']['massive']
    assert summary['planned']==38 and summary['terminal']==26 and summary['pending']==12
    assert len(FakeModel.calls)==26 and not FakeClient.calls
    assert all('TRAIN_NOT_FOR_TEST' not in str(item) for _,item in FakeModel.calls)
    first=list(FakeModel.calls)
    run.run_experiment(root,local=True)
    assert FakeModel.calls==first
    assert (root/'artifacts/massive/matrix.sqlite').exists()
    saved=json.loads((root/'artifacts/massive/summary.json').read_text())
    assert len(saved['by_model_repeat'])==19
    assert saved['primary_repeat']==0


def test_remote_failures_continue_with_clean_payload_and_known_budget_gate(run,tmp_path):
    root=fixture_bundle(tmp_path/'remote')
    credentials={provider:{'base_url':'https://unused.invalid','api_key':'FAKE_PRIVATE_KEY'} for provider in ['jev','gemini']}
    result=run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient)
    summary=result['datasets']['massive']
    assert summary['planned']==38 and summary['terminal']==12 and summary['failed']==1 and summary['pending']==26
    assert len(FakeClient.calls)==12 and not FakeModel.calls
    assert all('gold_label' not in json.dumps(payload) and 'query-0' not in json.dumps(payload) for _,payload in FakeClient.calls)
    assert 'FAKE_PRIVATE_KEY' not in (root/'artifacts/massive/summary.json').read_text()
    assert summary['cost_status']=='unknown'


def test_status_is_readonly_and_does_not_require_freeze(run,tmp_path,monkeypatch):
    root=fixture_bundle(tmp_path/'status');run.run_experiment(root,local=True)
    path=root/'artifacts/massive/matrix.sqlite';before=path.read_bytes()
    (root/'frozen.json').unlink()
    monkeypatch.setattr(run.joblib,'load',lambda _:pytest.fail('Status loaded model'))
    FakeModel.calls=[]
    result=run.read_status(root)
    assert result['datasets']['massive']['planned']==38
    assert path.read_bytes()==before and FakeModel.calls==[]


def test_unknown_cost_without_bounded_call_budget_refuses_remote(run,tmp_path):
    root=fixture_bundle(tmp_path/'budget',execution={'budget':{}})
    with pytest.raises(ValueError,match='budget'):
        run.run_experiment(root,remote=True,credentials={'jev':{'base_url':'unused','api_key':'key'},'gemini':{'base_url':'unused','api_key':'key'}},client_factory=FakeClient)
    assert not FakeClient.calls


def test_recovery_flag_requires_frozen_permission(run,tmp_path):
    root=fixture_bundle(tmp_path/'retry',execution={'allow_indeterminate_retry':False})
    with pytest.raises(ValueError):run.run_experiment(root,local=True,allow_indeterminate_retry=True)
    assert not FakeModel.calls


def test_journal_selected_models_preserves_unselected_pending(tmp_path):
    planned=[{'job_id':str(i),'dataset_id':'d','item_id':str(i),'group_id':str(i),'model':model,'repeat':0,
              'input':{'text':'safe'},'gold_label':'yes'} for i,model in enumerate(['local','remote'])]
    journal=DurableJournal(tmp_path/'selected.sqlite',planned,'hash')
    called=[]
    summary=execute_jobs(journal,lambda job:called.append(job['model']) or {'status':'valid','prediction':'yes'},selected_models={'local'})
    assert called==['local'] and summary['planned']==2 and summary['pending']==1


def test_dotenv_parser_is_literal_allowlisted_and_never_executes(run,tmp_path):
    path=tmp_path/'literal.env';marker=tmp_path/'executed'
    path.write_text(f'JEV_API_KEY="fake-key"\nJEV_API_URL=https://example.invalid/v1\nGEMINI_API_KEY=other-key\nGEMINI_API_URL=https://example.invalid/v1\nEVIL=$(touch {marker})\n')
    credentials=run.load_credentials_file(path)
    assert set(credentials)=={'jev','gemini'}
    assert credentials['jev']['api_key']=='fake-key'
    assert not marker.exists()


def test_budget_stops_before_next_client_call_and_preserves_pending(run,tmp_path):
    root=fixture_bundle(tmp_path/'bounded',execution={'budget':{'unknown_cost_policy':'bounded_calls','max_total_http_attempts':1}})
    credentials={provider:{'base_url':'https://unused.invalid','api_key':'secret'} for provider in ['jev','gemini']}
    result=run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient)
    assert len(FakeClient.calls)==1
    assert result['budget_exhausted'] and result['datasets']['massive']['terminal']==1
    assert result['datasets']['massive']['pending']==37
    result=run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient)
    assert len(FakeClient.calls)==1 and result['budget_exhausted']


def test_repeat_authorization_limit_cannot_be_exceeded(tmp_path):
    planned=[{'job_id':'0','dataset_id':'d','item_id':'0','group_id':'0','model':'remote','repeat':0,
              'input':{'text':'safe'},'gold_label':'yes'}]
    journal=DurableJournal(tmp_path/'bounded-recovery.sqlite',planned,'hash')
    journal.append_attempt('0',{'phase':'started','attempt_id':'first-http'})
    calls=[]
    def interrupt(job):
        calls.append(job['job_id'])
        journal.append_attempt('0',{'phase':'started','attempt_id':'second-http'})
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute_jobs(journal,interrupt,allow_indeterminate_retry=True,max_indeterminate_retries=1)
    result=execute_jobs(journal,interrupt,allow_indeterminate_retry=True,max_indeterminate_retries=1)
    assert calls==['0'] and result['pending']==1


def test_dataset_alias_and_full_id_cannot_duplicate_one_dataset(run,tmp_path):
    import hashlib
    root=fixture_bundle(tmp_path/'duplicate-dataset')
    bundle=json.loads((root/'frozen.json').read_text())
    bundle['metadata']['dataset_ids']=['massive','fake-official-dataset']
    body={key:value for key,value in bundle.items() if key!='bundle_sha256'}
    bundle['bundle_sha256']=hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'frozen.json').write_text(json.dumps(bundle))
    with pytest.raises(ValueError):run.run_experiment(root,local=True)
    assert not FakeModel.calls


def test_existing_writer_lock_prevents_concurrent_predictions(run,tmp_path):
    root=fixture_bundle(tmp_path/'locked')
    with run._writer_lock(root):
        with pytest.raises(RuntimeError):run.run_experiment(root,local=True)
    assert not FakeModel.calls


def test_resume_preserves_full_manifest_randomized_order(tmp_path):
    import random
    planned=[{'job_id':str(i),'dataset_id':'d','item_id':str(i),'group_id':str(i),'model':'local','repeat':0,
              'input':{'text':str(i)},'gold_label':'yes'} for i in range(8)]
    expected=[job['job_id'] for job in planned];random.Random(17).shuffle(expected)
    journal=DurableJournal(tmp_path/'order.sqlite',planned,'hash')
    first=[]
    def interrupt(job):
        first.append(job['job_id'])
        if len(first)==2:raise KeyboardInterrupt()
        return {'status':'valid','prediction':'yes'}
    with pytest.raises(KeyboardInterrupt):execute_jobs(journal,interrupt,17)
    resumed=[]
    execute_jobs(journal,lambda job:resumed.append(job['job_id']) or {'status':'valid','prediction':'yes'},17)
    assert first[:1]+resumed==expected


def test_cli_diagnostic_never_echoes_unexpected_credential_error(run,tmp_path,monkeypatch,capsys):
    def broken(*args,**kwargs):raise TypeError('FAKE_PRIVATE_CREDENTIAL')
    monkeypatch.setattr(run,'run_experiment',broken)
    assert run.main(['--root',str(tmp_path),'--local'])==2
    assert 'FAKE_PRIVATE_CREDENTIAL' not in capsys.readouterr().out


def rewrite_bundle(root,edit):
    bundle=json.loads((root/'frozen.json').read_text())
    edit(bundle)
    body={key:value for key,value in bundle.items() if key!='bundle_sha256'}
    bundle['bundle_sha256']=hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'frozen.json').write_text(json.dumps(bundle))


def test_missing_execution_source_binding_blocks_formal_prediction(run,tmp_path):
    root=fixture_bundle(tmp_path/'missing-source')
    rewrite_bundle(root,lambda bundle:bundle['metadata'].pop('execution_source_sha256'))
    with pytest.raises(ValueError,match='source'):run.run_experiment(root,local=True)
    assert not FakeModel.calls


def test_frozen_source_copy_cannot_replace_actual_executing_source(run,tmp_path):
    root=fixture_bundle(tmp_path/'fake-source')
    (root/'remote.py').write_text('# A frozen copy that differs from actual execution\n')
    fake_hash=hashlib.sha256((root/'remote.py').read_bytes()).hexdigest()
    def edit(bundle):
        bundle['metadata']['execution_source_sha256']['remote.py']=fake_hash
        next(item for item in bundle['artifacts'] if item['path']=='remote.py')['sha256']=fake_hash
    rewrite_bundle(root,edit)
    with pytest.raises(ValueError,match='source'):run.run_experiment(root,local=True)
    assert not FakeModel.calls


def test_execution_source_mapping_must_also_be_a_frozen_artifact(run,tmp_path):
    root=fixture_bundle(tmp_path/'missing-source-artifact')
    rewrite_bundle(root,lambda bundle:bundle.update(artifacts=[item for item in bundle['artifacts'] if item['path']!='statistics.py']))
    with pytest.raises(ValueError,match='source'):run.run_experiment(root,local=True)
    assert not FakeModel.calls


@pytest.mark.parametrize('phase',['started','finished'])
def test_run_attempt_persistence_failure_stops_matrix_without_terminal(run,tmp_path,monkeypatch,phase):
    root=fixture_bundle(tmp_path/('persist-'+phase))
    original=DurableJournal.append_attempt
    def fail(self,job_id,event):
        if event.get('phase')==phase and event.get('kind')!='job':
            raise OSError('temporary log failure FAKE_PRIVATE_KEY')
        return original(self,job_id,event)
    monkeypatch.setattr(DurableJournal,'append_attempt',fail)
    credentials={provider:{'base_url':'https://unused.invalid','api_key':'FAKE_PRIVATE_KEY'} for provider in ['jev','gemini']}
    result=run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient)
    assert result['durability_failure']
    assert result['datasets']['massive']['terminal']==0
    assert result['datasets']['massive']['pending']==38
    assert len(FakeClient.calls)==(0 if phase=='started' else 1)
    assert 'FAKE_PRIVATE_KEY' not in json.dumps(result)


def test_loaded_module_source_path_is_checked_not_just_sibling_copy(run,tmp_path,monkeypatch):
    import scientific.remote as remote
    root=fixture_bundle(tmp_path/'module-source')
    different=tmp_path/'different_remote.py';different.write_text('# changed actually imported remote source\n')
    monkeypatch.setattr(remote,'__file__',str(different))
    with pytest.raises(ValueError,match='source'):run.run_experiment(root,local=True)
    assert not FakeModel.calls


def test_executable_package_init_source_is_required(run,tmp_path):
    root=fixture_bundle(tmp_path/'missing-init')
    rewrite_bundle(root,lambda bundle:bundle['metadata']['execution_source_sha256'].pop('__init__.py'))
    with pytest.raises(ValueError,match='source'):run.run_experiment(root,local=True)
    assert not FakeModel.calls


def test_bool_is_not_an_http_attempt_budget(run,tmp_path):
    root=fixture_bundle(tmp_path/'bool-budget',execution={'budget':{'unknown_cost_policy':'bounded_calls','max_total_http_attempts':True}})
    credentials={provider:{'base_url':'https://unused.invalid','api_key':'secret'} for provider in ['jev','gemini']}
    with pytest.raises(ValueError,match='budget'):run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient)
    assert not FakeClient.calls


def test_cli_catches_direct_persistence_baseexception_with_safe_diagnostic(run,tmp_path,monkeypatch,capsys):
    from scientific.remote import JournalPersistenceFailure
    def failed(*args,**kwargs):raise JournalPersistenceFailure('finished')
    monkeypatch.setattr(run,'run_experiment',failed)
    assert run.main(['--root',str(tmp_path),'--local'])==4
    diagnostic=json.loads(capsys.readouterr().out)
    assert diagnostic['error_code']=='attempt_persistence_failed'


def test_gateway_base_urls_are_required_even_for_local_formal_execution(run,tmp_path):
    root=fixture_bundle(tmp_path/'missing-gateway')
    rewrite_bundle(root,lambda bundle:bundle['metadata']['execution'].pop('remote_base_urls'))
    with pytest.raises(ValueError,match='gateway|endpoint|URL'):run.run_experiment(root,local=True)
    assert not FakeModel.calls


@pytest.mark.parametrize('url',['','file:///tmp/gateway','https://user:secret@example.invalid/v1',
    'https://example.invalid/v1?api_key=PRIVATE_SECRET','https://example.invalid/v1#PRIVATE_SECRET',
    'https://','https://example.invalid:99999/v1','https://bad host.invalid/v1',
    'https://example.invalid/v1/api_key/PRIVATE_SECRET','https://example.invalid/v1;token=PRIVATE_SECRET'])
def test_unsafe_frozen_gateway_address_is_refused_before_prediction(run,tmp_path,url):
    root=fixture_bundle(tmp_path/'unsafe-gateway',execution={'remote_base_urls':{'jev':url,'gemini':'https://unused.invalid'}})
    with pytest.raises(ValueError):run.run_experiment(root,local=True)
    assert not FakeModel.calls and not FakeClient.calls


def test_injected_remote_gateway_cannot_change_frozen_service(run,tmp_path):
    root=fixture_bundle(tmp_path/'changed-gateway')
    credentials={provider:{'base_url':'https://different.invalid/v1','api_key':'secret'} for provider in ['jev','gemini']}
    with pytest.raises(ValueError,match='gateway|endpoint|URL'):run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient)
    assert not FakeClient.calls


def test_matching_gateway_trailing_slash_is_normalized_and_http_mock_allowed(run,tmp_path):
    root=fixture_bundle(tmp_path/'normalized-gateway',execution={'remote_base_urls':{'jev':'http://127.0.0.1:8766/v1/','gemini':'http://127.0.0.1:8766/v1/'}})
    credentials={provider:{'base_url':'http://127.0.0.1:8766/v1','api_key':'secret'} for provider in ['jev','gemini']}
    result=run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient)
    assert result['datasets']['massive']['terminal']==12 and len(FakeClient.calls)==12


def test_frozen_workers_are_fixed_and_cli_override_must_match(run,tmp_path):
    root=fixture_bundle(tmp_path/'workers',execution={'workers':3})
    credentials={provider:{'base_url':'https://unused.invalid','api_key':'secret'} for provider in ['jev','gemini']}
    with pytest.raises(ValueError,match='workers'):
        run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient,workers=2)
    result=run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient,workers=3)
    assert result['datasets']['massive']['terminal']==12


def test_concurrent_remote_budget_never_exceeds_frozen_cap(run,tmp_path):
    root=fixture_bundle(tmp_path/'workers-budget',execution={
        'workers':4,'budget':{'unknown_cost_policy':'bounded_calls','max_total_http_attempts':3}})
    credentials={provider:{'base_url':'https://unused.invalid','api_key':'secret'} for provider in ['jev','gemini']}
    result=run.run_experiment(root,remote=True,credentials=credentials,client_factory=FakeClient,workers=4)
    assert len(FakeClient.calls)<=3
    assert result['budget_exhausted'] and result['datasets']['massive']['pending']>=35
