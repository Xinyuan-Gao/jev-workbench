"""Execute only a hash-verified frozen matrix; status reads SQLite without writes.

Examples: python -m scientific.run --local; python -m scientific.run --status.
Remote credentials are injected programmatically, or explicitly read from the
user-selected --env-file with a literal allowlisted parser (never shell sourced).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from urllib.parse import urlsplit, urlunsplit, unquote

import joblib

from .freeze import REQUIRED_GATES, verify_freeze
from .journal import DurableJournal, execute_jobs
from .remote import AuditedHTTPClient, build_payload, _safe, JournalPersistenceFailure

ROOT=Path(__file__).resolve().parent
LOCAL_REPEATS={'majority':1,'lr':1,'linear_svc':1,'xgboost':5,'lightgbm':5}
EXECUTION_SOURCE_FILES=('__init__.py','run.py','remote.py','journal.py','statistics.py','datasets.py',
                        'local_models.py','freeze.py','task_schemas.py')


class BudgetLimitReached(BaseException):
    """Stop safely before another wire request; keep that job pending."""


class _AttemptBudget:
    """Process-wide atomic reservation for unknown-cost HTTP attempts."""

    def __init__(self, limit, used=0):
        self.limit = limit
        self.used = used
        self._lock = threading.Lock()

    def reserve(self):
        with self._lock:
            if self.used >= self.limit:
                raise BudgetLimitReached()
            self.used += 1

    @property
    def exhausted(self):
        with self._lock:
            return self.used >= self.limit


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic_json(path,value):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,target)


@contextmanager
def _writer_lock(root):
    with (root/'.matrix-run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another scientific matrix writer is active') from None
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def load_credentials_file(path):
    """Parse only four endpoint/credential variables, literally and without eval."""
    allowed={'JEV_API_URL','JEV_API_KEY','GEMINI_API_URL','GEMINI_API_KEY'}
    values={}
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line=raw.strip()
        if not line or line.startswith('#') or '=' not in line:continue
        key,value=line.split('=',1);key=key.strip()
        if key not in allowed:continue
        value=value.strip()
        if len(value)>=2 and value[0] in {'"',"'"} and value[-1]==value[0]:value=value[1:-1]
        values[key]=value
    return {provider:{'base_url':values.get(prefix+'_API_URL',''),'api_key':values.get(prefix+'_API_KEY','')}
            for provider,prefix in [('jev','JEV'),('gemini','GEMINI')]}


def normalize_gateway_url(value):
    """Canonical credential-free HTTP(S) service identity, without I/O."""
    if not isinstance(value,str) or not value or any(char.isspace() or ord(char)<32 for char in value):
        raise ValueError('Frozen gateway URL must be a nonempty HTTP(S) address')
    try:
        parsed=urlsplit(value)
        host=parsed.hostname
        port=parsed.port
    except ValueError:
        raise ValueError('Invalid frozen gateway URL') from None
    if (parsed.scheme.lower() not in {'http','https'} or not host or not parsed.netloc or
            parsed.username is not None or parsed.password is not None or '@' in parsed.netloc or
            '?' in value or '#' in value or parsed.netloc.endswith(':') or
            (port is not None and not 1<=port<=65535)):
        raise ValueError('Frozen gateway URL cannot contain userinfo, query, fragment or invalid port')
    decoded=unquote(parsed.path)
    if (any(char.isspace() or ord(char)<32 for char in decoded) or
            re.search(r'(?:^|[;/?:&])(?:api[-_]?key|(?:access|refresh|auth)[-_]?token|token|authorization|password|secret|key)(?:/|=|:)',decoded,re.I)):
        raise ValueError('Frozen gateway URL cannot contain credential-bearing path components')
    try:
        if ':' in host:
            ipaddress.IPv6Address(host)
            canonical_host='['+host.lower()+']'
        else:
            canonical_host=host.encode('idna').decode('ascii').lower().rstrip('.')
            if (len(canonical_host)>253 or any(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?',label) is None
                                            for label in canonical_host.split('.'))):
                raise ValueError('Invalid hostname')
    except (ValueError,UnicodeError):
        raise ValueError('Invalid frozen gateway URL hostname') from None
    scheme=parsed.scheme.lower()
    if port is not None and not ((scheme=='http' and port==80) or (scheme=='https' and port==443)):
        canonical_host+=':'+str(port)
    return urlunsplit((scheme,canonical_host,parsed.path.rstrip('/'),'',''))


def _frozen_inputs(root, freeze_path=None):
    path=Path(freeze_path) if freeze_path is not None else root/'frozen.json'
    if not path.is_absolute():
        path=root/path
    if not path.is_file():raise ValueError('frozen.json is required before any test prediction')
    bundle=_read(path)
    try:verification=verify_freeze(root,bundle)
    except Exception:raise ValueError('Invalid frozen bundle') from None
    if verification.get('valid') is not True:raise ValueError('Frozen artifact/hash verification failed')
    if any(not isinstance(bundle.get('gates',{}).get(gate),dict) or
           bundle['gates'][gate].get('passed') is not True or not bundle['gates'][gate].get('evidence')
           for gate in REQUIRED_GATES):
        raise ValueError('All frozen scientific gates must pass')
    metadata=bundle.get('metadata',{})
    sources=metadata.get('execution_source_sha256')
    artifacts={artifact['path']:artifact for artifact in bundle['artifacts']}
    if not isinstance(sources,dict) or not set(EXECUTION_SOURCE_FILES).issubset(sources):
        raise ValueError('Frozen execution source hash mapping is required')
    # Hash the actual package executing this entry point, rather than trusting
    # a supplied root/source_snapshot copy that may never be imported.
    actual_root=Path(__file__).resolve().parent
    for source in sources:
        if not isinstance(source,str) or re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*\.py',source) is None:
            raise ValueError('Execution source mapping requires package source filenames')
        expected=sources[source]
        if (not isinstance(expected,str) or re.fullmatch(r'[0-9a-f]{64}',expected) is None or
                source not in artifacts or artifacts[source].get('sha256')!=expected):
            raise ValueError('Execution source hash must match its frozen artifact')
        if source=='run.py':
            actual_path=Path(__file__).resolve()
        elif source in EXECUTION_SOURCE_FILES:
            module_name=__package__ if source=='__init__.py' else __package__+'.'+Path(source).stem
            actual_path=Path(importlib.import_module(module_name).__file__).resolve()
        else:
            # Extra frozen scripts (e.g. analyze.py) are checked without importing
            # them, so analysis/test-side effects cannot run during preflight.
            actual_path=actual_root/source
        if not actual_path.is_file() or hashlib.sha256(actual_path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Actual execution source differs from frozen source')
    ids=metadata.get('dataset_ids')
    if not isinstance(ids,list) or not ids or any(not isinstance(name,str) for name in ids) or len(set(ids))!=len(ids):
        raise ValueError('Frozen metadata.dataset_ids must be a nonempty unique list')
    paths={artifact['path'] for artifact in bundle['artifacts']}
    schema_path='data/task_schemas_candidate.json'
    if schema_path not in paths:raise ValueError('Task schemas must be frozen')
    schemas=_read(root/schema_path)
    candidates={}
    for candidate_path in sorted(paths):
        if re.fullmatch(r'data/[A-Za-z0-9_-]+_candidate\.json',candidate_path) and candidate_path!=schema_path:
            data=_read(root/candidate_path)
            if 'dataset_id' in data:
                alias=Path(candidate_path).name.removesuffix('_candidate.json')
                candidates[alias]=(candidate_path,data)
    datasets={}
    for requested in ids:
        matches=[(alias,path,data) for alias,(path,data) in candidates.items()
                 if requested in {alias,data['dataset_id']}]
        if len(matches)!=1:raise ValueError('Frozen dataset ID must resolve to exactly one candidate')
        alias,path,data=matches[0]
        if alias in datasets:raise ValueError('Two frozen IDs refer to the same dataset')
        if alias not in schemas:raise ValueError('Frozen dataset has no matching task schema')
        required={f'artifacts/{alias}/local_models.joblib',f'artifacts/{alias}/local_dev_report.json'}
        if not required.issubset(paths):raise ValueError('Local model and dev report must both be frozen')
        schema=schemas[alias]
        if list(schema.get('labels',[]))!=list(data.get('labels',[])):
            raise ValueError('Frozen schema label order differs from dataset')
        report=_read(root/f'artifacts/{alias}/local_dev_report.json')
        if report.get('status')!='complete' or report.get('test_used') is not False or report.get('labels')!=data['labels']:
            raise ValueError('Frozen local dev selection must be complete and test-free')
        if set(report.get('families',{}))!=set(LOCAL_REPEATS):
            raise ValueError('Frozen local baseline family manifest is incomplete')
        for family,count in LOCAL_REPEATS.items():
            value=report['families'][family]
            if not value.get('selection') or len(value.get('repeats',[]))!=count or not all(r.get('fit_performed') for r in value['repeats']):
                raise ValueError('Frozen dev report lacks completed real model repeats')
        records=[record for record in data['records'] if record.get('split')=='test']
        if not records or len({record['id'] for record in records})!=len(records):raise ValueError('Unique nonempty frozen test manifest required')
        if any(record.get('label') not in data['labels'] or not record.get('group_id') for record in records):
            raise ValueError('Test manifest label/group contract invalid')
        datasets[alias]={'data':data,'schema':schema,'test_records':records,'report':report}
    execution=metadata.get('execution')
    if not isinstance(execution,dict):raise ValueError('Frozen execution policy required')
    remote_models=execution.get('remote_models')
    if (not isinstance(remote_models,dict) or set(remote_models)!={'jev','gemini'} or
            any(not isinstance(model,str) or not model for model in remote_models.values())):
        raise ValueError('Frozen remote model identities required')
    base_urls=execution.get('remote_base_urls')
    if not isinstance(base_urls,dict) or set(base_urls)!={'jev','gemini'}:
        raise ValueError('Frozen remote gateway base URLs required')
    for value in base_urls.values():normalize_gateway_url(value)
    repeats=execution.get('remote_repeats')
    attempts=execution.get('max_http_attempts')
    if not isinstance(repeats,int) or isinstance(repeats,bool) or repeats<3:
        raise ValueError('At least three frozen remote repeats required')
    if not isinstance(attempts,int) or isinstance(attempts,bool) or not 1<=attempts<=3:
        raise ValueError('Frozen HTTP attempt limit must be 1 to 3')
    if not isinstance(execution.get('order_seed'),int):raise ValueError('Frozen order seed required')
    recovery=execution.get('max_indeterminate_retries_per_job',0)
    if not isinstance(recovery,int) or isinstance(recovery,bool) or not 0<=recovery<=2:
        raise ValueError('Frozen indeterminate recovery limit must be 0 to 2')
    workers=execution.get('workers')
    if workers is not None and (type(workers) is not int or not 1<=workers<=64):
        raise ValueError('Frozen workers must be an integer from 1 to 64')
    # A continuation freeze may deliberately resume an existing journal whose
    # rows were written under the parent freeze.  Keep that relationship
    # explicit and use the parent protocol hash only for journal compatibility;
    # the continuation bundle remains the identity used in new summaries.
    continuation=metadata.get('continuation')
    if continuation is not None:
        if (not isinstance(continuation,dict) or
                not re.fullmatch(r'[0-9a-f]{64}',continuation.get('parent_bundle_sha256','')) or
                not re.fullmatch(r'[0-9a-f]{64}',continuation.get('journal_protocol_hash','')) or
                continuation['parent_bundle_sha256'] != continuation['journal_protocol_hash']):
            raise ValueError('Invalid continuation freeze linkage')
        execution=dict(execution)
        execution['_journal_protocol_hash']=continuation['journal_protocol_hash']
    return bundle,datasets,execution


def _plan(dataset,execution):
    pairs=[(f'local:{family}',repeat) for family,count in LOCAL_REPEATS.items() for repeat in range(count)]
    pairs.extend((f'{provider}:{model}',repeat) for provider,model in execution['remote_models'].items()
                 for repeat in range(execution['remote_repeats']))
    data=dataset['data']
    result=[]
    for record in dataset['test_records']:
        clean_input=dict(record['input'])
        # Run the same strict payload whitelist/required-fields validator before
        # accepting any local or remote formal test combination.
        build_payload('jev',execution['remote_models']['jev'],clean_input,dataset['schema'])
        fields={'classification':('text',),'routing':('task',),'relevance':('query','passage')}[dataset['schema']['task']]
        clean_input={field:clean_input[field] for field in fields}
        for model,repeat in pairs:
            identity=json.dumps([data['dataset_id'],record['id'],model,repeat],ensure_ascii=False)
            result.append({'job_id':hashlib.sha256(identity.encode()).hexdigest(),
                'dataset_id':data['dataset_id'],'item_id':record['id'],'group_id':record['group_id'],
                'model':model,'repeat':repeat,'input':clean_input,'gold_label':record['label']})
    return result


def _summarize(records):
    grouped=defaultdict(list)
    for record in records:grouped[(record['model'],record['repeat'])].append(record)
    terminal=sum(record['outcome'] is not None for record in records)
    valid=sum(record['outcome'] is not None and record['outcome'].get('status')=='valid' for record in records)
    by_model_repeat=[]
    for (model,repeat),rows in sorted(grouped.items()):
        done=[row for row in rows if row['outcome'] is not None]
        legal=[row for row in done if row['outcome'].get('status')=='valid']
        correct=sum(row['outcome'].get('prediction')==row['gold_label'] for row in legal)
        by_model_repeat.append({'model':model,'repeat':repeat,'planned':len(rows),'terminal':len(done),
            'valid':len(legal),'failed':len(done)-len(legal),'pending':len(rows)-len(done),
            'independent_units':len({row['group_id'] for row in rows}),
            'correct':correct,'accuracy_all_submissions':correct/len(rows) if len(done)==len(rows) else None,
            'valid_output_rate':len(legal)/len(rows),'complete':len(done)==len(rows),
            'failures':dict(Counter(row['outcome']['status'] for row in done if row['outcome']['status']!='valid')),
            'cost':None,'cost_status':'not_applicable' if model.startswith('local:') else 'unknown'})
    return {'planned':len(records),'terminal':terminal,'valid':valid,'failed':terminal-valid,
            'pending':len(records)-terminal,'complete':terminal==len(records),
            'primary_repeat':0,'independent_units':len({row['group_id'] for row in records}),
            'cost':None,'cost_status':'unknown','by_model_repeat':by_model_repeat,
            'potential_duplicate_jobs':sum(bool(row.get('potential_duplicate')) for row in records)}


def read_status(root=ROOT):
    """Use mode=ro without creating or recovering a journal or loading models."""
    root=Path(root).resolve();datasets={}
    for path in sorted((root/'artifacts').glob('*/matrix.sqlite')):
        conn=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            conn.execute('PRAGMA query_only=ON')
            rows=[{**json.loads(job),'outcome':json.loads(outcome) if outcome else None}
                  for job,outcome in conn.execute('SELECT job_json,outcome_json FROM planned ORDER BY ordinal')]
            duplicates=set()
            for job_id,event in conn.execute('SELECT job_id,event_json FROM attempts'):
                if json.loads(event).get('potential_duplicate'):duplicates.add(job_id)
            for row in rows:row['potential_duplicate']=row['job_id'] in duplicates
            datasets[path.parent.name]=_summarize(rows)
        finally:conn.close()
    return {'datasets':datasets,'read_only':True}


def run_experiment(root=ROOT,*,local=False,remote=False,credentials=None,
                   client_factory=AuditedHTTPClient,allow_indeterminate_retry=False,workers=None,
                   freeze_path=None):
    root=Path(root).resolve()
    if not local and not remote:raise ValueError('Choose local and/or remote execution')
    bundle,datasets,execution=_frozen_inputs(root,freeze_path)
    frozen_workers=execution.get('workers')
    if workers is None:
        workers=frozen_workers or 1
    if type(workers) is not int or not 1<=workers<=64 or (frozen_workers is not None and workers != frozen_workers):
        raise ValueError('workers must match the frozen execution policy or be explicitly selected')
    if allow_indeterminate_retry and (execution.get('allow_indeterminate_retry') is not True or
                                     execution.get('max_indeterminate_retries_per_job',0)<1):
        raise ValueError('Indeterminate retry is not authorized by frozen execution policy')
    credentials=credentials or {}
    if remote:
        budget=execution.get('budget',{})
        if (budget.get('unknown_cost_policy')!='bounded_calls' or
                not isinstance(budget.get('max_total_http_attempts'),int) or
                isinstance(budget.get('max_total_http_attempts'),bool) or budget['max_total_http_attempts']<1):
            raise ValueError('Unknown-cost remote execution requires a frozen bounded-call budget')
        for provider in execution['remote_models']:
            settings=credentials.get(provider,{})
            if not settings.get('base_url') or not settings.get('api_key'):
                raise ValueError('Explicit remote endpoint/credential injection required for '+provider)
            if normalize_gateway_url(settings['base_url'])!=normalize_gateway_url(execution['remote_base_urls'][provider]):
                raise ValueError('Injected remote gateway URL differs from frozen endpoint')
    secrets=[settings.get('api_key') for settings in credentials.values() if isinstance(settings,dict) and settings.get('api_key')]
    attempt_budget=None
    result={'datasets':{},'bundle_sha256':bundle['bundle_sha256'],'budget_exhausted':False,
            'durability_failure':False}
    with _writer_lock(root):
        protocol_hash=execution.get('_journal_protocol_hash',bundle['bundle_sha256'])
        journals={alias:DurableJournal(root/f'artifacts/{alias}/matrix.sqlite',_plan(dataset,execution),
                                      protocol_hash,secrets=secrets) for alias,dataset in datasets.items()}
        used_attempts=sum(sum(event.get('phase')=='started' and event.get('kind')!='job' for event in row['attempts'])
                          for journal in journals.values() for row in journal.records())
        if remote:
            attempt_budget=_AttemptBudget(execution['budget']['max_total_http_attempts'], used_attempts)
        try:
            for alias,dataset in datasets.items():
                journal=journals[alias]
                models={}
                if local:
                    loaded=joblib.load(root/f'artifacts/{alias}/local_models.joblib')
                    models=loaded.get('models',loaded)
                    if set(models)!=set(LOCAL_REPEATS) or any(len(models[family])!=count for family,count in LOCAL_REPEATS.items()):
                        raise ValueError('Frozen local model repeat objects differ from planned manifest')
                clients={provider:client_factory(normalize_gateway_url(execution['remote_base_urls'][provider]),credentials[provider]['api_key'],provider,
                          max_retries=execution['max_http_attempts']-1,
                          timeout=execution.get('timeout_seconds',30),backoff_seconds=execution.get('backoff_seconds',1))
                         for provider in execution['remote_models']} if remote else {}
                selected={job['model'] for job in journal.pending_jobs()
                          if (local and job['model'].startswith('local:')) or (remote and not job['model'].startswith('local:'))}
                def predict(job):
                    if job['model'].startswith('local:'):
                        family=job['model'].split(':',1)[1]
                        outcome=models[family][job['repeat']].predict([{'input':job['input']}])[0]
                        if outcome.get('status')=='valid' and outcome.get('prediction') not in dataset['data']['labels']:
                            return {'status':'invalid_label','prediction':None,'raw_output':outcome}
                        return outcome
                    provider,model=job['model'].split(':',1)
                    def persist(event):
                        if event.get('phase')=='started' and event.get('kind')!='job':
                            attempt_budget.reserve()
                        try:
                            journal.append_attempt(job['job_id'],event)
                        except Exception:
                            # Covers injected/custom journal implementations too.
                            raise JournalPersistenceFailure(event.get('phase')) from None
                    payload=build_payload(provider,model,job['input'],dataset['schema'])
                    return clients[provider].send(payload,labels=dataset['data']['labels'],on_attempt=persist)
                execute_jobs(journal,predict,execution['order_seed'],selected_models=selected,
                             allow_indeterminate_retry=allow_indeterminate_retry,
                             max_indeterminate_retries=execution.get('max_indeterminate_retries_per_job',0),
                             workers=workers)
        except BudgetLimitReached:
            result['budget_exhausted']=True
        except JournalPersistenceFailure:
            result.update(durability_failure=True,error_code='attempt_persistence_failed')
        finally:
            for alias,journal in journals.items():
                summary=_summarize(journal.records())
                summary.update(bundle_sha256=bundle['bundle_sha256'],updated_at_utc=datetime.now(timezone.utc).isoformat())
                result['datasets'][alias]=summary
                _atomic_json(root/f'artifacts/{alias}/summary.json',_safe(summary,secrets))
    return _safe(result,secrets)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--local',action='store_true')
    parser.add_argument('--remote',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--allow-indeterminate-retry',action='store_true')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--freeze-path',type=Path,
                        help='Explicit frozen/continuation manifest; defaults to frozen.json')
    args=parser.parse_args(argv)
    if args.status:
        print(json.dumps(read_status(args.root),ensure_ascii=False,indent=2));return 0
    try:
        credentials=load_credentials_file(args.env_file) if args.remote and args.env_file else None
        result=run_experiment(args.root,local=args.local,remote=args.remote,credentials=credentials,
                              allow_indeterminate_retry=args.allow_indeterminate_retry,workers=args.workers,
                              freeze_path=args.freeze_path)
    except Exception:
        # Error strings may contain paths, raw environment values, or injected
        # client exception text. CLI exposes only this fixed safe diagnostic.
        print(json.dumps({'error':'Frozen execution/credential/artifact contract failed; no unsafe diagnostic emitted.'}));return 2
    except KeyboardInterrupt:
        print(json.dumps({'state':'interrupted','pending_preserved':True}));return 130
    except JournalPersistenceFailure:
        print(json.dumps({'error_code':'attempt_persistence_failed','pending_preserved':True}));return 4
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['durability_failure']:return 4
    return 3 if result['budget_exhausted'] else 0


if __name__=='__main__':raise SystemExit(main())
