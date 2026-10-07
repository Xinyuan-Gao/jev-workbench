"""Publish public allowlisted progress snapshots for the existing static server."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
FRONTEND=ROOT.parent/'frontend'
FAMILY_NAMES={'majority':'多数类','lr':'Logistic Regression','linear_svc':'LinearSVC',
              'xgboost':'XGBoost','lightgbm':'LightGBM'}


def _read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def snapshot(root=ROOT):
    root=Path(root);datasets=[];local=[]
    for name,title in [('massive','中文语音助手意图'),('t2','问题—片段信息相关性')]:
        data=_read(root/'data'/f'{name}_candidate.json')
        if data:
            audit=data['audit']
            datasets.append({'id':name,'title':title,'state':'候选数据已清理，待冻结',
                'counts':{k:{'records':v['records'],'groups':v['groups']} for k,v in audit['counts'].items()},
                'labels':len(data['labels']),'source':data['source'],
                'manifest_sha256':audit['manifest_sha256'],'removed_records':len(data.get('removals',[])),
                'structural_ready':audit['structural_ready'],
                'near_duplicate_audit':audit['near_duplicate_audit'],
                'limits':data.get('limitations',[])})
        else:datasets.append({'id':name,'title':title,'state':'原始数据核验中','counts':None})
        report=_read(root/'artifacts'/name/'local_dev_report.json',{})
        for family,display in FAMILY_NAMES.items():
            value=report.get('families',{}).get(family,{})
            candidates=value.get('candidates',[]);selected=value.get('selection') or {}
            # Report only explicitly safe audit fields, never arbitrary errors,
            # input text, credentials, raw HTTP or model serialization.
            local.append({'dataset':name,'model':display,'family':family,
                'state':value.get('status','complete' if family=='majority' and selected else 'waiting'),
                'candidates_done':len(candidates),'candidates_total':0 if family=='majority' else 12,
                'repeats_done':sum(bool(r.get('fit_performed')) for r in value.get('repeats',[])),
                'repeats_total':5 if family in {'xgboost','lightgbm'} else 1,
                'validation_macro_f1':selected.get('validation_macro_f1'),
                'params':selected.get('params'),
                'candidates':[{'index':c['candidate_index'],'params':c['params'],'status':c['status'],
                               'validation_macro_f1':c.get('validation_macro_f1'),
                               'model_fit_ms':c.get('model_fit_ms')} for c in candidates]})
    # Prefer an explicit continuation freeze when a resumed run exists, while
    # retaining the original frozen.json as the immutable parent record.
    frozen=_read(root/'frozen_continuation.json') or _read(root/'frozen.json')
    readiness=_read(root/'readiness.json',{'status':'preparing','gates':{}})
    sensitivity=_read(root/'audits/design_sensitivity.json',{})
    sample_scope={alias:{k:values.get(k) for k in ('test_rows','independent_groups',
        'min_rows_per_group','max_rows_per_group','exact_mcnemar_applicable_to_primary')}
        for alias,values in sensitivity.get('datasets',{}).items() if alias in {'massive','t2'}}
    download=_read('/tmp/jev_scientific_research/collection_download_progress.json',{})
    from .run import read_status
    formal=read_status(root)['datasets']
    formal_results=[{'dataset':alias,**row} for alias,summary in formal.items() for row in summary['by_model_repeat']]
    return {'updated_at_utc':datetime.now(timezone.utc).isoformat(),'phase':'正式实验' if frozen else '正式测试前准备',
            'frozen':bool(frozen),'freeze_hash':frozen.get('bundle_sha256') if frozen else None,
            'datasets':datasets,'local_validation':local,'readiness':readiness,
            'download':{k:download.get(k) for k in ('status','downloaded_bytes','total_bytes')},
            'formal_results':formal_results,'formal_progress':{key:sum(v[key] for v in formal.values()) for key in ('planned','terminal','valid','failed','pending')},
            'formal_results_state':'按完整计划提交评分' if formal else '尚未执行，验证集调优分数不是正式成绩',
            'sample_scope':sample_scope,
            'routing':{'state':'待真人独立标注核验','confirmatory':False},
            'remote':{'state':'验证集接入已完成；正式调用预算待明确' if readiness.get('gates',{}).get('remote_dev_compatibility',{}).get('passed') else '等待验证集接入检查',
                      'dev_calls':readiness.get('dev_remote_calls',0),'repeats_planned':3}}


def publish(root=ROOT,target=FRONTEND/'scientific-status.json'):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(snapshot(root),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,target)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        publish()
        if not args.watch:break
        time.sleep(3)
