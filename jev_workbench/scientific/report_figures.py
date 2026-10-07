"""Export traceable figures from complete final analysis, never from dev scores.

python -m jev_workbench.scientific.report_figures --analysis final_analysis.json --output final_figures
No predictions, test journals, model fitting, network calls or statistical recomputation.
Synthetic contract tests require an explicit Python API flag and are watermarked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FAILURE_COLUMN = 'NO_VALID_OUTPUT'
COLORS = {'local': '#7895A5', 'jev': '#846AA4', 'gemini': '#C07D72'}
LIMITATIONS = [
    'Frozen tasks/configurations and gateway services only; no universal model ranking.',
    'Repeated calls/training seeds do not increase independent sample size.',
    'Empirical bootstrap CI can degenerate; point intervals do not establish population certainty or equivalence.',
    'Macro-F1 intervals are exploratory; no F1 significance claim.',
]


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_score(score, labels, records, groups):
    if not isinstance(score, dict) or score.get('complete') is not True or score.get('provisional') is not False or score.get('pending') != 0:
        raise ValueError('Every primary and repeated score must be complete with no pending/provisional records')
    for key in ('planned', 'terminal', 'valid', 'failed', 'independent_groups'):
        if not _integer(score.get(key)):
            raise ValueError('Invalid score denominator')
    if score['planned'] != records or score['terminal'] != records or score['valid'] + score['failed'] != records or score['independent_groups'] != groups:
        raise ValueError('Score denominators differ from the complete dataset')
    statuses = score.get('status_counts', {})
    if not isinstance(statuses, dict) or not all(_integer(v) for v in statuses.values()) or sum(statuses.values()) != records or statuses.get('pending', 0) or statuses.get('valid', 0) != score['valid']:
        raise ValueError('Status counts inconsistent with score')
    if score.get('labels') != labels or score.get('confusion_columns') != labels + [FAILURE_COLUMN]:
        raise ValueError('Confusion labels must retain NO_VALID_OUTPUT')
    cm = score.get('confusion_matrix')
    if not isinstance(cm, list) or len(cm) != len(labels) or any(not isinstance(row, list) or len(row) != len(labels)+1 or not all(_integer(v) for v in row) for row in cm):
        raise ValueError('Invalid confusion dimensions/counts')
    if sum(map(sum, cm)) != records or sum(row[-1] for row in cm) != score['failed']:
        raise ValueError('Confusion matrix lost records or failures')
    correct = sum(cm[i][i] for i in range(len(labels)))
    for key, expected in [('accuracy', correct/records), ('valid_rate', score['valid']/records)]:
        if not _finite(score.get(key)) or not math.isclose(score[key], expected, abs_tol=1e-10):
            raise ValueError('Score estimate inconsistent with denominator')
    if not _finite(score.get('macro_f1')) or not 0 <= score['macro_f1'] <= 1:
        raise ValueError('Invalid Macro-F1')
    f1_values = []
    for i in range(len(labels)):
        support = sum(cm[i]); predicted = sum(row[i] for row in cm)
        f1_values.append(2*cm[i][i]/(support+predicted) if support+predicted else 0.)
    if not math.isclose(score['macro_f1'], sum(f1_values)/len(labels), abs_tol=1e-10):
        raise ValueError('Macro-F1 differs from failure-aware confusion counts')


def _validate(data, *, allow_synthetic):
    if not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('complete') is not True or data.get('pending', 0) != 0:
        raise ValueError('Only a complete schema-v1 final analysis is accepted')
    if data.get('synthetic_fixture') and not allow_synthetic:
        raise ValueError('synthetic data require explicit test mode')
    if not re.fullmatch(r'[0-9a-f]{64}', data.get('bundle_sha256', '')) or data.get('primary_repeat') != 0:
        raise ValueError('Frozen bundle SHA256 and primary repeat zero are required')
    policy = data.get('analysis_policy', {})
    if policy.get('primary_repeat') != 0 or not isinstance(data.get('datasets'), dict) or not data['datasets']:
        raise ValueError('Missing frozen dataset analysis policy')
    for alias, dataset in data['datasets'].items():
        if dataset.get('complete', True) is not True or dataset.get('pending', 0) != 0:
            raise ValueError('Every dataset must be complete')
        n, g, labels = dataset.get('test_records'), dataset.get('independent_groups'), dataset.get('labels')
        if not _integer(n) or n < 1 or not _integer(g) or not 0 < g <= n or not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
            raise ValueError('Invalid dataset denominators or label universe')
        models = dataset.get('models', {})
        best = policy.get('datasets', {}).get(alias, {}).get('best_local_family')
        if not best or 'local:'+best not in models or len([m for m in models if m.startswith('jev:')]) != 1 or len([m for m in models if m.startswith('gemini:')]) != 1:
            raise ValueError('Missing dev-selected local, JEV or Gemini model')
        for name, model in models.items():
            _check_score(model.get('score'), labels, n, g)
            for metric in ('accuracy', 'macro_f1'):
                interval = model.get('interval', {}).get(metric, {})
                ci = interval.get('ci95')
                if not _finite(interval.get('estimate')) or not math.isclose(interval['estimate'], model['score'][metric], abs_tol=1e-10) or not isinstance(ci, list) or len(ci) != 2 or not all(_finite(v) and 0 <= v <= 1 for v in ci) or ci[0] > ci[1]:
                    raise ValueError('Missing or inconsistent final metric confidence interval')
            stability = model.get('stability', {})
            rounds = stability.get('per_round', {})
            if not rounds or stability.get('primary_repeat') != 0 or stability.get('rounds') != len(rounds) or stability.get('unique_items') != n or stability.get('independent_groups') != g:
                raise ValueError('Missing complete repeated-round scores')
            if '0' not in {str(k) for k in rounds}:
                raise ValueError('Primary round zero is missing')
            for r, value in rounds.items():
                _check_score(value, labels, n, g)
                if str(r) == '0' and value != model['score']:
                    raise ValueError('Primary and repeated-round zero scores differ')
            op = model.get('operational', {})
            total = n * len(rounds)
            if op.get('logical_records') != total or op.get('unfinished_http_attempts') != 0:
                raise ValueError('Operational scope incomplete or contains unfinished attempts')
            covered = 0
            for key in ('logical_latency', 'valid_latency', 'failure_latency', 'first_http_attempt_latency', 'all_http_attempt_latency'):
                dist = op.get(key, {})
                count = dist.get('count')
                if not _integer(count): raise ValueError('Missing latency count')
                for metric in ('mean_ms', 'median_ms', 'p95_ms'):
                    value = dist.get(metric)
                    if (count and (not _finite(value) or value < 0)) or (not count and value is not None):
                        raise ValueError('Missing latency statistics cannot be reported as zero')
                if key in ('valid_latency', 'failure_latency'): covered += count
            valid_total = sum(r['valid'] for r in rounds.values()); fail_total = sum(r['failed'] for r in rounds.values())
            if op['valid_latency']['count'] > valid_total or op['failure_latency']['count'] > fail_total or op['logical_latency']['count'] != covered or op.get('latency_missing_logical_records') != total-covered:
                raise ValueError('Latency coverage does not preserve valid/failure/missing counts')


def _color(name):
    return COLORS.get(name.split(':')[0], '#7895A5')


def _stem(value):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_') or 'task'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _axes(title, height=4.8, width=10):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_title(textwrap.fill(title, 74), loc='left', fontsize=12, pad=16)
    fig.subplots_adjust(left=.30, right=.94, top=.84, bottom=.19)
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def export_report(analysis_path, output_dir, *, allow_synthetic=False):
    """Validate the entire input before creating figures or the output directory."""
    analysis_path, output_dir = Path(analysis_path).resolve(), Path(output_dir).resolve()
    raw = analysis_path.read_bytes()
    data = json.loads(raw)
    _validate(data, allow_synthetic=allow_synthetic)
    if output_dir == analysis_path.parent or output_dir in analysis_path.parents:
        raise ValueError('Output must be a separate figure directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'legend.frameon': False, 'axes.linewidth': .8})
    synthetic = bool(data.get('synthetic_fixture'))
    provenance = {'analysis_source': str(analysis_path), 'analysis_sha256': hashlib.sha256(raw).hexdigest(),
                  'bundle_sha256': data['bundle_sha256'], 'plotting_source_sha256': _sha(__file__)}
    manifest = {'schema_version': 1, 'purpose': 'synthetic_contract_test' if synthetic else 'complete_final_analysis',
                **provenance, 'primary_repeat': 0, 'figures': [],
                'final_visual_review_required': True, 'cost_and_calibration_computed': False}

    def save(fig, *, alias, kind, suffix='', title, caption, denominator, limitations, values):
        ident = _stem(alias)+'__'+kind+(('__'+_stem(suffix)) if suffix else '')
        if synthetic: fig.suptitle('SYNTHETIC CONTRACT TEST — NOT MODEL RESULTS', fontsize=11, color='#AD4C52', y=.97)
        else: fig.suptitle('Frozen test · primary round 0' if kind not in ('latency', 'round_scores', 'stability') else 'Frozen test · descriptive repeats', fontsize=9, y=.98)
        fig.canvas.draw()
        # Single-axes figures avoid comparison-panel layout ambiguities.
        for fmt in ('png', 'svg', 'pdf'):
            fig.savefig(output_dir/f'{ident}.{fmt}', dpi=300, facecolor='white')
        plt.close(fig)
        entry = {'id': ident, 'dataset': alias, 'kind': kind, 'title': title, 'caption': caption,
                 'denominators': denominator, 'limitations': list(dict.fromkeys(LIMITATIONS+limitations)),
                 **provenance, 'png':ident+'.png','svg':ident+'.svg','qa_pdf':ident+'.pdf',
                 'manifest':ident+'.manifest.json','source_data':ident+'.source_data.json','data':values,
                 'alignment': {'status': 'not_applicable', 'reason': 'One plot area; no multi-panel comparison'},
                 'rendered_qa': {'status': 'requires_rendered_pdf_and_visual_review'}}
        entry['export_hashes']={fmt:_sha(output_dir/entry[fmt]) for fmt in ('png','svg')}
        _dump(output_dir/entry['source_data'], {'data':values,'denominators':denominator,**provenance})
        _dump(output_dir/entry['manifest'],entry)
        manifest['figures'].append(entry)

    for alias, dataset in data['datasets'].items():
        names = list(dataset['models']); models=dataset['models']; n=dataset['test_records']; g=dataset['independent_groups']
        rag_task = alias in ('t2', 'rag', 'rag-relevance') or 't2' in str(dataset.get('dataset_id', '')).lower() or set(dataset['labels']) == {'relevant', 'irrelevant'}
        unit_note = ('pair-weighted point estimate; 95% CI resamples complete query groups' if rag_task else 'record-weighted point estimate; 95% CI follows frozen group/strata bootstrap')
        denom = {'primary_planned_records':n,'independent_groups':g,'labels':dataset['labels'],'primary_repeat':0}
        # Main all-planned effect and exploratory class-balance metric.
        for metric in ('accuracy','macro_f1'):
            title = f'{alias} | '+('All-planned Accuracy, 95% CI' if metric=='accuracy' else 'All-planned Macro-F1, exploratory 95% CI')
            fig,ax=_axes(title,height=max(4.8,.45*len(names)+2.1))
            plotdata={}
            for i,name in enumerate(names):
                est=models[name]['interval'][metric]['estimate']; lo,hi=models[name]['interval'][metric]['ci95']
                ax.plot([lo*100,hi*100],[i,i],color=_color(name),lw=2)
                ax.scatter(est*100,i,color=_color(name),s=32,zorder=3)
                ax.text(104,i,f'{est*100:.1f}%',va='center',fontsize=9)
                plotdata[name]={'estimate':est,'ci95':[lo,hi],'interval_metadata':models[name]['interval'],'valid':models[name]['score']['valid'],'failed':models[name]['score']['failed']}
            ax.set_yticks(range(len(names)), names);ax.invert_yaxis();ax.set_xlim(-2,119);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Percent of all planned records (%)' if metric=='accuracy' else 'Macro-F1 (%) · failures retain gold-class support');ax.grid(axis='x',alpha=.15)
            caption=f'{title}. Primary repeat 0; n={n} planned records, {g} independent groups. Failure outcomes remain in denominators. {unit_note}. '
            caption+=('Intervals are exploratory; no F1 significance or equivalence declaration. Fixed label universe includes unsupported classes as zero.' if metric=='macro_f1' else 'CI overlap is not a paired hypothesis test; equal-query mean accuracy is a separate point estimate without this CI.')
            save(fig,alias=alias,kind=metric,title=title,caption=caption,denominator=denom,limitations=['The plotted CI is the analysis interval, not a newly computed interval.', 'No CI is assigned to mean_group_accuracy.'],values=plotdata)

        title=f'{alias} | Valid and failed outputs · primary round 0'
        fig,ax=_axes(title,height=max(4.8,.5*len(names)+2));plotdata={}
        for i,name in enumerate(names):
            s=models[name]['score'];left=0
            ax.barh(i,s['valid']/n*100,height=.6,color='#8CA7B3')
            for status,count in s['status_counts'].items():
                if status=='valid' or not count:continue
                x=s['valid']/n*100+left
                ax.barh(i,count/n*100,left=x,height=.6,color='#BB7D70',hatch='///')
                left+=count/n*100
            ax.text(103,i,f"{s['valid']} valid / {s['failed']} fail",va='center',fontsize=8)
            plotdata[name]={'planned':n,'valid':s['valid'],'failed':s['failed'],'status_counts':s['status_counts']}
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlim(0,137);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Percent of all planned records (%) · hatched segment = failure')
        save(fig,alias=alias,kind='valid_failure',title=title,caption=f'Primary round 0; each model attempts the same {n} records. Valid and every failure type remain in the full denominator; source data retain separate failure statuses.',denominator=denom,limitations=['Output validity does not establish correctness.'],values=plotdata)

        title=f'{alias} | Per-round scores (descriptive, not extra samples)'
        ordered=[(name,repeat,value) for name in names for repeat,value in sorted(models[name]['stability']['per_round'].items(),key=lambda kv:int(kv[0]))]
        fig,ax=_axes(title,height=max(5,.27*len(ordered)+2.3));rounddata={}
        for i,(name,repeat,s) in enumerate(ordered):
            for key,marker,color,offset in [('accuracy','o','#587E94',-.13),('macro_f1','s','#957EB2',0),('valid_rate','^','#BB8765',.13)]:
                ax.scatter(s[key]*100,i+offset,marker=marker,color=color,s=24,label=key if i==0 else None)
            rounddata[f'{name}/repeat={repeat}']={'score':s}
        ax.set_yticks(range(len(ordered)),[f'{name} · round {int(r)+1}' for name,r,_ in ordered],fontsize=8);ax.invert_yaxis();ax.set_xlim(-2,102);ax.set_xlabel('Percent (%) · Accuracy / all-planned Macro-F1 / valid rate');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=8)
        save(fig,alias=alias,kind='round_scores',title=title,caption=f'Every recorded round is shown; each round has {n} records and {g} independent units. Scores across actual service rounds or fixed training seeds are descriptive. A one-round deterministic model is not duplicated.',denominator={**denom,'model_round_counts':{name:models[name]['stability']['rounds'] for name in names}},limitations=['Repeated scores are not independent task replications; no CI is estimated across rounds.'],values=rounddata)

        title=f'{alias} | Agreement with primary round'
        fig,ax=_axes(title,height=max(5,.42*len(names)+2));agreementdata={}
        for i,name in enumerate(names):
            st=models[name]['stability'];agreementdata[name]=st
            points=[('outcome_agreement_with_primary','o','#587E94',-.1),('prediction_agreement_conditional_on_both_valid','s','#BB8765',.1)]
            for key,marker,color,offset in points:
                vals=st.get(key,[])
                for j,x in enumerate(vals):
                    if x['rate'] is not None:ax.scatter(x['rate']*100,i+offset+j*.045,s=25,marker=marker,color=color,label=key if i==0 and j==0 else None)
            ax.text(105,i,f"{st['rounds']} actual round(s)",va='center',fontsize=8)
            if not st['assessable']:ax.text(4,i,'not assessable: one round',va='center',fontsize=8)
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlim(-2,139);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Agreement (%) · circles: outcome/status; squares: both-valid prediction')
        save(fig,alias=alias,kind='stability',title=title,caption=f'Agreement with primary repeat is descriptive. Outcome/status agreement uses all {n} planned pairs. Conditional prediction agreement uses only pairs valid in both rounds; each actual denominator is retained in source data. No repeated call increases independent n={g}.',denominator={**denom,'conditional_pair_denominators':{name:models[name]['stability'].get('prediction_agreement_conditional_on_both_valid',[]) for name in names}},limitations=['Conditional agreement can conceal failures; read with the full-output validity figure.'],values=agreementdata)

        best='local:'+data['analysis_policy']['datasets'][alias]['best_local_family']
        selected=[next(name for name in names if name.startswith('jev:')),next(name for name in names if name.startswith('gemini:')),best]
        for name in selected:
            s=models[name]['score'];matrix=np.array(s['confusion_matrix'])
            title=f'{alias} | {name} · all-planned confusion counts'
            size=max(6,len(dataset['labels'])*.20+3)
            fig,ax=_axes(title,height=size,width=size+1);fig.subplots_adjust(left=.23,right=.97,bottom=.28,top=.86)
            ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,int(matrix.max())),aspect='auto')
            ax.set_yticks(range(len(dataset['labels'])),dataset['labels'],fontsize=max(5,9-.08*len(dataset['labels'])))
            ax.set_xticks(range(len(s['confusion_columns'])),s['confusion_columns'],rotation=90,rotation_mode='anchor',ha='right',fontsize=max(5,9-.08*len(dataset['labels'])))
            ax.set_ylabel('Gold label');ax.set_xlabel('Predicted label; final column retains every invalid/failed output')
            if len(dataset['labels'])<=12:
                for row in range(matrix.shape[0]):
                    for col in range(matrix.shape[1]):ax.text(col,row,str(matrix[row,col]),ha='center',va='center',fontsize=9,color='white' if matrix[row,col]>matrix.max()*.55 else '#233E50')
            # Never trim zero-support labels or the all-zero failure column.
            ax.axvline(len(dataset['labels'])-.5,color='#BB765D',lw=1.2)
            save(fig,alias=alias,kind='confusion',suffix=name,title=title,caption=f'Primary repeat 0; {n} planned records. Rows are gold labels, columns are allowed predictions plus NO_VALID_OUTPUT. All fixed labels, including zero-support labels, are preserved. Local comparator {best} was selected on validation, not this test.',denominator={**denom,'valid':s['valid'],'failed':s['failed']},limitations=['Dense cells are not annotated; exact counts remain in adjacent source data.', 'Unanswered records are gold-class misses, not an additional gold class.'],values={'model':name,'rows':dataset['labels'],'columns':s['confusion_columns'],'matrix':s['confusion_matrix']})

        # Keep every model's latency scope separate: no local/gateway ratio league table.
        for name in names:
            op=models[name]['operational'];rounds=models[name]['stability']['per_round'];valid_count=sum(s['valid'] for s in rounds.values());fail_count=sum(s['failed'] for s in rounds.values())
            scopes=['logical_latency','valid_latency','failure_latency','first_http_attempt_latency','all_http_attempt_latency']
            counts=[op[k]['count'] for k in scopes]
            totals=[op['logical_records'],valid_count,fail_count,op['logical_records'] if not name.startswith('local:') else 0,op['finished_http_attempts']]
            title=f'{alias} | {name} · latency by scope, all actual rounds'
            fig,ax=_axes(title,height=5.4);max_value=max([op[k][m] for k in scopes for m in ('mean_ms','median_ms','p95_ms') if op[k][m] is not None] or [1])
            for i,k in enumerate(scopes):
                for metric,marker,color,offset in [('mean_ms','o','#587E94',-.11),('median_ms','s','#957EB2',0),('p95_ms','^','#BB8765',.11)]:
                    value=op[k][metric]
                    if value is not None:ax.scatter(value,i+offset,s=27,marker=marker,color=color,label=metric if i==0 else None)
                ax.text(max_value*1.04,i,f"{counts[i]}/{totals[i]} timed",fontsize=8,va='center')
                if not counts[i]:ax.text(max_value*.02,i,'no recorded timings',va='center',fontsize=8)
            ax.set_yticks(range(5),['Logical total incl. retry','Valid logical outputs','Failed logical outputs','First HTTP attempt','All finished HTTP attempts']);ax.invert_yaxis();ax.set_xlim(-max_value*.02,max_value*1.40);ax.set_xlabel('Milliseconds · mean (circle), median (square), p95 (triangle)');ax.grid(axis='x',alpha=.13)
            latencydenom={'logical_records':op['logical_records'],'valid_records':valid_count,'failed_records':fail_count,'latency_missing_logical_records':op['latency_missing_logical_records'],'finished_http_attempts':op['finished_http_attempts'],'scope_denominators':dict(zip(scopes,totals))}
            save(fig,alias=alias,kind='latency',suffix=name,title=title,caption=f'All actual rounds: {op["logical_records"]} logical records; {valid_count} valid and {fail_count} failed; {op["latency_missing_logical_records"]} logical timings missing. Counts on the plot are observed timed records / scope denominator. Local timings include feature transform/predict/probability and exclude training; gateway timings include transport/retries. No shared local-versus-remote scale or ratio ranking.',denominator=latencydenom,limitations=['Different latency scopes cannot be interpreted as pure provider inference speed.', 'First-attempt missing counts include logical records without finished first HTTP timing; absent local HTTP timing is not zero milliseconds.', 'Training/feature-fit latency and cost are not synthesized from this analysis.'],values=op)
    _dump(output_dir/'assets_manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try: result=export_report(args.analysis,args.output)
    except (ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'figure_export_refused','error':str(exc)}));return 2
    print(json.dumps({'status':'exported_requires_visual_review','bundle_sha256':result['bundle_sha256'],'figures':len(result['figures'])}));return 0


if __name__=='__main__':
    raise SystemExit(main())
