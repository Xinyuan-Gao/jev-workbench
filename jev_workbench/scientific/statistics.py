"""Failure-aware scoring and paired inference; repeated calls never increase n."""
from __future__ import annotations

from collections import Counter,defaultdict
import itertools
import numpy as np
from scipy.stats import binomtest


def _validate(rows, labels):
    if not labels or len(set(labels))!=len(labels):
        raise ValueError('Nonempty unique labels required')
    ids=[r['item_id'] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Select exactly one model/repeat before scoring')
    if any(r['gold_label'] not in labels or not r.get('group_id') for r in rows):
        raise ValueError('Invalid gold or independent group')


def _prediction(row, labels):
    out=row.get('outcome')
    if out is None: return None,'pending'
    if out.get('status')=='valid':
        if out.get('prediction') in labels: return out['prediction'],'valid'
        return None,'invalid_label'
    return None,out.get('status','execution_error')


def _counts(rows, labels):
    index={label:i for i,label in enumerate(labels)}
    matrix=np.zeros((len(labels),len(labels)+1),dtype=np.int64)
    statuses=Counter()
    for row in rows:
        pred,status=_prediction(row,labels)
        matrix[index[row['gold_label']], index[pred] if pred else len(labels)]+=1
        statuses[status]+=1
    return matrix,statuses


def _metrics(matrix, labels):
    tp=np.diag(matrix[:,:len(labels)]).astype(float)
    support=matrix.sum(axis=1).astype(float)
    predicted=matrix[:,:len(labels)].sum(axis=0).astype(float)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    denom=predicted+support
    f1=np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0)
    n=int(support.sum())
    return {'accuracy':float(tp.sum()/n) if n else None, 'macro_f1':float(f1.mean()) if n else None,
            'per_class':{label:{'precision':float(precision[i]),'recall':float(recall[i]),
                                'f1':float(f1[i]),'support':int(support[i]),'predicted':int(predicted[i])}
                         for i,label in enumerate(labels)}}


def score(rows:list[dict], labels:list[str])->dict:
    _validate(rows,labels)
    matrix,statuses=_counts(rows,labels)
    n=len(rows); valid=statuses['valid']; pending=statuses['pending']
    valid_matrix=matrix.copy();valid_matrix[:,-1]=0
    group_correct=defaultdict(list)
    for row in rows:group_correct[row['group_id']].append(_prediction(row,labels)[0]==row['gold_label'])
    return {**_metrics(matrix,labels),'planned':n,'terminal':n-pending,'pending':pending,
            'accuracy_weighting':'each planned row equally weighted',
            'mean_group_accuracy':float(np.mean([np.mean(v) for v in group_correct.values()])) if n else None,
            'valid':valid,'failed':n-pending-valid,'valid_rate':valid/n if n else None,
            'complete':pending==0 and n>0,'provisional':pending>0,
            'independent_groups':len({r['group_id'] for r in rows}),
            'status_counts':dict(statuses), 'valid_only':{**_metrics(valid_matrix,labels),'denominator':valid},
            'labels':labels, 'confusion_columns':labels+['NO_VALID_OUTPUT'], 'confusion_matrix':matrix.tolist(),
            'macro_policy':'fixed supplied label set; unsupported classes score zero'}


def _pair(a,b,labels):
    _validate(a,labels);_validate(b,labels)
    da={r['item_id']:r for r in a};db={r['item_id']:r for r in b}
    if set(da)!=set(db) or not da: raise ValueError('Nonempty matching planned items required')
    keys=sorted(da)
    if any((da[k]['gold_label'],da[k]['group_id'])!=(db[k]['gold_label'],db[k]['group_id']) for k in keys):
        raise ValueError('Paired gold or group mismatch')
    if any(r.get('outcome') is None for r in a+b): raise ValueError('Finish all planned items before inference')
    return [da[k] for k in keys],[db[k] for k in keys]


def _group_vectors(rows,labels,groups):
    index={label:i for i,label in enumerate(labels)};gindex={g:i for i,g in enumerate(groups)}
    gold=np.zeros((len(groups),len(labels)));predicted=gold.copy();tp=gold.copy()
    for row in rows:
        g=gindex[row['group_id']];y=index[row['gold_label']];gold[g,y]+=1
        pred,_=_prediction(row,labels)
        if pred is not None:
            p=index[pred];predicted[g,p]+=1
            if p==y: tp[g,y]+=1
    return gold,predicted,tp


def _bootstrap_metrics(weights,vectors):
    gold,pred,tp=(weights@v for v in vectors)
    denom=gold+pred
    f1=np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0).mean(axis=1)
    accuracy=tp.sum(axis=1)/gold.sum(axis=1)
    return accuracy,f1


def paired_bootstrap(a,b,labels,*,iterations=10000,seed=20261007,stratify=None):
    """Resample independent groups with replacement, retaining all their rows."""
    a,b=_pair(a,b,labels)
    if iterations<1: raise ValueError('Positive iterations required')
    groups=sorted({r['group_id'] for r in a});g=len(groups)
    va=_group_vectors(a,labels,groups);vb=_group_vectors(b,labels,groups)
    if stratify not in {None,'gold_label'}: raise ValueError('Unknown bootstrap stratification')
    strata={}
    if stratify=='gold_label':
        for gi,group in enumerate(groups):
            group_labels={r['gold_label'] for r in a if r['group_id']==group}
            if len(group_labels)!=1: raise ValueError('Gold stratification requires single-class groups')
            strata.setdefault(next(iter(group_labels)),[]).append(gi)
    else: strata={'all':list(range(g))}
    rng=np.random.default_rng(seed);acc_a=[];f1_a=[];acc_b=[];f1_b=[]
    for start in range(0,iterations,256):
        weights=np.zeros((min(256,iterations-start),g),dtype=int)
        for indices in strata.values():
            size=len(indices)
            weights[:,indices]=rng.multinomial(size,np.full(size,1/size),size=len(weights))
        aa,af=_bootstrap_metrics(weights,va);ba,bf=_bootstrap_metrics(weights,vb)
        acc_a.extend(aa);f1_a.extend(af);acc_b.extend(ba);f1_b.extend(bf)
    sa=score(a,labels);sb=score(b,labels)
    def result(values,estimate):
        return {'estimate':estimate,'ci95':np.quantile(values,[.025,.975]).tolist()}
    return {'method':'paired percentile cluster bootstrap','iterations':iterations,'seed':seed,
            'independent_groups':g,'planned_rows':len(a),'insufficient_independent_groups':g<2,
            'stratify':stratify,'stratum_support':{k:len(v) for k,v in strata.items()},
            'rare_strata':[k for k,v in strata.items() if len(v)<5],
            'interval_scope':'empirical sampling distribution conditional on fixed allocation; rare strata cannot establish population certainty',
            'a':{'accuracy':result(acc_a,sa['accuracy']),'macro_f1':result(f1_a,sa['macro_f1'])},
            'b':{'accuracy':result(acc_b,sb['accuracy']),'macro_f1':result(f1_b,sb['macro_f1'])},
            'accuracy_difference':result(np.array(acc_a)-np.array(acc_b),sa['accuracy']-sb['accuracy']),
            'macro_f1_difference':result(np.array(f1_a)-np.array(f1_b),sa['macro_f1']-sb['macro_f1']),
            'difference_direction':'a minus b'}


def mcnemar_exact(a,b,labels):
    a,b=_pair(a,b,labels)
    if len({r['group_id'] for r in a})!=len(a):
        raise ValueError('McNemar requires one independent observation per group')
    ac=bc=0
    for ra,rb in zip(a,b):
        ca=_prediction(ra,labels)[0]==ra['gold_label'];cb=_prediction(rb,labels)[0]==rb['gold_label']
        ac+=int(ca and not cb);bc+=int(cb and not ca)
    return {'method':'two-sided exact McNemar','a_only_correct':ac,'b_only_correct':bc,
            'p_value':float(binomtest(ac,ac+bc,.5).pvalue) if ac+bc else 1.0,'independent_groups':len(a)}


def paired_cluster_test(a,b,labels,*,iterations=10000,seed=20261007):
    """Two-sided paired permutation of group correctness sums (accuracy)."""
    a,b=_pair(a,b,labels);groups=sorted({r['group_id'] for r in a});d={g:0 for g in groups}
    for ra,rb in zip(a,b):
        d[ra['group_id']]+=int(_prediction(ra,labels)[0]==ra['gold_label'])-int(_prediction(rb,labels)[0]==rb['gold_label'])
    differences=np.array(list(d.values())); observed=abs(int(differences.sum()))
    if len(groups)<=18:
        stats=np.fromiter((abs(np.dot(sign,differences)) for sign in itertools.product((-1,1),repeat=len(groups))),dtype=float)
        p=float(np.mean(stats>=observed));method='exact group sign permutation'
    else:
        if iterations<1: raise ValueError('Positive iterations required')
        rng=np.random.default_rng(seed);exceed=0
        for start in range(0,iterations,256):
            signs=rng.choice((-1,1),size=(min(256,iterations-start),len(groups)))
            exceed+=int(np.sum(np.abs(signs@differences)>=observed))
        p=(exceed+1)/(iterations+1);method='Monte Carlo group sign permutation'
    return {'method':method,'p_value':p,'independent_groups':len(groups),'metric':'all-planned accuracy',
            'difference':float(differences.sum()/len(a)),'seed':seed,'iterations':len(stats) if len(groups)<=18 else iterations}


def holm_adjust(p_values):
    p=np.asarray(p_values,dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)): raise ValueError('Invalid p value')
    order=np.argsort(p,kind='stable');out=np.empty_like(p);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,float(p[index])*(len(p)-rank)))
        out[index]=running
    return out.tolist()


def repeat_stability(rows,labels):
    """Descriptive repeated-service/training outcomes, not extra independent n."""
    if len({r.get('model','unspecified') for r in rows})>1:
        raise ValueError('Select one model before repeated-round stability')
    rounds=defaultdict(list)
    for row in rows:rounds[row['repeat']].append(row)
    if not rounds:raise ValueError('Nonempty repeated rows required')
    indices=sorted(rounds);primary=rounds[indices[0]]
    per_round={};outcome_agreements=[];prediction_agreements=[]
    aligned={indices[0]:{r['item_id']:r for r in primary}}
    for repeat in indices:
        a,b=_pair(primary,rounds[repeat],labels)
        aligned[repeat]={r['item_id']:r for r in b}
        per_round[repeat]=score(b,labels)
        if repeat!=indices[0]:
            agreement=[];valid_pairs=[]
            for ra,rb in zip(a,b):
                pa,sa=_prediction(ra,labels);pb,sb=_prediction(rb,labels)
                agreement.append((pa,sa)==(pb,sb))
                if sa==sb=='valid':valid_pairs.append(pa==pb)
            outcome_agreements.append({'repeat':repeat,'rate':float(np.mean(agreement)),'denominator':len(a)})
            prediction_agreements.append({'repeat':repeat,'rate':float(np.mean(valid_pairs)) if valid_pairs else None,
                                          'denominator':len(valid_pairs)})
    all_correct=[];valid_round_counts=Counter()
    for item in aligned[indices[0]]:
        outcomes=[aligned[repeat][item] for repeat in indices]
        all_correct.append(all(_prediction(r,labels)[0]==r['gold_label'] for r in outcomes))
        valid_round_counts[sum(_prediction(r,labels)[1]=='valid' for r in outcomes)]+=1
    return {'rounds':len(indices),'unique_items':len(primary),
            'independent_groups':len({r['group_id'] for r in primary}),
            'assessable':len(indices)>1,'primary_repeat':indices[0],'per_round':per_round,
            'outcome_agreement_with_primary':outcome_agreements,
            'prediction_agreement_conditional_on_both_valid':prediction_agreements,
            'all_rounds_correct_rate':float(np.mean(all_correct)),
            'valid_round_count_distribution':dict(valid_round_counts),
            'inference_scope':'descriptive stability; repeated attempts do not add independent samples'}
