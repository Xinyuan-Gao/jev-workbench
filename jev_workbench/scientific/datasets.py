"""Dataset provenance, pre-evaluation cleaning and independent-unit gates."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


def normalized_text(value: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value)).casefold()).strip()


def input_key(record: dict) -> str:
    return json.dumps({k:normalized_text(v) for k,v in sorted(record['input'].items())},ensure_ascii=False,sort_keys=True)


def manifest_hash(records: list[dict]) -> str:
    canonical=json.dumps(sorted(records,key=lambda r:r['id']),ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def clean_exact_duplicates(records: list[dict]) -> tuple[list[dict],list[dict]]:
    """Preserve official test first; never choose duplicates using model outputs."""
    priority={'test':0,'validation':1,'train':2}
    grouped=defaultdict(list)
    for r in records: grouped[input_key(r)].append(r)
    kept,removed=[],[]
    for family in grouped.values():
        family.sort(key=lambda r:(priority[r['split']],r['id']))
        if len({r['label'] for r in family})>1:
            removed.extend({'id':r['id'],'split':r['split'],'reason':'conflicting_official_labels','kept_id':None} for r in family)
            continue
        winner=family[0]
        kept.append(deepcopy(winner))
        removed.extend({'id':r['id'],'split':r['split'],'reason':'exact_input_duplicate','kept_id':winner['id']} for r in family[1:])
    return sorted(kept,key=lambda r:r['id']),removed


def audit_dataset(records: list[dict], labels: list[str], *, min_test_groups: int=200) -> dict:
    errors=[]
    seen_ids=set()
    groups=defaultdict(set)
    inputs=defaultdict(list)
    counts={split:{'records':0,'groups':0,'labels':{}} for split in ('train','validation','test')}
    for r in records:
        if r['id'] in seen_ids: errors.append({'code':'duplicate_id','id':r['id']})
        seen_ids.add(r['id'])
        split=r.get('split')
        if split not in counts:
            errors.append({'code':'invalid_split','id':r['id']}); continue
        if r.get('label') not in labels: errors.append({'code':'invalid_label','id':r['id']})
        if not r.get('group_id'): errors.append({'code':'missing_group','id':r['id']})
        groups[r['group_id']].add(split)
        keys=set(r.get('input',{}))
        if keys not in ({'text'},{'task'},{'query','passage'}): errors.append({'code':'input_not_whitelisted','id':r['id']})
        if any(not isinstance(v,str) or not v.strip() for v in r.get('input',{}).values()):
            errors.append({'code':'invalid_or_empty_input','id':r['id']})
        inputs[input_key(r)].append(r)
        counts[split]['records']+=1
    for group,splits in groups.items():
        if len(splits)>1: errors.append({'code':'group_overlap','group_id':group,'splits':sorted(splits)})
    for family in inputs.values():
        if len({r['split'] for r in family})>1:
            errors.append({'code':'exact_input_overlap','ids':[r['id'] for r in family]})
    for split in counts:
        subset=[r for r in records if r['split']==split]
        counts[split]['groups']=len({r['group_id'] for r in subset})
        counts[split]['labels']=dict(sorted(Counter(r['label'] for r in subset).items()))
    if counts['test']['groups']<min_test_groups:
        errors.append({'code':'too_few_independent_test_units','required':min_test_groups,'actual':counts['test']['groups']})
    return {'structural_ready':not errors,'ready':False,'errors':errors,'counts':counts,'manifest_sha256':manifest_hash(records),
            'near_duplicate_audit':'not_yet_run','human_label_audit':'official_labels_preserved_not_independently_reannotated'}


def clean_near_duplicates(records: list[dict], *, threshold:float=.90, min_length:int=2,
                          batch_size:int=128) -> tuple[list[dict],list[dict],dict]:
    """Conservative lexical screening, independent of gold and model predictions.

    Test is retained before validation and train. For paired passages the query
    defines a unit; removal affects the entire query group. A corpus-wide TF-IDF
    here is ONLY an input audit; no feature fitted here is used by a classifier.
    Does not claim to detect semantic paraphrases or pretraining contamination.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not 0<threshold<=1 or min_length<1 or batch_size<1: raise ValueError('Invalid screening settings')
    priority={'test':0,'validation':1,'train':2};by_group=defaultdict(list)
    for row in records: by_group[row['group_id']].append(row)
    units=[]
    for group,rows in by_group.items():
        if len({r['split'] for r in rows})!=1: raise ValueError('Group crosses splits')
        texts={normalized_text(r['input'].get('query',r['input'].get('text',r['input'].get('task','')))) for r in rows}
        if len(texts)!=1: raise ValueError('Independent group has inconsistent query/text')
        units.append({'group':group,'split':rows[0]['split'],'text':next(iter(texts)),
                      'first_id':min(r['id'] for r in rows)})
    units.sort(key=lambda u:(priority[u['split']],u['first_id']))
    removed_groups={};candidate_pairs=0
    eligible=[u for u in units if len(u['text'])>=min_length]
    if eligible:
        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=False,norm='l2')
        matrix=vectorizer.fit_transform([u['text'] for u in eligible])
        screens=sum(u['split']!='train' for u in eligible)
        for start in range(0,screens,batch_size):
            end=min(start+batch_size,screens)
            similarities=(matrix[start:end]@matrix.T).tocsr()
            for local,i in enumerate(range(start,end)):
                winner=eligible[i]
                if winner['group'] in removed_groups: continue
                line=similarities.getrow(local)
                for j,similarity in sorted(zip(line.indices,line.data)):
                    if j<=i or similarity<threshold: continue
                    loser=eligible[j];candidate_pairs+=1
                    if loser['group'] not in removed_groups:
                        removed_groups[loser['group']]={'kept_group':winner['group'], 'kept_id':winner['first_id'],
                                                       'cosine':float(similarity)}
    # Include short exact query repeats, also when passages differ.
    seen_text={}
    for unit in units:
        if unit['group'] in removed_groups: continue
        if unit['text'] in seen_text:
            winner=seen_text[unit['text']]
            removed_groups[unit['group']]={'kept_group':winner['group'],'kept_id':winner['first_id'],'cosine':1.0}
        else: seen_text[unit['text']]=unit
    removed=[{'id':r['id'],'split':r['split'],'group_id':r['group_id'],'reason':'lexical_near_duplicate',
              **removed_groups[r['group_id']]} for r in records if r['group_id'] in removed_groups]
    kept=[r for r in records if r['group_id'] not in removed_groups]
    audit={'method':'character_tfidf_cosine','threshold':threshold,'min_length':min_length,'ngrams':[2,4],
           'fit_scope':'all input-only audit units; vectorizer never reused for model features',
           'priority':['test','validation','train'],'candidate_pairs':candidate_pairs,
           'short_units_exact_only':sum(len(u['text'])<min_length for u in units),
           'removed_records':len(removed),'removed_groups':len(removed_groups),
           'semantic_duplicate_detection':'not established','pretraining_contamination':'cannot exclude'}
    return kept,removed,audit


def load_massive(source: Path, *, test_cap_per_label: int|None=10, seed: int=20261007, min_test_groups: int=200) -> dict:
    original=[json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    rows=[]
    for r in original:
        if r['locale']!='zh-CN': continue
        split={'train':'train','dev':'validation','test':'test'}[r['partition']]
        rows.append({'id':'massive-zh-'+str(r['id']),'dataset_id':'massive-zh-CN-v1.1','source_id':str(r['id']),
                     'split':split,'group_id':'massive-original-'+str(r['id']),'label':r['intent'],
                     'scenario':r['scenario'],'input':{'text':r['utt']},
                     'label_source':'MASSIVE 1.1 official zh-CN intent; human localization judgments',
                     'construction_method':'official utterance, no generated variants'})
    labels=sorted({r['label'] for r in rows})
    cleaned,removed=clean_exact_duplicates(rows)
    candidates=defaultdict(list)
    for r in cleaned:
        if r['split']=='test': candidates[r['label']].append(r)
    rng=random.Random(seed)
    selected=[]
    for label in sorted(candidates):
        candidates[label].sort(key=lambda r:r['id'])
        rng.shuffle(candidates[label])
        selected.extend(candidates[label] if test_cap_per_label is None else candidates[label][:test_cap_per_label])
    selected_ids={r['id'] for r in selected}
    records=[r for r in cleaned if r['split']!='test' or r['id'] in selected_ids]
    audit=audit_dataset(records,labels,min_test_groups=min_test_groups)
    return {'dataset_id':'massive-zh-CN-v1.1','labels':labels,'records':records,'audit':audit,'removals':removed,
            'sampling':{'seed':seed,'test_cap_per_label':test_cap_per_label,'scheme':'capped stratified official test after input-only exact cleaning',
                        'source_test_labels':len(candidates),'selected_test_records':len(selected),'official_distribution_preserved':test_cap_per_label is None},
            'source':{'local_file':str(source),'sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                      'url':'https://github.com/alexa/massive','version':'1.1','license':'CC-BY-4.0 dataset / Apache-2.0 code'}}


def load_t2_source(source:Path, *, seed:int=20261007,min_test_groups:int=200)->dict:
    """Use actual original query/pid/grade records; never infer grades from order."""
    raw=[json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    groups=defaultdict(list)
    for row in raw:
        grade=row['human_grade']
        if isinstance(grade,bool) or grade not in (0,1,2,3):raise ValueError('T2 requires original 0..3 human grade')
        if row['binary_relevant'] is not (grade>=2):raise ValueError('Source binary label contradicts original grade')
        groups[(row['source_split'],str(row['source_qid']))].append(row)
    rows=[]
    for key,family in sorted(groups.items()):
        if len({r['internal_split'] for r in family})!=1:raise ValueError('Query source appears in multiple internal splits')
        for grade in range(4):
            candidates=sorted([r for r in family if r['human_grade']==grade],key=lambda r:int(r['source_pid']))
            if not candidates:continue
            rng=random.Random(f'{seed}:{key[0]}:{key[1]}:{grade}')
            chosen=candidates[rng.randrange(len(candidates))]
            ident=f't2-{key[0]}-{key[1]}-{chosen["source_pid"]}'
            rows.append({'id':ident,'dataset_id':'t2-original-dev-custom-query-split','source_id':chosen['id'],
                'source_qid':str(chosen['source_qid']),'source_pid':str(chosen['source_pid']),
                'source_split':chosen['source_split'],'source_revision':chosen['source_revision'],
                'split':chosen['internal_split'],'group_id':f't2-{key[0]}-query-{key[1]}',
                'grade':grade,'label':'relevant' if grade>=2 else 'irrelevant',
                'input':{'query':chosen['query'],'passage':chosen['passage']},
                'label_source':'T2Ranking original expert final qrels, grade>=2 relevance',
                'construction_method':'one seeded original passage per available grade; full text without truncation'})
    rows=sorted(rows,key=lambda r:r['id'])
    cleaned,removed=clean_exact_duplicates(rows)
    cleaned,near_removals,near=clean_near_duplicates(cleaned)
    audit=audit_dataset(cleaned,['irrelevant','relevant'],min_test_groups=min_test_groups)
    audit['near_duplicate_audit']=near
    return {'dataset_id':'t2-original-dev-custom-query-split','labels':['irrelevant','relevant'],
            'records':cleaned,'audit':audit,'removals':removed+near_removals,
            'sampling':{'seed':seed,'query_sampling':'1600 source-dev queries with both binary classes, uniform without replacement; 200 test / 200 validation / 1200 train before lexical cleaning',
                        'passage_sampling':'one uniformly chosen source passage per present original grade, no compensation for missing grades',
                        'input_policy':'full source text, never silently truncated','representative_natural_candidate_distribution':False},
            'source':{'local_file':str(source),'sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                      'url':'https://huggingface.co/datasets/THUIR/T2Ranking','version':'2a369a430a70979223f1b9a41b1919774d46b432',
                      'collection_sha256':'07b84e543e9ba696124a727c00629d5bce586631648c436c98dd6e9b146da212','license':'Apache-2.0'},
            'limitations':['原公开dev内部按query重新划分，不能称官方test成绩','每等级限额抽样，不代表自然检索候选分布',
                           '多个片段共享query，以query计独立样本量','公开数据的预训练污染无法排除','保留完整片段，超长输入若服务拒绝也计入失败']}
