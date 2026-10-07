"""Train-only local baselines with predeclared validation selection.

No test set is accepted by the training interface. Character TF-IDF vocabulary,
IDF, and randomized SVD are fitted exclusively to the supplied train partition.
LR/lbfgs and LinearSVC/dual=False are deterministic primary-only models; sampled
boosting and its randomized SVD are actually refitted for five distinct seeds.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import time
import warnings

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

DEFAULT_SEEDS = (20261007, 20261008, 20261009, 20261010, 20261011)
CANDIDATE_GRIDS = {
    'lr': tuple({'C': c, 'class_weight': weight} for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0) for weight in (None, 'balanced')),
    'linear_svc': tuple({'C': c, 'class_weight': weight} for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0) for weight in (None, 'balanced')),
    'xgboost': tuple({'n_estimators': n, 'max_depth': depth, 'learning_rate': rate}
                     for n in (60, 120, 200) for depth in (3, 6) for rate in (0.05, 0.1)),
    'lightgbm': tuple({'n_estimators': n, 'max_depth': depth, 'learning_rate': rate}
                      for n in (60, 120, 200) for depth in (3, 6) for rate in (0.05, 0.1)),
}
_TREE_FAMILIES = {'xgboost', 'lightgbm'}


def _inputs(records):
    inputs=[]
    task=None
    for record in records:
        item=record['input']
        current='relevance' if 'query' in item or 'passage' in item else 'text'
        fields=('query','passage') if current == 'relevance' else ('text',) if 'text' in item else ('task',)
        if any(not isinstance(item.get(field), str) or not item[field].strip() for field in fields):
            raise ValueError('Input requires nonempty task/text or query and passage strings')
        if task is not None and task != current:
            raise ValueError('Mixed text and relevance inputs')
        task=current
        inputs.append({field:item[field] for field in fields})
    return task, inputs


def _tokens(text):
    return set(re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9]+', text.casefold()))


class InputFeatures:
    def __init__(self, *, dense=False, svd_components=64, max_features=12000, seed=DEFAULT_SEEDS[0]):
        self.dense=dense
        self.svd_components=svd_components
        self.max_features=max_features
        self.seed=seed
        self.feature_names=[]
        self.svd=None

    def _vectorizer(self, vocabulary=None):
        return TfidfVectorizer(analyzer='char', ngram_range=(1,3), min_df=1,
                               max_features=self.max_features, sublinear_tf=True,
                               vocabulary=vocabulary, dtype=np.float64)

    def fit(self, train_records):
        if not train_records or any(r.get('split', 'train') != 'train' for r in train_records):
            raise ValueError('Feature fit accepts train records only')
        self.fit_record_ids=[r.get('id', str(i)) for i,r in enumerate(train_records)]
        self.task, items=_inputs(train_records)
        if self.task == 'relevance':
            queries=[item['query'] for item in items]; passages=[item['passage'] for item in items]
            self.vectorizer=self._vectorizer().fit(queries+passages)
            vocabulary=dict(self.vectorizer.vocabulary_)
            self.query_vectorizer=self._vectorizer(vocabulary).fit(queries)
            self.passage_vectorizer=self._vectorizer(vocabulary).fit(passages)
            query=self.query_vectorizer.transform(queries)
            passage=self.passage_vectorizer.transform(passages)
            svd_fit=sparse.vstack([query,passage],format='csr')
        else:
            texts=[next(iter(item.values())) for item in items]
            self.vectorizer=self._vectorizer().fit(texts)
            svd_fit=self.vectorizer.transform(texts)
        self.embedding_dimensions=len(self.vectorizer.vocabulary_)
        if self.dense:
            self.embedding_dimensions=max(1,min(self.svd_components,svd_fit.shape[0]-1,svd_fit.shape[1]-1))
            if svd_fit.shape[1] < 2:
                raise ValueError('SVD requires at least two character features')
            self.svd=TruncatedSVD(n_components=self.embedding_dimensions, random_state=self.seed, n_iter=5)
            with threadpool_limits(limits=1):
                self.svd.fit(svd_fit)
        if self.task == 'relevance':
            self.feature_names=['query_embedding','passage_embedding','absolute_difference','elementwise_product',
                                'cosine','token_jaccard','query_token_coverage']
        else:
            self.feature_names=['text_embedding']
        return self

    def transform(self, records):
        task,items=_inputs(records)
        if not items:
            width=4*self.embedding_dimensions+3 if self.task=='relevance' else self.embedding_dimensions
            return np.empty((0,width)) if self.dense else sparse.csr_matrix((0,width))
        if task != self.task:
            raise ValueError('Input task does not match fitted features')
        if self.task != 'relevance':
            matrix=self.vectorizer.transform([next(iter(item.values())) for item in items])
            with threadpool_limits(limits=1):
                return self.svd.transform(matrix) if self.dense else matrix
        query=self.query_vectorizer.transform([item['query'] for item in items])
        passage=self.passage_vectorizer.transform([item['passage'] for item in items])
        numerator=np.asarray(query.multiply(passage).sum(axis=1)).ravel()
        norm=np.sqrt(np.asarray(query.multiply(query).sum(axis=1)).ravel()*
                     np.asarray(passage.multiply(passage).sum(axis=1)).ravel())
        cosine=np.divide(numerator,norm,out=np.zeros_like(numerator),where=norm>0)
        interactions=[]
        for index,item in enumerate(items):
            query_tokens,passage_tokens=_tokens(item['query']),_tokens(item['passage'])
            intersection=len(query_tokens&passage_tokens)
            interactions.append([cosine[index],intersection/max(1,len(query_tokens|passage_tokens)),
                                 intersection/max(1,len(query_tokens))])
        interactions=np.asarray(interactions,dtype=float)
        if self.dense:
            with threadpool_limits(limits=1):
                q=self.svd.transform(query); p=self.svd.transform(passage)
            return np.hstack([q,p,np.abs(q-p),q*p,interactions])
        return sparse.hstack([query,passage,abs(query-passage),query.multiply(passage),sparse.csr_matrix(interactions)],format='csr')


def select_candidate(candidates):
    valid=[candidate for candidate in candidates if candidate['status']=='valid']
    if not valid:
        raise RuntimeError('All predeclared candidates failed')
    return min(valid,key=lambda candidate:(-candidate['validation_macro_f1'],candidate['candidate_index']))


def _estimator(family,params,seed,classes):
    if family=='lr':
        return LogisticRegression(**params,solver='lbfgs',max_iter=1500,tol=1e-5,random_state=seed)
    if family=='linear_svc':
        return LinearSVC(**params,dual=False,max_iter=5000,tol=1e-5,random_state=seed)
    if family=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**params,random_state=seed,n_jobs=1,subsample=0.8,colsample_bytree=0.8,
                             tree_method='hist',objective='binary:logistic' if classes==2 else 'multi:softprob',
                             eval_metric='logloss' if classes==2 else 'mlogloss')
    if family=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params,random_state=seed,n_jobs=1,subsample=0.8,subsample_freq=1,
                              colsample_bytree=0.8,num_leaves=min(31,2**params['max_depth']),
                              min_child_samples=10,verbosity=-1,force_col_wise=True,
                              objective='binary' if classes==2 else 'multiclass')
    raise ValueError('Unknown local model family')


def _predict_encoded(estimator,matrix):
    with warnings.catch_warnings(),threadpool_limits(limits=1):
        warnings.filterwarnings('ignore',message='X does not have valid feature names',category=UserWarning)
        return estimator.predict(matrix)


class FittedLocalModel:
    def __init__(self,family,seed,labels,*,features=None,estimator=None,majority_label=None):
        self.family,self.seed,self.labels=family,seed,list(labels)
        self.features,self.estimator,self.majority_label=features,estimator,majority_label

    def predict(self,records):
        """Return actual single-row feature + estimator latency without reading gold."""
        output=[]
        for record in records:
            started=time.perf_counter()
            confidence=None
            if self.family=='majority':
                prediction=self.majority_label
            else:
                matrix=self.features.transform([record])
                index=int(_predict_encoded(self.estimator,matrix)[0])
                prediction=self.labels[index]
                if hasattr(self.estimator,'predict_proba'):
                    with warnings.catch_warnings(),threadpool_limits(limits=1):
                        warnings.filterwarnings('ignore',message='X does not have valid feature names',category=UserWarning)
                        confidence=float(np.max(self.estimator.predict_proba(matrix)[0]))
            output.append({'status':'valid','prediction':prediction,'confidence':confidence,
                           'confidence_status':'valid' if confidence is not None else 'missing',
                           'latency_ms':(time.perf_counter()-started)*1000})
        return output


def _validate_partitions(train,validation,labels):
    if not train or not validation:
        raise ValueError('Nonempty train and validation are required')
    if not labels or len(set(labels))!=len(labels):
        raise ValueError('Unique label list required')
    for expected,records in [('train',train),('validation',validation)]:
        if any(record.get('split',expected)!=expected for record in records):
            raise ValueError(f'{expected} input contains a different split; test cannot enter tuning')
        if any(record['label'] not in labels for record in records):
            raise ValueError('Unknown training/validation label')
    if set(record['label'] for record in train)!=set(labels):
        raise ValueError('Every configured label must occur in train')
    train_ids={r.get('id') for r in train if r.get('id') is not None}
    if train_ids.intersection(r.get('id') for r in validation):
        raise ValueError('Train and validation record IDs overlap')
    train_groups={r.get('group_id') for r in train if r.get('group_id')}
    if train_groups.intersection(r.get('group_id') for r in validation if r.get('group_id')):
        raise ValueError('Train and validation groups overlap')
    _inputs(train); _inputs(validation)


def _versions():
    result={'python':platform.python_version()}
    for name in ('numpy','scipy','scikit-learn','xgboost','lightgbm','threadpoolctl'):
        try:
            result[name]=version(name)
        except PackageNotFoundError:
            result[name]='unavailable'
    return result


def _jsonsafe(value):
    if isinstance(value, dict):
        return {key:_jsonsafe(part) for key,part in value.items()}
    if isinstance(value, (list,tuple)):
        return [_jsonsafe(part) for part in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str,int,float,bool)):
        return value
    return repr(value)


def train_local_models(train_records,validation_records,labels,*,families=None,seeds=DEFAULT_SEEDS,output_path=None):
    """Select >=12 configs/family on validation; preserve every candidate audit.

    Return ``{'models': {family: [FittedLocalModel, ...]}, 'report': dict}``.
    If output_path is supplied, write only the JSON-safe audit report. Training
    feature limits (12k vocabulary, 64 SVD dimensions, one thread) are fixed here.
    Failed candidates retain their error and cannot become a zero-valued winner.
    """
    _validate_partitions(train_records,validation_records,labels)
    families=list(families) if families is not None else ['majority',*CANDIDATE_GRIDS]
    if len(set(families))!=len(families) or any(family not in {'majority',*CANDIDATE_GRIDS} for family in families):
        raise ValueError('Unknown or duplicate family')
    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(not isinstance(seed,int) for seed in seeds):
        raise ValueError('Distinct integer seeds required')
    if any(family in _TREE_FAMILIES for family in families) and len(seeds)<5:
        raise ValueError('Stochastic models require at least five distinct training seeds')
    label_to_index={label:index for index,label in enumerate(labels)}
    y_train=np.asarray([label_to_index[r['label']] for r in train_records])
    y_validation=np.asarray([label_to_index[r['label']] for r in validation_records])
    report={'created_at':datetime.now(timezone.utc).isoformat(),'status':'running',
            'fit_split':'train','selection_split':'validation',
            'selection_metric':'full_validation_macro_f1','test_used':False,'labels':list(labels),
            'train_record_ids':[r.get('id',str(i)) for i,r in enumerate(train_records)],
            'validation_record_ids':[r.get('id',str(i)) for i,r in enumerate(validation_records)],
            'seeds':list(seeds),'versions':_versions(),'thread_limit':1,
            'train_label_support':{label:sum(r['label']==label for r in train_records) for label in labels},
            'validation_label_support':{label:sum(r['label']==label for r in validation_records) for label in labels},
            'feature_parameters':{'analyzer':'char','ngram_range':[1,3],'max_features':12000,'svd_components':64},
            'grid_sha256':hashlib.sha256(json.dumps(CANDIDATE_GRIDS,sort_keys=True).encode()).hexdigest(),
            'tie_break':'highest Macro-F1, then predeclared candidate_index (not a calculated complexity measure)',
            'families':{}}
    def persist():
        if output_path is None:
            return
        target=Path(output_path); target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.tmp')
        with temporary.open('w',encoding='utf-8') as handle:
            json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,target)
    persist()
    models={}
    for family in families:
        report['active_family']=family
        if family=='majority':
            fit_start=time.perf_counter()
            counts=Counter(r['label'] for r in train_records)
            majority=min(labels,key=lambda label:(-counts[label],label_to_index[label]))
            fit_ms=(time.perf_counter()-fit_start)*1000
            prediction=np.full(len(validation_records),label_to_index[majority])
            score=float(f1_score(y_validation,prediction,labels=list(range(len(labels))),average='macro',zero_division=0))
            models[family]=[FittedLocalModel(family,seeds[0],labels,majority_label=majority)]
            report['families'][family]={'repeat_policy':'deterministic_primary_only','majority_label':majority,
                'train_class_counts':dict(counts),'selection':{'validation_macro_f1':score,'params':{}},
                'candidates':[],'repeats':[{'seed':seeds[0],'fit_performed':True,'model_fit_ms':fit_ms}]}
            persist()
            continue
        candidates=[]
        report['families'][family]={'candidates':candidates,'selection':None,'repeats':[],
                                  'status':'fitting_features'}
        persist()
        feature_start=time.perf_counter()
        try:
            features=InputFeatures(dense=family in _TREE_FAMILIES,seed=seeds[0]).fit(train_records)
        except Exception as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            report['families'][family]['status']='failed'
            persist()
            raise
        feature_fit_ms=(time.perf_counter()-feature_start)*1000
        try:
            transform_start=time.perf_counter(); x_train=features.transform(train_records)
            transform_train_ms=(time.perf_counter()-transform_start)*1000
            transform_start=time.perf_counter(); x_val=features.transform(validation_records)
            transform_validation_ms=(time.perf_counter()-transform_start)*1000
        except Exception as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            report['families'][family].update(status='failed',error_stage='feature_transform',
                                              feature_fit_ms=feature_fit_ms)
            persist()
            raise
        report['families'][family].update(status='tuning',feature_fit_ms=feature_fit_ms,
            feature_transform_train_ms=transform_train_ms,feature_transform_validation_ms=transform_validation_ms)
        fitted={}
        for index,params in enumerate(CANDIDATE_GRIDS[family]):
            candidate={'candidate_index':index,'params':dict(params),'seed':seeds[0],
                       'feature_fit_ms':feature_fit_ms,'status':'failed','validation_macro_f1':None,
                       'model_fit_ms':None,'validation_predict_ms':None}
            try:
                estimator=_estimator(family,params,seeds[0],len(labels))
                candidate['full_estimator_parameters']=_jsonsafe(estimator.get_params())
                fit_start=time.perf_counter()
                with threadpool_limits(limits=1):
                    estimator.fit(x_train,y_train)
                candidate['model_fit_ms']=(time.perf_counter()-fit_start)*1000
                predict_start=time.perf_counter()
                prediction=_predict_encoded(estimator,x_val)
                candidate['validation_predict_ms']=(time.perf_counter()-predict_start)*1000
                candidate['validation_macro_f1']=float(f1_score(y_validation,prediction,labels=list(range(len(labels))),average='macro',zero_division=0))
                candidate['status']='valid'
                # Keep only the current winner; tree grids can otherwise retain
                # twelve large 60-class ensembles without helping selection.
                if select_candidate(candidates+[candidate])['candidate_index']==index:
                    fitted={index:estimator}
            except Exception as exc:
                candidate.update(error_type=type(exc).__name__,error=str(exc))
            candidates.append(candidate)
            persist()
        try:
            selected=select_candidate(candidates)
        except RuntimeError as exc:
            report.update(status='failed',error=str(exc))
            report['families'][family]['status']='failed'
            persist()
            raise
        estimator=fitted[selected['candidate_index']]
        models[family]=[FittedLocalModel(family,seeds[0],labels,features=features,estimator=estimator)]
        repeats=[{'seed':seeds[0],'fit_performed':True,'reused_selected_train_fit':True,
                  'model_fit_ms':selected['model_fit_ms'],'feature_fit_ms':feature_fit_ms}]
        report['families'][family].update(selection=dict(selected),repeats=repeats,status='fitting_repeats')
        persist()
        if family in _TREE_FAMILIES:
            for seed in seeds[1:]:
                repeat={'seed':seed,'status':'fitting','fit_performed':False,
                        'reused_selected_train_fit':False,'model_fit_ms':None,'feature_fit_ms':None}
                repeats.append(repeat)
                persist()
                try:
                    feature_start=time.perf_counter()
                    repeat_features=InputFeatures(dense=True,seed=seed).fit(train_records)
                    repeat['feature_fit_ms']=(time.perf_counter()-feature_start)*1000
                    matrix=repeat_features.transform(train_records)
                    repeat_estimator=_estimator(family,selected['params'],seed,len(labels))
                    fit_start=time.perf_counter()
                    with threadpool_limits(limits=1):
                        repeat_estimator.fit(matrix,y_train)
                    repeat.update(model_fit_ms=(time.perf_counter()-fit_start)*1000,fit_performed=True,status='valid')
                    models[family].append(FittedLocalModel(family,seed,labels,features=repeat_features,estimator=repeat_estimator))
                except Exception as exc:
                    repeat.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                    report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                    report['families'][family]['status']='failed'
                    persist()
                    raise
                persist()
        report['families'][family].update({'status':'complete','candidates':candidates,'selection':dict(selected),
            'repeat_policy':'five_real_seed_fits' if family in _TREE_FAMILIES else 'deterministic_primary_only',
            'repeats':repeats,'feature_audit':{'fit_record_ids':features.fit_record_ids,
                'vocabulary_size':len(features.vectorizer.vocabulary_),'embedding_dimensions':features.embedding_dimensions,
                'feature_names':features.feature_names,'svd_fit_split':'train' if features.dense else None}})
        persist()
    report.update(status='complete',active_family=None)
    persist()
    return {'models':models,'report':report}
