import importlib
import importlib.util
import json


def module():
    assert importlib.util.find_spec('jev_workbench.scientific.datasets') is not None, 'Data contract not implemented'
    return importlib.import_module('jev_workbench.scientific.datasets')


def record(ident, split, group, text, label='a'):
    return {'id':ident,'dataset_id':'fixture','split':split,'group_id':group,'label':label,'input':{'text':text}}


def test_audit_rejects_shared_group_even_when_text_differs():
    m=module()
    result=m.audit_dataset([record('1','train','shared','训练消息'),record('2','test','shared','不同测试消息')],['a'],min_test_groups=1)
    assert not result['ready']
    assert any(e['code']=='group_overlap' for e in result['errors'])


def test_audit_counts_independent_queries_not_passage_rows():
    m=module()
    rows=[{'id':str(i),'dataset_id':'rag','split':'test','group_id':'same-query','label':'a','input':{'query':'退款需要多久','passage':str(i)}} for i in range(200)]
    result=m.audit_dataset(rows,['a'],min_test_groups=200)
    assert result['counts']['test']['records']==200
    assert result['counts']['test']['groups']==1
    assert not result['ready']


def test_cleaning_keeps_test_and_removes_identical_training_text():
    m=module()
    kept,removed=m.clean_exact_duplicates([record('1','train','g1','查询 天气'),record('2','test','g2','查询  天气')])
    assert [r['id'] for r in kept]==['2']
    assert removed[0]['reason']=='exact_input_duplicate'
    assert removed[0]['kept_id']=='2'


def test_manifest_hash_detects_label_or_input_change():
    m=module()
    rows=[record('1','test','g1','今天天气')]
    first=m.manifest_hash(rows)
    rows[0]['label']='b'
    assert m.manifest_hash(rows)!=first


def test_structural_audit_does_not_claim_all_scientific_gates_passed():
    audit=module().audit_dataset([record('1','test','g1','测试话语')],['a'],min_test_groups=1)
    assert audit['structural_ready']
    assert not audit['ready']


def test_structural_audit_rejects_nonstring_input():
    audit=module().audit_dataset([record('1','test','g1',None)],['a'],min_test_groups=1)
    assert not audit['structural_ready']


def test_near_cleaning_preserves_test_and_removes_similar_train():
    rows=[record('t','train','t','请帮我查询一下北京今天下午的天气情况怎么样啊'),
          record('e','test','e','请帮我查询一下北京今天下午的天气情况怎么样呀'),
          record('v','validation','v','把客厅的灯关闭')]
    cleaned,removed,audit=module().clean_near_duplicates(rows,threshold=.85,min_length=8)
    assert {r['id'] for r in cleaned}=={'e','v'}
    assert removed[0]['kept_id']=='e' and removed[0]['reason']=='lexical_near_duplicate'
    assert audit['method']=='character_tfidf_cosine'


def test_near_cleaning_uses_query_for_rag_and_removes_entire_group():
    rows=[]
    for split,qid,query in [('train','t','如何退还已经购买的会员订阅费用啊'),('test','e','如何退还已经购买的会员订阅费用呀')]:
        for p in range(2):
            rows.append({'id':qid+str(p),'split':split,'group_id':qid,'label':'a',
                         'input':{'query':query,'passage':'完全不同的片段'+qid+str(p)}})
    cleaned,removed,_=module().clean_near_duplicates(rows,threshold=.8,min_length=8)
    assert {r['group_id'] for r in cleaned}=={'e'}
    assert len(removed)==2


def test_massive_loader_preserves_official_splits_and_rare_classes(tmp_path):
    m=module()
    raw=[]
    for i,(partition,label,text) in enumerate([('train','a','训练A'),('train','b','训练B'),('dev','a','验证A'),('test','a','测试A1'),('test','a','测试A2'),('test','b','测试B')]):
        raw.append({'id':str(i),'locale':'zh-CN','partition':partition,'intent':label,'scenario':'fixture','utt':text,'worker_id':'not-for-model','judgments':[{}]})
    path=tmp_path/'source.jsonl'
    path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in raw))
    result=m.load_massive(path,test_cap_per_label=1,seed=42,min_test_groups=2)
    assert len([r for r in result['records'] if r['split']=='test'])==2
    assert {r['label'] for r in result['records'] if r['split']=='test'}=={'a','b'}
    assert next(r for r in result['records'] if r['source_id']=='2')['split']=='validation'
    assert all(set(r['input'])=={'text'} for r in result['records'])
    assert all('worker_id' not in r for r in result['records'])


def test_t2_loader_preserves_real_grade_and_excludes_topic_only_positive(tmp_path):
    raw=[{'id':'src-'+str(g),'source_revision':'fixed','source_split':'dev','source_qid':7,
          'source_pid':g,'internal_split':'test','query':'同一个问题','passage':'真实片段'+str(g),
          'human_grade':g,'binary_relevant':g>=2} for g in range(4)]
    path=tmp_path/'source.jsonl';path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in raw))
    result=module().load_t2_source(path,min_test_groups=1)
    assert len(result['records'])==4
    assert {r['grade']:r['label'] for r in result['records']}=={0:'irrelevant',1:'irrelevant',2:'relevant',3:'relevant'}
    assert result['audit']['counts']['test']['groups']==1
    assert all(set(r['input'])=={'query','passage'} for r in result['records'])
    assert all(r['source_split']=='dev' for r in result['records'])


def test_t2_sampling_chooses_one_per_available_grade_deterministically(tmp_path):
    raw=[{'id':str(g)+str(p),'source_revision':'fixed','source_split':'dev','source_qid':7,
          'source_pid':10*g+p,'internal_split':'test','query':'问题','passage':'片段'+str(g)+str(p),
          'human_grade':g,'binary_relevant':g>=2} for g in range(4) for p in range(3)]
    path=tmp_path/'source.jsonl';path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in raw))
    a=module().load_t2_source(path,min_test_groups=1,seed=7)
    path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in reversed(raw)))
    b=module().load_t2_source(path,min_test_groups=1,seed=7)
    assert a['records']==b['records']
    assert len(a['records'])==4
