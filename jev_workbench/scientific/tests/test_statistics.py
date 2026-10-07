import importlib
import pytest


def module():
    return importlib.import_module('jev_workbench.scientific.statistics')


def row(i, gold, pred, group=None, status=None):
    return {'item_id':str(i), 'group_id':group or str(i), 'gold_label':gold,
            'outcome':None if status=='pending' else {'status':status or ('valid' if pred else 'refusal'), 'prediction':pred}}


def test_failures_remain_in_accuracy_and_class_false_negatives():
    r=[row(1,'a','a'),row(2,'a','b'),row(3,'b','b'),row(4,'b',None)]
    m=module().score(r,['a','b'])
    assert m['accuracy']==.5
    assert m['macro_f1']==pytest.approx(7/12)
    assert m['valid_only']['macro_f1']==pytest.approx(2/3)
    assert m['valid_rate']==.75
    assert m['per_class']['b']['support']==2
    assert m['per_class']['b']['recall']==.5
    assert m['complete']


def test_pending_is_counted_but_never_called_complete():
    m=module().score([row(1,'a','a'),row(2,'a',None,status='pending')],['a'])
    assert m['planned']==2 and m['terminal']==1 and not m['complete']
    assert m['accuracy']==.5
    assert m['provisional']


def test_malformed_valid_output_does_not_count_as_valid():
    m=module().score([row(1,'a','illegal',status='valid')],['a'])
    assert m['valid_rate']==0 and m['status_counts']['invalid_label']==1


def test_paired_bootstrap_is_by_group_and_preserves_pairing():
    a=[row(i,'a','a','one-query') for i in range(200)]
    b=[row(i,'a',None,'one-query') for i in range(200)]
    m=module().paired_bootstrap(a,b,['a'],iterations=100,seed=7)
    assert m['independent_groups']==1
    assert m['accuracy_difference']['estimate']==1
    assert m['accuracy_difference']['ci95']==[1,1]
    assert m['macro_f1_difference']['ci95']==[1,1]
    assert m['insufficient_independent_groups']


def test_paired_comparison_rejects_different_records_or_gold():
    with pytest.raises(ValueError):
        module().paired_bootstrap([row(1,'a','a')],[row(2,'a','a')],['a'],iterations=10)
    with pytest.raises(ValueError):
        module().paired_bootstrap([row(1,'a','a')],[row(1,'b','b')],['a','b'],iterations=10)


def test_mcnemar_uses_only_independent_items_and_exact_tail():
    a=[row(i,'a','a') for i in range(6)]
    b=[row(i,'a',None) for i in range(6)]
    result=module().mcnemar_exact(a,b,['a'])
    assert result['a_only_correct']==6 and result['b_only_correct']==0
    assert result['p_value']==pytest.approx(2/64)
    with pytest.raises(ValueError):
        module().mcnemar_exact([row(1,'a','a','same'),row(2,'a','a','same')],
                                [row(1,'a','a','same'),row(2,'a','a','same')],['a'])


def test_holm_is_order_preserving_and_monotone_after_sort():
    adjusted=module().holm_adjust([.04,.01,.03])
    assert adjusted==pytest.approx([.06,.03,.06])


def test_cluster_permutation_does_not_inflate_n_from_passages():
    a=[row(i,'a','a','query') for i in range(100)]
    b=[row(i,'a',None,'query') for i in range(100)]
    result=module().paired_cluster_test(a,b,['a'],iterations=100)
    assert result['independent_groups']==1 and result['p_value']==1


def test_different_repeat_keys_must_be_filtered_before_pairing():
    rows=[row(1,'a','a'),row(1,'a','a')]
    with pytest.raises(ValueError): module().paired_bootstrap(rows,rows,['a'],iterations=10)


def test_stratified_bootstrap_preserves_sample_allocation_and_reports_rare_classes():
    labels=[str(i) for i in range(60)]
    rows=[row(i,str(i),str(i)) for i in range(60)]
    result=module().paired_bootstrap(rows,rows,labels,iterations=200,stratify='gold_label')
    assert result['a']['macro_f1']['ci95']==[1,1]
    assert len(result['rare_strata'])==60
    assert result['stratify']=='gold_label'


def test_stratification_rejects_query_with_multiple_gold_classes():
    rows=[row(1,'a','a','query'),row(2,'b','b','query')]
    with pytest.raises(ValueError): module().paired_bootstrap(rows,rows,['a','b'],iterations=10,stratify='gold_label')


def test_query_mean_and_passage_weighted_accuracy_are_distinguished():
    rows=[row(1,'a','a','small')]+[row(i,'a',None,'large') for i in range(2,5)]
    result=module().score(rows,['a'])
    assert result['accuracy']==.25
    assert result['mean_group_accuracy']==.5


def test_repeat_stability_does_not_treat_repeats_as_independent_examples():
    rows=[]
    for repeat,predictions in [(0,['a','a']),(1,['a',None]),(2,['a','b'])]:
        rows.extend({**row(i,'a',p),'repeat':repeat,'model':'fixture'} for i,p in enumerate(predictions))
    result=module().repeat_stability(rows,['a','b'])
    assert result['independent_groups']==2 and result['unique_items']==2 and result['rounds']==3
    assert result['all_rounds_correct_rate']==.5
    assert result['per_round'][1]['accuracy']==.5


def test_missing_repeat_item_prevents_stability_claim():
    rows=[{**row(1,'a','a'),'repeat':0},{**row(2,'a','a'),'repeat':0},{**row(1,'a','a'),'repeat':1}]
    with pytest.raises(ValueError):module().repeat_stability(rows,['a'])
