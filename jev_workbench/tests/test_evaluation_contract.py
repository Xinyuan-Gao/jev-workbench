import json

from backend.adapters import GeminiAdapter, JEVAdapter
from backend.experiments import get_experiment
import backend.experiments as experiments


def _response():
    return {"choices": [{"message": {"content": json.dumps({"prediction": "relevant", "confidence": 0.7})}}]}


def test_gemini_prediction_payload_does_not_include_gold_label():
    adapter = GeminiAdapter(api_url="https://example.test/v1", api_key="test", api_path="/chat/completions")
    seen = {}
    adapter._request = lambda payload: (seen.update(payload=payload) or _response())
    item = {"id": "rag-001", "query": "如何退款", "passage": "订单页提交退款", "label": "relevant"}
    adapter.classify(item, {"labels": ["relevant", "irrelevant"], "task": "relevance"})
    content = seen["payload"]["messages"][1]["content"]
    assert '"label"' not in content
    assert "如何退款" in content
    assert "订单页提交退款" in content


def test_jev_rag_payload_contains_query_and_passage_and_excludes_gold_label():
    adapter = JEVAdapter(api_url="https://example.test/v1", api_key="test", api_path="/chat/completions")
    seen = {}
    adapter._request = lambda payload: (seen.update(payload=payload) or _response())
    item = {"id": "rag-001", "query": "如何退款", "passage": "订单页提交退款", "label": "relevant"}
    adapter.classify(item, {"labels": ["relevant", "irrelevant"], "task": "relevance"})
    content = seen["payload"]["messages"][0]["content"]
    assert "如何退款" in content
    assert "订单页提交退款" in content
    assert '"label"' not in content


def test_group_split_has_disjoint_groups_and_all_labels():
    experiment = get_experiment("intent-classification")
    splits = experiments.split_records(experiment["records"], seed=42)
    assert set(splits) == {"train", "validation", "test"}
    groups = {name: {record["group_id"] for record in records} for name, records in splits.items()}
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
    labels = set(experiment["labels"])
    assert all({record["label"] for record in records} == labels for records in splits.values())


def test_runner_scores_only_group_held_out_test_split():
    from backend.runner import Run
    experiment = get_experiment("intent-classification")
    run = Run("test-run", experiment, "simulation", adapter=None)
    run.start()
    run.thread.join(timeout=5)
    assert run.status == "completed"
    assert run.state()["total"] == len(experiments.split_records(experiment["records"])["test"])
    assert run.state()["dataset_total"] == len(experiment["records"])


def test_classification_macro_f1_is_not_accuracy_for_asymmetric_errors():
    import backend.evaluation as evaluation
    records = [
        {'status':'completed','gold_label':'a','prediction':'a','latency_ms':1},
        {'status':'completed','gold_label':'a','prediction':'a','latency_ms':2},
        {'status':'completed','gold_label':'a','prediction':'a','latency_ms':3},
        {'status':'completed','gold_label':'b','prediction':'a','latency_ms':4},
    ]
    m = evaluation.classification_metrics(records, ['a','b'])
    assert m['accuracy_completed'] == .75
    assert abs(m['macro_f1_completed'] - 3/7) < 1e-10


def test_prediction_request_uses_input_field_whitelist():
    adapter = GeminiAdapter(api_url='https://example.test/v1', api_key='test')
    seen = {}
    adapter._request = lambda payload: (seen.update(payload=payload) or _response())
    adapter.classify({'query':'问题', 'passage':'片段', 'gold_label':'ANSWER_SENTINEL',
                      'annotation_note':'ANSWER_SENTINEL'},
                     {'labels':['relevant','irrelevant'], 'task':'relevance'})
    assert 'ANSWER_SENTINEL' not in json.dumps(seen,ensure_ascii=False)


def test_invalid_remote_prediction_is_rejected():
    import pytest
    adapter = GeminiAdapter(api_url='https://example.test/v1', api_key='test')
    adapter._request = lambda payload: {'choices':[{'message':{'content':json.dumps({'prediction':'unknown','confidence':.9})}}]}
    with pytest.raises(ValueError, match='允许标签'):
        adapter.classify({'text':'文本'}, {'labels':['a','b'],'task':'classification'})


def test_metrics_distinguish_valid_output_accuracy_from_all_attempts():
    from backend.evaluation import classification_metrics
    rows = [
        {'status':'completed','prediction':'a','gold_label':'a','latency_ms':1},
        {'status':'completed','prediction':'b','gold_label':'b','latency_ms':2},
        {'status':'failed','prediction':None,'gold_label':'a','latency_ms':3},
    ]
    metrics = classification_metrics(rows, ['a','b'])
    assert metrics['accuracy_completed'] == 1
    assert metrics['accuracy_attempted'] == 2/3
    assert metrics['valid_output_rate'] == 2/3
