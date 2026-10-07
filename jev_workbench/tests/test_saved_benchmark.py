import json
from pathlib import Path
from backend.runner import ExperimentRunner
import backend.runner as runner_module


def _saved():
    return {
        'run_id': 'corrected-test', 'experiment': 'intent-classification',
        'experiment_name': '中文意图分类', 'mode': 'gemini', 'status': 'completed',
        'dataset_total': 216, 'train_count': 126, 'validation_count': 36,
        'test_count': 1, 'split_group_counts': {'train':21,'validation':6,'test':9},
        'metrics': {'completed':1,'failed':0},
        'results': [{'id':'intent-001','input':{'text':'测试'},'gold_label':'technical',
                     'prediction':'technical','status':'completed','confidence':.9,'latency_ms':2}],
    }


def test_saved_completed_runs_survive_runner_restart(tmp_path):
    (tmp_path/'corrected-test.json').write_text(json.dumps(_saved(),ensure_ascii=False))
    restored = ExperimentRunner(history_dir=tmp_path)
    assert restored.summary()['runs'][0]['run_id'] == 'corrected-test'
    assert restored.get('corrected-test').state()['total'] == 1
    assert restored.get('corrected-test').results[0]['input']['label'] == 'technical'
    assert ExperimentRunner(history_dir=tmp_path).get('corrected-test').status == 'completed'


def test_benchmark_view_reads_only_run_files(tmp_path):
    (tmp_path/'corrected-test.json').write_text(json.dumps(_saved(),ensure_ascii=False))
    (tmp_path/'manifest.json').write_text('{}')
    data = runner_module.benchmark_summary(tmp_path)
    assert data['expected_runs'] == 15
    assert data['completed_runs'] == 1
    assert len(data['runs']) == 1
