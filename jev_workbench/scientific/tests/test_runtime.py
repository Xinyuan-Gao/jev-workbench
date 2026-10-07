"""Offline contract tests for durable scientific experiment execution."""
import importlib
import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def runtime():
    try:
        journal = importlib.import_module('scientific.journal')
        remote = importlib.import_module('scientific.remote')
    except ImportError as exc:
        pytest.fail(f'Scientific runtime is not implemented: {exc}')
    return journal, remote


SCHEMA = {'task': 'classification', 'labels': ['yes', 'no'],
          'definitions': {'yes': '符合要求', 'no': '不符合要求'},
          'instructions': '判断文本是否满足给定条件。'}


def jobs(count=3):
    return [{'job_id': str(i), 'dataset_id': 'd', 'item_id': str(i),
             'group_id': 'g', 'model': 'm', 'repeat': 0,
             'input': {'text': f'文本 {i}'}, 'gold_label': 'yes'} for i in range(count)]


def test_refusal_and_exception_continue_fixed_denominator_and_resume(runtime, tmp_path):
    journal_mod, remote = runtime
    planned = jobs()
    journal = journal_mod.DurableJournal(tmp_path / 'audit.sqlite', planned, 'hash')
    called = []
    def predict(job):
        called.append(job['job_id'])
        if job['job_id'] == '0':
            return remote.parse_response('jev', {'refusal': 'cannot comply'}, SCHEMA['labels'])
        if job['job_id'] == '1':
            raise RuntimeError('broken sample')
        return {'status': 'valid', 'prediction': 'yes', 'confidence': None,
                'confidence_status': 'missing'}
    summary = journal_mod.execute_jobs(journal, predict, order_seed=1)
    assert summary == {'planned': 3, 'terminal': 3, 'valid': 1, 'failed': 2,
                       'pending': 0, 'complete': True}
    assert len(journal.records()) == 3
    assert all(row['attempts'][0]['phase'] == 'started' for row in journal.records())
    resumed = journal_mod.DurableJournal(tmp_path / 'audit.sqlite', planned, 'hash')
    assert journal_mod.execute_jobs(resumed, predict, order_seed=7) == summary
    assert len(called) == 3
    resumed.record_terminal('0', {'status': 'valid', 'prediction': 'yes'})
    assert resumed.records()[0]['outcome']['status'] == 'refusal'


def test_workers_execute_each_pending_job_once_and_keep_manifest_order(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'parallel.sqlite', jobs(24), 'hash')
    calls = []
    lock = threading.Lock()

    def predict(job):
        time.sleep(.002)
        with lock:
            calls.append(job['job_id'])
        return {'status': 'valid', 'prediction': 'yes'}

    summary = journal_mod.execute_jobs(journal, predict, order_seed=3, workers=4)
    assert summary['complete'] and summary['terminal'] == 24
    assert len(calls) == len(set(calls)) == 24
    assert {row['job_id'] for row in journal.records()} == {str(i) for i in range(24)}


def test_interrupt_leaves_all_planned_and_resume_skips_terminal(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'audit.sqlite', jobs(), 'hash')
    calls = []
    def predict(job):
        calls.append(job['job_id'])
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return {'status': 'valid', 'prediction': 'yes'}
    with pytest.raises(KeyboardInterrupt):
        journal_mod.execute_jobs(journal, predict)
    assert journal.summary()['planned'] == 3
    assert journal.summary()['terminal'] == 1
    assert journal.summary()['pending'] == 2
    assert len(journal.records()) == 3
    previous = set(calls[:1])
    resumed = journal_mod.DurableJournal(tmp_path / 'audit.sqlite', jobs(), 'hash')
    resumed_calls = []
    journal_mod.execute_jobs(resumed, lambda job: resumed_calls.append(job['job_id']) or
                             {'status': 'valid', 'prediction': 'no'})
    assert not previous.intersection(resumed_calls)
    assert resumed.summary()['complete']


def test_manifest_mismatch_and_duplicate_rejected(runtime, tmp_path):
    journal_mod, _ = runtime
    path = tmp_path / 'audit.sqlite'
    journal_mod.DurableJournal(path, jobs(), 'hash')
    with pytest.raises(ValueError):
        journal_mod.DurableJournal(path, jobs(), 'different')
    changed = jobs(); changed[0]['gold_label'] = 'no'
    with pytest.raises(ValueError):
        journal_mod.DurableJournal(path, changed, 'hash')
    with pytest.raises(ValueError):
        journal_mod.DurableJournal(tmp_path / 'dup.sqlite', [jobs()[0], jobs()[0]], 'hash')


def test_payload_whitelist_same_definitions_and_relevance_required(runtime):
    _, remote = runtime
    item = {'text': 'safe', 'gold': 'PRIVATE_GOLD', 'label': 'PRIVATE_LABEL',
            'group': 'PRIVATE_GROUP', 'metadata': {'secret': 'PRIVATE_META'},
            'task': 'PRIVATE_TASK', 'query': 'PRIVATE_QUERY'}
    jev = remote.build_payload('jev', 'jev-latest', item, SCHEMA)
    gemini = remote.build_payload('gemini', 'gemini-test', item, SCHEMA)
    for payload in [jev, gemini]:
        assert 'PRIVATE_' not in json.dumps(payload)
        assert '判断文本' in json.dumps(payload, ensure_ascii=False)
        assert '符合要求' in json.dumps(payload, ensure_ascii=False)
    envelope = json.loads(jev['messages'][0]['content'])
    assert envelope['state'] == 'safe'
    assert envelope['questions']['classification']['criteria'] == SCHEMA['definitions']
    assert 'temperature' not in jev
    assert gemini['temperature'] == 0 and gemini['max_tokens'] == 256
    with pytest.raises(ValueError):
        remote.build_payload('jev', 'm', {'query': 'q'}, {**SCHEMA, 'task': 'relevance'})


@pytest.mark.parametrize('confidence,status', [(None, 'missing'), ('bad', 'invalid'),
                                             (float('nan'), 'invalid'), (1.1, 'invalid'),
                                             (True, 'invalid'), (0.4, 'valid')])
def test_confidence_does_not_erase_legal_prediction(runtime, confidence, status):
    _, remote = runtime
    response = {'answers': {'classification': {'choice': 'yes', 'confidence': confidence}}}
    outcome = remote.parse_response('jev', response, SCHEMA['labels'])
    assert outcome['status'] == 'valid' and outcome['prediction'] == 'yes'
    assert outcome['confidence_status'] == status
    if status != 'valid':
        assert outcome['confidence'] is None


@pytest.mark.parametrize('response,status', [
    ({'choices': [{'message': {'content': 'I cannot assist with that request.'}}]}, 'refusal'),
    ({'choices': [{'message': {'content': '{}'}, 'finish_reason': 'content_filter'}]}, 'refusal'),
    ({'choices': [{'message': {'content': '{"prediction":"yes"'}, 'finish_reason': 'length'}]}, 'truncated_output'),
    ({'answers': {'classification': {'choice': 'unknown'}}}, 'invalid_label'),
    ({'choices': [{'message': {'content': 'unstructured text'}}]}, 'malformed_response')])
def test_failure_categories_keep_output(runtime, response, status):
    _, remote = runtime
    outcome = remote.parse_response('jev', response, SCHEMA['labels'])
    assert outcome['status'] == status and outcome['prediction'] is None
    assert outcome['raw_output'] is not None


def test_gateway_named_answer_and_openai_json(runtime):
    _, remote = runtime
    for content in [{'intent': {'type': 'choice', 'choice': 'no'}}, {'prediction': 'no'}]:
        response = {'choices': [{'message': {'content': json.dumps(content)}}]}
        assert remote.parse_response('jev', response, SCHEMA['labels'])['prediction'] == 'no'
    assert remote.parse_response('gemini', {'choices': [{'message': {'content': '{"prediction":"yes"}'}}]}, ['yes'])['status'] == 'valid'


@contextmanager
def local_gateway(replies):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.rfile.read(int(self.headers['Content-Length'])).decode()))
            status, headers, body = replies[min(len(requests) - 1, len(replies) - 1)]
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body.encode())
        def log_message(self, *_):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown(); worker.join(); server.server_close()


def test_retry_raw_body_redaction_callback_and_unknown_cost(runtime):
    _, remote = runtime
    secret = 'SUPER_SECRET_CREDENTIAL'
    success = json.dumps({'model': 'served-model', 'choices': [{'message': {'content': '{"prediction":"yes"}'}}],
                          'nested': {'api_key': secret, 'token': 'OTHER_SECRET'}, 'echo': secret})
    replies = [(429, {'Retry-After': '9999', 'Authorization': secret}, 'retry raw ' + secret),
               (200, {'x-request-id': 'req1', 'Set-Cookie': secret}, success)]
    events = []; waits = []
    with local_gateway(replies) as (url, requests):
        client = remote.AuditedHTTPClient(url, secret, 'gemini', sleep_fn=waits.append)
        payload = remote.build_payload('gemini', 'm', {'text': 'safe'}, SCHEMA)
        result = client.send(payload, on_attempt=events.append)
    assert len(requests) == 2 and requests[0][0] == '/v1/chat/completions'
    assert requests[0][1] == requests[1][1]
    assert len(result['attempts']) == 2
    assert result['attempts'][0]['raw_body'].startswith('retry raw ')
    assert result['attempts'][0]['retry_reason'] == 'http_429'
    assert waits and 0 <= waits[0] <= 30
    assert [event['phase'] for event in events] == ['started', 'finished', 'started', 'finished']
    serialized = json.dumps(result) + json.dumps(events)
    assert secret not in serialized and 'OTHER_SECRET' not in serialized
    assert 'Set-Cookie' not in serialized and 'Authorization' not in serialized
    assert result['metadata']['cost_status'] == 'unknown'
    assert result['attempts'][1]['response_model'] == 'served-model'
    assert result['outcome']['status'] == 'valid'


@pytest.mark.parametrize('status,body,expected', [
    (400, 'bad request', 'http_error'), (200, 'not json', 'malformed_response'),
    (200, '{"refusal":"no"}', 'refusal')])
def test_nonretryable_and_parse_failures_do_not_retry(runtime, status, body, expected):
    _, remote = runtime
    with local_gateway([(status, {}, body)]) as (url, requests):
        result = remote.AuditedHTTPClient(url, 'secret', 'jev', sleep_fn=lambda _: None).send(
            remote.build_payload('jev', 'm', {'text': 'safe'}, SCHEMA))
    assert len(requests) == 1
    assert result['outcome']['status'] == expected


def test_network_timeout_retries_and_preserves_attempts(runtime, monkeypatch):
    _, remote = runtime
    def timeout(*args, **kwargs):
        raise socket.timeout('timeout secret')
    monkeypatch.setattr(remote.urllib.request, 'urlopen', timeout)
    result = remote.AuditedHTTPClient('http://unused', 'secret', 'jev', sleep_fn=lambda _: None).send(
        remote.build_payload('jev', 'm', {'text': 'safe'}, SCHEMA))
    assert len(result['attempts']) == 3
    assert result['outcome']['status'] == 'network_error'
    assert 'secret' not in json.dumps(result)


def test_execute_preserves_invalid_confidence_status(runtime, tmp_path):
    journal_mod, remote = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'confidence.sqlite', jobs(1), 'hash')
    journal_mod.execute_jobs(journal, lambda _: remote.parse_response('jev',
        {'answers': {'classification': {'choice': 'yes', 'confidence': 'broken'}}}, ['yes']))
    result = journal.records()[0]['outcome']
    assert result['status'] == 'valid'
    assert result['confidence_status'] == 'invalid'


def test_named_answer_without_type(runtime):
    _, remote = runtime
    response = {'choices': [{'message': {'content': '{"intent":{"choice":"yes"}}'}}]}
    assert remote.parse_response('jev', response, ['yes'])['prediction'] == 'yes'


def test_json_string_credentials_and_plaintext_header_redaction(runtime):
    _, remote = runtime
    content = json.dumps({'prediction': 'yes', 'token': 'INNER_SECRET'})
    response = json.dumps({'choices': [{'message': {'content': content}}]})
    with local_gateway([(200, {}, response)]) as (url, _):
        result = remote.AuditedHTTPClient(url, 'api-secret', 'gemini').send(
            remote.build_payload('gemini', 'm', {'text': 'safe'}, SCHEMA))
    assert 'INNER_SECRET' not in json.dumps(result)
    with local_gateway([(400, {}, 'Authorization: Basic EXTRA_SECRET\nX-Api-Key: EXTRA_KEY')]) as (url, _):
        result = remote.AuditedHTTPClient(url, 'api-secret', 'gemini').send(
            remote.build_payload('gemini', 'm', {'text': 'safe'}, SCHEMA))
    assert 'EXTRA_SECRET' not in json.dumps(result)
    assert 'EXTRA_KEY' not in json.dumps(result)


def test_nonserializable_outcome_is_terminal_sample_failure(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'serialize.sqlite', jobs(2), 'hash')
    journal_mod.execute_jobs(journal, lambda job: {'status': 'valid', 'prediction': 'yes',
        'raw_output': object()} if job['job_id'] == '0' else {'status': 'valid', 'prediction': 'yes'})
    assert journal.summary()['terminal'] == 2
    assert journal.summary()['valid'] == 1
    assert journal.summary()['failed'] == 1


def test_idempotent_terminal_ignores_unserializable_replacement(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'idempotent.sqlite', jobs(1), 'hash')
    journal.record_terminal('0', {'status': 'refusal', 'prediction': None})
    journal.record_terminal('0', {'status': 'valid', 'prediction': 'yes', 'raw_output': object()})
    assert journal.records()[0]['outcome']['status'] == 'refusal'


def test_incomplete_network_body_retries(runtime, monkeypatch):
    import http.client
    _, remote = runtime
    def incomplete(*args, **kwargs):
        raise http.client.IncompleteRead(b'partial', 10)
    monkeypatch.setattr(remote.urllib.request, 'urlopen', incomplete)
    result = remote.AuditedHTTPClient('http://unused', 'secret', 'jev', sleep_fn=lambda _: None).send(
        remote.build_payload('jev', 'm', {'text': 'safe'}, SCHEMA))
    assert len(result['attempts']) == 3
    assert result['outcome']['status'] == 'network_error'


def test_package_exports_public_runtime(runtime):
    import scientific
    assert scientific.DurableJournal is runtime[0].DurableJournal
    assert scientific.execute_jobs is runtime[0].execute_jobs
    assert scientific.AuditedHTTPClient is runtime[1].AuditedHTTPClient
    assert scientific.build_payload is runtime[1].build_payload
    assert scientific.parse_response is runtime[1].parse_response


@pytest.mark.parametrize('text', ['This content has been blocked for safety reasons.',
                                  'I must decline this request.'])
def test_explicit_filter_and_refusal_text(runtime, text):
    _, remote = runtime
    outcome = remote.parse_response('gemini', {'choices': [{'message': {'content': text}}]}, ['yes'])
    assert outcome['status'] == 'refusal'
    assert outcome['prediction'] is None


def test_unexpected_parser_error_returns_audited_failure(runtime, monkeypatch):
    _, remote = runtime
    def broken_parser(*args):
        raise RuntimeError('parse failed api-secret')
    monkeypatch.setattr(remote, 'parse_response', broken_parser)
    with local_gateway([(200, {}, '{"prediction":"yes"}')]) as (url, requests):
        result = remote.AuditedHTTPClient(url, 'api-secret', 'gemini').send(
            remote.build_payload('gemini', 'm', {'text': 'safe'}, SCHEMA))
    assert len(requests) == 1 and len(result['attempts']) == 1
    assert result['outcome']['status'] == 'malformed_response'
    assert 'api-secret' not in json.dumps(result)


def test_legal_class_is_not_refusal_because_explanation_quotes_refusal(runtime):
    _, remote = runtime
    content = json.dumps({'prediction': 'yes', 'reason': 'The sample says I cannot assist with that request.'})
    outcome = remote.parse_response('gemini', {'choices': [{'message': {'content': content}}]}, ['yes'])
    assert outcome['status'] == 'valid'
    assert outcome['prediction'] == 'yes'


def test_predictor_exception_redacts_credentials(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'secret.sqlite', jobs(1), 'hash')
    def predict(_):
        raise RuntimeError('Authorization Bearer FAKE_SECRET_KEY')
    journal_mod.execute_jobs(journal, predict)
    assert 'FAKE_SECRET_KEY' not in json.dumps(journal.records())


def test_overflow_confidence_does_not_erase_prediction(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'overflow.sqlite', jobs(1), 'hash')
    journal_mod.execute_jobs(journal, lambda _: {'status': 'valid', 'prediction': 'yes', 'confidence': 10 ** 400})
    outcome = journal.records()[0]['outcome']
    assert outcome['status'] == 'valid' and outcome['confidence_status'] == 'invalid'


def test_real_gemini_filter_wording(runtime):
    _, remote = runtime
    outcome = remote.parse_response('gemini', {'choices': [{'message': {
        'content': "This request was blocked by Gemini's filters due to the content of the prompt."}}]}, ['yes'])
    assert outcome['status'] == 'refusal'


def test_interrupted_http_requires_explicit_duplicate_retry(runtime, tmp_path):
    journal_mod, _ = runtime
    path = tmp_path / 'interrupted.sqlite'
    journal = journal_mod.DurableJournal(path, jobs(1), 'hash')
    journal.append_attempt('0', {'attempt_id': 'unfinished-http', 'phase': 'started', 'request_body': '{}'})
    resumed = journal_mod.DurableJournal(path, jobs(1), 'hash')
    calls = []
    predict = lambda job: calls.append(job['job_id']) or {'status': 'valid', 'prediction': 'yes'}
    summary = journal_mod.execute_jobs(resumed, predict)
    assert calls == [] and summary['pending'] == 1
    record = resumed.records()[0]
    assert record['potential_duplicate']
    assert any(event['phase'] == 'indeterminate' for event in record['attempts'])
    journal_mod.execute_jobs(resumed, predict, allow_indeterminate_retry=True)
    assert calls == ['0']
    assert any(event.get('kind') == 'duplicate_retry_authorization' for event in resumed.records()[0]['attempts'])


def test_finished_http_recovers_terminal_without_another_charge(runtime, tmp_path):
    journal_mod, remote = runtime
    path = tmp_path / 'finished.sqlite'
    journal = journal_mod.DurableJournal(path, jobs(1), 'hash')
    def persist_then_interrupt(event):
        journal.append_attempt('0', event)
        if event['phase'] == 'finished':
            raise KeyboardInterrupt()
    with local_gateway([(200, {}, '{"prediction":"yes"}')]) as (url, requests):
        client = remote.AuditedHTTPClient(url, 'secret', 'gemini')
        with pytest.raises(KeyboardInterrupt):
            client.send(remote.build_payload('gemini', 'm', {'text': 'safe'}, SCHEMA),
                        on_attempt=persist_then_interrupt)
        resumed = journal_mod.DurableJournal(path, jobs(1), 'hash')
        journal_mod.execute_jobs(resumed, lambda _: pytest.fail('Must not call recovered HTTP job'))
    assert len(requests) == 1
    assert resumed.records()[0]['outcome']['prediction'] == 'yes'
    assert resumed.summary()['terminal'] == 1


def test_started_http_has_endpoint_provider_and_requested_model(runtime):
    _, remote = runtime
    events = []
    with local_gateway([(200, {}, '{"prediction":"yes"}')]) as (url, _):
        remote.AuditedHTTPClient(url, 'secret', 'gemini').send(
            remote.build_payload('gemini', 'm', {'text': 'safe'}, SCHEMA), on_attempt=events.append)
    assert events[0]['provider'] == 'gemini'
    assert events[0]['endpoint'].endswith('/v1/chat/completions')
    assert events[0]['request_model'] == 'm'


def test_job_http_execution_id_is_linked(runtime, tmp_path):
    journal_mod, remote = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'linked.sqlite', jobs(1), 'hash')
    with local_gateway([(200, {}, '{"prediction":"yes"}')]) as (url, _):
        client = remote.AuditedHTTPClient(url, 'secret', 'gemini')
        journal_mod.execute_jobs(journal, lambda job: client.send(
            remote.build_payload('gemini', 'm', job['input'], SCHEMA),
            on_attempt=lambda event: journal.append_attempt(job['job_id'], event)))
    events = journal.records()[0]['attempts']
    assert events[0]['execution_id'] == events[1]['execution_id'] == events[2]['execution_id']


def test_injected_plain_secret_scrubbed_at_sqlite_boundary(runtime, tmp_path):
    journal_mod, _ = runtime
    path = tmp_path / 'plain-secret.sqlite'
    journal = journal_mod.DurableJournal(path, jobs(1), 'hash', secrets=['RAW_FAKE_KEY'])
    journal.append_attempt('0', {'phase': 'custom', 'nested': {'error': 'failure RAW_FAKE_KEY'}})
    journal_mod.execute_jobs(journal, lambda _: (_ for _ in ()).throw(RuntimeError('RAW_FAKE_KEY')))
    assert b'RAW_FAKE_KEY' not in path.read_bytes()


def test_running_cannot_count_as_terminal(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'running.sqlite', jobs(1), 'hash')
    with pytest.raises(ValueError):
        journal.record_terminal('0', {'status': 'running'})
    assert journal.summary()['pending'] == 1


def test_same_journal_object_does_not_bypass_indeterminate_guard(runtime, tmp_path):
    journal_mod, _ = runtime
    journal = journal_mod.DurableJournal(tmp_path / 'same-object.sqlite', jobs(1), 'hash')
    journal.append_attempt('0', {'attempt_id':'inflight-http','phase':'started','request_body':'{}'})
    calls=[]
    summary=journal_mod.execute_jobs(journal,lambda job:calls.append(job['job_id']) or {'status':'valid','prediction':'yes'})
    assert calls==[] and summary['pending']==1
    assert journal.records()[0]['potential_duplicate']


def test_same_journal_object_recovers_durable_finished_outcome(runtime, tmp_path):
    journal_mod, remote = runtime
    journal=journal_mod.DurableJournal(tmp_path/'same-finished.sqlite',jobs(1),'hash')
    def persist_then_interrupt(event):
        journal.append_attempt('0',event)
        if event['phase']=='finished':
            raise KeyboardInterrupt()
    with local_gateway([(200,{},'{"prediction":"yes"}')]) as (url,requests):
        with pytest.raises(KeyboardInterrupt):
            journal_mod.execute_jobs(journal,lambda _:remote.AuditedHTTPClient(url,'secret','gemini').send(
                remote.build_payload('gemini','m',{'text':'safe'},SCHEMA),on_attempt=persist_then_interrupt))
        journal_mod.execute_jobs(journal,lambda _:pytest.fail('Must recover without another HTTP call'))
    assert len(requests)==1 and journal.summary()['valid']==1


@pytest.mark.parametrize('phase',['started','finished'])
def test_public_http_callback_failure_is_durability_failure_not_business_terminal(runtime,tmp_path,phase):
    journal_mod,remote=runtime
    journal=journal_mod.DurableJournal(tmp_path/'audit-failure.sqlite',jobs(1),'hash')
    def persist(event):
        if event['phase']==phase:
            raise OSError('temporary persistence fault PRIVATE_KEY')
        journal.append_attempt('0',event)
    with local_gateway([(200,{},'{"prediction":"yes","usage":{"total_tokens":5}}')]) as (url,requests):
        with pytest.raises(BaseException) as failure:
            journal_mod.execute_jobs(journal,lambda _:remote.AuditedHTTPClient(url,'PRIVATE_KEY','gemini').send(
                remote.build_payload('gemini','m',{'text':'safe'},SCHEMA),on_attempt=persist))
    assert type(failure.value).__name__=='JournalPersistenceFailure'
    assert 'PRIVATE_KEY' not in str(failure.value)
    assert len(requests)==(0 if phase=='started' else 1)
    assert journal.summary()['terminal']==0 and journal.summary()['pending']==1
    journal.recover_pending()
    assert journal.records()[0]['potential_duplicate']==(phase=='finished')


def test_direct_journal_append_io_failure_is_baseexception(runtime,tmp_path,monkeypatch):
    journal_mod,_=runtime
    journal=journal_mod.DurableJournal(tmp_path/'sqlite-failure.sqlite',jobs(1),'hash')
    def unavailable(*args,**kwargs):raise OSError('PRIVATE_KEY')
    monkeypatch.setattr(journal,'_connect',unavailable)
    with pytest.raises(BaseException) as failure:journal.append_attempt('0',{'phase':'started'})
    assert type(failure.value).__name__=='JournalPersistenceFailure'
    assert not isinstance(failure.value,Exception)
    assert 'PRIVATE_KEY' not in str(failure.value)
