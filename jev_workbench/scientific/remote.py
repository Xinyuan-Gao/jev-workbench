"""Provider payloads, response classification, and credential-safe HTTP auditing.

Credentials and endpoint configuration are injected by the caller. This module
never reads environment files and performs no network request on import.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
import http.client
import json
import math
import re
import socket
import time
import urllib.error
import urllib.request
import uuid
from typing import Callable

_EXECUTION_ID = ContextVar('scientific_execution_id', default=None)


class JournalPersistenceFailure(BaseException):
    """Audit storage failure must abort execution, never become sample failure."""

    def __init__(self, phase=None):
        super().__init__('Scientific audit persistence failed; pending preserved')
        self.phase = phase


def _notify_attempt(callback, event):
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        # No exception text/cause is emitted: a storage exception can contain
        # credential values. BaseException controls (budget/interruption) pass.
        raise JournalPersistenceFailure(event.get('phase')) from None


_SECRET_KEY = re.compile(r'^(?:x[-_])?(?:authorization|proxy[-_]authorization|api[-_]?key|(?:access|refresh|auth|bearer|session)[-_]?token|token|password|secret|cookie|set[-_]cookie)$', re.I)
_REFUSAL = re.compile(r"(?:\b(?:i\s+)?(?:cannot|can't|unable to)\s+(?:assist|help|comply|provide|fulfill)|\bi\s+(?:must|have to)\s+decline\b|\brequest\s+(?:was|is|has been)\s+blocked\s+by\b|\b(?:content|safety)\s+(?:(?:has been|is|was)\s+)?(?:filter(?:ed)?|blocked)|拒绝(?:回答|处理)|无法(?:协助|提供|回答)|不能(?:协助|提供|回答)|内容(?:已被|被)?(?:过滤|拦截))", re.I)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secrets(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, part in value.items():
            if _SECRET_KEY.fullmatch(str(key)):
                if isinstance(part, (str, int, float)) and str(part):
                    found.add(str(part))
            found.update(_secrets(part))
    elif isinstance(value, list):
        for part in value:
            found.update(_secrets(part))
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, (dict, list)):
                found.update(_secrets(decoded))
        except (ValueError, TypeError):
            pass
    return found


def _safe(value, secrets=()):
    if isinstance(value, dict):
        return {key: '[REDACTED]' if _SECRET_KEY.fullmatch(str(key)) else _safe(part, secrets)
                for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(part, secrets) for part in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, (dict, list)):
                cleaned = _safe(decoded, set(secrets) | _secrets(decoded))
                if cleaned != decoded:
                    value = json.dumps(cleaned, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            pass
        for secret in sorted(set(secrets), key=len, reverse=True):
            if secret:
                value = value.replace(secret, '[REDACTED]')
        # Also catches credentials in echoed headers or plain-text error bodies.
        value = re.sub(r'(?i)\bBearer\s+[^\s"\',;<>]+', 'Bearer [REDACTED]', value)
        return re.sub(r'(?im)(\b(?:authorization|(?:x[-_])?api[-_]?key|(?:access[-_])?token)\s*:\s*)[^\r\n]+',
                      r'\1[REDACTED]', value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _safe_body(body: str, secrets) -> str:
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return _safe(body, secrets)
    cleaned = _safe(parsed, set(secrets) | _secrets(parsed))
    if cleaned == parsed:
        return _safe(body, secrets)
    return json.dumps(cleaned, ensure_ascii=False, allow_nan=False)


def build_payload(provider: str, model: str, item: dict, schema: dict) -> dict:
    if provider not in {'jev', 'gemini'}:
        raise ValueError('Unknown provider')
    task = schema.get('task')
    fields = {'classification': ('text',), 'routing': ('task',), 'relevance': ('query', 'passage')}.get(task)
    if fields is None:
        raise ValueError('Unknown task')
    labels = schema.get('labels')
    definitions = schema.get('definitions')
    if (not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x for x in labels)
            or len(set(labels)) != len(labels) or not isinstance(definitions, dict)
            or any(not isinstance(definitions.get(label), str) or not definitions[label].strip() for label in labels)
            or not isinstance(schema.get('instructions'), str) or not schema['instructions'].strip()):
        raise ValueError('Task schema requires labels, definitions, and instructions')
    if not isinstance(item, dict):
        raise ValueError('Item must be a dictionary')
    clean = {field: item[field] for field in fields if field in item}
    if any(not isinstance(clean.get(field), str) or not clean[field].strip() for field in fields):
        raise ValueError(f'{task} requires nonempty string fields: {fields}')
    criteria = {label: definitions[label] for label in labels}
    instructions = f"任务类型：{task}。\n{schema['instructions']}\n根据输入选择且仅选择一个允许标签。标签定义：" + json.dumps(criteria, ensure_ascii=False)
    if provider == 'jev':
        state = json.dumps(clean, ensure_ascii=False) if task == 'relevance' else clean[fields[0]]
        envelope = {'state': state, 'questions': {'classification': {
            'type': 'choice', 'instructions': instructions, 'criteria': criteria}}}
        return {'model': model, 'messages': [{'role': 'user', 'content': json.dumps(envelope, ensure_ascii=False)}]}
    system = {'role': 'system', 'content': instructions + '\n只输出 JSON 对象，包含 prediction（允许标签）与可选 confidence（0 到 1）。'}
    user = {'role': 'user', 'content': json.dumps({'input': clean, 'task': task,
             'instructions': instructions, 'labels': labels, 'definitions': criteria}, ensure_ascii=False)}
    return {'model': model, 'messages': [system, user], 'temperature': 0, 'max_tokens': 256}


def _failure(status: str, raw_output, **extra) -> dict:
    return {'status': status, 'prediction': None, 'confidence': None,
            'confidence_status': 'missing', 'raw_output': _safe(raw_output), **extra}


def _has_refusal(value) -> bool:
    if isinstance(value, dict):
        for key, part in value.items():
            if str(key).lower() in {'refusal', 'blocked', 'content_filter'} and part:
                return True
            if str(key).lower() in {'finish_reason', 'finishreason'} and str(part).lower() in {
                    'content_filter', 'safety', 'blocklist', 'prohibited_content', 'recitation'}:
                return True
            if _has_refusal(part):
                return True
    elif isinstance(value, list):
        return any(_has_refusal(part) for part in value)
    return False


def _answer(value):
    if not isinstance(value, dict):
        return None
    if any(key in value for key in ('prediction', 'choice', 'label')):
        return value
    if isinstance(value.get('answers'), dict):
        answers = value['answers']
        if isinstance(answers.get('classification'), dict):
            return _answer(answers['classification'])
        for answer in answers.values():
            found = _answer(answer)
            if found is not None:
                return found
    for answer in value.values():
        if isinstance(answer, dict) and (answer.get('type') == 'choice' or
                                        any(key in answer for key in ('choice', 'prediction', 'label'))):
            return _answer(answer)
    return None


def parse_response(provider: str, response: dict, labels: list[str]) -> dict:
    if provider not in {'jev', 'gemini'}:
        raise ValueError('Unknown provider')
    if not isinstance(response, dict):
        return _failure('malformed_response', response)
    choices = response.get('choices')
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get('message') if isinstance(choice.get('message'), dict) else {}
    raw = message.get('content', response.get('answers', response))
    if _has_refusal(response):
        return _failure('refusal', raw)
    if str(choice.get('finish_reason', '')).lower() in {'length', 'max_tokens'}:
        return _failure('truncated_output', raw)
    parsed = raw
    if isinstance(raw, str):
        fenced = re.fullmatch(r'\s*```(?:json)?\s*(.*?)\s*```\s*', raw, flags=re.S | re.I)
        try:
            parsed = json.loads(fenced.group(1) if fenced else raw)
        except (ValueError, TypeError):
            return _failure('refusal' if _REFUSAL.search(raw) else 'malformed_response', raw)
    if isinstance(parsed, str) and _REFUSAL.search(parsed):
        return _failure('refusal', raw)
    if _has_refusal(parsed):
        return _failure('refusal', raw)
    answer = _answer(response) if 'answers' in response else _answer(parsed)
    if answer is None:
        return _failure('malformed_response', raw)
    prediction = answer.get('choice', answer.get('prediction', answer.get('label')))
    if not isinstance(prediction, str) or prediction not in labels:
        return _failure('refusal' if isinstance(prediction, str) and _REFUSAL.search(prediction)
                        else 'invalid_label', raw)
    confidence = answer.get('confidence')
    confidence_status = 'missing' if confidence is None else 'invalid'
    normalized = None
    if confidence is not None and not isinstance(confidence, bool):
        try:
            number = float(confidence)
            if math.isfinite(number) and 0 <= number <= 1:
                confidence_status, normalized = 'valid', number
        except (ValueError, TypeError, OverflowError):
            pass
    return {'status': 'valid', 'prediction': prediction, 'confidence': normalized,
            'confidence_status': confidence_status, 'raw_output': _safe(raw)}


class AuditedHTTPClient:
    SAFE_HEADERS = frozenset({'content-type', 'x-request-id', 'request-id', 'openai-request-id',
                              'retry-after', 'x-ratelimit-remaining-requests', 'x-ratelimit-reset-requests'})

    def __init__(self, base_url: str, api_key: str, provider: str, timeout: float = 30,
                 max_retries: int = 2, backoff_seconds: float = 1,
                 sleep_fn: Callable[[float], None] = time.sleep):
        if provider not in {'jev', 'gemini'}:
            raise ValueError('Unknown provider')
        if not base_url or not api_key:
            raise ValueError('Endpoint and API key must be injected')
        self.base_url = base_url.rstrip('/')
        self.api_key, self.provider, self.timeout = api_key, provider, timeout
        self.max_retries = min(2, max(0, int(max_retries)))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.sleep_fn = sleep_fn

    def _wait(self, retry_index: int, headers) -> float:
        delay = self.backoff_seconds * (2 ** retry_index)
        value = headers.get('Retry-After') if headers else None
        if value:
            try:
                delay = float(value)
            except (ValueError, TypeError):
                try:
                    when = parsedate_to_datetime(value)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    delay = when.timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    pass
        return min(30.0, max(0.0, delay)) if math.isfinite(delay) else 30.0

    def _labels(self, payload: dict) -> list[str]:
        try:
            user = json.loads(next(message['content'] for message in payload['messages'] if message['role'] == 'user'))
            if self.provider == 'jev':
                return list(user['questions']['classification']['criteria'])
            return list(user['labels'])
        except (ValueError, TypeError, KeyError, StopIteration):
            return []

    def send(self, payload: dict, *, labels: list[str] | None = None,
             on_attempt: Callable[[dict], None] | None = None, execution_id: str | None = None) -> dict:
        execution_id = execution_id or _EXECUTION_ID.get() or str(uuid.uuid4())
        labels = self._labels(payload) if labels is None else labels
        endpoint = self.base_url if self.base_url.endswith('/chat/completions') else self.base_url + '/chat/completions'
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        request = urllib.request.Request(endpoint, data=body, method='POST',
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'})
        secrets = {self.api_key} | _secrets(payload)
        attempts = []
        total_start = time.perf_counter()
        response_json = None
        outcome = None
        for index in range(self.max_retries + 1):
            started = {'attempt_id': str(uuid.uuid4()), 'attempt': index + 1, 'phase': 'started',
                       'execution_id': execution_id,
                       'provider': self.provider, 'endpoint': endpoint, 'request_model': payload.get('model'),
                       'timeout_seconds': self.timeout,
                       'started_at': _utc(), 'request_body': _safe_body(body.decode(), secrets),
                       'request_headers': {'content-type': 'application/json'}}
            _notify_attempt(on_attempt, _safe(started, secrets))
            attempt_start = time.perf_counter()
            status = None; raw_body = ''; headers = {}; error = None; network_error = False
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status = response.status
                    headers = dict(response.headers.items())
                    raw_body = response.read().decode('utf-8', errors='replace')
            except urllib.error.HTTPError as exc:
                status = exc.code
                headers = dict(exc.headers.items()) if exc.headers else {}
                try:
                    raw_body = exc.read().decode('utf-8', errors='replace')
                except Exception as read_error:
                    error = f'{type(read_error).__name__}: {read_error}'
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError,
                    http.client.IncompleteRead, http.client.RemoteDisconnected) as exc:
                network_error = True
                error = f'{type(exc).__name__}: {exc}'
                if isinstance(exc, http.client.IncompleteRead):
                    raw_body = exc.partial.decode('utf-8', errors='replace')
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            try:
                parsed = json.loads(raw_body)
                response_json = parsed if isinstance(parsed, dict) else None
            except (ValueError, TypeError):
                response_json = None
            secrets.update(_secrets(response_json))
            retryable = network_error or status in {408, 425, 429} or (status is not None and 500 <= status <= 599)
            retry = retryable and index < self.max_retries
            reason = 'network_error' if network_error else f'http_{status}' if retryable else None
            wait = self._wait(index, headers) if retry else 0.0
            if not retry:
                if network_error:
                    outcome = _failure('network_error', raw_body, error=_safe(error, secrets))
                elif error and status is None:
                    outcome = _failure('transport_error', raw_body, error=_safe(error, secrets))
                elif status is None or not 200 <= status <= 299:
                    outcome = _failure('http_error', raw_body, http_status=status)
                elif response_json is None:
                    outcome = _failure('malformed_response', raw_body)
                else:
                    try:
                        outcome = parse_response(self.provider, response_json, labels)
                    except Exception as exc:
                        outcome = _failure('malformed_response', raw_body,
                                           error=_safe(f'{type(exc).__name__}: {exc}', secrets))
            attempt = {**started, 'phase': 'finished', 'finished_at': _utc(),
                       'outcome': _safe(outcome, secrets) if not retry else None,
                       'duration_ms': (time.perf_counter() - attempt_start) * 1000,
                       'http_status': status, 'raw_body': _safe_body(raw_body, secrets),
                       'response_headers': {key.lower(): _safe(value, secrets) for key, value in headers.items()
                                            if key.lower() in self.SAFE_HEADERS},
                       'usage': _safe(response_json.get('usage'), secrets) if response_json else None,
                       'response_model': _safe(response_json.get('model'), secrets) if response_json else None,
                       'retry_reason': reason, 'will_retry': retry, 'wait_seconds': wait, 'error': _safe(error, secrets)}
            attempt = _safe(attempt, secrets)
            attempts.append(attempt)
            _notify_attempt(on_attempt, attempt)
            if retry:
                try:
                    self.sleep_fn(wait)
                except Exception as exc:
                    outcome = _failure('transport_error', raw_body, error=_safe(str(exc), secrets))
                    break
                continue
            break
        return _safe({'outcome': outcome, 'response': response_json, 'attempts': attempts,
                      'metadata': {'provider': self.provider, 'endpoint': endpoint,
                                   'request_model': payload.get('model'),
                                   'temperature': payload.get('temperature'),
                                   'temperature_status': 'specified' if 'temperature' in payload else 'unspecified',
                                   'cost': None, 'cost_status': 'unknown',
                                   'usage_status': 'present' if response_json and response_json.get('usage') is not None else 'missing'},
                      'total_latency_ms': (time.perf_counter() - total_start) * 1000}, secrets)
