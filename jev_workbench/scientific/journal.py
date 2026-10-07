"""SQLite journal: every planned job remains in the denominator after failure.

One CLI writer is expected; SQLite transactions also protect individual writes.
A KeyboardInterrupt intentionally leaves the in-flight job pending for resume.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import uuid
from typing import Callable
from .remote import _safe, _secrets, _EXECUTION_ID, JournalPersistenceFailure


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class DurableJournal:
    TERMINAL_STATUSES = frozenset({'valid', 'refusal', 'invalid_label', 'malformed_response',
                                  'truncated_output', 'http_error', 'network_error', 'transport_error',
                                  'execution_error', 'failed', 'invalid_input'})

    def __init__(self, path: Path, planned: list[dict], protocol_hash: str, *, secrets=()):
        self.path = Path(path)
        self.secrets = tuple(secret for secret in secrets if secret)
        required = {'job_id', 'dataset_id', 'item_id', 'group_id', 'model', 'repeat', 'input', 'gold_label'}
        ids = []
        for job in planned:
            if not required.issubset(job) or not isinstance(job['input'], dict):
                raise ValueError('Planned job lacks required fields or dictionary input')
            if not isinstance(job['job_id'], str) or not job['job_id'] or not isinstance(job['gold_label'], str):
                raise ValueError('job_id and gold_label must be strings')
            ids.append(job['job_id'])
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate job_id in planned manifest')
        if not isinstance(protocol_hash, str) or not protocol_hash:
            raise ValueError('protocol_hash must be nonempty')
        manifest = _json(sorted(planned, key=lambda job: job['job_id']))
        manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS planned (job_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, job_json TEXT NOT NULL, outcome_json TEXT, terminal_at TEXT)')
            conn.execute('CREATE TABLE IF NOT EXISTS attempts (event_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES planned(job_id), event_json TEXT NOT NULL, recorded_at TEXT NOT NULL)')
            existing = dict(conn.execute('SELECT key,value FROM metadata'))
            if existing:
                if existing.get('protocol_hash') != protocol_hash or existing.get('manifest_hash') != manifest_hash:
                    raise ValueError('Resume protocol or planned manifest mismatch')
                stored = [json.loads(row[0]) for row in conn.execute('SELECT job_json FROM planned')]
                if _json(sorted(stored, key=lambda job: job['job_id'])) != manifest:
                    raise ValueError('Stored planned manifest is inconsistent')
            else:
                conn.executemany('INSERT INTO metadata(key,value) VALUES (?,?)',
                                 [('protocol_hash', protocol_hash), ('manifest_hash', manifest_hash)])
                conn.executemany('INSERT INTO planned(job_id,ordinal,job_json) VALUES (?,?,?)',
                                 [(job['job_id'], index, _json(job)) for index, job in enumerate(planned)])
            self._recover(conn)

    def _redact(self, value):
        return _safe(value, set(self.secrets) | _secrets(value))

    def _recover(self, conn):
        for (job_id,) in conn.execute('SELECT job_id FROM planned WHERE outcome_json IS NULL').fetchall():
            events = [json.loads(row[0]) for row in conn.execute(
                'SELECT event_json FROM attempts WHERE job_id=? ORDER BY event_id', (job_id,))]
            http_starts = [event for event in events if event.get('phase') == 'started' and event.get('kind') != 'job']
            if not http_starts:
                continue
            last_id = http_starts[-1].get('attempt_id')
            finished = next((event for event in reversed(events) if event.get('phase') == 'finished'
                             and event.get('attempt_id') == last_id), None)
            if finished and not finished.get('will_retry') and isinstance(finished.get('outcome'), dict):
                outcome = {**finished['outcome'], 'recovered_from_attempt': last_id,
                           'transport': {'attempts': [event for event in events if event.get('phase') == 'finished'],
                                         'recovered': True}}
                conn.execute('UPDATE planned SET outcome_json=?, terminal_at=? WHERE job_id=?',
                             (_json(self._redact(outcome)), _utc(), job_id))
                recovery = {'kind': 'recovery', 'phase': 'recovered_terminal', 'recovery_for': last_id,
                            'potential_duplicate': False, 'recorded_at': _utc()}
            else:
                if any(event.get('recovery_for') == last_id for event in events):
                    continue
                recovery = {'kind': 'recovery', 'phase': 'indeterminate', 'recovery_for': last_id,
                            'potential_duplicate': True, 'previous_attempt_ids': [event.get('attempt_id') for event in http_starts],
                            'recorded_at': _utc()}
            conn.execute('INSERT INTO attempts(job_id,event_json,recorded_at) VALUES (?,?,?)',
                         (job_id, _json(recovery), _utc()))

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA synchronous=FULL')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def pending_jobs(self) -> list[dict]:
        with self._connect() as conn:
            return [json.loads(row[0]) for row in conn.execute(
                'SELECT job_json FROM planned WHERE outcome_json IS NULL ORDER BY ordinal')]

    def recover_pending(self) -> None:
        """Refresh durable HTTP recovery even when reusing this journal object."""
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._recover(conn)

    def record_terminal(self, job_id: str, outcome: dict) -> None:
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            existing = conn.execute('SELECT outcome_json FROM planned WHERE job_id=?', (job_id,)).fetchone()
            if existing is None:
                raise KeyError(job_id)
            if existing[0] is not None:
                return
            if not isinstance(outcome, dict) or outcome.get('status') not in self.TERMINAL_STATUSES:
                raise ValueError('Terminal outcome requires terminal status')
            value = _json(self._redact(outcome))
            conn.execute('UPDATE planned SET outcome_json=?, terminal_at=? WHERE job_id=? AND outcome_json IS NULL',
                         (value, _utc(), job_id))

    def append_attempt(self, job_id: str, attempt: dict) -> None:
        try:
            with self._connect() as conn:
                if not conn.execute('SELECT 1 FROM planned WHERE job_id=?', (job_id,)).fetchone():
                    raise KeyError(job_id)
                conn.execute('INSERT INTO attempts(job_id,event_json,recorded_at) VALUES (?,?,?)',
                             (job_id, _json(self._redact(attempt)), _utc()))
        except Exception:
            raise JournalPersistenceFailure(attempt.get('phase')) from None

    def records(self) -> list[dict]:
        with self._connect() as conn:
            attempts = {}
            for job_id, event in conn.execute('SELECT job_id,event_json FROM attempts ORDER BY event_id'):
                attempts.setdefault(job_id, []).append(json.loads(event))
            result = []
            for job_id, job, outcome, terminal_at in conn.execute(
                    'SELECT job_id,job_json,outcome_json,terminal_at FROM planned ORDER BY ordinal'):
                result.append({**json.loads(job), 'outcome': json.loads(outcome) if outcome else None,
                               'terminal_at': terminal_at, 'attempts': attempts.get(job_id, []),
                               'potential_duplicate': any(event.get('potential_duplicate') for event in attempts.get(job_id, []))})
            return result

    def summary(self) -> dict:
        records = self.records()
        terminal = sum(row['outcome'] is not None for row in records)
        valid = sum(row['outcome'] is not None and row['outcome'].get('status') == 'valid' for row in records)
        return {'planned': len(records), 'terminal': terminal, 'valid': valid,
                'failed': terminal - valid, 'pending': len(records) - terminal,
                'complete': terminal == len(records)}


def execute_jobs(journal: DurableJournal, predict_fn: Callable[[dict], dict], order_seed: int = 0,
                 *, allow_indeterminate_retry: bool = False, selected_models=None,
                 max_indeterminate_retries: int | None = None, workers: int = 1) -> dict:
    """Predict each pending job once; persist sample failures and continue.

    Predictors can bind HTTP ``on_attempt`` to ``journal.append_attempt(job_id, event)``
    for immediate HTTP-level durability. Each job also gets a started event before
    its predictor runs. BaseException (including interruption) is never swallowed.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    journal.recover_pending()
    pending = journal.pending_jobs()
    if selected_models is not None:
        selected_models = set(selected_models)
        pending = [job for job in pending if job['model'] in selected_models]
    records = journal.records()
    retry_counts = {record['job_id']: sum(event.get('kind') == 'duplicate_retry_authorization'
                                        for event in record['attempts']) for record in records}
    uncertain = {record['job_id'] for record in records
                 if record['outcome'] is None and record['potential_duplicate']}
    # Shuffle the frozen full manifest, then filter terminal/unselected jobs.
    # Shuffling only pending rows changes the registered order after a resume.
    full_order = [record['job_id'] for record in records]
    random.Random(order_seed).shuffle(full_order)
    pending_by_id = {job['job_id']: job for job in pending}
    pending = [pending_by_id[job_id] for job_id in full_order if job_id in pending_by_id]
    work = []
    for job in pending:
        if job['job_id'] in uncertain:
            if (not allow_indeterminate_retry or
                    (max_indeterminate_retries is not None and retry_counts[job['job_id']] >= max_indeterminate_retries)):
                continue
            journal.append_attempt(job['job_id'], {'kind': 'duplicate_retry_authorization',
                'phase': 'authorized', 'potential_duplicate': True, 'recorded_at': _utc()})
        work.append(job)

    def process(job):
        execution_id = str(uuid.uuid4())
        journal.append_attempt(job['job_id'], {'phase': 'started', 'kind': 'job',
                                              'attempt_id': execution_id, 'execution_id': execution_id,
                                              'started_at': _utc()})
        context_token = _EXECUTION_ID.set(execution_id)
        try:
            result = predict_fn(job)
            if not isinstance(result, dict):
                raise ValueError('Predictor must return a dictionary')
            if isinstance(result.get('outcome'), dict):
                outcome = dict(result['outcome'])
                outcome['transport'] = {key: value for key, value in result.items() if key != 'outcome'}
            else:
                outcome = dict(result)
            if outcome.get('status') not in journal.TERMINAL_STATUSES:
                raise ValueError('Predictor outcome lacks status')
            if outcome.get('status') == 'valid':
                if not isinstance(outcome.get('prediction'), str) or not outcome['prediction']:
                    raise ValueError('Valid outcome lacks prediction')
                confidence = outcome.get('confidence')
                if confidence is None:
                    outcome.update(confidence=None, confidence_status=(
                        'invalid' if outcome.get('confidence_status') == 'invalid' else 'missing'))
                elif isinstance(confidence, bool):
                    outcome.update(confidence=None, confidence_status='invalid')
                else:
                    try:
                        confidence = float(confidence)
                        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                            raise ValueError('Invalid confidence')
                        outcome.update(confidence=confidence, confidence_status='valid')
                    except (ValueError, TypeError, OverflowError):
                        outcome.update(confidence=None, confidence_status='invalid')
            # Serialization errors are sample failures, while persistence errors
            # must propagate rather than pretend that a durable write succeeded.
            _json(outcome)
        except Exception as exc:
            outcome = {'status': 'execution_error', 'prediction': None, 'confidence': None,
                       'confidence_status': 'missing', 'error_type': type(exc).__name__, 'error': str(exc)}
        finally:
            _EXECUTION_ID.reset(context_token)
        journal.record_terminal(job['job_id'], outcome)

    if workers == 1:
        for job in work:
            process(job)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='scientific-job')
        futures = [executor.submit(process, job) for job in work]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return journal.summary()
