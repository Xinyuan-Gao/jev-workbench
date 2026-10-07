"""Adapters for OpenAI-compatible JEV endpoints."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


def load_jev_env() -> Dict[str, str]:
    """Read JEV settings from environment and the workbench's optional .env.local."""
    values: Dict[str, str] = {}
    env_path = Path(__file__).resolve().parents[1] / ".env.local"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in (
        "JEV_API_URL", "JEV_API_KEY", "JEV_API_PATH", "JEV_API_PROTOCOL",
        "GEMINI_API_URL", "GEMINI_API_KEY", "GEMINI_API_PATH", "GEMINI_MODEL",
    ):
        if os.getenv(key):
            values[key] = os.environ[key]
    return values


class JEVAdapter:
    """Small OpenAI-compatible client; secrets never enter ``last_log``."""

    _RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, api_path: Optional[str] = None, timeout: float = 30.0, model: str = "jev-latest", protocol: Optional[str] = None, provider: str = "JEV", max_retries: int = 3, backoff_seconds: float = 1.0, max_backoff_seconds: float = 30.0):
        env = load_jev_env()
        self.api_url = (api_url or env.get("JEV_API_URL", "")).rstrip("/")
        self.api_key = api_key or env.get("JEV_API_KEY", "")
        self.api_path = api_path or env.get("JEV_API_PATH", "/chat/completions")
        self.protocol = (protocol or env.get("JEV_API_PROTOCOL", "systemone")).strip().lower()
        self.model = model
        self.provider = provider
        if not self.api_path.startswith("/"):
            self.api_path = "/" + self.api_path
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.max_backoff_seconds = max(self.backoff_seconds, float(max_backoff_seconds))
        self.last_log = ""
        self._lock = threading.Lock()

    @classmethod
    def _retry_after_seconds(cls, exc: urllib.error.HTTPError) -> Optional[float]:
        """Return a non-negative Retry-After value, supporting seconds or HTTP dates."""
        header = exc.headers.get("Retry-After") if exc.headers else None
        if not header:
            return None
        try:
            return max(0.0, float(header.strip()))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(header)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.astimezone()
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    def _retry_delay(self, attempt: int, exc: Optional[urllib.error.HTTPError] = None) -> float:
        """Compute exponential backoff, preferring a server-provided Retry-After."""
        requested = self._retry_after_seconds(exc) if exc is not None else None
        if requested is None:
            requested = self.backoff_seconds * (2 ** attempt)
        return min(self.max_backoff_seconds, max(0.0, requested))

    def _wait_before_retry(self, attempt: int, exc: Optional[urllib.error.HTTPError] = None) -> None:
        delay = self._retry_delay(attempt, exc)
        with self._lock:
            self.last_log = f"retry attempt={attempt + 1}/{self.max_retries} delay={delay:.3f}s"
        if delay:
            time.sleep(delay)

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_url or not self.api_key:
            raise RuntimeError(f"{self.provider}_API_URL/{self.provider}_API_KEY 未配置")
        url = self.api_url + self.api_path
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    with self._lock:
                        self.last_log = f"POST {url} status={response.status}"
                    return body
            except urllib.error.HTTPError as exc:
                retryable = exc.code in self._RETRYABLE_HTTP_CODES
                with self._lock:
                    self.last_log = f"POST {url} status={exc.code} error={exc.reason}"
                if retryable and attempt < self.max_retries:
                    self._wait_before_retry(attempt, exc)
                    continue
                raise RuntimeError(f"{self.provider} 请求失败（HTTP {exc.code}）") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                with self._lock:
                    self.last_log = f"POST {url} error={type(exc).__name__}"
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise RuntimeError(f"{self.provider} 请求失败：{exc}") from exc

    @staticmethod
    def _content(response: Dict[str, Any]) -> Any:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("JEV 返回缺少 choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, (dict, list)):
            return content
        text = str(content).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
            if fenced:
                try:
                    return json.loads(fenced.group(1))
                except json.JSONDecodeError:
                    pass
            return text

    def classify(self, item: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        labels = list(schema.get("labels") or [])
        if not labels and schema.get("label"):
            labels = [str(schema["label"])]
        # Jev's typed envelope works on both the official /systemone route and
        # 5yuantoken's /chat/completions gateway. Set JEV_API_PROTOCOL=openai only
        # for a genuinely OpenAI-shaped proxy.
        criteria = {label: None for label in labels}
        input_fields = {"classification": ("text",), "routing": ("task",), "relevance": ("query", "passage")}
        fields = input_fields.get(str(schema.get("task", "classification")), ("text", "query", "passage", "task"))
        clean_item = {key: item[key] for key in fields if key in item}
        if schema.get("task") == "relevance" and not all(str(clean_item.get(key, "")).strip() for key in ("query", "passage")):
            raise ValueError("RAG 输入必须同时包含 query 和 passage")
        typed_payload = {
            "model": self.model,
            "state": clean_item,
            "questions": {
                "classification": {
                    "type": "choice",
                    "instructions": f"根据输入选择最合适的类别。任务类型：{schema.get('task', 'classification')}。",
                    "criteria": criteria,
                }
            },
        }
        if self.protocol == "openai":
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个结构化分类器。只输出 JSON，包含 prediction 和 confidence。"},
                    {"role": "user", "content": json.dumps({"item": clean_item, "schema": schema}, ensure_ascii=False)},
                ],
                "temperature": 0,
            }
        elif self.api_path.rstrip("/").endswith("/chat/completions"):
            # 5yuantoken's OpenAI route adapts the *user message* into Jev state.
            # Its gateway accepts state only as text, so extract the human-readable field.
            if isinstance(item, str):
                state_text = item
            elif isinstance(item, dict):
                task = str(schema.get("task", "classification"))
                if task == "relevance":
                    state_text = json.dumps({"query": item.get("query", ""), "passage": item.get("passage", "")}, ensure_ascii=False)
                elif "text" in item:
                    state_text = str(item["text"])
                elif "task" in item:
                    state_text = str(item["task"])
                else:
                    state_text = json.dumps(clean_item, ensure_ascii=False)
            else:
                state_text = str(item)
            envelope = {"state": state_text, "questions": typed_payload["questions"]}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": json.dumps(envelope, ensure_ascii=False)}],
            }
        else:
            payload = typed_payload
        response = self._request(payload)
        parsed = None if isinstance(response.get("answers"), dict) else self._content(response)
        if isinstance(response.get("answers"), dict):
            answers = response["answers"]
            answer = answers.get("classification") or next(iter(answers.values()), {})
            prediction = answer.get("choice", answer.get("prediction", answer.get("label")))
            confidence = answer.get("confidence", 0.0)
            probabilities = answer.get("probabilities", {})
        elif isinstance(parsed, dict):
            # Gateway responses may return a named answer object directly, e.g.
            # {"intent": {"type": "choice", "choice": "billing", ...}}.
            answer = None
            for value in parsed.values():
                if isinstance(value, dict) and value.get("type") in {"choice", "noul", "score"}:
                    answer = value
                    break
            if answer is not None:
                prediction = answer.get("choice", answer.get("prediction", answer.get("label")))
                confidence = answer.get("confidence", 0.0)
                probabilities = answer.get("probabilities", {})
            else:
                prediction = parsed.get("prediction", parsed.get("label", parsed.get("choice")))
                confidence = parsed.get("confidence", 0.0)
                probabilities = parsed.get("probabilities", {})
        else:
            prediction, confidence, probabilities = str(parsed), 0.0, {}
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            raise ValueError("confidence 不是有效数值")
        if prediction not in labels:
            raise ValueError("预测不属于允许标签")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence 必须是 0 到 1 的有限数值")
        return {"prediction": prediction, "confidence": confidence, "probabilities": probabilities}


class GeminiAdapter(JEVAdapter):
    """OpenAI-compatible Gemini adapter using a separate credential."""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None,
                 api_path: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = 30.0, max_retries: int = 3,
                 backoff_seconds: float = 1.0, max_backoff_seconds: float = 30.0):
        env = load_jev_env()
        super().__init__(
            api_url=api_url or env.get("GEMINI_API_URL", ""),
            api_key=api_key or env.get("GEMINI_API_KEY", ""),
            api_path=api_path or env.get("GEMINI_API_PATH", "/chat/completions"),
            timeout=timeout,
            model=model or env.get("GEMINI_MODEL", "gemini-3.8-flash"),
            protocol="openai",
            provider="GEMINI",
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
