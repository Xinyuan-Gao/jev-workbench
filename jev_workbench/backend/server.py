"""Local HTTP API for the JEV experiment workbench."""
from __future__ import annotations

import json
import mimetypes
import os
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from .experiments import list_experiments
from .runner import ExperimentRunner, benchmark_summary
from .evaluation import classification_metrics


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


class WorkbenchHandler(BaseHTTPRequestHandler):
    server: "WorkbenchServer"

    def _send(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return value

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def _serve_frontend(self, path: str) -> bool:
        """Serve the bundled static UI without allowing path traversal."""
        relative = "index.html" if path in {"", "/"} else unquote(path.lstrip("/"))
        candidate = (FRONTEND_DIR / relative).resolve()
        try:
            candidate.relative_to(FRONTEND_DIR.resolve())
        except ValueError:
            self._send(404, {"error": "未找到文件"})
            return True
        if not candidate.is_file():
            return False
        data = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") or content_type == "application/javascript" else content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if not path.startswith("/api/") and self._serve_frontend(path):
                return
            if path == "/api/experiments":
                return self._send(200, list_experiments())
            if path == "/api/state":
                return self._send(200, self.server.runner.summary())
            if path == "/api/benchmark":
                return self._send(200, benchmark_summary(self.server.runner.history_dir))
            parts = path.split("/")
            if len(parts) == 5 and parts[:3] == ["", "api", "runs"]:
                run = self.server.runner.get(parts[3])
                if parts[4] == "state":
                    return self._send(200, run.state())
                if parts[4] == "events":
                    with run._lock:
                        events = list(run.events)
                    try:
                        after = max(0, int(parse_qs(parsed.query).get("after", ["0"])[0]))
                    except (TypeError, ValueError):
                        after = 0
                    return self._send(200, {"run_id": run.id, "status": run.status, "events": events[after:], "next_cursor": len(events)})
                if parts[4] == "results":
                    with run._lock:
                        rows = [{**row, "status": row.get("status", "completed"), "gold_label": row["input"]["label"]} for row in run.results]
                        experiment = next(item for item in list_experiments() if item["id"] == run.state()["experiment_id"])
                        metrics = classification_metrics(rows, experiment["labels"])
                        names = {"jev": "JEV", "gemini": "Gemini 3.8 Flash", "xgboost": "TF-IDF + XGBoost", "lightgbm": "TF-IDF + LightGBM", "simulation": "规则模拟"}
                        model = {"model": names[run.mode], "accuracy": metrics["accuracy_completed"], "macro_f1": metrics["macro_f1_completed"],
                                 "p95_latency_ms": metrics["p95_latency_ms_completed"], "estimated_cost_usd": None}
                        return self._send(200, {"run_id": run.id, "status": run.status, "results": list(run.results), "models": [model] if rows else []})
            return self._send(404, {"error": "未找到路径"})
        except KeyError:
            return self._send(404, {"error": "run 不存在"})
        except Exception as exc:
            return self._send(500, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/api/runs":
                body = self._read_json()
                run = self.server.runner.start(
                    body.get("experiment_id", ""),
                    mode=body.get("mode", "simulation"),
                    delay_seconds=body.get("delay_seconds", 0.0),
                )
                return self._send(202, {"run_id": run.id, "status": run.status, "experiment_id": run.experiment["id"], "mode": run.mode})
            parts = path.split("/")
            if len(parts) == 5 and parts[:3] == ["", "api", "runs"] and parts[4] == "stop":
                run = self.server.runner.get(parts[3])
                run.stop()
                return self._send(202, run.state())
            return self._send(404, {"error": "未找到路径"})
        except (ValueError, KeyError) as exc:
            return self._send(400 if isinstance(exc, ValueError) else 404, {"error": str(exc)})
        except Exception as exc:
            return self._send(500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        # Keep API logs useful while ensuring secrets cannot be printed.
        super().log_message(format, *args)


class WorkbenchServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int] = ("127.0.0.1", 8765), runner: ExperimentRunner | None = None):
        self.runner = runner or ExperimentRunner()
        super().__init__(server_address, WorkbenchHandler)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = WorkbenchServer((host, port))
    print(f"JEV workbench listening on http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve(os.getenv("WORKBENCH_HOST", "127.0.0.1"), int(os.getenv("WORKBENCH_PORT", "8765")))
