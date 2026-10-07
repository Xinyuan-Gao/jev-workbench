"""Asynchronous experiment execution and event storage."""
from __future__ import annotations

import threading
import time
import uuid
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters import GeminiAdapter, JEVAdapter
from .experiments import get_experiment, simulate_prediction, split_records
from .traditional import TraditionalModel
from .evaluation import classification_metrics

# Optional local history for the SavedRun viewer; empty unless the user saves runs here.
HISTORY_DIR = Path(__file__).resolve().parents[1] / "reports" / "saved_runs"


def benchmark_summary(history_dir: Path = HISTORY_DIR) -> Dict[str, Any]:
    runs = []
    for path in sorted(history_dir.glob("corrected-*.json"), key=lambda path: path.stat().st_mtime):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("run_id") and record.get("metrics") and isinstance(record.get("results"), list):
                runs.append(record)
        except (ValueError, OSError):
            # A file still being written is read on the next poll.
            continue
    selected = {}
    for run in runs:
        selected[(run["experiment"], run["mode"])] = run
    runs = list(selected.values())
    return {"version": "corrected-group-heldout-v1", "expected_runs": 15,
            "total_runs": len(runs), "completed_runs": sum(r["status"] == "completed" for r in runs),
            "finished_runs": sum(r["status"] in {"completed", "partial", "failed", "stopped"} for r in runs),
            "finished": len(runs) == 15 and all(r["status"] in {"completed", "partial", "failed", "stopped"} for r in runs), "runs": runs}


class SavedRun:
    """A completed on-disk run presented through the existing workbench API."""
    def __init__(self, record: Dict[str, Any]):
        self.record = record
        self.id = record["run_id"]
        self.status = record["status"]
        self.mode = record["mode"]
        self._lock = threading.RLock()
        self.results = [{**row, "input": {**row["input"], "label": row["gold_label"]}} for row in record["results"]]
        self.events = [{"seq": 0, "type": "restored", "message": "已从磁盘恢复测试集结果", "run_id": self.id}]

    def state(self) -> Dict[str, Any]:
        r = self.record
        return {"run_id": self.id, "experiment_id": r["experiment"], "experiment_name": r["experiment_name"],
                "mode": self.mode, "status": self.status, "count": r["metrics"]["completed"],
                "completed": r["metrics"]["completed"], "total": r["test_count"], "total_samples": r["test_count"],
                "dataset_total": r["dataset_total"], "evaluation_split": "test",
                "split_counts": {"train": r["train_count"], "validation": r["validation_count"], "test": r["test_count"]}}

    def stop(self) -> None:
        pass


class Run:
    def __init__(self, run_id: str, experiment: Dict[str, Any], mode: str, adapter: Optional[JEVAdapter], delay_seconds: float = 0.0, history_dir: Optional[Path] = None):
        self.id = run_id
        self.experiment = experiment
        self.mode = mode
        self.adapter = adapter
        self.history_dir = history_dir
        self.splits = split_records(experiment["records"])
        self.evaluation_records = self.splits["test"]
        self.local_model = None
        self.delay_seconds = delay_seconds
        self.status = "queued"
        self.events: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self.thread = threading.Thread(target=self._execute, name=f"jev-run-{run_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self.status in {"queued", "running"}:
                self._event("stopping", {})

    def _event(self, event_type: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events), "type": event_type, "ts": time.time(), **data})

    def _execute(self) -> None:
        with self._lock:
            self.status = "running"
            self.started_at = time.time()
        self._event("started", {"run_id": self.id, "experiment_id": self.experiment["id"], "mode": self.mode})
        try:
            if self.mode in {"xgboost", "lightgbm"}:
                self.local_model = TraditionalModel(self.mode)
                self.local_model.fit(self.splits["train"])
            for index, item in enumerate(self.evaluation_records):
                if self._stop.is_set():
                    with self._lock:
                        self.status = "stopped"
                    self._event("stopped", {"processed": len(self.results)})
                    return
                started = time.perf_counter()
                self._event("record_started", {"record_id": item["id"], "index": index})
                if self.mode == "jev":
                    prediction = self.adapter.classify(item, {"labels": self.experiment["labels"], "task": self.experiment["task"]}) if self.adapter else None
                elif self.mode == "gemini":
                    prediction = self.adapter.classify(item, {"labels": self.experiment["labels"], "task": self.experiment["task"]}) if self.adapter else None
                elif self.local_model is not None:
                    prediction = self.local_model.predict(item)
                else:
                    prediction = simulate_prediction(self.experiment, item)
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                record = {
                    "id": item["id"],
                    "input": item,
                    "prediction": prediction.get("prediction"),
                    "confidence": prediction.get("confidence", 0.0),
                    "probabilities": prediction.get("probabilities", {}),
                    "latency_ms": latency_ms,
                    "mode": self.mode,
                }
                with self._lock:
                    self.results.append(record)
                self._event("prediction", {"record": record})
                if self.delay_seconds:
                    self._stop.wait(self.delay_seconds)
            with self._lock:
                self.status = "completed"
        except Exception as exc:
            with self._lock:
                self.status = "failed"
            # Error text is surfaced to the UI, but no Authorization header/key is included.
            self._event("error", {"message": str(exc)})
        finally:
            self.finished_at = time.time()
            self._event("finished", {"status": self.status, "count": len(self.results)})
            if self.history_dir is not None:
                self.history_dir.mkdir(parents=True, exist_ok=True)
                rows = [{**row, "status": "completed", "gold_label": row["input"]["label"]} for row in self.results]
                record = {"run_id": self.id, "experiment": self.experiment["id"], "experiment_name": self.experiment["name"],
                          "mode": self.mode, "status": self.status, "dataset_total": len(self.experiment["records"]),
                          "train_count": len(self.splits["train"]), "validation_count": len(self.splits["validation"]),
                          "test_count": len(self.splits["test"]), "results": rows,
                          "metrics": classification_metrics(rows, self.experiment["labels"])}
                path = self.history_dir / f"{self.id}.json"
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.id,
                "experiment_id": self.experiment["id"],
                "experiment_name": self.experiment["name"],
                "mode": self.mode,
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "count": len(self.results),
                "completed": len(self.results),
                "total": len(self.evaluation_records),
                "dataset_total": len(self.experiment["records"]),
                "evaluation_split": "test",
                "split_counts": {name: len(records) for name, records in self.splits.items()},
                "total_samples": len(self.evaluation_records),
            }


class ExperimentRunner:
    def __init__(self, history_dir: Path = HISTORY_DIR):
        self.history_dir = history_dir
        self.runs: Dict[str, Run] = {}
        self._lock = threading.RLock()
        for record in benchmark_summary(history_dir)["runs"]:
            self.runs[record["run_id"]] = SavedRun(record)

    def start(self, experiment_id: str, mode: str = "simulation", delay_seconds: float = 0.0) -> Run:
        experiment = get_experiment(experiment_id)
        if mode not in {"simulation", "xgboost", "lightgbm", "jev", "gemini"}:
            raise ValueError("mode 必须是 simulation、xgboost、lightgbm、jev 或 gemini")
        adapter = JEVAdapter() if mode == "jev" else GeminiAdapter() if mode == "gemini" else None
        run = Run("corrected-" + uuid.uuid4().hex[:12], experiment, mode, adapter, max(0.0, float(delay_seconds)), self.history_dir)
        with self._lock:
            self.runs[run.id] = run
        run.start()
        return run

    def get(self, run_id: str) -> Run:
        with self._lock:
            if run_id not in self.runs:
                raise KeyError(run_id)
            return self.runs[run_id]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            states = [run.state() for run in self.runs.values()]
        return {"runs": states, "active": sum(item["status"] in {"queued", "running"} for item in states)}
