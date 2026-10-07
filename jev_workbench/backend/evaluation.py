"""Classification metrics with explicit completed-request denominators."""
from __future__ import annotations
import math
import statistics
from typing import Any, Dict, List

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    rank = (len(values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (rank - low)


def classification_metrics(records: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    completed = [r for r in records if r["status"] == "completed"]
    per_class = {}
    f1s = []
    for label in labels:
        tp = sum(r.get("prediction") == label and r.get("gold_label") == label for r in completed)
        fp = sum(r.get("prediction") == label and r.get("gold_label") != label for r in completed)
        fn = sum(r.get("prediction") != label and r.get("gold_label") == label for r in completed)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1s.append(f1)
    latencies = [r["latency_ms"] for r in completed]
    return {
        "total": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "completion_rate": len(completed) / len(records) if records else 0.0,
        "valid_output_rate": len(completed) / len(records) if records else 0.0,
        "accuracy_attempted": sum(r.get("prediction") == r.get("gold_label") for r in completed) / len(records) if records else 0.0,
        "accuracy_completed": sum(r.get("prediction") == r.get("gold_label") for r in completed) / len(completed) if completed else 0.0,
        "macro_f1_completed": statistics.fmean(f1s) if f1s else 0.0,
        "per_class_completed": per_class,
        "avg_latency_ms_completed": statistics.fmean(latencies) if latencies else None,
        "p95_latency_ms_completed": percentile(latencies, 0.95) if latencies else None,
    }
