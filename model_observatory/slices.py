from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .metrics import accuracy, evaluate
from .schema import PredictionRecord


def analyze_slices(
    records: Iterable[PredictionRecord], min_count: int = 20
) -> list[dict[str, object]]:
    data = list(records)
    if not data:
        raise ValueError("at least one prediction record is required")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    overall_accuracy = accuracy(data)
    groups: dict[tuple[str, str], list[PredictionRecord]] = defaultdict(list)
    for record in data:
        for name, value in record.slices.items():
            groups[(name, value)].append(record)

    rows: list[dict[str, object]] = []
    for (name, value), group in groups.items():
        if len(group) < min_count:
            continue
        metrics = evaluate(group)
        rows.append(
            {
                "slice": name,
                "value": value,
                "count": len(group),
                "accuracy": metrics["accuracy"],
                "accuracy_gap": float(metrics["accuracy"]) - overall_accuracy,
                "macro_f1": metrics["macro_f1"],
                "ece": metrics["ece"],
                "mean_confidence": sum(item.confidence for item in group) / len(group),
            }
        )
    return sorted(rows, key=lambda row: (float(row["accuracy_gap"]), -int(row["count"])))
