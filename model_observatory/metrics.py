from __future__ import annotations

import math
from typing import Iterable

from .schema import PredictionRecord


def _require_records(records: Iterable[PredictionRecord]) -> list[PredictionRecord]:
    materialized = list(records)
    if not materialized:
        raise ValueError("at least one prediction record is required")
    return materialized


def confusion_matrix(records: Iterable[PredictionRecord]) -> dict[str, dict[str, int]]:
    data = _require_records(records)
    labels = sorted(data[0].probabilities)
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for record in data:
        matrix[record.label][record.prediction] += 1
    return matrix


def accuracy(records: Iterable[PredictionRecord]) -> float:
    data = _require_records(records)
    return sum(record.correct for record in data) / len(data)


def class_metrics(records: Iterable[PredictionRecord]) -> dict[str, dict[str, float | int]]:
    data = _require_records(records)
    matrix = confusion_matrix(data)
    result: dict[str, dict[str, float | int]] = {}
    for label in matrix:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[actual][label] for actual in matrix if actual != label)
        false_negative = sum(matrix[label][predicted] for predicted in matrix if predicted != label)
        support = sum(matrix[label].values())
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def log_loss(records: Iterable[PredictionRecord], epsilon: float = 1e-15) -> float:
    data = _require_records(records)
    losses = [
        -math.log(min(1.0 - epsilon, max(epsilon, record.probabilities[record.label])))
        for record in data
    ]
    return sum(losses) / len(losses)


def multiclass_brier(records: Iterable[PredictionRecord]) -> float:
    data = _require_records(records)
    total = 0.0
    for record in data:
        total += sum(
            (probability - (1.0 if label == record.label else 0.0)) ** 2
            for label, probability in record.probabilities.items()
        )
    return total / len(data)


def calibration(records: Iterable[PredictionRecord], bins: int = 10) -> dict[str, object]:
    data = _require_records(records)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    bucketed: list[list[PredictionRecord]] = [[] for _ in range(bins)]
    for record in data:
        index = min(bins - 1, int(record.confidence * bins))
        bucketed[index].append(record)

    expected_calibration_error = 0.0
    rows: list[dict[str, float | int]] = []
    for index, bucket in enumerate(bucketed):
        if bucket:
            mean_confidence = sum(item.confidence for item in bucket) / len(bucket)
            mean_accuracy = sum(item.correct for item in bucket) / len(bucket)
            expected_calibration_error += (
                len(bucket) / len(data) * abs(mean_confidence - mean_accuracy)
            )
        else:
            mean_confidence = 0.0
            mean_accuracy = 0.0
        rows.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(bucket),
                "confidence": mean_confidence,
                "accuracy": mean_accuracy,
            }
        )
    return {"ece": expected_calibration_error, "bins": rows}


def risk_coverage(records: Iterable[PredictionRecord], points: int = 10) -> list[dict[str, float | int]]:
    data = sorted(_require_records(records), key=lambda item: item.confidence, reverse=True)
    if points < 2:
        raise ValueError("points must be at least 2")
    rows: list[dict[str, float | int]] = []
    for step in range(1, points + 1):
        count = max(1, round(len(data) * step / points))
        accepted = data[:count]
        rows.append(
            {
                "coverage": count / len(data),
                "risk": 1.0 - sum(record.correct for record in accepted) / count,
                "count": count,
                "threshold": accepted[-1].confidence,
            }
        )
    return rows


def evaluate(records: Iterable[PredictionRecord], bins: int = 10) -> dict[str, object]:
    data = _require_records(records)
    per_class = class_metrics(data)
    macro_f1 = sum(float(row["f1"]) for row in per_class.values()) / len(per_class)
    balanced_accuracy = sum(float(row["recall"]) for row in per_class.values()) / len(per_class)
    calibration_result = calibration(data, bins=bins)
    return {
        "count": len(data),
        "classes": sorted(data[0].probabilities),
        "accuracy": accuracy(data),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "log_loss": log_loss(data),
        "brier_score": multiclass_brier(data),
        "ece": calibration_result["ece"],
        "calibration_bins": calibration_result["bins"],
        "risk_coverage": risk_coverage(data),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(data),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
