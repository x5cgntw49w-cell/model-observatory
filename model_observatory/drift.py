from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping

from .schema import PredictionRecord


def compare_distributions(
    reference: Iterable[PredictionRecord], current: Iterable[PredictionRecord]
) -> dict[str, object]:
    reference_data = list(reference)
    current_data = list(current)
    if not reference_data or not current_data:
        raise ValueError("reference and current records are required")
    if set(reference_data[0].probabilities) != set(current_data[0].probabilities):
        raise ValueError("reference and current class spaces must match")

    labels = sorted(reference_data[0].probabilities)
    reference_predictions = _distribution(
        Counter(item.prediction for item in reference_data), labels
    )
    current_predictions = _distribution(
        Counter(item.prediction for item in current_data), labels
    )
    reference_labels = _distribution(Counter(item.label for item in reference_data), labels)
    current_labels = _distribution(Counter(item.label for item in current_data), labels)

    return {
        "reference_count": len(reference_data),
        "current_count": len(current_data),
        "prediction_js_divergence": jensen_shannon(
            reference_predictions, current_predictions
        ),
        "label_js_divergence": jensen_shannon(reference_labels, current_labels),
        "confidence_psi": population_stability_index(
            [item.confidence for item in reference_data],
            [item.confidence for item in current_data],
        ),
        "prediction_distribution": {
            "reference": dict(zip(labels, reference_predictions)),
            "current": dict(zip(labels, current_predictions)),
        },
        "slice_drift": _slice_drift(reference_data, current_data),
    }


def jensen_shannon(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("distributions must be non-empty and have equal length")
    midpoint = [(left + right) / 2.0 for left, right in zip(first, second)]
    return 0.5 * _kl_divergence(first, midpoint) + 0.5 * _kl_divergence(second, midpoint)


def population_stability_index(
    reference: list[float], current: list[float], bins: int = 10
) -> float:
    if not reference or not current:
        raise ValueError("reference and current values are required")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    epsilon = 1e-8
    reference_counts = [0] * bins
    current_counts = [0] * bins
    for value in reference:
        reference_counts[min(bins - 1, int(value * bins))] += 1
    for value in current:
        current_counts[min(bins - 1, int(value * bins))] += 1

    score = 0.0
    for reference_count, current_count in zip(reference_counts, current_counts):
        reference_share = max(epsilon, reference_count / len(reference))
        current_share = max(epsilon, current_count / len(current))
        score += (current_share - reference_share) * math.log(
            current_share / reference_share
        )
    return score


def _distribution(counts: Mapping[str, int], labels: list[str]) -> list[float]:
    total = sum(counts.values())
    return [counts.get(label, 0) / total for label in labels]


def _kl_divergence(first: list[float], second: list[float]) -> float:
    return sum(
        left * math.log(left / right)
        for left, right in zip(first, second)
        if left > 0.0 and right > 0.0
    )


def _slice_drift(
    reference: list[PredictionRecord], current: list[PredictionRecord]
) -> list[dict[str, object]]:
    reference_groups: dict[str, Counter[str]] = defaultdict(Counter)
    current_groups: dict[str, Counter[str]] = defaultdict(Counter)
    for record in reference:
        for name, value in record.slices.items():
            reference_groups[name][value] += 1
    for record in current:
        for name, value in record.slices.items():
            current_groups[name][value] += 1

    rows: list[dict[str, object]] = []
    for name in sorted(set(reference_groups) & set(current_groups)):
        values = sorted(set(reference_groups[name]) | set(current_groups[name]))
        reference_distribution = _distribution(reference_groups[name], values)
        current_distribution = _distribution(current_groups[name], values)
        rows.append(
            {
                "slice": name,
                "js_divergence": jensen_shannon(
                    reference_distribution, current_distribution
                ),
                "reference": dict(zip(values, reference_distribution)),
                "current": dict(zip(values, current_distribution)),
            }
        )
    return sorted(rows, key=lambda row: float(row["js_divergence"]), reverse=True)
