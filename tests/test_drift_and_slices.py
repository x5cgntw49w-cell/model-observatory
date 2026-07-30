from __future__ import annotations

import pytest

from model_observatory.drift import (
    compare_distributions,
    jensen_shannon,
    population_stability_index,
)
from model_observatory.schema import PredictionRecord
from model_observatory.slices import analyze_slices


def test_identical_distributions_have_zero_js() -> None:
    assert jensen_shannon([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_js_is_symmetric() -> None:
    first = jensen_shannon([0.9, 0.1], [0.2, 0.8])
    second = jensen_shannon([0.2, 0.8], [0.9, 0.1])
    assert first == pytest.approx(second)
    assert first > 0.0


def test_identical_confidence_has_zero_psi() -> None:
    values = [0.55, 0.62, 0.73, 0.88]
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_compare_detects_prediction_shift() -> None:
    reference = [
        PredictionRecord(str(index), "a", {"a": 0.9, "b": 0.1})
        for index in range(8)
    ]
    current = [
        PredictionRecord(str(index), "b", {"a": 0.1, "b": 0.9})
        for index in range(8)
    ]
    drift = compare_distributions(reference, current)
    assert drift["prediction_js_divergence"] > 0.5


def test_slice_analysis_is_sorted_worst_first(records) -> None:
    rows = analyze_slices(records, min_count=2)
    assert rows[0]["slice"] == "device"
    assert rows[0]["value"] == "mobile"
    assert rows[0]["accuracy_gap"] < 0.0


def test_slice_analysis_filters_small_groups(records) -> None:
    assert analyze_slices(records, min_count=3) == []


def test_slice_analysis_rejects_invalid_min_count(records) -> None:
    with pytest.raises(ValueError, match="positive"):
        analyze_slices(records, min_count=0)
