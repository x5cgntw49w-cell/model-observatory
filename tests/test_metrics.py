from __future__ import annotations

import math

import pytest

from model_observatory.metrics import (
    accuracy,
    calibration,
    class_metrics,
    evaluate,
    log_loss,
    multiclass_brier,
    risk_coverage,
)


def test_accuracy(records) -> None:
    assert accuracy(records) == pytest.approx(0.75)


def test_class_metrics(records) -> None:
    metrics = class_metrics(records)
    assert metrics["a"]["precision"] == pytest.approx(2 / 3)
    assert metrics["a"]["recall"] == pytest.approx(1.0)
    assert metrics["b"]["precision"] == pytest.approx(1.0)
    assert metrics["b"]["recall"] == pytest.approx(0.5)


def test_log_loss(records) -> None:
    expected = -sum(math.log(value) for value in [0.9, 0.6, 0.3, 0.8]) / 4
    assert log_loss(records) == pytest.approx(expected)


def test_multiclass_brier(records) -> None:
    assert multiclass_brier(records) == pytest.approx(0.35)


def test_calibration_weights_non_empty_bins(records) -> None:
    result = calibration(records, bins=5)
    assert 0.0 <= result["ece"] <= 1.0
    assert sum(row["count"] for row in result["bins"]) == 4


def test_risk_decreases_at_lower_coverage(records) -> None:
    curve = risk_coverage(records, points=4)
    assert curve[0]["coverage"] == pytest.approx(0.25)
    assert curve[0]["risk"] <= curve[-1]["risk"]
    assert curve[-1]["coverage"] == pytest.approx(1.0)


def test_evaluate_returns_machine_readable_summary(records) -> None:
    summary = evaluate(records)
    assert summary["count"] == 4
    assert summary["classes"] == ["a", "b"]
    assert summary["accuracy"] == pytest.approx(0.75)
    assert set(summary["confusion_matrix"]) == {"a", "b"}


def test_empty_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate([])
