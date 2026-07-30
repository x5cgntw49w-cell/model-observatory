from __future__ import annotations

import pytest

from model_observatory.schema import PredictionRecord


def make_record(
    record_id: str,
    label: str,
    probabilities: dict[str, float],
    **slices: str,
) -> PredictionRecord:
    return PredictionRecord(record_id, label, probabilities, slices)


@pytest.fixture
def records() -> list[PredictionRecord]:
    return [
        make_record("1", "a", {"a": 0.9, "b": 0.1}, device="desktop"),
        make_record("2", "a", {"a": 0.6, "b": 0.4}, device="mobile"),
        make_record("3", "b", {"a": 0.7, "b": 0.3}, device="mobile"),
        make_record("4", "b", {"a": 0.2, "b": 0.8}, device="desktop"),
    ]
