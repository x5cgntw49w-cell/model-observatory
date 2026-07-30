from __future__ import annotations

import json

import pytest

from model_observatory.io import load_jsonl, write_jsonl
from model_observatory.schema import PredictionRecord, ValidationError


def valid_mapping() -> dict[str, object]:
    return {
        "id": "example-1",
        "label": "yes",
        "probabilities": {"yes": 0.75, "no": 0.25},
        "slices": {"region": "north"},
    }


def test_record_derives_prediction_and_confidence() -> None:
    record = PredictionRecord.from_mapping(valid_mapping())
    assert record.prediction == "yes"
    assert record.confidence == pytest.approx(0.75)
    assert record.correct is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "", "id must be"),
        ("label", None, "label must be"),
        ("probabilities", {"yes": 1.0}, "at least two"),
        ("slices", [], "slices must be"),
    ],
)
def test_invalid_top_level_fields(field: str, value: object, message: str) -> None:
    mapping = valid_mapping()
    mapping[field] = value
    with pytest.raises(ValidationError, match=message):
        PredictionRecord.from_mapping(mapping)


def test_probabilities_must_sum_to_one() -> None:
    mapping = valid_mapping()
    mapping["probabilities"] = {"yes": 0.8, "no": 0.3}
    with pytest.raises(ValidationError, match="sum to 1.0"):
        PredictionRecord.from_mapping(mapping)


def test_label_must_be_in_class_space() -> None:
    mapping = valid_mapping()
    mapping["label"] = "maybe"
    with pytest.raises(ValidationError, match="present in probabilities"):
        PredictionRecord.from_mapping(mapping)


def test_jsonl_round_trip(tmp_path) -> None:
    records = [PredictionRecord.from_mapping(valid_mapping())]
    path = tmp_path / "records.jsonl"
    write_jsonl(path, records)
    loaded = load_jsonl(path)
    assert loaded == records


def test_jsonl_reports_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(valid_mapping()) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValidationError, match=r"bad.jsonl:2"):
        load_jsonl(path)


def test_jsonl_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "duplicates.jsonl"
    row = json.dumps(valid_mapping())
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate id"):
        load_jsonl(path)
