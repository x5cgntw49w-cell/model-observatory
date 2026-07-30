from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import PredictionRecord, ValidationError


def load_jsonl(path: str | Path) -> list[PredictionRecord]:
    source = Path(path)
    records: list[PredictionRecord] = []
    seen_ids: set[str] = set()

    with source.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                data = json.loads(raw_line)
                record = PredictionRecord.from_mapping(data)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValidationError(f"{source}:{line_number}: {error}") from error
            if record.record_id in seen_ids:
                raise ValidationError(
                    f"{source}:{line_number}: duplicate id {record.record_id!r}"
                )
            seen_ids.add(record.record_id)
            records.append(record)

    if not records:
        raise ValidationError(f"{source}: no prediction records found")
    _validate_class_space(records, source)
    return records


def write_jsonl(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_mapping(), sort_keys=True) + "\n")


def _validate_class_space(records: list[PredictionRecord], source: Path) -> None:
    expected = set(records[0].probabilities)
    for index, record in enumerate(records[1:], start=2):
        if set(record.probabilities) != expected:
            raise ValidationError(
                f"{source}:{index}: probability classes differ from the first record"
            )
