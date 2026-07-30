from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping


class ValidationError(ValueError):
    """Raised when a prediction record violates the public schema."""


@dataclass(frozen=True)
class PredictionRecord:
    record_id: str
    label: str
    probabilities: dict[str, float]
    slices: dict[str, str] = field(default_factory=dict)

    @property
    def prediction(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.prediction]

    @property
    def correct(self) -> bool:
        return self.prediction == self.label

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PredictionRecord":
        if not isinstance(data, Mapping):
            raise ValidationError("record must be a JSON object")

        record_id = data.get("id")
        label = data.get("label")
        probabilities = data.get("probabilities")
        slices = data.get("slices", {})

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValidationError("id must be a non-empty string")
        if not isinstance(label, str) or not label.strip():
            raise ValidationError("label must be a non-empty string")
        if not isinstance(probabilities, Mapping) or len(probabilities) < 2:
            raise ValidationError("probabilities must contain at least two classes")

        clean_probabilities: dict[str, float] = {}
        for name, value in probabilities.items():
            if not isinstance(name, str) or not name:
                raise ValidationError("probability class names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"probability for {name!r} must be numeric")
            probability = float(value)
            if not isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValidationError(f"probability for {name!r} must be in [0, 1]")
            clean_probabilities[name] = probability

        total = sum(clean_probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValidationError(f"probabilities must sum to 1.0, got {total:.8f}")
        if label not in clean_probabilities:
            raise ValidationError("label must be present in probabilities")
        if not isinstance(slices, Mapping):
            raise ValidationError("slices must be a JSON object")

        clean_slices: dict[str, str] = {}
        for key, value in slices.items():
            if not isinstance(key, str) or not key:
                raise ValidationError("slice names must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)):
                raise ValidationError(f"slice value for {key!r} must be scalar")
            clean_slices[key] = str(value)

        return cls(
            record_id=record_id.strip(),
            label=label.strip(),
            probabilities=clean_probabilities,
            slices=clean_slices,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.record_id,
            "label": self.label,
            "probabilities": self.probabilities,
            "slices": self.slices,
        }
