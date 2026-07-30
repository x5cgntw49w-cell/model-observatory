from __future__ import annotations

import random
from pathlib import Path

from .io import write_jsonl
from .schema import PredictionRecord


CLASSES = ("approve", "review", "reject")


def generate_demo(
    output_dir: str | Path, size: int = 600, seed: int = 17
) -> tuple[Path, Path]:
    if size < 30:
        raise ValueError("size must be at least 30")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(seed)
    reference = [
        _make_record(random_generator, index, shifted=False) for index in range(size)
    ]
    current = [
        _make_record(random_generator, index, shifted=True) for index in range(size)
    ]
    reference_path = destination / "reference.jsonl"
    current_path = destination / "current.jsonl"
    write_jsonl(reference_path, reference)
    write_jsonl(current_path, current)
    return reference_path, current_path


def _make_record(
    random_generator: random.Random, index: int, shifted: bool
) -> PredictionRecord:
    device = random_generator.choices(
        ["desktop", "mobile"], weights=[0.68, 0.32] if not shifted else [0.49, 0.51]
    )[0]
    region = random_generator.choices(
        ["north", "south", "west"],
        weights=[0.42, 0.38, 0.20] if not shifted else [0.31, 0.31, 0.38],
    )[0]
    label = random_generator.choices(CLASSES, weights=[0.46, 0.30, 0.24])[0]

    error_rate = 0.14
    if shifted:
        error_rate += 0.08
    if device == "mobile":
        error_rate += 0.08
    if region == "west" and shifted:
        error_rate += 0.10

    correct = random_generator.random() > error_rate
    prediction = label if correct else random_generator.choice([item for item in CLASSES if item != label])
    base_confidence = random_generator.uniform(0.64, 0.93)
    if shifted and not correct:
        base_confidence = random_generator.uniform(0.68, 0.88)
    remainder = 1.0 - base_confidence
    other_classes = [item for item in CLASSES if item != prediction]
    split = random_generator.uniform(0.25, 0.75)
    probabilities = {
        prediction: base_confidence,
        other_classes[0]: remainder * split,
        other_classes[1]: remainder * (1.0 - split),
    }
    return PredictionRecord(
        record_id=f"{'cur' if shifted else 'ref'}-{index:05d}",
        label=label,
        probabilities=probabilities,
        slices={"device": device, "region": region},
    )
