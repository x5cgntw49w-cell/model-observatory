from __future__ import annotations

import json

from model_observatory.cli import main
from model_observatory.demo import generate_demo
from model_observatory.io import load_jsonl


def test_demo_generation_is_deterministic(tmp_path) -> None:
    first_reference, first_current = generate_demo(tmp_path / "first", size=40, seed=9)
    second_reference, second_current = generate_demo(tmp_path / "second", size=40, seed=9)
    assert first_reference.read_text() == second_reference.read_text()
    assert first_current.read_text() == second_current.read_text()


def test_demo_contains_expected_slices(tmp_path) -> None:
    reference_path, _ = generate_demo(tmp_path, size=40)
    record = load_jsonl(reference_path)[0]
    assert set(record.slices) == {"device", "region"}


def test_demo_cli_writes_report_and_summary(tmp_path) -> None:
    output = tmp_path / "output"
    result = main(["demo", "--output-dir", str(output), "--size", "60", "--seed", "4"])
    assert result == 0
    assert (output / "report.html").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["reference"]["count"] == 60
    assert summary["current"]["count"] == 60
    assert "confidence_psi" in summary["drift"]


def test_report_escapes_slice_values(tmp_path) -> None:
    output = tmp_path / "output"
    main(["demo", "--output-dir", str(output), "--size", "60"])
    report = (output / "report.html").read_text()
    assert "Model Observatory Report" in report
    assert "Worst-first slice analysis" in report
    assert "Generated deterministically" in report
