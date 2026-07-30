from __future__ import annotations

import argparse
from pathlib import Path

from .demo import generate_demo
from .drift import compare_distributions
from .io import load_jsonl
from .metrics import evaluate
from .report import write_report
from .slices import analyze_slices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-observatory",
        description="Evaluate classification quality, calibration, slices, and drift.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate one JSONL dataset")
    evaluate_parser.add_argument("input", type=Path)
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("report"))
    evaluate_parser.add_argument("--min-slice-count", type=int, default=20)

    compare_parser = subparsers.add_parser("compare", help="compare reference and current datasets")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--current", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, default=Path("report"))
    compare_parser.add_argument("--min-slice-count", type=int, default=20)

    demo_parser = subparsers.add_parser("demo", help="generate shifted data and a report")
    demo_parser.add_argument("--output-dir", type=Path, default=Path("demo-output"))
    demo_parser.add_argument("--size", type=int, default=600)
    demo_parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        data_dir = args.output_dir / "data"
        reference_path, current_path = generate_demo(data_dir, args.size, args.seed)
        return _compare(
            reference_path,
            current_path,
            args.output_dir,
            min_slice_count=max(5, args.size // 20),
        )
    if args.command == "evaluate":
        records = load_jsonl(args.input)
        summary = {
            "current": evaluate(records),
            "slices": analyze_slices(records, args.min_slice_count),
        }
        html_path, json_path = write_report(args.output_dir, summary)
        print(f"report: {html_path}")
        print(f"summary: {json_path}")
        return 0
    return _compare(
        args.reference,
        args.current,
        args.output_dir,
        args.min_slice_count,
    )


def _compare(
    reference_path: Path,
    current_path: Path,
    output_dir: Path,
    min_slice_count: int,
) -> int:
    reference = load_jsonl(reference_path)
    current = load_jsonl(current_path)
    summary = {
        "reference": evaluate(reference),
        "current": evaluate(current),
        "drift": compare_distributions(reference, current),
        "slices": analyze_slices(current, min_slice_count),
    }
    html_path, json_path = write_report(output_dir, summary)
    print(f"report: {html_path}")
    print(f"summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
