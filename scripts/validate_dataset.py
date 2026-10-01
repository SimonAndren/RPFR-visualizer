#!/usr/bin/env python3
"""Validate a QM9s-RPFR HDF5 dataset and optionally write QC reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to an HDF5 dataset file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for qc_report.json, qc_summary.csv, and qc_thresholds.json",
    )
    parser.add_argument(
        "--format",
        default="json,csv",
        help="Comma-separated export formats: json, csv (default: json,csv)",
    )
    parser.add_argument(
        "--fail-on-tier-d",
        action="store_true",
        help="Exit with code 1 when any molecule is tier D",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run dataset QC from the command line."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.path.exists():
        parser.error(f"HDF5 file not found: {args.path}")

    from rpfr_visualizer.qc.engine import run_dataset_qc
    from rpfr_visualizer.qc.export import export_with_qc

    report = run_dataset_qc(args.path)

    print(f"Dataset: {args.path}")
    print(f"Molecules: {report.n_molecules}")
    print("Tier histogram:")
    for tier in ("A", "B", "C", "D"):
        print(f"  {tier}: {report.tier_histogram.get(tier, 0)}")

    if report.failure_counts:
        print("Top failure/warn checks:")
        for check_id, count in sorted(
            report.failure_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]:
            print(f"  {check_id}: {count}")

    formats = {item.strip().lower() for item in args.format.split(",") if item.strip()}
    if args.output_dir is not None and formats:
        paths = export_with_qc(args.path, args.output_dir, dataset_qc=report)
        print("Wrote:")
        for name, path in paths.items():
            if (
                (name == "qc_report" and "json" in formats)
                or (name == "qc_summary" and "csv" in formats)
                or name == "qc_thresholds"
            ):
                print(f"  {path}")

    if args.fail_on_tier_d and report.tier_histogram.get("D", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
