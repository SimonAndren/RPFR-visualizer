"""Write QC bundles alongside dataset exports."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd

from rpfr_visualizer.qc.engine import run_dataset_qc
from rpfr_visualizer.qc.models import DatasetQC
from rpfr_visualizer.qc.thresholds import QcThresholds, load_thresholds


def export_with_qc(
    path: Path | str,
    output_dir: Path | str,
    *,
    thresholds: QcThresholds | None = None,
    dataset_qc: DatasetQC | None = None,
) -> dict[str, Path]:
    """Write qc_report.json, qc_summary.csv, and qc_thresholds.json."""
    dataset_path = Path(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = dataset_qc or run_dataset_qc(dataset_path, thresholds=thresholds)
    if thresholds is None:
        with h5py.File(dataset_path, "r") as h5:
            resolved_thresholds = load_thresholds(h5)
    else:
        resolved_thresholds = thresholds

    report_path = out / "qc_report.json"
    summary_path = out / "qc_summary.csv"
    thresholds_path = out / "qc_thresholds.json"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)

    rows = [report.summary_row(mol_id) for mol_id in sorted(report.molecules, key=int)]
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    with thresholds_path.open("w", encoding="utf-8") as handle:
        json.dump(resolved_thresholds.to_dict(), handle, indent=2, sort_keys=True)

    return {
        "qc_report": report_path,
        "qc_summary": summary_path,
        "qc_thresholds": thresholds_path,
    }
