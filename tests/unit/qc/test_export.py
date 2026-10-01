"""Tests for qc/export.py."""

import json
from pathlib import Path

from rpfr_visualizer.qc.export import export_with_qc


def test_export_with_qc_writes_bundle(synthetic_h5_path, tmp_path):
    paths = export_with_qc(synthetic_h5_path, tmp_path)
    assert paths["qc_report"].exists()
    assert paths["qc_summary"].exists()
    assert paths["qc_thresholds"].exists()

    with paths["qc_report"].open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["n_molecules"] == 5
    assert "1" in payload["molecules"]
