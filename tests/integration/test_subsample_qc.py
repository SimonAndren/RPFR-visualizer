"""Integration tests for QC on the local subsample."""

import pytest

from rpfr_visualizer.io.loader import QM9sDataset


def test_subsample_qc_report(subsample_h5_path):
    if subsample_h5_path is None:
        pytest.skip("Local 1% subsample not available")
    report = QM9sDataset(subsample_h5_path).qc()
    assert report.n_molecules > 0
    sample = next(iter(report.molecules.values()))
    check_ids = {check.check_id for check in sample.checks}
    assert "structure.schema_version" in check_ids
    assert "frequencies.tr_residual" in check_ids
    assert "rpfr.finite_positive" in check_ids
