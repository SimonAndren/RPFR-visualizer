"""Tests for io/validate.py."""

from rpfr_visualizer.io.loader import QM9sDataset


def test_validate_dataset(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    report = ds.validate()
    assert report.n_molecules == 5
    assert report.n_qc_pass == 3
    assert 3 in report.flagged_molecule_ids
    assert 5 in report.flagged_molecule_ids


def test_validate_summary(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    report = ds.validate()
    summary = report.summary()
    assert summary["n_molecules"] == 5
    assert summary["n_qc_fail"] >= 1
    assert "flagged_molecule_ids" in summary
