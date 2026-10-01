"""Tests for qc/engine.py."""

import h5py

from rpfr_visualizer.io.loader import QM9sDataset
from rpfr_visualizer.qc.engine import run_dataset_qc


def test_run_dataset_qc_covers_all_molecules(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    report = ds.qc()
    assert report.n_molecules == 5
    assert set(report.molecules) == {"1", "2", "3", "4", "5"}
    for mol_qc in report.molecules.values():
        assert mol_qc.checks
        assert mol_qc.overall_tier in {"A", "B", "C", "D"}


def test_tr_warn_molecule_still_present(synthetic_h5_path):
    report = QM9sDataset(synthetic_h5_path).qc()
    mol4 = report.molecules["4"]
    tr_check = mol4.check_by_id("frequencies.tr_residual")
    assert tr_check is not None
    assert tr_check.status == "warn"
    assert mol4.overall_tier in {"B", "C", "D"}


def test_failed_qc_molecule_is_tier_d(synthetic_h5_path):
    report = QM9sDataset(synthetic_h5_path).qc()
    mol3 = report.molecules["3"]
    assert mol3.overall_tier == "D"


def test_run_dataset_qc_ignores_non_molecule_groups(synthetic_h5_path):
    """Non-numeric top-level groups must not break molecule iteration."""
    with h5py.File(synthetic_h5_path, "a") as h5:
        if "metadata" not in h5:
            h5.create_group("metadata")

    report = run_dataset_qc(synthetic_h5_path)
    assert report.n_molecules == 5
    assert "metadata" not in report.molecules
