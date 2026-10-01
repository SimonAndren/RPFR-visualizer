"""Tests for the QC strip helpers used by the explorer."""

from __future__ import annotations

from pathlib import Path

from rpfr_visualizer.io.providers import HDF5Provider
from rpfr_visualizer.viz.qc_strip import run_qc_for_key, stored_qc_flags


def test_stored_qc_flags(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    flags = stored_qc_flags(mol)
    assert isinstance(flags.qc_pass, bool)
    assert "Stored verdict" in "\n".join(flags.format_lines())


def test_run_qc_for_key(synthetic_h5_path: Path) -> None:
    report = run_qc_for_key(synthetic_h5_path, "1")
    assert report.mol_id
    assert report.overall_tier in {"A", "B", "C", "D"}
    assert report.checks


def test_a_geometry_mismatch_is_reported_once_in_words() -> None:
    """It showed as "RDKit could not read the structure (geometry_mismatch)"
    -- the stored status code -- and again as a separate note claiming it was
    not part of the verdict, when it was exactly why the verdict failed."""
    from rpfr_visualizer.viz.qc_strip import StoredQcFlags

    flags = StoredQcFlags(
        qc_pass=False,
        freq_qc_ok=True,
        rpfr_qc_ok=True,
        tr_converged=True,
        rdkit_ok=False,
        smiles_order_match=True,
        n_imag_vib=0,
        rdkit_status="geometry_mismatch",
        geometry_match=False,
    )
    text = "\n".join(flags.format_lines())
    assert "needs checking" in text
    assert text.count("does not match the geometry") == 1
    assert "geometry_mismatch" not in text
    assert "not part of the verdict" not in text
