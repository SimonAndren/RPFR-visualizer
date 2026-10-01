"""Tests for qc/tiers.py."""

from rpfr_visualizer.qc.models import CheckResult
from rpfr_visualizer.qc.tiers import derive_usable_for, finalize_molecule_qc


def _check(check_id: str, status: str, *, value=None, message: str = "") -> CheckResult:
    category = check_id.split(".", maxsplit=1)[0]
    return CheckResult(
        check_id=check_id,
        category=category,
        status=status,
        value=value,
        threshold=None,
        message=message or check_id,
        source="computed",
    )


def test_smiles_order_match_excludes_site_labeled():
    checks = [
        _check("structure.finite_arrays", "pass"),
        _check("rpfr.finite_positive", "pass"),
        _check("assignment.smiles_order_match", "fail"),
        _check("annotation.rdkit_parse", "pass"),
    ]
    mol_qc = finalize_molecule_qc("5", checks)
    assert "site_labeled_analysis" in mol_qc.exclude_from
    assert "local_env" in mol_qc.exclude_from
    assert "ml_features" in mol_qc.exclude_from
    assert "rpfr_bulk" in mol_qc.usable_for


def test_tr_warn_still_allows_rpfr_bulk():
    checks = [
        _check("structure.finite_arrays", "pass"),
        _check("rpfr.finite_positive", "pass"),
        _check("frequencies.tr_residual", "warn", value=23.0),
        _check("assignment.smiles_order_match", "pass"),
    ]
    usable, excluded = derive_usable_for(checks)
    assert "rpfr_bulk" in usable
    assert "site_labeled_analysis" not in excluded
