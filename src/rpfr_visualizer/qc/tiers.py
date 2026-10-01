"""Derive overall tiers and use-case tags from check results."""

from __future__ import annotations

from rpfr_visualizer.qc.models import CheckResult, MoleculeQC, OverallTier

SITE_LABELED_TAGS = ("site_labeled_analysis", "local_env", "ml_features")
RPFR_SITE_TAGS = ("rpfr_site", "clumping")
ALL_USABLE_TAGS = (
    "rpfr_bulk",
    "rpfr_site",
    "clumping",
    "local_env",
    "ml_features",
    "site_labeled_analysis",
)

STATUS_SEVERITY = {"pass": 0, "info": 1, "warn": 2, "fail": 3}


def _check(checks: list[CheckResult], check_id: str) -> CheckResult | None:
    for check in checks:
        if check.check_id == check_id:
            return check
    return None


def _status(checks: list[CheckResult], check_id: str) -> str:
    found = _check(checks, check_id)
    return found.status if found else "pass"


def _failed(check: CheckResult | None) -> bool:
    return check is not None and check.status == "fail"


def derive_overall_tier(checks: list[CheckResult]) -> OverallTier:
    """Map check severities to an overall A/B/C/D tier."""
    max_severity = max((STATUS_SEVERITY.get(check.status, 0) for check in checks), default=0)
    if max_severity >= STATUS_SEVERITY["fail"]:
        return "D"
    if max_severity >= STATUS_SEVERITY["warn"]:
        return "C"
    if _status(checks, "assignment.smiles_order_match") == "fail":
        return "B"
    if max_severity >= STATUS_SEVERITY["info"]:
        return "B"
    return "A"


def derive_usable_for(checks: list[CheckResult]) -> tuple[list[str], list[str]]:
    """Derive usable_for and exclude_from tags from check results."""
    usable: set[str] = set()
    excluded: set[str] = set()

    structure_ok = all(
        _status(checks, check_id) != "fail"
        for check_id in (
            "structure.schema_version",
            "structure.required_datasets",
            "structure.shape_consistency",
            "structure.finite_arrays",
        )
    )
    rpfr_ok = _status(checks, "rpfr.finite_positive") != "fail"

    if structure_ok and rpfr_ok:
        usable.update(("rpfr_bulk", "rpfr_site"))

    imag_worst = _check(checks, "frequencies.imaginary_vib_worst")
    imag_ok = True
    if imag_worst and imag_worst.value is not None:
        imag_ok = float(imag_worst.value) >= -50.0
    elif _status(checks, "frequencies.imaginary_vib_count") == "fail":
        imag_ok = False
    if imag_ok:
        usable.add("clumping")
    else:
        excluded.update(("clumping", "rpfr_site"))

    smiles_match = _status(checks, "assignment.smiles_order_match") == "pass"
    rdkit_ok = _status(checks, "annotation.rdkit_parse") in {"pass", "warn"}
    if smiles_match and rdkit_ok:
        usable.update(("local_env", "ml_features", "site_labeled_analysis"))
    elif _failed(_check(checks, "assignment.smiles_order_match")):
        excluded.update(SITE_LABELED_TAGS)

    tr_status = _status(checks, "frequencies.tr_residual")
    if tr_status == "warn" and "rpfr_site" in usable:
        pass

    return sorted(usable), sorted(excluded)


def build_summary(checks: list[CheckResult], overall_tier: OverallTier) -> str:
    """Build a one-line summary for UI/export."""
    notable: list[str] = []
    for check_id in (
        "frequencies.tr_residual",
        "frequencies.imaginary_vib_worst",
        "assignment.smiles_order_match",
        "rpfr.finite_positive",
    ):
        check = _check(checks, check_id)
        if check and check.status in {"warn", "fail"}:
            notable.append(check.message)
    if notable:
        return f"Tier {overall_tier}: {notable[0]}"
    return f"Tier {overall_tier}: all critical checks passed"


def finalize_molecule_qc(mol_id: str, checks: list[CheckResult]) -> MoleculeQC:
    """Apply tier and use-case rules to a list of checks."""
    overall_tier = derive_overall_tier(checks)
    usable_for, exclude_from = derive_usable_for(checks)

    if _failed(_check(checks, "assignment.smiles_order_match")) and overall_tier == "A":
        overall_tier = "B"

    imag_worst = _check(checks, "frequencies.imaginary_vib_worst")
    if imag_worst and imag_worst.value is not None and float(imag_worst.value) < -50.0:
        if overall_tier not in {"D"}:
            overall_tier = "D"
        exclude_from = sorted(set(exclude_from) | set(RPFR_SITE_TAGS))

    return MoleculeQC(
        mol_id=mol_id,
        overall_tier=overall_tier,
        checks=checks,
        usable_for=usable_for,
        exclude_from=exclude_from,
        summary=build_summary(checks, overall_tier),
    )
