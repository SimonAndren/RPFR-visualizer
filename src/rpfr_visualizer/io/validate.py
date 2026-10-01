"""Quality control validation for QM9s-RPFR datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rpfr_visualizer.io.schema import HEAVY_ELEMENTS

if TYPE_CHECKING:
    from rpfr_visualizer.io.loader import QM9sDataset


@dataclass
class ValidationIssue:
    """A single QC issue for a molecule."""

    molecule_id: int
    issue_type: str
    message: str


@dataclass
class ValidationReport:
    """Aggregated QC report for a dataset."""

    n_molecules: int
    n_qc_pass: int
    n_qc_fail: int
    n_nan_beta: int
    n_unphysical_beta: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def qc_pass_rate(self) -> float:
        if self.n_molecules == 0:
            return 0.0
        return self.n_qc_pass / self.n_molecules

    @property
    def flagged_molecule_ids(self) -> list[int]:
        return sorted({issue.molecule_id for issue in self.issues})

    def summary(self) -> dict[str, int | float | list[int]]:
        """Return a JSON-serializable summary."""
        return {
            "n_molecules": self.n_molecules,
            "n_qc_pass": self.n_qc_pass,
            "n_qc_fail": self.n_qc_fail,
            "qc_pass_rate": self.qc_pass_rate,
            "n_nan_beta": self.n_nan_beta,
            "n_unphysical_beta": self.n_unphysical_beta,
            "flagged_molecule_ids": self.flagged_molecule_ids,
        }


def derive_qc_pass(
    *,
    freq_qc_ok: bool,
    rpfr_qc_ok: bool,
    tr_converged: bool,
    rdkit_ok: bool,
    smiles_order_match: bool,
    n_imag_vib: int,
) -> bool:
    """Derive a single QC pass flag from molecule attributes."""
    return (
        freq_qc_ok
        and rpfr_qc_ok
        and tr_converged
        and rdkit_ok
        and smiles_order_match
        and n_imag_vib == 0
    )


def _beta_has_nan_or_inf(beta: tuple[float, ...]) -> bool:
    return any(not math.isfinite(v) for v in beta)


def _beta_is_unphysical(element: str, beta: tuple[float, ...]) -> bool:
    """Flag unphysical beta values (heavy elements should have beta >= 1)."""
    if element in HEAVY_ELEMENTS:
        return any(v < 1.0 for v in beta if math.isfinite(v))
    # Hydrogen RPFR values are typically >> 1; flag non-positive values.
    return any(v <= 0.0 for v in beta if math.isfinite(v))


def validate_molecule(mol) -> list[ValidationIssue]:
    """Validate a single molecule and return any issues found."""
    issues: list[ValidationIssue] = []

    if not mol.qc_pass:
        reasons = []
        if not mol.freq_qc_ok:
            reasons.append("freq_qc_ok=False")
        if not mol.rpfr_qc_ok:
            reasons.append("rpfr_qc_ok=False")
        if not mol.tr_converged:
            reasons.append("tr_converged=False")
        if not mol.rdkit_ok:
            reasons.append("rdkit_ok=False")
        if not mol.smiles_order_match:
            reasons.append("smiles_order_match=False")
        if mol.n_imag_vib != 0:
            reasons.append(f"n_imag_vib={mol.n_imag_vib}")
        issues.append(
            ValidationIssue(
                molecule_id=mol.molecule_id,
                issue_type="qc_flag",
                message="; ".join(reasons),
            )
        )

    for site in mol.sites:
        if _beta_has_nan_or_inf(site.beta):
            issues.append(
                ValidationIssue(
                    molecule_id=mol.molecule_id,
                    issue_type="nan_beta",
                    message=f"NaN/Inf beta at atom_index={site.atom_index}",
                )
            )
        elif _beta_is_unphysical(site.element, site.beta):
            issues.append(
                ValidationIssue(
                    molecule_id=mol.molecule_id,
                    issue_type="unphysical_beta",
                    message=(
                        f"Unphysical beta for {site.element} at "
                        f"atom_index={site.atom_index}: min={min(site.beta):.4f}"
                    ),
                )
            )

    return issues


def validate_dataset(dataset: QM9sDataset) -> ValidationReport:
    """Validate all molecules in a dataset and return an aggregated report."""
    n_qc_pass = 0
    n_nan_beta = 0
    n_unphysical_beta = 0
    all_issues: list[ValidationIssue] = []

    for mol in dataset:
        if mol.qc_pass:
            n_qc_pass += 1
        mol_issues = validate_molecule(mol)
        for issue in mol_issues:
            if issue.issue_type == "nan_beta":
                n_nan_beta += 1
            elif issue.issue_type == "unphysical_beta":
                n_unphysical_beta += 1
        all_issues.extend(mol_issues)

    n_molecules = len(dataset)
    molecules_with_issues = {issue.molecule_id for issue in all_issues}
    n_qc_fail = len(molecules_with_issues)

    return ValidationReport(
        n_molecules=n_molecules,
        n_qc_pass=n_qc_pass,
        n_qc_fail=n_qc_fail,
        n_nan_beta=n_nan_beta,
        n_unphysical_beta=n_unphysical_beta,
        issues=all_issues,
    )
