"""Fractionation factor calculations from RPFR beta values."""

from __future__ import annotations

import math

from rpfr_visualizer.analysis.models import (
    AnalysisKind,
    AnalysisRequest,
    MoleculeAnalysisResult,
    SiteAnalysisResult,
)
from rpfr_visualizer.analysis.reference import ReferenceFrame
from rpfr_visualizer.io.loader import Molecule


def alpha(beta_target: float, beta_ref: float) -> float:
    """Compute fractionation factor alpha = beta_target / beta_ref."""
    if beta_ref == 0:
        raise ValueError("Reference beta must be non-zero")
    return beta_target / beta_ref


def epsilon_per_mil(alpha_value: float) -> float:
    """Compute epsilon in per mil: 1000 * (alpha - 1)."""
    return 1000.0 * (alpha_value - 1.0)


def ln_alpha(alpha_value: float) -> float:
    """Compute 1000 * ln(alpha), common in isotope geochemistry."""
    if alpha_value <= 0:
        raise ValueError("alpha must be positive for ln(alpha)")
    return 1000.0 * math.log(alpha_value)


def compute_fractionation(
    molecule: Molecule,
    *,
    kind: AnalysisKind,
    temperature: float,
    reference: ReferenceFrame | None = None,
    element: str | None = None,
    method: str | None = None,
) -> MoleculeAnalysisResult:
    """Compute per-site fractionation relative to a reference frame.

    *method* names the off-grid β interpolant and is used for both the target
    site and its reference, so α is a ratio of two comparable numbers.
    """
    if kind == AnalysisKind.BETA:
        return _beta_result(molecule, temperature=temperature, element=element, method=method)
    if reference is None:
        raise ValueError(f"Reference frame required for {kind.value}")

    sites: list[SiteAnalysisResult] = []
    for site in molecule.sites:
        if element is not None and site.element != element:
            continue
        beta_target = site.beta_at(temperature, method=method)
        beta_ref = reference.resolve_beta(
            target_molecule=molecule,
            target_atom_index=site.atom_index,
            element=site.element,
            temperature=temperature,
            method=method,
        )
        if beta_ref is None:
            sites.append(
                SiteAnalysisResult(
                    atom_index=site.atom_index,
                    element=site.element,
                    value=None,
                    reference_value=None,
                )
            )
            continue

        alpha_val = alpha(beta_target, beta_ref)
        if kind == AnalysisKind.ALPHA:
            value: float | None = alpha_val
        elif kind == AnalysisKind.EPSILON:
            value = epsilon_per_mil(alpha_val)
        elif kind == AnalysisKind.LN_ALPHA:
            value = ln_alpha(alpha_val)
        else:
            raise ValueError(f"Unsupported fractionation kind: {kind}")

        sites.append(
            SiteAnalysisResult(
                atom_index=site.atom_index,
                element=site.element,
                value=value,
                reference_value=beta_ref,
            )
        )

    return MoleculeAnalysisResult(
        molecule_id=molecule.molecule_id,
        kind=kind,
        temperature=temperature,
        sites=sites,
    )


def run_analysis(molecule: Molecule, request: AnalysisRequest) -> MoleculeAnalysisResult:
    """Dispatch an :class:`AnalysisRequest` to the appropriate calculator."""
    if request.kind in {
        AnalysisKind.BETA,
        AnalysisKind.ALPHA,
        AnalysisKind.EPSILON,
        AnalysisKind.LN_ALPHA,
    }:
        return compute_fractionation(
            molecule,
            kind=request.kind,
            temperature=request.temperature,
            reference=request.reference,
            element=request.element,
            method=request.beta_method,
        )
    raise ValueError(f"Use clumping or mass-law modules for {request.kind.value}")


def _beta_result(
    molecule: Molecule,
    *,
    temperature: float,
    element: str | None,
    method: str | None = None,
) -> MoleculeAnalysisResult:
    sites = []
    for site in molecule.sites:
        if element is not None and site.element != element:
            continue
        sites.append(
            SiteAnalysisResult(
                atom_index=site.atom_index,
                element=site.element,
                value=site.beta_at(temperature, method=method),
            )
        )
    return MoleculeAnalysisResult(
        molecule_id=molecule.molecule_id,
        kind=AnalysisKind.BETA,
        temperature=temperature,
        sites=sites,
    )


__all__ = [
    "alpha",
    "compute_fractionation",
    "epsilon_per_mil",
    "ln_alpha",
    "run_analysis",
]
