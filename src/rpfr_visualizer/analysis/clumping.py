"""Clumping factor analysis (Delta = ln C)."""

from __future__ import annotations

import math

import numpy as np

from rpfr_visualizer.analysis.models import AnalysisKind, MoleculeAnalysisResult, SiteAnalysisResult
from rpfr_visualizer.io.loader import Molecule
from rpfr_visualizer.io.providers import MoleculeArrays


def clumping_delta(c_value: float) -> float | None:
    """Compute Delta = ln(C). Returns None for invalid pair factors."""
    if c_value <= 0:
        return None
    if c_value == 1.0:
        return 0.0
    return math.log(c_value)


def _temp_index(temperatures: tuple[float, ...], temperature: float) -> int:
    for idx, temp in enumerate(temperatures):
        if abs(temp - temperature) < 1e-6:
            return idx
    available = ", ".join(str(t) for t in temperatures)
    raise ValueError(f"Temperature {temperature} K is not on the grid. Available: [{available}]")


def compute_clumping(
    molecule: Molecule,
    arrays: MoleculeArrays,
    *,
    temperature: float,
    aggregation: str = "mean",
) -> MoleculeAnalysisResult:
    """Summarize pair clumping per atom as mean Delta over partners.

    Parameters
    ----------
    molecule:
        Loaded molecule record.
    arrays:
        Optional arrays including ``clumping`` of shape (N, N, T).
    temperature:
        Temperature in Kelvin on the dataset grid.
    aggregation:
        ``mean`` or ``max`` over off-diagonal pair factors involving each atom.
    """
    if arrays.clumping is None:
        raise ValueError("Clumping array not available for this molecule")

    clumping = np.asarray(arrays.clumping, dtype=np.float64)
    n_atoms = molecule.n_atoms
    if clumping.shape[0] != n_atoms or clumping.shape[1] != n_atoms:
        raise ValueError(f"Clumping shape {clumping.shape[:2]} does not match n_atoms={n_atoms}")

    temp_idx = _temp_index(molecule.sites[0].temperatures, temperature)
    pair_slice = clumping[:, :, temp_idx]

    sites: list[SiteAnalysisResult] = []
    for atom_idx, site in enumerate(molecule.sites):
        partner_deltas: list[float] = []
        for partner_idx in range(n_atoms):
            if partner_idx == atom_idx:
                continue
            delta = clumping_delta(float(pair_slice[atom_idx, partner_idx]))
            if delta is not None:
                partner_deltas.append(delta)

        if not partner_deltas:
            value: float | None = None
        elif aggregation == "mean":
            value = float(np.mean(partner_deltas))
        elif aggregation == "max":
            value = float(np.max(partner_deltas))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        sites.append(
            SiteAnalysisResult(
                atom_index=site.atom_index,
                element=site.element,
                value=value,
            )
        )

    return MoleculeAnalysisResult(
        molecule_id=molecule.molecule_id,
        kind=AnalysisKind.CLUMPING,
        temperature=temperature,
        sites=sites,
        pair_values=pair_slice,
        metadata={"aggregation": aggregation},
    )


__all__ = ["clumping_delta", "compute_clumping"]
