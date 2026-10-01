"""Mass-law exponent analysis from per-site lambda arrays."""

from __future__ import annotations

import numpy as np

from rpfr_visualizer.analysis.clumping import _temp_index
from rpfr_visualizer.analysis.models import AnalysisKind, MoleculeAnalysisResult, SiteAnalysisResult
from rpfr_visualizer.io.loader import Molecule
from rpfr_visualizer.io.providers import MoleculeArrays


def compute_mass_law(
    molecule: Molecule,
    arrays: MoleculeArrays,
    *,
    temperature: float,
) -> MoleculeAnalysisResult:
    """Return per-site mass-law exponent lambda at a grid temperature."""
    if arrays.mass_law is None:
        raise ValueError("mass_law array not available for this molecule")

    mass_law = np.asarray(arrays.mass_law, dtype=np.float64)
    if mass_law.shape[0] != molecule.n_atoms:
        raise ValueError(
            f"mass_law shape {mass_law.shape} does not match n_atoms={molecule.n_atoms}"
        )

    temp_idx = _temp_index(molecule.sites[0].temperatures, temperature)
    sites: list[SiteAnalysisResult] = []
    for site in molecule.sites:
        value = float(mass_law[site.atom_index, temp_idx])
        sites.append(
            SiteAnalysisResult(
                atom_index=site.atom_index,
                element=site.element,
                value=value,
            )
        )

    return MoleculeAnalysisResult(
        molecule_id=molecule.molecule_id,
        kind=AnalysisKind.MASS_LAW,
        temperature=temperature,
        sites=sites,
    )


__all__ = ["compute_mass_law"]
