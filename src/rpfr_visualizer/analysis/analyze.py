"""Central dispatcher for visualizer analysis requests."""

from __future__ import annotations

from rpfr_visualizer.analysis.clumping import compute_clumping
from rpfr_visualizer.analysis.fractionation import run_analysis as run_fractionation
from rpfr_visualizer.analysis.mass_law import compute_mass_law
from rpfr_visualizer.analysis.models import AnalysisKind, AnalysisRequest, MoleculeAnalysisResult
from rpfr_visualizer.io.loader import Molecule
from rpfr_visualizer.io.providers import MoleculeArrays


def run_molecule_analysis(
    molecule: Molecule,
    request: AnalysisRequest,
    *,
    arrays: MoleculeArrays | None = None,
) -> MoleculeAnalysisResult:
    """Run the requested analysis on a molecule."""
    if request.kind == AnalysisKind.CLUMPING:
        if arrays is None:
            raise ValueError("Clumping analysis requires MoleculeArrays")
        return compute_clumping(
            molecule,
            arrays,
            temperature=request.temperature,
            aggregation=request.aggregation,
        )
    if request.kind == AnalysisKind.MASS_LAW:
        if arrays is None:
            raise ValueError("Mass-law analysis requires MoleculeArrays")
        return compute_mass_law(molecule, arrays, temperature=request.temperature)
    return run_fractionation(molecule, request)


__all__ = ["run_molecule_analysis"]
