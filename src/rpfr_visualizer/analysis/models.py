"""Analysis request and result models for the visualizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from rpfr_visualizer.analysis.reference import ReferenceFrame


class AnalysisKind(str, Enum):
    """Supported isotope analysis quantities."""

    BETA = "beta"
    ALPHA = "alpha"
    EPSILON = "epsilon"
    LN_ALPHA = "ln_alpha"
    CLUMPING = "clumping"
    MASS_LAW = "mass_law"


@dataclass(frozen=True)
class AnalysisRequest:
    """Declarative description of an analysis to run."""

    kind: AnalysisKind
    temperature: float = 300.0
    reference: ReferenceFrame | None = None
    element: str | None = None
    #: How a clumping matrix collapses to one value per atom: ``mean`` or
    #: ``max`` over that atom's partners. Ignored by every other kind.
    #: compute_clumping has always supported both; without this a caller going
    #: through the dispatcher had no way to pick.
    aggregation: str = "mean"
    #: Which interpolant answers an off-grid *temperature*; None is the
    #: package default. Ignored at a stored temperature, which is read.
    beta_method: str | None = None


@dataclass
class SiteAnalysisResult:
    """Per-site analysis output."""

    atom_index: int
    element: str
    value: float | None
    reference_value: float | None = None


@dataclass
class MoleculeAnalysisResult:
    """Analysis output for one molecule."""

    molecule_id: int
    kind: AnalysisKind
    temperature: float
    sites: list[SiteAnalysisResult] = field(default_factory=list)
    pair_values: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def values_by_index(self) -> dict[int, float | None]:
        """Map atom index to analysis value."""
        return {site.atom_index: site.value for site in self.sites}

    def per_atom_array(self, n_atoms: int) -> np.ndarray:
        """Return a length-*n_atoms* array with NaN for undefined sites."""
        out = np.full(n_atoms, np.nan, dtype=np.float64)
        for site in self.sites:
            if site.value is not None:
                out[site.atom_index] = site.value
        return out
