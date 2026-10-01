"""What the explorer is currently showing.

All user-settable options live here as plain data. The controller reads this
state; the widgets write to it. Keeping it in one dataclass is what lets the
analysis logic be tested without constructing any UI, and what stops settings
from hiding in widget closures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.analysis.temperature import DEFAULT_METHOD

#: How alpha/epsilon are referenced.
#:
#: ``none`` and ``element_mean`` both use the in-molecule element mean; ``none``
#: exists because beta needs no reference at all.
ReferenceMode = Literal["none", "element_mean", "standard", "mol_b", "self"]

#: Elements carrying RPFR data in these datasets.
ELEMENTS: tuple[str, ...] = ("H", "C", "N", "O")


@dataclass(frozen=True)
class LoadedMolecule:
    """A molecule pulled from a dataset, with its optional array data."""

    dataset: str
    key: str
    molecule: Any
    arrays: Any = None

    @property
    def label(self) -> str:
        return f"{self.dataset}:{self.key}"


@dataclass
class ExplorerState:
    """Everything the user has chosen. No widgets, no data, no I/O."""

    dataset: str = ""

    # Selection
    mol_a: LoadedMolecule | None = None
    mol_b: LoadedMolecule | None = None
    ref_atom_index_b: int = 0
    anchor_atom_index: int = 0

    # What to compute
    kind: AnalysisKind = AnalysisKind.BETA
    temperature: float = 298.15
    #: Which interpolant answers an off-grid temperature. One setting for the
    #: whole page: the β curve, the atom colours and the CSV all read β at the
    #: same temperature, and showing two different numbers for it would be a
    #: bug rather than a choice. Ignored at a stored temperature.
    beta_method: str = DEFAULT_METHOD
    element: str | None = "C"
    reference_mode: ReferenceMode = "element_mean"
    standard_overrides: dict[str, float] = field(default_factory=dict)

    # How to draw it
    show_hydrogens: bool = True
    #: Draw the atom identity (C1, O3) on each focus atom. Not the value --
    #: colour carries that, and printing both crowded the picture.
    show_labels: bool = True
    use_3d: bool = True
    use_rdkit_2d: bool = True

    # Derived properties (the temperature / clumping / mass-law panel)
    #: Sites drawn on the curves. Empty means "choose for me" -- the strongest
    #: few under the current element lock, since every atom at once is
    #: unreadable on a log axis spanning twenty orders of magnitude.
    derived_sites: tuple[int, ...] = ()
    #: Atom pair highlighted from the clumping matrix, as (i, j).
    highlight_pair: tuple[int, int] | None = None
    #: How a clumping matrix collapses per atom: "mean" or "max".
    derived_aggregation: str = "mean"

    #: Temperatures the CSV export covers. Empty means "just the one on
    #: screen". A separate field rather than a reuse of *temperature*: the
    #: export is the one place where asking for a dozen temperatures at once
    #: is the normal thing to want, and the figure can only show one.
    download_temperatures: tuple[float, ...] = ()

    # Ranking
    top_n: int = 15
    heavy_only: bool = True
    use_match_pool: bool = True

    # Functional-group filters
    fg_preset: str | None = None
    fg_text: str = ""
    fg_observed_r1: str | None = None
    max_fg_hits: int = 200

    def temperature_is_on_grid(self, grid: Sequence[float], *, tol: float = 1e-6) -> bool:
        """True when the chosen temperature is a stored grid point.

        Off-grid temperatures are interpolated rather than read, which is a
        provenance distinction the UI surfaces.
        """
        return any(abs(float(t) - float(self.temperature)) <= tol for t in grid)

    def needs_reference(self) -> bool:
        """True when the chosen quantity is defined against a reference."""
        return self.kind != AnalysisKind.BETA

    def describe_quantity(self) -> str:
        """Short human label for the current quantity and temperature."""
        return f"{self.kind.value} @ {self.temperature:g} K"


__all__ = ["ELEMENTS", "ExplorerState", "LoadedMolecule", "ReferenceMode"]
