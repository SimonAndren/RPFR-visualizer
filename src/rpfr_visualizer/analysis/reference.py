"""Reference frames for isotope enrichment comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rpfr_visualizer.io.loader import Molecule

if TYPE_CHECKING:
    from rpfr_visualizer.analysis.element_stats import ElementStatsCache


class ReferenceFrame(ABC):
    """Resolve a baseline beta for comparing enrichment within an element."""

    @abstractmethod
    def resolve_beta(
        self,
        *,
        target_molecule: Molecule,
        target_atom_index: int,
        element: str,
        temperature: float,
        method: str | None = None,
    ) -> float | None:
        """Return reference beta, or None if undefined (e.g. cross-element).

        *method* names the off-grid interpolant. It must reach here: the
        target and the reference are divided by one another, so reading them
        with different interpolants puts the difference between two fits into
        a number meant to carry a physical one.
        """


@dataclass(frozen=True)
class SelfReference(ReferenceFrame):
    """Compare each site to a baseline site within the same molecule."""

    anchor_atom_index: int

    def resolve_beta(
        self,
        *,
        target_molecule: Molecule,
        target_atom_index: int,
        element: str,
        temperature: float,
        method: str | None = None,
    ) -> float | None:
        del target_atom_index
        anchor = target_molecule.sites[self.anchor_atom_index]
        if anchor.element != element:
            return None
        return anchor.beta_at(temperature, method=method)


@dataclass(frozen=True)
class MoleculeReference(ReferenceFrame):
    """Compare to a site in another molecule (possibly another dataset)."""

    reference_molecule: Molecule
    reference_atom_index: int

    def resolve_beta(
        self,
        *,
        target_molecule: Molecule,
        target_atom_index: int,
        element: str,
        temperature: float,
        method: str | None = None,
    ) -> float | None:
        del target_molecule, target_atom_index
        ref_site = self.reference_molecule.sites[self.reference_atom_index]
        if ref_site.element != element:
            return None
        return ref_site.beta_at(temperature, method=method)


@dataclass(frozen=True)
class ElementMeanReference(ReferenceFrame):
    """Compare each site to the mean beta of same-element sites in the molecule."""

    def resolve_beta(
        self,
        *,
        target_molecule: Molecule,
        target_atom_index: int,
        element: str,
        temperature: float,
        method: str | None = None,
    ) -> float | None:
        del target_atom_index
        values = [
            site.beta_at(temperature, method=method)
            for site in target_molecule.sites
            if site.element == element
        ]
        if not values:
            return None
        return sum(values) / len(values)


@dataclass(frozen=True)
class StandardReference(ReferenceFrame):
    """Compare to a constant β per element and/or dataset median β(T).

    Lookup order:
    1. ``beta_by_element`` constant override (UI-editable session value)
    2. dataset median β(T) from ``stats`` when provided
    """

    beta_by_element: dict[str, float] = field(default_factory=dict)
    stats: ElementStatsCache | None = None

    def resolve_beta(
        self,
        *,
        target_molecule: Molecule,
        target_atom_index: int,
        element: str,
        temperature: float,
        method: str | None = None,
    ) -> float | None:
        del target_molecule, target_atom_index
        if element in self.beta_by_element:
            return float(self.beta_by_element[element])
        if self.stats is not None:
            return self.stats.median_beta(element, temperature, warn=False, method=method)
        return None

    def with_override(self, element: str, beta: float) -> StandardReference:
        """Return a copy with an edited constant β for *element*."""
        merged = dict(self.beta_by_element)
        merged[element] = float(beta)
        return StandardReference(beta_by_element=merged, stats=self.stats)

    def without_override(self, element: str) -> StandardReference:
        """Return a copy that falls back to dataset median for *element*."""
        merged = dict(self.beta_by_element)
        merged.pop(element, None)
        return StandardReference(beta_by_element=merged, stats=self.stats)


__all__ = [
    "ElementMeanReference",
    "MoleculeReference",
    "ReferenceFrame",
    "SelfReference",
    "StandardReference",
]
