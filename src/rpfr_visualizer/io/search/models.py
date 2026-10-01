"""Records returned by molecule search.

These live below both :mod:`rpfr_visualizer.io.molecule_index` and the search backends
so the backends never have to import the facade that wraps them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MoleculeIndexEntry:
    """One searchable molecule record."""

    dataset: str
    key: str
    smiles: str
    canonical_smiles: str
    n_atoms: int
    label: str
    formula: str = ""
    inchikey: str = ""
    element_counts: dict[str, int] = field(default_factory=dict)
    names: tuple[str, ...] = ()
    #: Stored QC verdict. ``None`` means the index has no verdict for this
    #: molecule, which is distinct from a recorded failure.
    qc_pass: bool | None = None


@dataclass(frozen=True)
class RowFilter:
    """Constraints AND'd onto a query rather than OR'd with it.

    Comma-separated search tokens are alternatives; these are restrictions.
    ``fg:alcohol, C:5-9, qc:pass`` means alcohols *that also* have five to nine
    carbons *and also* passed QC. Everything here is pushed into SQL rather
    than applied to the returned page, so a limited query still reports the
    first *n* molecules satisfying every part.

    *element_counts* maps an element to an inclusive ``(min, max)`` range;
    *column_counts* does the same for ``n_heavy`` / ``n_atoms``. Both are
    tuples rather than dicts so the filter stays hashable.

    *qc_pass* selects on the stored QC verdict: ``True`` keeps molecules that
    passed, ``False`` keeps recorded failures, and ``None`` does not filter.
    Molecules with no stored verdict are excluded by either setting — "not
    known to have failed" is not the same claim as "passed".
    """

    element_counts: tuple[tuple[str, int, int], ...] = ()
    column_counts: tuple[tuple[str, int, int], ...] = ()
    qc_pass: bool | None = None

    def __bool__(self) -> bool:
        return bool(self.element_counts or self.column_counts or self.qc_pass is not None)

    def merge(self, other: RowFilter) -> RowFilter:
        """Combine two filters by ANDing their constraints.

        A later QC constraint wins over an earlier one; two contradictory ones
        in a single query would otherwise have to return nothing, which reads
        as a bug rather than as an answer.
        """
        return RowFilter(
            element_counts=self.element_counts + other.element_counts,
            column_counts=self.column_counts + other.column_counts,
            qc_pass=other.qc_pass if other.qc_pass is not None else self.qc_pass,
        )

    def describe(self) -> str:
        """Human-readable summary, for explaining why a hit was returned."""
        parts = [
            f"{name} {lo}" if lo == hi else f"{name} {lo}-{hi}"
            for name, lo, hi in (*self.element_counts, *self.column_counts)
        ]
        if self.qc_pass is not None:
            parts.append("QC pass" if self.qc_pass else "QC fail")
        return ", ".join(parts)


#: Former name, from when composition was the only thing that could be
#: constrained. Kept so existing imports and documentation keep working.
SizeFilter = RowFilter


@dataclass
class MoleculeSearchHit:
    """A molecule matched by a search query.

    *match_type* records how the match was made (``formula``, ``smarts``,
    ``name:pubchem``, …) so the UI can show why a molecule was returned.
    """

    entry: MoleculeIndexEntry
    query: str
    match_type: str


__all__ = ["MoleculeIndexEntry", "MoleculeSearchHit", "RowFilter", "SizeFilter"]
