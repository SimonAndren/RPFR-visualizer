"""In-memory molecule search over a list of entries.

Serves tests, ad-hoc entry lists, and small catalogs. It implements the same
interface as :class:`~rpfr_visualizer.io.search.sqlite_backend.SqliteSearchBackend`
so the facade above them is backend-agnostic and the two can be checked against
each other (``tests/unit/io/test_search_equivalence.py``).

For the published data use the SQLite backend: this one holds every entry as a
dataclass and scans linearly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from rpfr_visualizer.io.search.chem import canonical_smiles, normalize_formula
from rpfr_visualizer.io.search.models import MoleculeIndexEntry, SizeFilter
from rpfr_visualizer.io.search.screening import (
    compute_pattern_fp,
    compute_query_fp,
    fingerprint_matrix,
    screen_candidates,
    verify_matches,
)


class MemorySearchBackend:
    """Query a plain list of :class:`MoleculeIndexEntry` records."""

    def __init__(self, entries: Sequence[MoleculeIndexEntry] | None = None) -> None:
        self.entries: list[MoleculeIndexEntry] = list(entries or [])
        self._by_canonical: dict[str, list[MoleculeIndexEntry]] | None = None
        self._by_formula: dict[str, list[MoleculeIndexEntry]] | None = None
        self._by_ident: dict[tuple[str, str], MoleculeIndexEntry] | None = None
        self._fp_matrix = None
        self._fp_entries: list[MoleculeIndexEntry] | None = None

    # ------------------------------------------------------------- inventory

    def datasets(self) -> tuple[str, ...]:
        return tuple(sorted({entry.dataset for entry in self.entries}))

    def count(self, *, datasets: Sequence[str] | None = None) -> int:
        return len(self._pool(datasets))

    def iter_entries(
        self, *, datasets: Sequence[str] | None = None
    ) -> Iterator[MoleculeIndexEntry]:
        yield from self._pool(datasets)

    # --------------------------------------------------------------- lookups

    def get(self, dataset: str, key: str) -> MoleculeIndexEntry | None:
        return self._ident_index().get((dataset, key))

    def get_many(
        self, pairs: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], MoleculeIndexEntry]:
        index = self._ident_index()
        return {pair: index[pair] for pair in pairs if pair in index}

    def by_key(
        self,
        token: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        return self._collect(
            (e for e in self._pool(datasets) if e.key == token), limit=limit, size=size
        )

    def by_key_substring(
        self,
        token: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        needle = token.lower()
        return self._collect(
            (e for e in self._pool(datasets) if needle in e.key.lower()), limit=limit, size=size
        )

    def by_formula(
        self,
        formula: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        wanted = {formula, normalize_formula(formula)}
        wanted.discard("")
        index = self._formula_index()
        allowed = self._allowed(datasets)
        found: list[MoleculeIndexEntry] = []
        for key in wanted:
            found.extend(e for e in index.get(key, []) if allowed is None or e.dataset in allowed)
        return self._collect(found, limit=limit, size=size)

    def by_canonical_smiles(
        self,
        canon: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        allowed = self._allowed(datasets)
        found = (
            e
            for e in self._canonical_index().get(canon, [])
            if allowed is None or e.dataset in allowed
        )
        return self._collect(found, limit=limit, size=size)

    def by_smiles_substring(
        self,
        fragment: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        needle = fragment.lower()
        return self._collect(
            (e for e in self._pool(datasets) if e.smiles and needle in e.smiles.lower()),
            limit=limit,
            size=size,
        )

    def by_inchikey(
        self,
        inchikey: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        key = inchikey.strip().upper()
        skeleton = len(key) <= 14
        found = (
            e
            for e in self._pool(datasets)
            if e.inchikey and (e.inchikey.split("-")[0] == key if skeleton else e.inchikey == key)
        )
        return self._collect(found, limit=limit, size=size)

    def by_name(
        self,
        name: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        needle = name.strip().lower()
        found = (e for e in self._pool(datasets) if needle in {n.lower() for n in e.names})
        return self._collect(found, limit=limit, size=size)

    def search_names(
        self,
        text: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Prefix name search, matching the SQLite backend's semantics."""
        terms = [part.lower() for part in text.split() if part]
        if not terms:
            return []
        found = []
        for entry in self._pool(datasets):
            lowered = [n.lower() for n in entry.names]
            if all(any(_prefix_hit(name, term) for name in lowered) for term in terms):
                found.append(entry)
        return self._collect(found, limit=limit, size=size)

    def by_elements(
        self,
        *,
        must_have: Sequence[str] = (),
        must_not_have: Sequence[str] = (),
        counts: dict[str, tuple[int, int]] | None = None,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        if not (must_have or must_not_have or counts or size):
            return []
        found: list[MoleculeIndexEntry] = []
        for entry in self._pool(datasets):
            present = entry.element_counts
            if any(el not in present for el in must_have):
                continue
            if any(el in present for el in must_not_have):
                continue
            if any(
                el not in present or not (low <= present[el] <= high)
                for el, (low, high) in (counts or {}).items()
            ):
                continue
            found.append(entry)
        return self._collect(found, limit=limit, size=size)

    def by_functional_group(
        self,
        preset: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Match a curated preset by running its SMARTS (nothing is precomputed here)."""
        from rpfr_visualizer.analysis.functional_groups import preset_by_name

        try:
            found = preset_by_name(preset)
        except KeyError:
            return []
        if not found.smarts:
            return []
        return self.by_smarts(found.smarts, datasets=datasets, limit=limit, size=size)

    def by_functional_groups(
        self,
        presets: Sequence[str],
        *,
        mode: str = "all",
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Same contract as the SQLite backend, by running each SMARTS here.

        Intersecting identities rather than entries: two runs return equal-valued
        but distinct records, so set operations have to key on (dataset, key).
        """
        names = tuple(dict.fromkeys(presets))
        if not names:
            return []
        if len(names) == 1:
            return self.by_functional_group(names[0], datasets=datasets, limit=limit, size=size)
        if mode not in {"all", "any"}:
            raise ValueError(f"Unknown functional-group mode: {mode!r}")

        # Unlimited per group, then combine: a per-group limit would truncate
        # before the intersection and lose molecules that satisfy every group.
        per_group = [
            self.by_functional_group(name, datasets=datasets, limit=None, size=size)
            for name in names
        ]
        ident_sets = [{(e.dataset, e.key) for e in entries} for entries in per_group]
        keep = set.intersection(*ident_sets) if mode == "all" else set.union(*ident_sets)

        seen: set[tuple[str, str]] = set()
        out: list[MoleculeIndexEntry] = []
        for entries in per_group:
            for entry in entries:
                ident = (entry.dataset, entry.key)
                if ident in keep and ident not in seen:
                    seen.add(ident)
                    out.append(entry)
        out.sort(key=lambda e: (e.dataset, e.key))
        return out[:limit] if limit is not None else out

    def has_functional_group(self, preset: str) -> bool:
        from rpfr_visualizer.analysis.functional_groups import preset_by_name

        try:
            return bool(preset_by_name(preset).smarts)
        except KeyError:
            return False

    def functional_group_counts(self, *, datasets: Sequence[str] | None = None) -> dict[str, int]:
        """``{preset_name: n_molecules}`` for every curated group.

        Nothing is precomputed here, so this runs each preset's SMARTS. It is
        only viable because this backend exists for small catalogs; the SQLite
        backend answers the same question from one aggregate. Groups with no
        members are absent rather than zero, matching the SQLite behaviour.
        """
        from rpfr_visualizer.analysis.functional_groups import primary_fg_presets

        counts: dict[str, int] = {}
        for preset in primary_fg_presets():
            n = len(self.by_functional_group(preset.name, datasets=datasets))
            if n:
                counts[preset.name] = n
        return counts

    def dataset_inventory(self) -> dict[str, dict[str, int]]:
        """Per-dataset molecule, structure-coverage and QC counts."""
        inventory: dict[str, dict[str, int]] = {}
        for entry in self.entries:
            row = inventory.setdefault(
                entry.dataset,
                {
                    "n": 0,
                    "n_with_structure": 0,
                    "n_qc_pass": 0,
                    "n_qc_fail": 0,
                    "n_qc_unknown": 0,
                },
            )
            row["n"] += 1
            if entry.smiles:
                row["n_with_structure"] += 1
            # This backend scans plain entries, which carry no QC verdict.
            row["n_qc_unknown"] += 1
        return inventory

    def by_smarts(
        self,
        smarts: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        screen: bool = True,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        try:
            from rdkit import Chem, RDLogger

            RDLogger.DisableLog("rdApp.*")
        except ImportError as exc:  # pragma: no cover - RDKit is a viz extra
            raise ImportError(
                "RDKit is required for SMARTS search. Install with: uv sync --extra viz"
            ) from exc

        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            return []

        pool = self._pool(datasets)
        # Narrow before verification, matching the SQLite backend: RDKit
        # matching is the expensive step, so the size filter is applied to
        # candidates rather than to the results.
        if size:
            pool = [entry for entry in pool if _size_ok(entry, size)]
        if not pool:
            return []

        if screen:
            matrix = self._fingerprints_for(pool)
            keep = screen_candidates(matrix, compute_query_fp(pattern))
            candidates = [(int(i), pool[int(i)].smiles) for i in keep]
        else:
            candidates = [(i, entry.smiles) for i, entry in enumerate(pool)]

        matched = verify_matches(candidates, smarts, limit=limit)
        return [pool[i] for i in matched]

    # ------------------------------------------------------------- internals

    def _pool(self, datasets: Sequence[str] | None) -> list[MoleculeIndexEntry]:
        allowed = self._allowed(datasets)
        if allowed is None:
            return self.entries
        return [entry for entry in self.entries if entry.dataset in allowed]

    @staticmethod
    def _allowed(datasets: Sequence[str] | None) -> set[str] | None:
        return set(datasets) if datasets else None

    @staticmethod
    def _collect(
        entries, *, limit: int | None, size: SizeFilter | None = None
    ) -> list[MoleculeIndexEntry]:
        seen: set[tuple[str, str]] = set()
        out: list[MoleculeIndexEntry] = []
        for entry in entries:
            ident = (entry.dataset, entry.key)
            if ident in seen:
                continue
            if not _size_ok(entry, size):
                continue
            seen.add(ident)
            out.append(entry)
            if limit is not None and len(out) >= limit:
                break
        return out

    def _ident_index(self) -> dict[tuple[str, str], MoleculeIndexEntry]:
        if self._by_ident is None:
            self._by_ident = {(e.dataset, e.key): e for e in self.entries}
        return self._by_ident

    def _canonical_index(self) -> dict[str, list[MoleculeIndexEntry]]:
        if self._by_canonical is None:
            index: dict[str, list[MoleculeIndexEntry]] = {}
            for entry in self.entries:
                canon = entry.canonical_smiles or (
                    canonical_smiles(entry.smiles) if entry.smiles else ""
                )
                if canon:
                    index.setdefault(canon, []).append(entry)
            self._by_canonical = index
        return self._by_canonical

    def _formula_index(self) -> dict[str, list[MoleculeIndexEntry]]:
        if self._by_formula is None:
            index: dict[str, list[MoleculeIndexEntry]] = {}
            for entry in self.entries:
                if not entry.formula:
                    continue
                index.setdefault(entry.formula, []).append(entry)
                normalized = normalize_formula(entry.formula)
                if normalized and normalized != entry.formula:
                    index.setdefault(normalized, []).append(entry)
            self._by_formula = index
        return self._by_formula

    def _fingerprints_for(self, pool: list[MoleculeIndexEntry]):
        if self._fp_matrix is None or self._fp_entries is not pool:
            self._fp_entries = pool
            self._fp_matrix = fingerprint_matrix(
                [compute_pattern_fp(entry.smiles) for entry in pool]
            )
        return self._fp_matrix


def _size_ok(entry: MoleculeIndexEntry, size: SizeFilter | None) -> bool:
    """Whether *entry* satisfies a composition constraint.

    The SQLite backend pushes this into SQL; here it is a predicate over the
    entry's own counts, so both backends answer identically. ``n_heavy`` is
    derived rather than stored, since an in-memory entry carries only
    ``element_counts`` and ``n_atoms``.
    """
    if not size:
        return True
    if size.qc_pass is not None and entry.qc_pass is not size.qc_pass:
        # `is not` rather than `!=`: None must match neither branch, and
        # None != False would otherwise be True.
        return False
    counts = entry.element_counts
    for element, low, high in size.element_counts:
        if not low <= counts.get(element, 0) <= high:
            return False
    for column, low, high in size.column_counts:
        if column == "n_heavy":
            value = sum(n for el, n in counts.items() if el != "H")
        elif column == "n_atoms":
            value = entry.n_atoms
        else:
            raise ValueError(f"Unknown size column: {column}")
        if not low <= value <= high:
            return False
    return True


def _prefix_hit(name: str, term: str) -> bool:
    """True when any whitespace-separated token of *name* starts with *term*."""
    return any(token.startswith(term) for token in name.split())


__all__ = ["MemorySearchBackend"]
