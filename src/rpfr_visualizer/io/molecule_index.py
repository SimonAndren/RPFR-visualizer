"""Searchable molecule index across registered HDF5 datasets.

Facade over the search backends in :mod:`rpfr_visualizer.io.search`. A query token is
classified (key, formula, SMARTS, SMILES, name) and dispatched to the backend;
the backend decides whether that is a SQL lookup against a prebuilt index or a
scan over an in-memory entry list.

For the published data, open the shipped index::

    index = MoleculeIndex.open("data/cache/qm9s_rpfr_index.sqlite")
    hits = index.search("smarts:[CX3](=O)[OX2H1]")

Chemistry helpers, name resolution and the record types now live in
:mod:`rpfr_visualizer.io.search`; they are re-exported here so existing imports keep
working.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import h5py

from rpfr_visualizer.io.providers import DatasetCatalog, HDF5Provider
from rpfr_visualizer.io.search.chem import (
    canonical_smiles,
    looks_like_formula,
    looks_like_smiles,
    normalize_formula,
    parse_element_counts,
    query_as_smarts,
    smiles_descriptors,
    split_search_query,
    split_size_tokens,
    stored_qc_pass,
)
from rpfr_visualizer.io.search.memory_backend import MemorySearchBackend
from rpfr_visualizer.io.search.models import MoleculeIndexEntry, MoleculeSearchHit, SizeFilter
from rpfr_visualizer.io.search.names import (
    load_name_resolve_cache,
    name_resolve_cache_path,
    resolve_common_name,
    resolve_name_hybrid,
    save_name_resolve_cache,
    synonym_seed,
)
from rpfr_visualizer.io.search.sqlite_backend import SqliteSearchBackend

#: Environment variable that points at the external drive holding the full
#: datasets. Set it and nothing else in the package needs to know where the
#: data lives.
DRIVE_ROOT_ENV = "RPFR_VISUALIZER_DRIVE"

#: Fallback root when neither :data:`DRIVE_ROOT_ENV` nor an explicit path is
#: given. A guess, not a contract: set the variable or pass the path.
DEFAULT_DRIVE_ROOT = Path.home() / "RPFR-Datasets" / "Processed-Data"


def default_drive_root() -> Path:
    """Where to look for the full datasets when no drive root is given.

    Reads :data:`DRIVE_ROOT_ENV` at call time rather than at import, so setting
    the variable works without reimporting the package.

    Returns:
        The path from the environment if set and non-empty, else
        :data:`DEFAULT_DRIVE_ROOT`.
    """
    override = os.environ.get(DRIVE_ROOT_ENV)
    return Path(override) if override else DEFAULT_DRIVE_ROOT


#: Version of the legacy JSON entry cache (distinct from the SQLite
#: index_version in :mod:`rpfr_visualizer.io.search.schema`).
INDEX_VERSION = 2

#: Warn when materializing more entries than this — a sign that caller should
#: be streaming with :meth:`MoleculeIndex.iter_entries` instead.
ENTRIES_WARN_THRESHOLD = 50_000


def _decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value) if value is not None else ""


class MoleculeIndex:
    """Searchable index of molecules across one or more datasets.

    Wraps a backend (SQLite for the published data, in-memory for tests and
    small catalogs) and turns free-text queries into typed hits.
    """

    def __init__(
        self,
        entries: Sequence[MoleculeIndexEntry] | None = None,
        *,
        backend: Any | None = None,
        resolve_cache_path: Path | str | None = None,
        allow_pubchem: bool = True,
    ) -> None:
        if backend is None:
            backend = MemorySearchBackend(entries or [])
        self.backend = backend
        self.resolve_cache_path = (
            Path(resolve_cache_path) if resolve_cache_path else name_resolve_cache_path()
        )
        self.allow_pubchem = allow_pubchem

    # ------------------------------------------------------------ construction

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        resolve_cache_path: Path | str | None = None,
        allow_pubchem: bool = False,
    ) -> MoleculeIndex:
        """Open a prebuilt SQLite index read-only.

        *allow_pubchem* defaults to False here: a published index is meant to
        answer offline, and a panel should never block on a network request.
        """
        return cls(
            backend=SqliteSearchBackend(path),
            resolve_cache_path=resolve_cache_path,
            allow_pubchem=allow_pubchem,
        )

    @classmethod
    def build_from_catalog(
        cls,
        catalog: DatasetCatalog,
        *,
        datasets: tuple[str, ...] | None = None,
        canonicalize: bool = True,
        resolve_cache_path: Path | str | None = None,
    ) -> MoleculeIndex:
        """Scan HDF5 attrs into an in-memory index.

        Fine for a subsample; for the published data build a SQLite index with
        :func:`rpfr_visualizer.io.search.build_index` and use :meth:`open`.
        """
        selected = datasets or catalog.datasets
        seed = synonym_seed()
        names_by_smiles: dict[str, list[str]] = {}
        for name, smiles in seed.items():
            canon = canonical_smiles(smiles) or smiles
            names_by_smiles.setdefault(canon, []).append(name)
            names_by_smiles.setdefault(smiles, []).append(name)

        entries: list[MoleculeIndexEntry] = []
        for dataset in selected:
            provider = catalog.get_provider(dataset)
            if not isinstance(provider, HDF5Provider):
                continue
            with h5py.File(provider.path, "r") as h5:
                for key in provider.list_keys():
                    group = h5[key]
                    smiles = _decode_attr(group.attrs.get("smiles", ""))
                    n_atoms = int(group.attrs.get("n_atoms", 0))
                    desc = (
                        smiles_descriptors(smiles)
                        if canonicalize and smiles
                        else {
                            "canonical_smiles": "",
                            "formula": "",
                            "inchikey": "",
                            "element_counts": {},
                        }
                    )
                    canon = desc["canonical_smiles"] or ""
                    known_names = tuple(
                        sorted(
                            set(names_by_smiles.get(canon, []) + names_by_smiles.get(smiles, []))
                        )
                    )
                    label = f"{dataset}:{key}"
                    if smiles:
                        label = f"{dataset}:{key} ({smiles[:40]}{'…' if len(smiles) > 40 else ''})"
                    entries.append(
                        MoleculeIndexEntry(
                            dataset=dataset,
                            key=key,
                            smiles=smiles,
                            canonical_smiles=canon,
                            n_atoms=n_atoms,
                            label=label,
                            formula=str(desc["formula"] or ""),
                            inchikey=str(desc["inchikey"] or ""),
                            element_counts=dict(desc["element_counts"] or {}),
                            names=known_names,
                            qc_pass=stored_qc_pass(group),
                        )
                    )
        return cls(entries, resolve_cache_path=resolve_cache_path)

    # -------------------------------------------------------------- inventory

    @property
    def datasets(self) -> tuple[str, ...]:
        """Datasets covered by this index."""
        return self.backend.datasets()

    def __len__(self) -> int:
        return self.backend.count()

    def iter_entries(
        self, *, datasets: tuple[str, ...] | None = None
    ) -> Iterator[MoleculeIndexEntry]:
        """Stream entries without materializing them all."""
        return self.backend.iter_entries(datasets=datasets)

    @property
    def entries(self) -> list[MoleculeIndexEntry]:
        """Every entry as a list.

        Prefer :meth:`iter_entries`, :meth:`get` or :meth:`get_many`: at
        published-data scale this materializes hundreds of thousands of
        records, and warns when it does.
        """
        total = len(self)
        if total > ENTRIES_WARN_THRESHOLD:
            import warnings

            warnings.warn(
                f"Materializing {total} index entries; use iter_entries() or get_many() instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        return list(self.backend.iter_entries())

    def get(self, dataset: str, key: str) -> MoleculeIndexEntry | None:
        """Look up one molecule by dataset and key."""
        return self.backend.get(dataset, key)

    def get_many(
        self, pairs: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], MoleculeIndexEntry]:
        """Look up many ``(dataset, key)`` pairs in one go."""
        return self.backend.get_many(pairs)

    # ----------------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        datasets: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[MoleculeSearchHit]:
        """Search by name, formula, SMILES, SMARTS, key, or numeric id.

        A SMILES token is parsed the same way the index was built, then
        matched by canonical SMILES — ``OCC`` and ``CCO`` are the same
        molecule, not every structure that contains that fragment. Use
        ``smarts:`` or ``fg:`` for substructure / group search.

        Comma-separated tokens are OR'd (except inside a ``smarts:`` query,
        where commas are part of the pattern).
        """
        tokens = split_search_query(query)
        if not tokens:
            return []

        # Size tokens are AND'd onto every other token rather than OR'd
        # alongside them: "fg:alcohol, C:5-9" means alcohols that also have
        # five to nine carbons. A query of nothing but size tokens is a
        # composition search in its own right.
        tokens, size = split_size_tokens(tokens)
        if not tokens:
            return self._search_size_only(size, datasets=datasets, limit=limit)

        hits: list[MoleculeSearchHit] = []
        # Keyed on the molecule, not on (molecule, query). A molecule matching
        # two OR'd tokens is still one molecule: listing it once per reason put
        # duplicates in the result list, spent the hit budget twice on the same
        # thing, and -- once an export stopped being capped -- wrote duplicate
        # rows into the CSV. The first token to match keeps its reason.
        seen: set[tuple[str, str]] = set()
        for token in tokens:
            for hit in self._search_token(token, datasets=datasets, limit=limit, size=size):
                ident = (hit.entry.dataset, hit.entry.key)
                if ident in seen:
                    continue
                seen.add(ident)
                hits.append(hit)
                if limit is not None and len(hits) >= limit:
                    return hits
        return hits

    def search_functional_group(
        self,
        preset: str,
        *,
        datasets: tuple[str, ...] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeSearchHit]:
        """Look up a curated functional group by name.

        Uses precomputed membership when the backend has it, so this is a table
        lookup rather than a substructure scan.
        """
        entries = self.backend.by_functional_group(
            preset, datasets=datasets, limit=limit, size=size
        )
        return [MoleculeSearchHit(e, f"fg:{preset}", "functional_group") for e in entries]

    def search_functional_groups(
        self,
        presets: Sequence[str],
        *,
        mode: str = "all",
        datasets: tuple[str, ...] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeSearchHit]:
        """Molecules carrying every listed group, or any of them.

        ``any`` is already expressible as comma-separated ``fg:`` tokens, which
        the OR path handles; ``all`` is the case that needed a backend of its own.
        """
        finder = getattr(self.backend, "by_functional_groups", None)
        if finder is None:  # pragma: no cover - both shipped backends have it
            raise ValueError("This index cannot answer multi-group queries")
        entries = finder(presets, mode=mode, datasets=datasets, limit=limit, size=size)
        label = ("&" if mode == "all" else ", ").join(presets)
        match_type = "functional_groups_all" if mode == "all" else "functional_groups_any"
        return [MoleculeSearchHit(e, label, match_type) for e in entries]

    def functional_group_counts(self, *, datasets: tuple[str, ...] | None = None) -> dict[str, int]:
        """``{preset_name: n_molecules}`` for every curated group.

        Groups with no members are absent rather than zero, so callers that want
        to show "none here" should iterate the presets, not this mapping.
        """
        counter = getattr(self.backend, "functional_group_counts", None)
        if counter is not None:
            return dict(counter(datasets=datasets))
        # A backend without the aggregate: fall back to counting hits.
        from rpfr_visualizer.analysis.functional_groups import primary_fg_presets

        counts: dict[str, int] = {}
        for preset in primary_fg_presets():
            n = len(self.search_functional_group(preset.name, datasets=datasets))
            if n:
                counts[preset.name] = n
        return counts

    def dataset_inventory(self) -> dict[str, dict[str, int]]:
        """Per-dataset molecule, structure-coverage and QC counts."""
        inventory = getattr(self.backend, "dataset_inventory", None)
        if inventory is None:
            return {}
        return dict(inventory())

    def _search_size_only(
        self,
        size: SizeFilter,
        *,
        datasets: tuple[str, ...] | None,
        limit: int | None,
    ) -> list[MoleculeSearchHit]:
        """Answer a query made only of constraints, e.g. ``C:5-9, O:1``.

        The element counts go in as the query proper; whatever else the filter
        carries -- column ranges, a QC verdict -- rides along as the AND'd
        remainder, so a bare ``qc:pass`` is still answerable.
        """
        if not size:
            return []
        entries = self.backend.by_elements(
            counts={element: (low, high) for element, low, high in size.element_counts},
            datasets=datasets,
            limit=limit,
            size=SizeFilter(column_counts=size.column_counts, qc_pass=size.qc_pass),
        )
        return [MoleculeSearchHit(e, size.describe(), "size") for e in entries]

    def _search_token(
        self,
        token: str,
        *,
        datasets: tuple[str, ...] | None,
        limit: int | None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeSearchHit]:
        """Classify one token and dispatch it. Order matters — first hit wins."""
        backend = self.backend

        # 0. Explicit prefixes
        lowered = token.lower()
        if lowered.startswith("smarts:"):
            smarts = token[7:].strip()
            entries = backend.by_smarts(smarts, datasets=datasets, limit=limit, size=size)
            return [MoleculeSearchHit(e, token, "smarts") for e in entries]
        if lowered.startswith("fg:"):
            # "fg:a&b" means both. A comma would mean either, but commas already
            # separate OR'd tokens, so the conjunction needs a delimiter of its
            # own -- and "&" appears in neither SMILES nor SMARTS, so it cannot
            # collide with a pattern body.
            names = [part.strip() for part in token[3:].split("&") if part.strip()]
            if len(names) > 1:
                return self.search_functional_groups(
                    names, mode="all", datasets=datasets, limit=limit, size=size
                )
            return self.search_functional_group(
                token[3:].strip(), datasets=datasets, limit=limit, size=size
            )
        if lowered.startswith("inchikey:"):
            entries = backend.by_inchikey(
                token[9:].strip(), datasets=datasets, limit=limit, size=size
            )
            return [MoleculeSearchHit(e, token, "inchikey") for e in entries]

        # 1. Direct key / id match
        entries = backend.by_key(token, datasets=datasets, limit=limit, size=size)
        if entries:
            return [MoleculeSearchHit(e, token, "key") for e in entries]

        # 2. Formula
        if looks_like_formula(token):
            entries = backend.by_formula(token, datasets=datasets, limit=limit, size=size)
            if entries:
                return [MoleculeSearchHit(e, token, "formula") for e in entries]

        # 3. InChIKey (recognised by shape: 14-char skeleton or full 27-char)
        if _looks_like_inchikey(token):
            entries = backend.by_inchikey(token, datasets=datasets, limit=limit, size=size)
            if entries:
                return [MoleculeSearchHit(e, token, "inchikey") for e in entries]

        # 4. SMILES: exact molecule (canonical graph). Unprefixed SMARTS that
        #    is not valid SMILES still goes through the substructure path.
        if looks_like_smiles(token):
            query_canon = canonical_smiles(token)
            if query_canon:
                entries = backend.by_canonical_smiles(
                    query_canon, datasets=datasets, limit=limit, size=size
                )
                return [MoleculeSearchHit(e, token, "canonical_smiles") for e in entries]
            smarts = query_as_smarts(token)
            if smarts:
                entries = backend.by_smarts(smarts, datasets=datasets, limit=limit, size=size)
                return [MoleculeSearchHit(e, token, "substructure") for e in entries]

        # 5. Name: stored synonyms, then seed → cache → PubChem resolution
        if not looks_like_smiles(token):
            hits = [
                MoleculeSearchHit(e, token, "name:local")
                for e in backend.by_name(token, datasets=datasets, limit=limit, size=size)
            ]
            if hits:
                return hits
            resolved, source = resolve_name_hybrid(
                token,
                resolve_cache_path=self.resolve_cache_path,
                allow_pubchem=self.allow_pubchem,
            )
            if resolved:
                resolved_canon = canonical_smiles(resolved) or resolved
                hits = [
                    MoleculeSearchHit(e, token, f"name:{source}")
                    for e in backend.by_canonical_smiles(
                        resolved_canon, datasets=datasets, limit=limit, size=size
                    )
                ]
                if hits:
                    return hits
            prefix_hits = [
                MoleculeSearchHit(e, token, "name:prefix")
                for e in backend.search_names(token, datasets=datasets, limit=limit, size=size)
            ]
            if prefix_hits:
                return prefix_hits

        # 6. Substring fallback across SMILES, key and formula
        hits = [
            MoleculeSearchHit(e, token, "smiles_substring")
            for e in backend.by_smiles_substring(token, datasets=datasets, limit=limit, size=size)
        ]
        hits.extend(
            MoleculeSearchHit(e, token, "key_substring")
            for e in backend.by_key_substring(token, datasets=datasets, limit=limit, size=size)
        )
        hits.extend(
            MoleculeSearchHit(e, token, "formula")
            for e in backend.by_formula(token, datasets=datasets, limit=limit, size=size)
        )
        if not hits:
            query_canon = canonical_smiles(token)
            if query_canon:
                hits = [
                    MoleculeSearchHit(e, token, "canonical_smiles")
                    for e in backend.by_canonical_smiles(
                        query_canon, datasets=datasets, limit=limit, size=size
                    )
                ]
        return _dedupe(hits, limit=limit)

    def filter_elements(
        self,
        *,
        must_have: Sequence[str] = (),
        must_not_have: Sequence[str] = (),
        counts: dict[str, tuple[int, int]] | None = None,
        datasets: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Filter molecules by element composition."""
        return self.backend.by_elements(
            must_have=must_have,
            must_not_have=must_not_have,
            counts=counts,
            datasets=datasets,
            limit=limit,
        )

    # ------------------------------------------------------------ legacy cache

    def save_cache(self, path: Path | str) -> None:
        """Persist entries as JSON (legacy format; prefer a SQLite index)."""
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_version": INDEX_VERSION,
            "entries": [
                {
                    "dataset": e.dataset,
                    "key": e.key,
                    "smiles": e.smiles,
                    "canonical_smiles": e.canonical_smiles,
                    "n_atoms": e.n_atoms,
                    "formula": e.formula,
                    "inchikey": e.inchikey,
                    "element_counts": e.element_counts,
                    "names": list(e.names),
                    "label": e.label,
                }
                for e in self.backend.iter_entries()
            ],
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load_cache(
        cls,
        path: Path | str,
        *,
        resolve_cache_path: Path | str | None = None,
    ) -> MoleculeIndex:
        """Load entries written by :meth:`save_cache`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):  # back-compat: caches used to be a bare list
            items = raw
        else:
            items = raw.get("entries", [])

        entries = []
        for item in items:
            formula = item.get("formula", "")
            element_counts = item.get("element_counts") or (
                parse_element_counts(formula) if formula else {}
            )
            entries.append(
                MoleculeIndexEntry(
                    dataset=item["dataset"],
                    key=item["key"],
                    smiles=item.get("smiles", ""),
                    canonical_smiles=item.get("canonical_smiles", ""),
                    n_atoms=int(item.get("n_atoms", 0)),
                    label=item.get("label") or f"{item['dataset']}:{item['key']}",
                    formula=formula,
                    inchikey=item.get("inchikey", ""),
                    element_counts=dict(element_counts),
                    names=tuple(item.get("names") or ()),
                )
            )
        return cls(entries, resolve_cache_path=resolve_cache_path)

    def to_frame(self, *, limit: int | None = None):
        """Return a pandas DataFrame summary of the index."""
        import pandas as pd

        rows = []
        for entry in self.backend.iter_entries():
            rows.append(
                {
                    "dataset": entry.dataset,
                    "key": entry.key,
                    "smiles": entry.smiles,
                    "canonical_smiles": entry.canonical_smiles,
                    "formula": entry.formula,
                    "n_atoms": entry.n_atoms,
                    "names": ",".join(entry.names),
                    "label": entry.label,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        return pd.DataFrame(rows)


def _looks_like_inchikey(token: str) -> bool:
    """True for a full InChIKey (14-10-1) or its 14-character skeleton block."""
    text = token.strip().upper()
    if not text.isalpha() and "-" not in text:
        return False
    parts = text.split("-")
    if len(parts) == 3:
        return (
            len(parts[0]) == 14
            and len(parts[1]) == 10
            and len(parts[2]) == 1
            and all(p.isalpha() for p in parts)
        )
    return len(text) == 14 and text.isalpha()


def _dedupe(hits: list[MoleculeSearchHit], *, limit: int | None = None) -> list[MoleculeSearchHit]:
    seen: set[tuple[str, str]] = set()
    out: list[MoleculeSearchHit] = []
    for hit in hits:
        ident = (hit.entry.dataset, hit.entry.key)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(hit)
        if limit is not None and len(out) >= limit:
            break
    return out


def index_cache_path(data_dir: Path | str) -> Path:
    """Default cache file path under ``data/cache``."""
    data = Path(data_dir)
    if data.name == "raw":
        data = data.parent
    return data / "cache" / "molecule_index.json"


def build_or_load_index(
    catalog: DatasetCatalog,
    cache_path: Path | str | None = None,
    *,
    force_rebuild: bool = False,
) -> MoleculeIndex:
    """Return a usable index: prebuilt SQLite if present, else the JSON cache.

    A SQLite index sitting beside the JSON cache always wins — it is the
    published artifact and the only one that scales to the full data.
    """
    from rpfr_visualizer.io.search import find_index

    if cache_path is None:
        cache_path = index_cache_path(Path("data/raw"))
    cache_path = Path(cache_path)
    resolve_path = name_resolve_cache_path(cache_path.parent.parent)

    if not force_rebuild:
        sqlite_path = find_index(data_dir=cache_path.parent.parent)
        if sqlite_path is not None:
            return MoleculeIndex.open(sqlite_path, resolve_cache_path=resolve_path)

    catalog_sig = {
        "index_version": INDEX_VERSION,
        "datasets": [
            (name, str(catalog.get_provider(name).path.resolve())) for name in catalog.datasets
        ],
    }
    sig_path = cache_path.with_suffix(".sig")
    sig_text = json.dumps(catalog_sig, sort_keys=True)

    if (
        not force_rebuild
        and cache_path.exists()
        and sig_path.exists()
        and sig_path.read_text() == sig_text
    ):
        return MoleculeIndex.load_cache(cache_path, resolve_cache_path=resolve_path)

    index = MoleculeIndex.build_from_catalog(catalog, resolve_cache_path=resolve_path)
    index.save_cache(cache_path)
    sig_path.write_text(sig_text, encoding="utf-8")
    return index


#: Filename fragments stripped when deriving a dataset name. Applied in order.
_NAME_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_tr_project", ""),
    ("_tr_skip", "_skip"),
    # Schema 2.2 dropped the tr_ naming; the full file is qm9s_rpfr_v2.2.h5 and
    # the subsample qm9s_v2.2_1pt.h5, and both have to land on qm9s_v2.2 or the
    # release tier and the full index become two unrelated datasets.
    ("qm9s_rpfr_v2.2", "qm9s_v2.2"),
)

#: Subsample markers stripped so a 1% file and the full file share one name.
#: That is what lets a subsample index and a full index be ranked against each
#: other by coverage, instead of being two datasets that never meet.
#:
#: ``_1pt`` is the v2.2 spelling of ``_1pct``; both are stripped so that
#: ``qm9s_v2.2_1pt.h5`` and the full ``qm9s_rpfr_v2.2.h5`` meet at one name.
#: Longer markers come first: stripping ``_1pt`` out of ``_1pct`` would leave
#: a stray ``c`` behind.
_SAMPLE_MARKERS: tuple[str, ...] = (
    "_1pct",
    "_5pct",
    "_10pct",
    "_1pt",
    "_subsample",
    "_sample",
)


def dataset_name_for(path: Path | str) -> str:
    """The dataset name an HDF5 file registers under.

    Derived from the filename, never from the HDF5 ``dataset`` attribute: that
    attribute is not unique — both ``qm9s_tr_*`` files carry ``"qm9s"`` — so
    using it as a catalog key silently loses one file of every pair.

    ``qm9s_tr_project.h5`` and ``qm9s_tr_project_1pct.h5`` both give ``qm9s``;
    ``qm9s_tr_skip_1pct.h5`` gives ``qm9s_skip``.
    """
    name = Path(path).stem
    for fragment, replacement in _NAME_SUFFIXES:
        name = name.replace(fragment, replacement)
    for marker in _SAMPLE_MARKERS:
        name = name.replace(marker, "")
    return name


def register_all(catalog: DatasetCatalog, files: Sequence[Path]) -> list[str]:
    """Register every file under a unique name; return notes about collisions.

    Two files deriving the same name used to mean the second was dropped without
    a word. Suffixing keeps both openable and says which is which.
    """
    notes: list[str] = []
    for h5 in files:
        name = dataset_name_for(h5)
        if name in catalog.datasets:
            suffixed = name
            n = 2
            while suffixed in catalog.datasets:
                suffixed = f"{name}_{n}"
                n += 1
            notes.append(f"{h5.name} also derives {name!r}; registered it as {suffixed!r}.")
            name = suffixed
        catalog.register(name, h5)
    return notes


def register_subsample_catalog(raw_dir) -> DatasetCatalog:
    """Register the HDF5 files under *raw_dir*, named by :func:`dataset_name_for`.

    This used to match a hardcoded list of seven ``_1pct`` filenames, so a folder
    holding full-size files came back empty and the caller fell through to a
    layout branch that happened to work for a different reason.
    """
    catalog = DatasetCatalog()
    files = sorted(p for p in Path(raw_dir).glob("*.h5") if not p.name.startswith("._"))
    register_all(catalog, files)
    return catalog


def register_drive_catalog(
    drive_root: Path | str | None = None,
) -> DatasetCatalog:
    """Register full dataset files from an external drive layout."""
    root = Path(drive_root) if drive_root else default_drive_root()
    if not root.exists():
        raise FileNotFoundError(
            f"External drive not found at {root}. "
            f"Connect it, pass drive_root=, or set ${DRIVE_ROOT_ENV}."
        )
    mapping = {
        "qm9s_v2.2": "qm9s/qm9s_rpfr_v2.2.h5",
    }
    catalog = DatasetCatalog()
    for name, rel in mapping.items():
        path = root / rel
        if path.exists():
            catalog.register(name, path)
    if not catalog.datasets:
        raise FileNotFoundError(f"No HDF5 files found under {root}")
    return catalog


__all__ = [
    "DEFAULT_DRIVE_ROOT",
    "DRIVE_ROOT_ENV",
    "ENTRIES_WARN_THRESHOLD",
    "INDEX_VERSION",
    "MemorySearchBackend",
    "MoleculeIndex",
    "MoleculeIndexEntry",
    "MoleculeSearchHit",
    "SqliteSearchBackend",
    "build_or_load_index",
    "canonical_smiles",
    "default_drive_root",
    "index_cache_path",
    "load_name_resolve_cache",
    "looks_like_formula",
    "looks_like_smiles",
    "name_resolve_cache_path",
    "normalize_formula",
    "parse_element_counts",
    "register_drive_catalog",
    "register_subsample_catalog",
    "resolve_common_name",
    "resolve_name_hybrid",
    "save_name_resolve_cache",
    "smiles_descriptors",
    "split_search_query",
    "synonym_seed",
]
