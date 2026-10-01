"""SQLite-backed molecule search.

Opens a prebuilt index read-only and answers lookups with indexed SQL.
Substructure queries go through the fingerprint screen in
:mod:`rpfr_visualizer.io.search.screening` before RDKit verification.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rpfr_visualizer.io.search.chem import normalize_formula
from rpfr_visualizer.io.search.keys import compile_query
from rpfr_visualizer.io.search.models import MoleculeIndexEntry, SizeFilter
from rpfr_visualizer.io.search.schema import (
    FP_BYTES,
    INDEX_VERSION,
    META_HAS_FTS,
    META_INDEX_VERSION,
    META_RDKIT_VERSION,
    META_SOURCE_SIGNATURE,
)
from rpfr_visualizer.io.search.screening import (
    compute_query_fp,
    fingerprint_matrix,
    screen_candidates,
    verify_matches,
)

#: Candidates verified per SMILES fetch during substructure search.
_VERIFY_CHUNK = 5_000

#: Columns every entry query selects. Synonyms come along as one joined string
#: via a correlated subquery — fetching them per row turned a single result set
#: into hundreds of extra queries, and selecting them separately meant the id
#: was missing and names silently came back empty.
_NAMES_SUBQUERY = (
    "(SELECT group_concat(n.name, '\x1f') FROM names n WHERE n.mol = {alias}id) AS names"
)

_BASE_COLUMNS = (
    "dataset, key, smiles, canonical_smiles, n_atoms, label, formula, inchikey, "
    "element_counts, qc_pass"
)

_ENTRY_COLUMNS = _BASE_COLUMNS + ", " + _NAMES_SUBQUERY.format(alias="molecules.")

#: Columns a size filter may constrain. Interpolated into SQL, so it is an
#: allow-list rather than anything derived from user input.
_SIZE_COLUMNS_ALLOWED = frozenset({"n_heavy", "n_atoms"})


class IndexVersionError(RuntimeError):
    """Raised when an index file was written by an incompatible builder."""


class SqliteSearchBackend:
    """Query a prebuilt molecule index."""

    def __init__(self, path: Path | str, *, check_version: bool = True) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Search index not found: {self.path}")
        # Read-only URI: a shipped index must never be mutated by a query.
        self._conn = sqlite3.connect(
            f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._meta: dict[str, str] | None = None
        self._fp_matrix: Any | None = None
        self._fp_ids: list[int] | None = None
        self._has_fts: bool | None = None
        self._key_vocab: list[str] | None = None
        self._mol_bin: bool | None = None
        self._key_cache: dict[str, np.ndarray | None] = {}
        if check_version:
            self._check_version()

    # ------------------------------------------------------------------ meta

    def _check_version(self) -> None:
        stored = int(self.meta().get(META_INDEX_VERSION, 0) or 0)
        if stored != INDEX_VERSION:
            raise IndexVersionError(
                f"Search index {self.path} was built with index_version={stored}, "
                f"but this package expects {INDEX_VERSION}. Rebuild it with:\n"
                f"  rpfr-visualizer index build --out {self.path}"
            )

    def meta(self) -> dict[str, str]:
        """Return the index's meta table as a dict."""
        if self._meta is None:
            rows = self._conn.execute("SELECT key, value FROM meta").fetchall()
            self._meta = {row["key"]: row["value"] for row in rows}
        return self._meta

    def rdkit_version(self) -> str:
        return self.meta().get(META_RDKIT_VERSION, "")

    def source_signature(self) -> list[list[Any]]:
        raw = self.meta().get(META_SOURCE_SIGNATURE, "[]")
        try:
            return list(json.loads(raw))
        except json.JSONDecodeError:
            return []

    def _fts_available(self) -> bool:
        if self._has_fts is None:
            if self.meta().get(META_HAS_FTS) == "0":
                self._has_fts = False
            else:
                row = self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='names_fts'"
                ).fetchone()
                self._has_fts = row is not None
        return self._has_fts

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------- inventory

    def datasets(self) -> tuple[str, ...]:
        rows = self._conn.execute("SELECT dataset FROM datasets ORDER BY dataset").fetchall()
        return tuple(row["dataset"] for row in rows)

    def count(self, *, datasets: Sequence[str] | None = None) -> int:
        where, params = self._dataset_clause(datasets)
        sql = f"SELECT COUNT(*) AS n FROM molecules{where}"
        row = self._conn.execute(sql, params).fetchone()
        return int(row["n"])

    def iter_entries(
        self, *, datasets: Sequence[str] | None = None
    ) -> Iterator[MoleculeIndexEntry]:
        """Stream every entry without materializing the whole index."""
        where, params = self._dataset_clause(datasets)
        cursor = self._conn.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM molecules{where} ORDER BY id", params
        )
        for row in cursor:
            yield self._row_to_entry(row)

    # --------------------------------------------------------------- lookups

    def get(self, dataset: str, key: str) -> MoleculeIndexEntry | None:
        row = self._conn.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM molecules WHERE dataset = ? AND key = ?",
            (dataset, key),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_many(
        self, pairs: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], MoleculeIndexEntry]:
        """Look up many ``(dataset, key)`` pairs at once.

        The panel needs this to turn a page of search hits into entries; doing
        it one query at a time, or by scanning every entry, is what made the
        old in-memory index unusable at full dataset size.
        """
        found: dict[tuple[str, str], MoleculeIndexEntry] = {}
        pairs = list(pairs)
        for chunk in _chunks(pairs, 500):
            placeholders = ",".join("(?,?)" for _ in chunk)
            flat: list[str] = []
            for dataset, key in chunk:
                flat.extend((dataset, key))
            rows = self._conn.execute(
                f"SELECT {_ENTRY_COLUMNS} FROM molecules WHERE (dataset, key) IN ({placeholders})",
                flat,
            ).fetchall()
            for row in rows:
                entry = self._row_to_entry(row)
                found[(entry.dataset, entry.key)] = entry
        return found

    def by_key(
        self,
        token: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        return self._select("key = ?", (token,), datasets=datasets, limit=limit, size=size)

    def by_key_substring(
        self,
        token: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        return self._select(
            "key LIKE ?", (f"%{token}%",), datasets=datasets, limit=limit, size=size
        )

    def by_formula(
        self,
        formula: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        candidates = {formula, normalize_formula(formula)}
        candidates.discard("")
        if not candidates:
            return []
        placeholders = ",".join("?" for _ in candidates)
        return self._select(
            f"formula IN ({placeholders})",
            tuple(candidates),
            datasets=datasets,
            limit=limit,
            size=size,
        )

    def by_canonical_smiles(
        self,
        canon: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        return self._select(
            "canonical_smiles = ?", (canon,), datasets=datasets, limit=limit, size=size
        )

    def by_smiles_substring(
        self,
        fragment: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        return self._select(
            "smiles LIKE ?", (f"%{fragment}%",), datasets=datasets, limit=limit, size=size
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
        column = "inchikey" if len(key) > 14 else "inchikey14"
        return self._select(f"{column} = ?", (key,), datasets=datasets, limit=limit, size=size)

    def by_name(
        self,
        name: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Exact (case-insensitive) name lookup against the shipped synonyms."""
        where, params = self._dataset_clause(datasets, prefix="AND")
        size_sql, size_params = _size_suffix(size, alias="m")
        sql = (
            f"SELECT {_prefixed('m')} FROM molecules m "
            "JOIN names n ON n.mol = m.id WHERE n.name = ?"
            + where
            + size_sql
            # One molecule can carry several synonyms, so the join multiplies
            # rows; collapse before LIMIT or the limit would count duplicates.
            + " GROUP BY m.id ORDER BY m.id"
            + _limit_clause(limit)
        )
        rows = self._conn.execute(sql, (name, *params, *size_params)).fetchall()
        return self._rows_to_entries(rows, limit=limit)

    def search_names(
        self,
        text: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Prefix name search: FTS5 when available, LIKE otherwise.

        Typing "acet" finds acetaldehyde — each whitespace-separated term is a
        prefix, which is what a user typing into a search box expects.
        """
        text = text.strip()
        if not text:
            return []
        where, params = self._dataset_clause(datasets, prefix="AND")
        size_sql, size_params = _size_suffix(size, alias="m")
        if self._fts_available():
            sql = (
                f"SELECT {_prefixed('m')} FROM names_fts f "
                "JOIN names n ON n.id = f.rowid JOIN molecules m ON m.id = n.mol "
                "WHERE names_fts MATCH ?"
                + where
                + size_sql
                + " GROUP BY m.id ORDER BY m.id"
                + _limit_clause(limit)
            )
            # Quote each term (so punctuation cannot be read as FTS syntax) and
            # append * to make it a prefix match.
            query = " ".join(f'"{part}"*' for part in text.split() if part)
            if query:
                try:
                    rows = self._conn.execute(sql, (query, *params, *size_params)).fetchall()
                    return self._rows_to_entries(rows, limit=limit)
                except sqlite3.OperationalError:
                    self._has_fts = False  # fall through to LIKE
        sql = (
            f"SELECT {_prefixed('m')} FROM molecules m "
            "JOIN names n ON n.mol = m.id WHERE n.name LIKE ?"
            + where
            + size_sql
            + " GROUP BY m.id ORDER BY m.id"
            + _limit_clause(limit)
        )
        rows = self._conn.execute(sql, (f"%{text}%", *params, *size_params)).fetchall()
        return self._rows_to_entries(rows, limit=limit)

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
        """Filter by element composition.

        *counts* maps an element to an inclusive ``(min, max)`` range. *size*
        carries the same information plus ``n_heavy`` / ``n_atoms`` bounds, and
        is ANDed onto *counts*.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for element in must_have:
            clauses.append(
                "EXISTS (SELECT 1 FROM element_counts e WHERE e.mol = m.id AND e.element = ?)"
            )
            params.append(element)
        for element in must_not_have:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM element_counts e WHERE e.mol = m.id AND e.element = ?)"
            )
            params.append(element)
        combined = SizeFilter(
            element_counts=tuple(
                (element, int(low), int(high)) for element, (low, high) in (counts or {}).items()
            )
        )
        if size is not None:
            combined = combined.merge(size)
        size_sql, size_params = _size_clauses(combined, alias="m")
        clauses.extend(size_sql)
        params.extend(size_params)
        if not clauses:
            return []
        where, ds_params = self._dataset_clause(datasets, prefix="AND", alias="m")
        # Every clause is an EXISTS subquery, so a molecule appears at most once
        # and LIMIT can go straight into SQL.
        sql = (
            f"SELECT {_prefixed('m')} FROM molecules m "
            f"WHERE {' AND '.join(clauses)}{where} ORDER BY m.id" + _limit_clause(limit)
        )
        rows = self._conn.execute(sql, (*params, *ds_params)).fetchall()
        return self._rows_to_entries(rows, limit=limit)

    def by_functional_group(
        self,
        preset: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Look up a curated functional group from the precomputed membership table."""
        where, params = self._dataset_clause(datasets, prefix="AND", alias="m")
        size_sql, size_params = _size_suffix(size, alias="m")
        # ``fg_membership`` is keyed on (fg, mol) and ``p.name`` pins a single
        # fg, so the join cannot multiply rows — LIMIT is safe in SQL. Applying
        # it in Python instead measured 926 ms against ~5 ms here.
        sql = (
            f"SELECT {_prefixed('m')} FROM molecules m "
            "JOIN fg_membership fm ON fm.mol = m.id "
            "JOIN fg_presets p ON p.id = fm.fg WHERE p.name = ?"
            + where
            + size_sql
            + " ORDER BY m.id"
            + _limit_clause(limit)
        )
        rows = self._conn.execute(sql, (preset, *params, *size_params)).fetchall()
        return self._rows_to_entries(rows, limit=limit)

    def by_functional_groups(
        self,
        presets: Sequence[str],
        *,
        mode: str = "all",
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Molecules carrying every one of *presets* (``mode="all"``), or any.

        ``all`` is the case the single-preset lookup cannot express. It counts
        distinct matched preset names per molecule and keeps those reaching the
        full set — one pass over ``fg_membership``, which is keyed on (fg, mol)
        and indexed on mol, rather than N queries intersected in Python.
        """
        names = tuple(dict.fromkeys(presets))
        if not names:
            return []
        if len(names) == 1:
            return self.by_functional_group(names[0], datasets=datasets, limit=limit, size=size)
        if mode not in {"all", "any"}:
            raise ValueError(f"Unknown functional-group mode: {mode!r}")

        where, params = self._dataset_clause(datasets, prefix="AND", alias="m")
        size_sql, size_params = _size_suffix(size, alias="m")
        placeholders = ", ".join("?" for _ in names)
        # GROUP BY collapses the one row per matched group back to one per
        # molecule, so LIMIT stays safe in SQL for both modes.
        having = " HAVING COUNT(DISTINCT p.name) = ?" if mode == "all" else ""
        sql = (
            f"SELECT {_prefixed('m')} FROM molecules m "
            "JOIN fg_membership fm ON fm.mol = m.id "
            f"JOIN fg_presets p ON p.id = fm.fg WHERE p.name IN ({placeholders})"
            + where
            + size_sql
            + " GROUP BY m.id"
            + having
            + " ORDER BY m.id"
            + _limit_clause(limit)
        )
        args: list[Any] = [*names, *params, *size_params]
        if mode == "all":
            args.append(len(names))
        rows = self._conn.execute(sql, args).fetchall()
        return self._rows_to_entries(rows, limit=limit)

    def has_functional_group(self, preset: str) -> bool:
        """True when *preset* has precomputed membership in this index."""
        row = self._conn.execute("SELECT 1 FROM fg_presets WHERE name = ?", (preset,)).fetchone()
        return row is not None

    def functional_group_counts(self, *, datasets: Sequence[str] | None = None) -> dict[str, int]:
        """``{preset_name: n_molecules}`` for every precomputed group.

        One aggregate, building no entries. Counting by calling
        :meth:`by_functional_group` and taking ``len()`` materialises a full
        record per membership row — including a correlated synonym subquery and
        a JSON parse — which measured 2.8 s against 119 ms here on the full
        index. Groups with no members are absent rather than zero.
        """
        if not datasets:
            sql = (
                "SELECT p.name AS name, COUNT(*) AS n FROM fg_membership fm "
                "JOIN fg_presets p ON p.id = fm.fg GROUP BY p.name"
            )
            params: tuple[str, ...] = ()
        else:
            where, params = self._dataset_clause(datasets, prefix="AND", alias="m")
            sql = (
                "SELECT p.name AS name, COUNT(*) AS n FROM fg_membership fm "
                "JOIN fg_presets p ON p.id = fm.fg "
                "JOIN molecules m ON m.id = fm.mol "
                f"WHERE 1=1{where} GROUP BY p.name"
            )
        return {row["name"]: int(row["n"]) for row in self._conn.execute(sql, params)}

    def dataset_inventory(self) -> dict[str, dict[str, int]]:
        """Per-dataset molecule, structure-coverage and QC counts.

        ``n_with_structure`` is the number that can be found by name, formula,
        structure or functional group at all. Without it, a zero-hit structural
        search over molecules with no stored SMILES is indistinguishable from
        "there are none".

        ``smiles`` is ``NOT NULL DEFAULT ''``, so absence is the empty string.
        """
        sql = (
            "SELECT dataset, COUNT(*) AS n, "
            "SUM(CASE WHEN smiles = '' THEN 0 ELSE 1 END) AS n_with_structure, "
            "SUM(CASE WHEN qc_pass = 1 THEN 1 ELSE 0 END) AS n_qc_pass, "
            "SUM(CASE WHEN qc_pass = 0 THEN 1 ELSE 0 END) AS n_qc_fail, "
            "SUM(CASE WHEN qc_pass IS NULL THEN 1 ELSE 0 END) AS n_qc_unknown "
            "FROM molecules GROUP BY dataset"
        )
        return {
            row["dataset"]: {
                "n": int(row["n"]),
                "n_with_structure": int(row["n_with_structure"]),
                "n_qc_pass": int(row["n_qc_pass"]),
                "n_qc_fail": int(row["n_qc_fail"]),
                "n_qc_unknown": int(row["n_qc_unknown"]),
            }
            for row in self._conn.execute(sql)
        }

    # ----------------------------------------------------------- substructure

    def by_smarts(
        self,
        smarts: str,
        *,
        datasets: Sequence[str] | None = None,
        limit: int | None = None,
        screen: bool = True,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        """Substructure search, screened by pattern fingerprint then verified.

        Set *screen* False to verify every molecule — slower, and only useful
        for proving the screen is not dropping matches.

        A *size* filter is applied to the candidates before verification, not
        to the results after it: RDKit verification is the expensive step, so
        narrowing the candidate set first is what makes the combination cheap.
        """
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

        if screen:
            # Deliberately not calling _ids_for here: when screening, the id
            # pool is only needed to answer "is it empty?", and materialising
            # all 881,757 ids to do that measured 838 ms per query.
            if not self._any_rows(datasets):
                return []
            candidate_ids = self._screen_by_keys(pattern, datasets)
            if candidate_ids is None:
                # No topological screen available for this query (or this
                # index predates it): fall back to the pattern fingerprint.
                matrix, positions = self._fingerprints_for(datasets)
                keep = screen_candidates(matrix, compute_query_fp(pattern))
                candidate_ids = [positions[int(i)] for i in keep]
            if not candidate_ids:
                return []
        else:
            candidate_ids = self._ids_for(datasets)
            if not candidate_ids:
                return []

        if size:
            allowed = self._ids_for_size(size, datasets)
            candidate_ids = [mol_id for mol_id in candidate_ids if mol_id in allowed]
            if not candidate_ids:
                return []

        # SMILES are fetched per chunk during verification rather than for the
        # whole index up front: with a limit, a query usually stops after a few
        # thousand candidates, and pulling ~900k strings first costs more than
        # the screen it follows.
        matched: list[int] = []
        for chunk in _chunks(candidate_ids, _VERIFY_CHUNK):
            smiles_by_id, binaries = self._structures_for(chunk)
            pairs = [(mol_id, smiles_by_id.get(mol_id, "")) for mol_id in chunk]
            remaining = None if limit is None else limit - len(matched)
            matched.extend(verify_matches(pairs, smarts, limit=remaining, binaries=binaries))
            if limit is not None and len(matched) >= limit:
                break

        if not matched:
            return []
        return self._entries_by_id(matched)

    # --------------------------------------------------------------- internals

    def _ids_for(self, datasets: Sequence[str] | None) -> list[int]:
        """Row ids in the (optionally dataset-filtered) pool, ascending."""
        where, params = self._dataset_clause(datasets)
        rows = self._conn.execute(f"SELECT id FROM molecules{where} ORDER BY id", params).fetchall()
        return [int(row["id"]) for row in rows]

    def _any_rows(self, datasets: Sequence[str] | None) -> bool:
        """True when the (optionally dataset-filtered) pool holds anything."""
        where, params = self._dataset_clause(datasets)
        row = self._conn.execute(f"SELECT 1 FROM molecules{where} LIMIT 1", params).fetchone()
        return row is not None

    def _key_vocabulary(self) -> list[str]:
        """Every key token this index knows, cached for the process lifetime."""
        if self._key_vocab is None:
            try:
                rows = self._conn.execute("SELECT token FROM key_bitmaps ORDER BY token").fetchall()
            except sqlite3.OperationalError:
                rows = []  # index predates the topological screen
            self._key_vocab = [row["token"] for row in rows]
        return self._key_vocab

    def _key_bitmap(self, token: str) -> np.ndarray | None:
        """The bitmap for one key, cached. Only the keys a query needs load."""
        if token in self._key_cache:
            return self._key_cache[token]
        row = self._conn.execute(
            "SELECT bits FROM key_bitmaps WHERE token = ?", (token,)
        ).fetchone()
        bits = (
            np.frombuffer(zlib.decompress(row["bits"]), dtype=np.uint8) if row is not None else None
        )
        self._key_cache[token] = bits
        return bits

    def _screen_by_keys(self, pattern: Any, datasets: Sequence[str] | None) -> list[int] | None:
        """Candidate ids from the topological screen, or None to fall back.

        For each screenable feature of the pattern the compatible keys are
        OR'd together, and those unions are AND'd across features. Only the
        handful of bitmaps a query actually compiles to are read, so this
        costs a few milliseconds rather than the ~1.3 s the full
        fingerprint matrix took to load.
        """
        vocabulary = self._key_vocabulary()
        if not vocabulary:
            return None
        per_feature = compile_query(pattern, vocabulary)
        if per_feature is None:
            return None

        survivors: np.ndarray | None = None
        for tokens in per_feature:
            if not tokens:
                return []  # nothing in the index can satisfy this feature
            union: np.ndarray | None = None
            for token in tokens:
                bits = self._key_bitmap(token)
                if bits is None:
                    continue
                union = bits.copy() if union is None else (union | bits)
            if union is None:
                return []
            survivors = union if survivors is None else (survivors & union)
        if survivors is None:
            return None

        # Bit (id - 1) is set, so the id is the bit position plus one.
        ids = (np.flatnonzero(np.unpackbits(survivors, bitorder="little")) + 1).tolist()
        if datasets:
            allowed = set(self._ids_for(datasets))
            ids = [mol_id for mol_id in ids if mol_id in allowed]
        return ids

    def _ids_for_size(self, size: SizeFilter, datasets: Sequence[str] | None) -> set[int]:
        """Row ids satisfying a composition constraint."""
        where, ds_params = self._dataset_clause(datasets, prefix="AND")
        clauses, params = _size_clauses(size)
        sql = f"SELECT id FROM molecules WHERE {' AND '.join(clauses)}{where}"
        return {int(row["id"]) for row in self._conn.execute(sql, (*params, *ds_params))}

    def _id_mask(self, datasets: Sequence[str] | None) -> np.ndarray:
        """Boolean mask over ``self._fp_ids`` selecting *datasets*."""
        wanted = np.fromiter(self._ids_for(datasets), dtype=np.int64)
        return np.isin(np.asarray(self._fp_ids, dtype=np.int64), wanted)

    def _smiles_for(self, ids: Sequence[int]) -> dict[int, str]:
        """SMILES for a batch of row ids."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, smiles FROM molecules WHERE id IN ({placeholders})", tuple(ids)
        ).fetchall()
        return {int(row["id"]): (row["smiles"] or "") for row in rows}

    def _structures_for(self, ids: Sequence[int]) -> tuple[dict[int, str], dict[int, bytes] | None]:
        """SMILES and, when the index has them, binary molecules for a batch.

        Both come from one query: fetching them separately doubled the number
        of 5,000-parameter IN lookups for no benefit.
        """
        if not ids:
            return {}, None
        if not self._has_mol_bin():
            return self._smiles_for(ids), None
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, smiles, mol_bin FROM molecules WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        smiles: dict[int, str] = {}
        binaries: dict[int, bytes] = {}
        for row in rows:
            mol_id = int(row["id"])
            smiles[mol_id] = row["smiles"] or ""
            if row["mol_bin"]:
                binaries[mol_id] = row["mol_bin"]
        return smiles, binaries

    def _has_mol_bin(self) -> bool:
        """Whether this index stores prebuilt binary molecules."""
        if self._mol_bin is None:
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(molecules)")}
            self._mol_bin = "mol_bin" in columns
        return self._mol_bin

    def _fingerprints_for(self, datasets: Sequence[str] | None):
        """Return ``(matrix, row_index_to_id)`` for the pool.

        The full-index matrix is loaded once and cached (~113 MB at 880k
        molecules); a dataset filter selects rows from it rather than
        re-reading the column.
        """
        if self._fp_matrix is None:
            rows = self._conn.execute("SELECT id, pattern_fp FROM molecules ORDER BY id").fetchall()
            self._fp_ids = [int(row["id"]) for row in rows]
            self._fp_matrix = fingerprint_matrix(
                [row["pattern_fp"] for row in rows], fp_bytes=FP_BYTES
            )
        if not datasets:
            return self._fp_matrix, self._fp_ids

        # Vectorised: the equivalent Python loop ran once per indexed molecule
        # (881,757 iterations) and measured ~750 ms per dataset-filtered query.
        mask = self._id_mask(datasets)
        positions = [mol_id for mol_id, keep in zip(self._fp_ids or [], mask, strict=True) if keep]
        return self._fp_matrix[mask], positions

    def _entries_by_id(self, ids: Sequence[int]) -> list[MoleculeIndexEntry]:
        entries: list[MoleculeIndexEntry] = []
        for chunk in _chunks(list(ids), 500):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT id, {_ENTRY_COLUMNS} FROM molecules WHERE id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            by_id = {int(row["id"]): row for row in rows}
            entries.extend(self._row_to_entry(by_id[i]) for i in chunk if i in by_id)
        return entries

    def _select(
        self,
        condition: str,
        params: tuple[Any, ...],
        *,
        datasets: Sequence[str] | None,
        limit: int | None,
        size: SizeFilter | None = None,
    ) -> list[MoleculeIndexEntry]:
        where, ds_params = self._dataset_clause(datasets, prefix="AND")
        size_sql, size_params = _size_suffix(size)
        sql = (
            f"SELECT {_ENTRY_COLUMNS} FROM molecules "
            f"WHERE {condition}{where}{size_sql} ORDER BY id" + _limit_clause(limit)
        )
        rows = self._conn.execute(sql, (*params, *ds_params, *size_params)).fetchall()
        return [self._row_to_entry(row) for row in rows]

    @staticmethod
    def _dataset_clause(
        datasets: Sequence[str] | None,
        *,
        prefix: str = "WHERE",
        alias: str = "",
    ) -> tuple[str, tuple[str, ...]]:
        if not datasets:
            return "", ()
        column = f"{alias}.dataset" if alias else "dataset"
        placeholders = ",".join("?" for _ in datasets)
        return f" {prefix} {column} IN ({placeholders})", tuple(datasets)

    def _rows_to_entries(
        self, rows: Sequence[sqlite3.Row], *, limit: int | None
    ) -> list[MoleculeIndexEntry]:
        seen: set[tuple[str, str]] = set()
        entries: list[MoleculeIndexEntry] = []
        for row in rows:
            entry = self._row_to_entry(row)
            ident = (entry.dataset, entry.key)
            if ident in seen:
                continue
            seen.add(ident)
            entries.append(entry)
            if limit is not None and len(entries) >= limit:
                break
        return entries

    def _row_to_entry(self, row: sqlite3.Row) -> MoleculeIndexEntry:
        try:
            element_counts = json.loads(row["element_counts"] or "{}")
        except json.JSONDecodeError:
            element_counts = {}
        return MoleculeIndexEntry(
            dataset=row["dataset"],
            key=row["key"],
            smiles=row["smiles"] or "",
            canonical_smiles=row["canonical_smiles"] or "",
            n_atoms=int(row["n_atoms"] or 0),
            label=row["label"] or f"{row['dataset']}:{row['key']}",
            formula=row["formula"] or "",
            inchikey=row["inchikey"] or "",
            element_counts=dict(element_counts),
            names=self._names_for(row),
            qc_pass=None if row["qc_pass"] is None else bool(row["qc_pass"]),
        )

    @staticmethod
    def _names_for(row: sqlite3.Row) -> tuple[str, ...]:
        """Split the joined synonyms selected alongside the entry."""
        if "names" not in row.keys():
            return ()
        joined = row["names"]
        if not joined:
            return ()
        return tuple(sorted(part for part in str(joined).split("\x1f") if part))


def _limit_clause(limit: int | None) -> str:
    """``LIMIT n`` when a limit is set, else the empty string."""
    return f" LIMIT {int(limit)}" if limit is not None else ""


def _size_clauses(size: SizeFilter | None, *, alias: str = "") -> tuple[list[str], list[Any]]:
    """SQL conditions and parameters for a composition constraint.

    Element counts go through an EXISTS against ``element_counts``, which
    ``idx_elem`` serves as a range scan; ``n_heavy`` / ``n_atoms`` compare
    directly. Both are conditions rather than joins, so a molecule still
    appears at most once and LIMIT stays safe in SQL.
    """
    if not size:
        return [], []
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    for element, low, high in size.element_counts:
        clauses.append(
            f"EXISTS (SELECT 1 FROM element_counts e WHERE e.mol = {prefix}id "
            "AND e.element = ? AND e.count BETWEEN ? AND ?)"
        )
        params.extend((element, int(low), int(high)))
    for column, low, high in size.column_counts:
        # Interpolated into SQL, so it must come from the fixed allow-list
        # rather than from anything a user typed.
        if column not in _SIZE_COLUMNS_ALLOWED:
            raise ValueError(f"Unknown size column: {column}")
        clauses.append(f"{prefix}{column} BETWEEN ? AND ?")
        params.extend((int(low), int(high)))
    if size.qc_pass is not None:
        # "= ?" rather than "IS ?": a NULL verdict must not satisfy either
        # branch, because an unrecorded result is not a recorded one.
        clauses.append(f"{prefix}qc_pass = ?")
        params.append(1 if size.qc_pass else 0)
    return clauses, params


def _size_suffix(size: SizeFilter | None, *, alias: str = "") -> tuple[str, list[Any]]:
    """:func:`_size_clauses` rendered as an ``AND ...`` suffix."""
    clauses, params = _size_clauses(size, alias=alias)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def _prefixed(alias: str) -> str:
    """Entry columns qualified with a table alias (for joined queries)."""
    plain = ", ".join(f"{alias}.{c.strip()}" for c in _BASE_COLUMNS.split(","))
    return plain + ", " + _NAMES_SUBQUERY.format(alias=f"{alias}.")


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


__all__ = ["IndexVersionError", "SqliteSearchBackend"]
