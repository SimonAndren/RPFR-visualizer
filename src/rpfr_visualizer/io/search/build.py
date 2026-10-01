"""Build the SQLite molecule search index from HDF5 datasets.

Run once per data release; the resulting file is published alongside the HDF5
data so users never build it. The expensive parts are RDKit descriptor
derivation and fingerprinting (~0.4 ms/molecule together), which dominate the
runtime at full dataset size.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py

from rpfr_visualizer.io.search.chem import normalize_formula, smiles_descriptors, stored_qc_pass
from rpfr_visualizer.io.search.schema import (
    FP_KIND,
    FP_SIZE,
    FTS_REBUILD_SQL,
    FTS_SQL,
    INDEX_VERSION,
    META_BUILT_AT,
    META_FG_CATALOG_HASH,
    META_FP_KIND,
    META_FP_SIZE,
    META_HAS_FTS,
    META_INDEX_VERSION,
    META_RDKIT_VERSION,
    META_SOURCE_SIGNATURE,
    SCHEMA_SQL,
)
from rpfr_visualizer.io.search.screening import (
    compute_pattern_fp,
    compute_query_fp,
    fingerprint_matrix,
    screen_candidates,
    verify_matches,
)

if TYPE_CHECKING:
    from rpfr_visualizer.io.providers import DatasetCatalog


@dataclass
class BuildReport:
    """What a build produced."""

    path: Path
    n_molecules: int = 0
    n_datasets: int = 0
    n_names: int = 0
    n_fg_links: int = 0
    has_fts: bool = False
    per_dataset: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"{self.n_molecules} molecules across {self.n_datasets} dataset(s)",
            f"{self.n_names} name(s)",
            f"{self.n_fg_links} functional-group link(s)",
            f"FTS5 {'on' if self.has_fts else 'off'}",
        ]
        return f"{self.path}: " + ", ".join(parts)


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None else str(value)


def _fg_catalog_hash(presets: Any) -> str:
    payload = json.dumps(
        [[p.name, p.smarts or ""] for p in presets], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_index(
    catalog: DatasetCatalog,
    out_path: Path | str,
    *,
    datasets: tuple[str, ...] | None = None,
    with_functional_groups: bool = True,
    fp_size: int = FP_SIZE,
    progress: bool = True,
    overwrite: bool = True,
) -> BuildReport:
    """Build a search index from every registered HDF5 dataset.

    Parameters
    ----------
    catalog:
        Registered HDF5 providers to index.
    out_path:
        Destination ``.sqlite`` file.
    with_functional_groups:
        Precompute membership for the curated preset catalog so the UI's
        functional-group filter is a table lookup instead of a scan.
    """
    from rpfr_visualizer.io.providers import HDF5Provider

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Index already exists: {out} (pass overwrite=True)")
        out.unlink()

    selected = datasets or catalog.datasets
    report = BuildReport(path=out)

    conn = sqlite3.connect(out)
    try:
        conn.executescript(SCHEMA_SQL)
        has_fts = _try_create_fts(conn)
        report.has_fts = has_fts

        signature: list[list[Any]] = []
        mol_rows: list[tuple] = []
        element_rows: list[tuple] = []
        next_id = 1

        for dataset in selected:
            provider = catalog.get_provider(dataset)
            if not isinstance(provider, HDF5Provider):
                continue
            keys = provider.list_keys()
            if progress:
                print(f"[{dataset}] indexing {len(keys)} molecules from {provider.path.name}")

            with h5py.File(provider.path, "r") as h5:
                schema_version = _decode(h5.attrs.get("schema_version", ""))
                for key in keys:
                    group = h5[key]
                    smiles = _decode(group.attrs.get("smiles", ""))
                    n_atoms = int(group.attrs.get("n_atoms", 0) or 0)
                    desc = smiles_descriptors(smiles) if smiles else {}
                    canon = str(desc.get("canonical_smiles") or "")
                    formula = str(desc.get("formula") or "")
                    inchikey = str(desc.get("inchikey") or "")
                    counts = dict(desc.get("element_counts") or {})
                    n_heavy = sum(n for el, n in counts.items() if el != "H")
                    label = f"{dataset}:{key}"
                    if smiles:
                        trimmed = smiles[:40] + ("…" if len(smiles) > 40 else "")
                        label = f"{label} ({trimmed})"
                    mol_rows.append(
                        (
                            next_id,
                            dataset,
                            key,
                            smiles,
                            canon,
                            normalize_formula(formula) or formula,
                            inchikey,
                            inchikey.split("-")[0] if inchikey else "",
                            n_atoms,
                            n_heavy,
                            json.dumps(counts, sort_keys=True),
                            label,
                            _qc_pass(group),
                            compute_pattern_fp(smiles, fp_size=fp_size) if smiles else None,
                        )
                    )
                    element_rows.extend(
                        (next_id, element, int(count)) for element, count in counts.items()
                    )
                    next_id += 1

            n_in_dataset = len(keys)
            report.per_dataset[dataset] = n_in_dataset
            temperatures = json.dumps(list(provider.temperatures))
            conn.execute(
                "INSERT INTO datasets (dataset, source_file, n_molecules, temperatures, "
                "schema_version) VALUES (?, ?, ?, ?, ?)",
                (dataset, provider.path.name, n_in_dataset, temperatures, schema_version),
            )
            # Basename only: a published index must survive the user putting
            # their HDF5 files somewhere else entirely.
            signature.append([dataset, provider.path.name, n_in_dataset])

        conn.executemany(
            "INSERT INTO molecules (id, dataset, key, smiles, canonical_smiles, formula, "
            "inchikey, inchikey14, n_atoms, n_heavy, element_counts, label, qc_pass, "
            "pattern_fp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            mol_rows,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO element_counts (mol, element, count) VALUES (?,?,?)",
            element_rows,
        )
        report.n_molecules = len(mol_rows)
        report.n_datasets = len(report.per_dataset)

        report.n_names = _populate_names(conn, progress=progress)
        if has_fts:
            conn.execute(FTS_REBUILD_SQL)

        fg_hash = ""
        if with_functional_groups:
            report.n_fg_links, fg_hash = _populate_functional_groups(conn, progress=progress)

        _write_meta(
            conn,
            {
                META_INDEX_VERSION: str(INDEX_VERSION),
                META_BUILT_AT: datetime.now(timezone.utc).isoformat(timespec="seconds"),
                META_RDKIT_VERSION: _rdkit_version(),
                META_FP_KIND: FP_KIND,
                META_FP_SIZE: str(fp_size),
                META_HAS_FTS: "1" if has_fts else "0",
                META_FG_CATALOG_HASH: fg_hash,
                META_SOURCE_SIGNATURE: json.dumps(signature),
            },
        )
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()

    # VACUUM needs its own connection outside the transaction.
    vacuum = sqlite3.connect(out)
    try:
        vacuum.execute("VACUUM")
    finally:
        vacuum.close()

    if progress:
        print(report.summary())
    return report


def _qc_pass(group: Any) -> int | None:
    """Read the stored QC verdict, if the file records one.

    Thin wrapper over :func:`rpfr_visualizer.io.search.chem.stored_qc_pass` kept for
    the ``int | None`` shape the SQLite column wants.
    """
    verdict = stored_qc_pass(group)
    return None if verdict is None else int(verdict)


def _try_create_fts(conn: sqlite3.Connection) -> bool:
    """Create the FTS5 name table; report whether this build has it."""
    try:
        conn.executescript(FTS_SQL)
    except sqlite3.OperationalError:
        return False
    return True


def _populate_names(conn: sqlite3.Connection, *, progress: bool = True) -> int:
    """Attach the shipped offline synonyms to molecules by canonical SMILES."""
    from rpfr_visualizer.io.search.chem import canonical_smiles
    from rpfr_visualizer.io.search.names import synonym_seed

    seed = synonym_seed()
    if not seed:
        return 0

    by_canonical: dict[str, list[str]] = {}
    for name, smiles in seed.items():
        canon = canonical_smiles(smiles) or smiles
        by_canonical.setdefault(canon, []).append(name)

    rows: list[tuple] = []
    for canon, names in by_canonical.items():
        matches = conn.execute(
            "SELECT id FROM molecules WHERE canonical_smiles = ?", (canon,)
        ).fetchall()
        for (mol_id,) in matches:
            rows.extend((mol_id, name, "synonym_seed") for name in names)

    conn.executemany("INSERT OR IGNORE INTO names (mol, name, source) VALUES (?,?,?)", rows)
    if progress and rows:
        print(f"[names] linked {len(rows)} synonym(s)")
    return len(rows)


def _populate_functional_groups(
    conn: sqlite3.Connection, *, progress: bool = True
) -> tuple[int, str]:
    """Precompute curated functional-group membership for every molecule."""
    from rpfr_visualizer.analysis.functional_groups import FUNCTIONAL_GROUP_PRESETS

    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        if progress:
            print("[fg] RDKit unavailable — skipping functional-group membership")
        return 0, ""

    presets = [p for p in FUNCTIONAL_GROUP_PRESETS if p.smarts]
    conn.executemany(
        "INSERT OR IGNORE INTO fg_presets (name, description, smarts, element) VALUES (?,?,?,?)",
        [(p.name, p.description, p.smarts, p.element) for p in presets],
    )
    preset_ids = {name: fg_id for fg_id, name in conn.execute("SELECT id, name FROM fg_presets")}

    rows = conn.execute("SELECT id, smiles, pattern_fp FROM molecules ORDER BY id").fetchall()
    ids = [int(r[0]) for r in rows]
    smiles_by_id = {int(r[0]): (r[1] or "") for r in rows}
    matrix = fingerprint_matrix([r[2] for r in rows])

    links: list[tuple] = []
    for preset in presets:
        pattern = Chem.MolFromSmarts(preset.smarts)
        if pattern is None:
            continue
        keep = screen_candidates(matrix, compute_query_fp(pattern))
        candidates = [(ids[i], smiles_by_id[ids[i]]) for i in keep]
        matched = list(verify_matches(candidates, preset.smarts))
        links.extend((preset_ids[preset.name], mol_id, 1) for mol_id in matched)
        if progress:
            print(f"[fg] {preset.name}: {len(matched)} molecule(s)")

    conn.executemany(
        "INSERT OR IGNORE INTO fg_membership (fg, mol, n_matches) VALUES (?,?,?)", links
    )
    return len(links), _fg_catalog_hash(presets)


def _write_meta(conn: sqlite3.Connection, meta: dict[str, str]) -> None:
    conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", sorted(meta.items()))


def _rdkit_version() -> str:
    try:
        import rdkit

        return str(rdkit.__version__)
    except ImportError:
        return ""


__all__ = ["BuildReport", "build_index"]
