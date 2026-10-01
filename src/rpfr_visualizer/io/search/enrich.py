"""Add the topological screen and binary molecules to an existing index.

Everything this writes is derivable from ``molecules.canonical_smiles``, which
the index already stores, so enriching does not need the HDF5 files or the
drive they live on. That matters: walking the seven HDF5 datasets is ~85% of a
full build (measured 1,857 molecules/s, about eight minutes for 881,757), while
the RDKit work here runs at 30-50k molecules/s and finishes in about two.

Written as a separate pass rather than folded only into :mod:`build` so an
already-published index can be upgraded in place, and so the same population
helpers serve both paths and cannot drift apart.
"""

from __future__ import annotations

import sqlite3
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rpfr_visualizer.io.search.keys import mol_keys
from rpfr_visualizer.io.search.schema import (
    INDEX_VERSION,
    KEY_KIND,
    META_HAS_MOL_BIN,
    META_INDEX_VERSION,
    META_KEY_BITS,
    META_KEY_KIND,
    SCHEMA_SQL,
)

#: Molecules per transaction while writing mol_bin.
_WRITE_CHUNK = 20_000

#: zlib level for key bitmaps. Level 6 already takes them from 238 MB to
#: 3.4 MB; 9 buys under 2% more for several times the build cost.
_COMPRESSION = 6


@dataclass
class EnrichReport:
    """What an enrich pass did."""

    path: Path
    n_molecules: int = 0
    n_parsed: int = 0
    n_mol_bin: int = 0
    n_keys: int = 0
    key_bytes: int = 0
    mol_bin_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.path.name}: {self.n_parsed:,}/{self.n_molecules:,} molecules parsed, "
            f"{self.n_keys:,} keys ({self.key_bytes / 1e6:.1f} MB), "
            f"{self.n_mol_bin:,} binary mols ({self.mol_bin_bytes / 1e6:.1f} MB)"
        )


def enrich_index(
    path: Path | str,
    *,
    progress: Callable[[str], None] | None = None,
) -> EnrichReport:
    """Add ``key_bitmaps`` and ``molecules.mol_bin`` to the index at *path*.

    Idempotent: rerunning replaces both rather than appending to them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Search index not found: {path}")
    report = EnrichReport(path=path)

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        # Bring an older file up to the current shape. Every statement is
        # CREATE ... IF NOT EXISTS, so this only adds what is missing.
        conn.executescript(SCHEMA_SQL)
        _add_mol_bin_column(conn)

        row = conn.execute("SELECT COUNT(*) AS n, MAX(id) AS top FROM molecules").fetchone()
        report.n_molecules = int(row["n"] or 0)
        max_id = int(row["top"] or 0)
        if max_id == 0:
            report.warnings.append("index holds no molecules; nothing to enrich")
            return report

        say(f"reading {report.n_molecules:,} molecules")
        n_bytes = (max_id + 7) // 8
        bitmaps: dict[str, np.ndarray] = {}
        pending: list[tuple[bytes, int]] = []

        for mol_id, mol in _iter_mols(conn):
            report.n_parsed += 1
            for token in mol_keys(mol):
                bits = bitmaps.get(token)
                if bits is None:
                    bits = np.zeros(n_bytes, dtype=np.uint8)
                    bitmaps[token] = bits
                # Bit (id - 1): ids are 1-based, so molecule 1 is bit 0.
                offset = mol_id - 1
                bits[offset // 8] |= 1 << (offset % 8)

            blob = mol.ToBinary()
            pending.append((blob, mol_id))
            report.mol_bin_bytes += len(blob)
            if len(pending) >= _WRITE_CHUNK:
                _write_mol_bin(conn, pending)
                report.n_mol_bin += len(pending)
                pending.clear()
                say(f"  {report.n_parsed:,} parsed")

        if pending:
            _write_mol_bin(conn, pending)
            report.n_mol_bin += len(pending)

        say(f"writing {len(bitmaps):,} key bitmaps")
        _write_bitmaps(conn, bitmaps, n_molecules=max_id)
        report.n_keys = len(bitmaps)
        report.key_bytes = int(
            conn.execute("SELECT COALESCE(SUM(LENGTH(bits)), 0) AS n FROM key_bitmaps").fetchone()[
                "n"
            ]
        )

        _write_meta(conn, n_bits=max_id, has_mol_bin=report.n_mol_bin > 0)
        conn.commit()
    finally:
        conn.close()

    # VACUUM has to run outside the write transaction, on its own connection.
    say("compacting")
    vacuum = sqlite3.connect(path)
    try:
        vacuum.execute("ANALYZE")
        vacuum.execute("VACUUM")
    finally:
        vacuum.close()
    return report


def _add_mol_bin_column(conn: sqlite3.Connection) -> None:
    """``ALTER TABLE`` for indexes built before mol_bin existed."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(molecules)")}
    if "mol_bin" not in columns:
        conn.execute("ALTER TABLE molecules ADD COLUMN mol_bin BLOB")


def _iter_mols(conn: sqlite3.Connection) -> Iterator[tuple[int, object]]:
    """Yield ``(id, Mol)`` for every molecule with a parseable structure.

    Rows with no SMILES are skipped. They cannot match a substructure query anyway, and leaving their bits
    clear is what makes the screen correctly exclude them.

    Canonical SMILES are preferred but **not trusted to parse**. A handful of
    stored canonical forms do not round-trip through RDKit — aromatic
    aluminium and phosphorus heterocycles such as ``c1c[cH][Al][n]c1``,
    written from ``C1=CN=[Al]C=C1`` — and taking the canonical form on faith
    skipped those molecules outright, leaving them with neither keys nor a
    binary mol so the screen silently dropped them from every query. Falling
    back to the raw SMILES recovers them.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    cursor = conn.execute(
        "SELECT id, canonical_smiles, smiles FROM molecules ORDER BY id",
    )
    for row in cursor:
        mol = None
        for smiles in (row["canonical_smiles"], row["smiles"]):
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                break
        if mol is None:
            continue
        yield int(row["id"]), mol


def _write_mol_bin(conn: sqlite3.Connection, pending: list[tuple[bytes, int]]) -> None:
    conn.executemany("UPDATE molecules SET mol_bin = ? WHERE id = ?", pending)


def _write_bitmaps(
    conn: sqlite3.Connection, bitmaps: dict[str, np.ndarray], *, n_molecules: int
) -> None:
    conn.execute("DELETE FROM key_bitmaps")
    conn.executemany(
        "INSERT INTO key_bitmaps(token, kind, n_mol, n_set, bits) VALUES (?, ?, ?, ?, ?)",
        [
            (
                token,
                token.partition("|")[0],
                n_molecules,
                int(np.unpackbits(bits).sum()),
                zlib.compress(bits.tobytes(), _COMPRESSION),
            )
            for token, bits in sorted(bitmaps.items())
        ],
    )


def _write_meta(conn: sqlite3.Connection, *, n_bits: int, has_mol_bin: bool) -> None:
    conn.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [
            (META_INDEX_VERSION, str(INDEX_VERSION)),
            (META_KEY_KIND, KEY_KIND),
            (META_KEY_BITS, str(n_bits)),
            (META_HAS_MOL_BIN, "1" if has_mol_bin else "0"),
        ],
    )
    conn.execute(f"PRAGMA user_version = {INDEX_VERSION}")


__all__ = ["EnrichReport", "enrich_index"]
