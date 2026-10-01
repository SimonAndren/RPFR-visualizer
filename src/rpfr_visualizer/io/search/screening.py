"""Substructure screening with RDKit pattern fingerprints.

A SMARTS query over 190k molecules is far too slow to run RDKit's
``HasSubstructMatch`` on every row. Instead each molecule stores a pattern
fingerprint at build time; a query fingerprint is computed once, and any
molecule whose fingerprint lacks a bit the query sets cannot possibly match.
That screen is a single vectorised pass over a ``(n, FP_BYTES)`` byte matrix,
and only the survivors are verified with RDKit.

**The screen is a filter, never an answer.** It can only ever return a superset
of the true matches, and every survivor is verified before it is reported, so
results are identical to brute force. The one way that guarantee could break is
a query whose fingerprint is unavailable or empty — RDKit cannot encode every
SMARTS construct — so those cases fall back to verifying everything rather than
silently dropping hits. ``tests/unit/io/test_smarts_screening.py`` pins the
equivalence.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from rpfr_visualizer.io.search.schema import FP_BYTES, FP_SIZE


def compute_pattern_fp(smiles: str, *, fp_size: int = FP_SIZE) -> bytes | None:
    """Return the pattern fingerprint of *smiles* as raw bytes, or None."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _fp_bytes(mol, fp_size=fp_size)


def compute_query_fp(pattern: Any, *, fp_size: int = FP_SIZE) -> bytes | None:
    """Return the screening fingerprint for a parsed SMARTS query.

    Returns None when RDKit cannot build one — the caller must then verify
    every candidate instead of screening.
    """
    if pattern is None:
        return None
    try:
        return _fp_bytes(pattern, fp_size=fp_size)
    except Exception:
        return None


def _fp_bytes(mol: Any, *, fp_size: int) -> bytes | None:
    """Pack a pattern fingerprint into bytes (bit i → byte i//8, bit i%8)."""
    from rdkit import Chem, DataStructs

    try:
        fp = Chem.PatternFingerprint(mol, fpSize=fp_size)
    except Exception:
        return None
    if fp is None:
        return None
    return bytes(DataStructs.BitVectToBinaryText(fp))


def fingerprint_matrix(blobs: Sequence[bytes | None], *, fp_bytes: int = FP_BYTES) -> np.ndarray:
    """Stack stored fingerprints into an ``(n, fp_bytes)`` uint8 matrix.

    A missing fingerprint becomes all-zero, which never survives a non-empty
    screen — correct, because a molecule with no parseable SMILES cannot match
    a substructure query anyway.
    """
    matrix = np.zeros((len(blobs), fp_bytes), dtype=np.uint8)
    for i, blob in enumerate(blobs):
        if not blob:
            continue
        row = np.frombuffer(blob, dtype=np.uint8)
        matrix[i, : row.shape[0]] = row[:fp_bytes]
    return matrix


def screen_candidates(matrix: np.ndarray, query_fp: bytes | None) -> np.ndarray:
    """Return row indices whose fingerprint is a superset of *query_fp*.

    With no usable query fingerprint every row is a candidate, so the caller
    verifies everything rather than losing matches.
    """
    n = matrix.shape[0]
    if query_fp is None:
        return np.arange(n)
    query = np.frombuffer(query_fp, dtype=np.uint8)
    columns = np.nonzero(query)[0]
    if columns.size == 0:
        return np.arange(n)
    columns = columns[columns < matrix.shape[1]]
    if columns.size == 0:
        return np.arange(n)
    wanted = query[columns]
    return np.nonzero(((matrix[:, columns] & wanted) == wanted).all(axis=1))[0]


def verify_matches(
    smiles_by_index: Sequence[tuple[int, str]],
    smarts: str,
    *,
    limit: int | None = None,
    binaries: Mapping[int, bytes] | None = None,
) -> Iterator[int]:
    """Yield indices whose structure genuinely contains *smarts*.

    Stops early once *limit* matches are found — an interactive panel asking for
    the first 200 hits should not pay for the full dataset.

    Rebuilding molecules is the dominant cost here, not the matching:
    ``HasSubstructMatch`` runs at ~630k mol/s once a molecule exists, while
    ``MolFromSmiles`` manages ~50k/s and accounted for 80% of a substructure
    query. *binaries* supplies ``Mol.ToBinary()`` blobs for the same indices,
    which deserialise at ~219k/s — 4.3x faster. Any index missing from it
    falls back to parsing its SMILES, so an index built before ``mol_bin``
    existed still works.
    """
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError as exc:  # pragma: no cover - RDKit is a viz extra
        raise ImportError(
            "RDKit is required for substructure search. Install with: uv sync --extra viz"
        ) from exc

    query = Chem.MolFromSmarts(smarts)
    if query is None:
        return

    found = 0
    for index, smiles in smiles_by_index:
        mol = None
        blob = None if binaries is None else binaries.get(index)
        if blob:
            try:
                mol = Chem.Mol(blob)
            except Exception:  # noqa: BLE001 — a stale blob must not kill the query
                mol = None
        if mol is None:
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if mol.HasSubstructMatch(query):
            yield index
            found += 1
            if limit is not None and found >= limit:
                return


__all__ = [
    "compute_pattern_fp",
    "compute_query_fp",
    "fingerprint_matrix",
    "screen_candidates",
    "verify_matches",
]
