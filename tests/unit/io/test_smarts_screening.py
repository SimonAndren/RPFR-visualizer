"""The fingerprint screen must never change a substructure result.

Screening is an optimisation: it prunes candidates before RDKit verification.
If it ever drops a true match the panel silently under-reports, which is worse
than being slow. These tests pin screened results to brute force.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.analysis.functional_groups import FUNCTIONAL_GROUP_PRESETS
from rpfr_visualizer.io.search import SqliteSearchBackend
from rpfr_visualizer.io.search.screening import (
    compute_pattern_fp,
    compute_query_fp,
    fingerprint_matrix,
    screen_candidates,
)

pytest.importorskip("rdkit")

# Constructs chosen to stress what a pattern fingerprint can and cannot encode:
# recursive SMARTS, atom lists, ring membership/size, degree and H counts.
TRICKY_SMARTS: tuple[str, ...] = (
    "[CX3](=O)[OX2H1]",
    "[CX3]=O",
    "c1ccccc1",
    "[NX3;!$(NC=O)]",
    "[$([NX3](C)(C)C)]",
    "[CX2]#[NX1]",
    "[OX2H]",
    "[OD2]([#6])[#6]",
    "C1OC1",
    "[r3]",
    "[r5]",
    "[R1]",
    "[R2]",
    "[CH3]",
    "[CH2]",
    "[#6;X4]",
    "[C,N]",
    "[!C;!c]",
    "[$(C=O)]",
    "[CX4H3][CX4H2][OX2H]",
    "[#7]~[#6]",
    "[*]",
)


def _identity(entries) -> set[tuple[str, str]]:
    return {(e.dataset, e.key) for e in entries}


@pytest.mark.parametrize("smarts", TRICKY_SMARTS)
def test_screened_smarts_equals_brute_force(search_index_path: Path, smarts: str) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert _identity(backend.by_smarts(smarts, screen=True)) == _identity(
        backend.by_smarts(smarts, screen=False)
    )


@pytest.mark.parametrize("preset", [p for p in FUNCTIONAL_GROUP_PRESETS if p.smarts])
def test_every_catalog_preset_screens_correctly(search_index_path: Path, preset) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert _identity(backend.by_smarts(preset.smarts, screen=True)) == _identity(
        backend.by_smarts(preset.smarts, screen=False)
    )


def test_precomputed_membership_matches_live_matching(search_index_path: Path) -> None:
    """The stored fg_membership table must agree with running the SMARTS now."""
    backend = SqliteSearchBackend(search_index_path)
    for preset in FUNCTIONAL_GROUP_PRESETS:
        if not preset.smarts or not backend.has_functional_group(preset.name):
            continue
        stored = _identity(backend.by_functional_group(preset.name))
        live = _identity(backend.by_smarts(preset.smarts, screen=False))
        assert stored == live, f"{preset.name} membership drifted from its SMARTS"


def test_screen_returns_a_superset_never_a_subset() -> None:
    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC#N", ""]
    matrix = fingerprint_matrix([compute_pattern_fp(s) for s in smiles])
    from rdkit import Chem

    for smarts in TRICKY_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        truth = {
            i
            for i, s in enumerate(smiles)
            if s and (m := Chem.MolFromSmiles(s)) and m.HasSubstructMatch(pattern)
        }
        candidates = set(screen_candidates(matrix, compute_query_fp(pattern)).tolist())
        assert truth <= candidates, f"{smarts} screened out a real match"


def test_unusable_query_fingerprint_keeps_every_candidate() -> None:
    """No fingerprint means verify everything, never silently return nothing."""
    matrix = fingerprint_matrix([compute_pattern_fp(s) for s in ["CCO", "CCN"]])
    assert screen_candidates(matrix, None).tolist() == [0, 1]
    assert screen_candidates(matrix, bytes(128)).tolist() == [0, 1]


def test_missing_fingerprint_is_never_a_false_match() -> None:
    """A molecule with no parseable SMILES cannot satisfy a substructure query."""
    matrix = fingerprint_matrix([compute_pattern_fp("CCO"), None])
    from rdkit import Chem

    query = compute_query_fp(Chem.MolFromSmarts("[OX2H]"))
    assert 1 not in screen_candidates(matrix, query).tolist()
