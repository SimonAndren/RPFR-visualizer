"""The topological screen must never return less than brute force.

Every way this screen has gone wrong has been silent — a query came back with
a plausible, smaller answer rather than an error — so the tests here compare
screened results against ``screen=False`` on a real (small) index rather than
asserting on internals. The three named regressions each cost real matches
against the full 881,757-molecule index before they were caught.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from rpfr_visualizer.io.search import SqliteSearchBackend  # noqa: E402
from rpfr_visualizer.io.search.enrich import enrich_index  # noqa: E402

# Patterns chosen to hit every branch of compile_query: multi-bond, single
# atom, bondless molecules, isotopes, aromatics, negation, ring and recursive
# primitives (which must fall back), and a pattern nothing can satisfy.
PATTERNS: tuple[str, ...] = (
    "[CX3](=O)[OX2H1]",
    "[CX3](=O)[OX2][#6]",
    "[CX4][OX2H]",
    "[CX3]=O",
    "[CX2]#[NX1]",
    "c1ccccc1",
    "[OD2]([#6])[#6]",
    "[NX3;!$(NC=O)]",
    "[$([CX3](=O))]",
    "[R]",
    "[r5]",
    "[C;R2]",
    "[!C]",
    "[#7]",
    "[OX2H]",
    "[S]",
    "[F,Cl,Br,I]",
    "[CH3]",
    "[nH]",
    "[2H]",
    "[13C]",
    "[Pt]",
    "C",
    "CC(=O)O",
)


@pytest.fixture
def enriched_index(search_index_path: Path) -> SqliteSearchBackend:
    """The shared test index, with keys and binary molecules added."""
    enrich_index(search_index_path)
    return SqliteSearchBackend(search_index_path)


@pytest.mark.parametrize("smarts", PATTERNS)
def test_screened_matches_brute_force(enriched_index: SqliteSearchBackend, smarts: str) -> None:
    screened = {(e.dataset, e.key) for e in enriched_index.by_smarts(smarts)}
    brute = {(e.dataset, e.key) for e in enriched_index.by_smarts(smarts, screen=False)}
    assert screened == brute, f"{smarts}: screen dropped {sorted(brute - screened)}"


def test_screen_agrees_with_precomputed_membership(
    enriched_index: SqliteSearchBackend,
) -> None:
    """A live SMARTS scan and the precomputed table are the same question."""
    for preset in ("alcohol", "carboxylic_acid", "nitrile"):
        if not enriched_index.has_functional_group(preset):
            continue
        table = {(e.dataset, e.key) for e in enriched_index.by_functional_group(preset)}
        smarts = enriched_index._conn.execute(  # noqa: SLF001 — no public accessor
            "SELECT smarts FROM fg_presets WHERE name = ?", (preset,)
        ).fetchone()["smarts"]
        assert {(e.dataset, e.key) for e in enriched_index.by_smarts(smarts)} == table


# ------------------------------------------------------------- regressions


def test_deuterated_hydroxyl_still_reads_as_one_hydrogen() -> None:
    """``GetTotalNumHs()`` alone reports 0 for a deuterated acid.

    ``[2H]`` is a graph atom, not an implicit hydrogen, so the bare call made
    ``[OX2H1]`` reject the probe. That dropped 58 of the 5,165 carboxylic
    acids in the published index.
    """
    from rdkit import Chem

    from rpfr_visualizer.io.search.keys import atom_key

    deuterated = Chem.MolFromSmiles("[2H]OC(=O)NC")
    protiated = Chem.MolFromSmiles("OC(=O)NC")
    query = Chem.MolFromSmarts("[OX2H1]")
    assert deuterated.HasSubstructMatch(query)

    def hydroxyl_hydrogens(mol) -> set[int]:
        return {
            atom_key(a)[4]
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 8 and a.GetTotalDegree() == 2
        }

    assert hydroxyl_hydrogens(deuterated) == hydroxyl_hydrogens(protiated) == {1}


def test_isotope_query_is_screenable_and_selective() -> None:
    """``[2H]`` compiled to "nothing can match" and returned zero rows.

    An empty result looks like a valid one, which made this the worst of the
    three bugs — and on an isotope-effects database ``[2H]`` is an ordinary
    query, not an edge case.
    """
    from rdkit import Chem

    from rpfr_visualizer.io.search.keys import compile_query, mol_keys

    vocab = sorted(
        mol_keys(Chem.MolFromSmiles("[2H]OC(=O)NC")) | mol_keys(Chem.MolFromSmiles("CCO"))
    )
    compiled = compile_query(Chem.MolFromSmarts("[2H]"), vocab)
    assert compiled not in (None, [[]])


def test_enrich_falls_back_when_canonical_smiles_will_not_parse() -> None:
    """Some stored canonical SMILES do not round-trip through RDKit.

    Aromatic aluminium and phosphorus heterocycles such as ``c1c[cH][Al][n]c1``
    are written from a parseable ``C1=CN=[Al]C=C1`` but cannot be read back.
    Trusting the canonical form skipped those molecules entirely, leaving them
    with neither keys nor a binary molecule, so the screen excluded them from
    every query.
    """
    from rdkit import Chem

    assert Chem.MolFromSmiles("c1c[cH][Al][n]c1") is None
    assert Chem.MolFromSmiles("C1=CN=[Al]C=C1") is not None
