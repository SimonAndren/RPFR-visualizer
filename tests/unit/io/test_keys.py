"""Unit tests for the topological key screen.

The screen is only safe if it never returns less than brute force, and every
way it has gone wrong so far has been silent: a query returned a plausible,
smaller answer instead of raising. So these tests weight two things heavily —
that ``QueryAtom.Match`` agrees with the probe atoms we rebuild from stored
keys, and that the three bugs found against the full index stay fixed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem, RDLogger  # noqa: E402

from rpfr_visualizer.io.search.keys import (  # noqa: E402
    _probe_atom,  # noqa: E402
    atom_key,
    atom_token,
    compile_query,
    mol_keys,
    parse_token,
)

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------- probe atoms

# (smarts, element, isotope, aromatic, total_degree, n_hydrogens, expected)
#
# SMARTS ``X`` is *total* degree, so it counts hydrogens: a probe for
# (C, X3, H1) needs two heavy neighbours plus one H, not three plus one.
# Getting that convention backwards makes the probe disagree with the query
# it is being tested against, which drops matches without any error.
PROBE_CASES = [
    ("[CX3]", 6, 0, 0, 3, 0, True),
    ("[CX3]", 6, 0, 0, 4, 0, False),
    ("[CX3]", 6, 0, 0, 3, 1, True),
    ("[CX4]", 6, 0, 0, 4, 0, True),
    ("[CX4]", 6, 0, 0, 4, 3, True),
    ("[CX4]", 6, 0, 0, 3, 0, False),
    ("[OX2H1]", 8, 0, 0, 2, 1, True),
    ("[OX2H1]", 8, 0, 0, 2, 0, False),
    ("[OX2H1]", 8, 0, 0, 1, 1, False),
    ("[OX2H]", 8, 0, 0, 2, 1, True),
    ("[OD2]", 8, 0, 0, 2, 0, True),
    ("[OD2]", 8, 0, 0, 2, 1, False),
    ("[NX3]", 7, 0, 0, 3, 0, True),
    ("[NX3]", 7, 0, 0, 3, 2, True),
    ("[NX3]", 7, 0, 0, 2, 0, False),
    ("[NX1]", 7, 0, 0, 1, 0, True),
    ("[CX2]", 6, 0, 0, 2, 0, True),
    ("[CX3H1]", 6, 0, 0, 3, 1, True),
    ("[CX3H1]", 6, 0, 0, 3, 0, False),
    ("[#6]", 6, 0, 0, 4, 0, True),
    ("c", 6, 0, 1, 3, 1, True),
    ("c", 6, 0, 0, 3, 1, False),
    # Isotopes: [2H] is AtomAtomicNum 1 AND AtomIsotope 2, so a probe left at
    # the default isotope fails the second half.
    ("[2H]", 1, 2, 0, 1, 0, True),
    ("[2H]", 1, 0, 0, 1, 0, False),
    ("[13C]", 6, 13, 0, 4, 0, True),
    ("[13C]", 6, 0, 0, 4, 0, False),
    ("[#1]", 1, 2, 0, 1, 0, True),
]


@pytest.mark.parametrize(
    ("smarts", "element", "isotope", "aromatic", "degree", "hydrogens", "expected"),
    PROBE_CASES,
)
def test_probe_atom_agrees_with_smarts(
    smarts: str,
    element: int,
    isotope: int,
    aromatic: int,
    degree: int,
    hydrogens: int,
    expected: bool,
) -> None:
    query_atom = Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)
    probe = _probe_atom(element, isotope, aromatic, degree, hydrogens)
    assert probe is not None
    assert bool(query_atom.Match(probe)) is expected


def test_probe_rejects_impossible_key() -> None:
    """More hydrogens than total connections cannot describe a real atom."""
    assert _probe_atom(6, 0, 0, 2, 5) is None


# ------------------------------------------------------------ key extraction


def test_atom_key_counts_total_degree_not_heavy_degree() -> None:
    mol = Chem.MolFromSmiles("CC=O")
    carbonyl = next(a for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetTotalNumHs() == 1)
    element, isotope, aromatic, degree, hydrogens = atom_key(carbonyl)
    assert (element, isotope, aromatic) == (6, 0, 0)
    assert degree == 3  # two heavy neighbours + one hydrogen
    assert hydrogens == 1


def test_tokens_round_trip() -> None:
    """Tokens are text, not hashes, so parsing one must give back its parts."""
    mol = Chem.MolFromSmiles("CC(=O)O")
    tokens = mol_keys(mol)
    assert tokens

    atoms = {atom_key(a) for a in mol.GetAtoms()}
    bonds = {
        tuple(sorted((atom_key(b.GetBeginAtom()), atom_key(b.GetEndAtom()))))
        + (int(b.GetBondType()),)
        for b in mol.GetBonds()
    }

    seen_atoms, seen_bonds = set(), set()
    for token in tokens:
        kind, parts = parse_token(token)
        if kind == "a":
            assert atom_token(parts[0]) == token
            seen_atoms.add(parts[0])
        else:
            seen_bonds.add((parts[0], parts[1], parts[2]))
    assert seen_atoms == atoms
    assert seen_bonds == bonds


def test_token_is_orientation_independent() -> None:
    """A bond must key the same whichever end it is walked from."""
    assert mol_keys(Chem.MolFromSmiles("CCO")) == mol_keys(Chem.MolFromSmiles("OCC"))


def test_bondless_molecule_still_has_keys() -> None:
    """Methane has no bonds in RDKit, so a bond-only index would lose it."""
    methane = Chem.MolFromSmiles("C")
    assert methane.GetNumBonds() == 0
    keys = mol_keys(methane)
    assert keys
    assert all(token.startswith("a|") for token in keys)


# ------------------------------------------------------------ query compiling


def _vocabulary(*smiles: str) -> list[str]:
    vocab: set[str] = set()
    for text in smiles:
        vocab |= mol_keys(Chem.MolFromSmiles(text))
    return sorted(vocab)


def test_compile_returns_none_for_environment_dependent_primitives() -> None:
    """Ring and recursive queries cannot be judged from a lone probe."""
    vocab = _vocabulary("CC(=O)O", "c1ccccc1", "CCO")
    for smarts in ("[R]", "[r5]", "[$([CX3](=O))]", "[C;R2]"):
        assert compile_query(Chem.MolFromSmarts(smarts), vocab) is None


def test_compile_reports_impossible_query_distinctly_from_unscreenable() -> None:
    """``[[]]`` means "nothing can match"; ``None`` means "cannot screen"."""
    vocab = _vocabulary("CCO")
    assert compile_query(Chem.MolFromSmarts("[Pt]"), vocab) == [[]]


def test_single_atom_query_is_screenable() -> None:
    vocab = _vocabulary("CCO", "CCS")
    compiled = compile_query(Chem.MolFromSmarts("[S]"), vocab)
    assert compiled is not None
    assert compiled != [[]]
