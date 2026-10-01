"""Topological keys: a substructure screen that knows about connectivity.

The pattern-fingerprint screen in :mod:`rpfr_visualizer.io.search.screening` is
information-limited, not bit-limited. ``Chem.PatternFingerprint`` hashes small
subgraphs, so a three-atom query like ``[CX3](=O)[OX2H1]`` sets bits that stand
for generic C-O and C=O paths — fragments a third of the database genuinely
contains. Measured against the 881,757-molecule index, that screen leaves
34.6% of all molecules to verify in order to find 0.6% true hits, and raising
``fpSize`` from 1024 to 8192 does not move that number at all.

What the fingerprint discards is exactly what a functional-group query is made
of: how many neighbours an atom has, and how many hydrogens. This module keeps
those instead. Every atom contributes a key

    (element, aromatic, total_degree, n_hydrogens)

and every bond contributes a key pairing its two atom keys with a bond order.
A molecule survives the screen when it holds at least one key compatible with
each bond (or, for a single-atom query, each atom) of the pattern. On the same
index that takes the carboxylic-acid candidate set from 34.6% to 1.9% of the
database.

**The screen is a filter, never an answer**, exactly as before. It returns a
superset and every survivor is still verified with RDKit. Three things
guarantee that:

* Compatibility is decided by RDKit itself. Each stored key is turned back
  into a probe atom carrying those four properties, and the query's own
  ``QueryAtom.Match`` is asked whether it accepts it. The screen therefore
  inherits SMARTS semantics rather than reimplementing them.
* Some SMARTS primitives cannot be judged from an isolated atom — recursive
  environments (``[$(...)]``), ring membership and ring size all depend on the
  rest of the molecule. Those are detected and the query falls back to the
  pattern fingerprint rather than silently dropping matches.
* Atom keys are stored as well as bond keys, so a molecule with no bonds at
  all (methane is one) still has keys and can still match a single-atom query.
  A bond-only index would have excluded it.
"""

from __future__ import annotations

from typing import Any

#: Primitives whose truth depends on the rest of the molecule, so a
#: single-atom probe cannot decide them. A query atom using any of these
#: makes its bond unscreenable.
_ENVIRONMENT_PRIMITIVES = (
    "RecursiveStructure",
    "AtomInNRings",
    "AtomMinRingSize",
    "AtomRingBondCount",
    "AtomInRing",
    "AtomRingSize",
)

_ATOM_PREFIX = "a"
_BOND_PREFIX = "b"


def atom_key(atom: Any) -> tuple[int, int, int, int, int]:
    """The five invariants stored per atom.

    Three RDKit subtleties decide these, and getting any wrong silently drops
    matches rather than failing loudly:

    ``GetTotalDegree`` rather than ``GetDegree``, because SMARTS ``X`` counts
    hydrogens — ``[CX3]`` matches the carbon in both ``CC=O`` and ``C=O``.

    ``GetTotalNumHs(includeNeighbors=True)`` rather than the bare call, because
    an isotopic hydrogen is a graph atom, not an implicit one. In
    ``[2H]OC(=O)NC`` the hydroxyl oxygen reports ``GetTotalNumHs() == 0`` while
    RDKit's own matcher still accepts ``[OX2H1]``; counting neighbours makes
    the deuterated and protiated forms share a key, which is what SMARTS
    semantics require. This database is about isotope effects, so deuterated
    molecules are not an edge case here — the bare call dropped 58 of the
    5,165 carboxylic acids.

    GetIsotope is stored because this is an isotope-effects database and
    [2H] is an ordinary query here. SMARTS reads it as
    AtomAtomicNum 1 AND AtomIsotope 2, so a probe left at the default
    isotope 0 fails the second half and the screen returns nothing at all —
    the worst kind of wrong answer, since an empty result looks like a valid
    one. Keeping the isotope turns [2H] into a 125x win instead.
    """
    return (
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        int(atom.GetIsAromatic()),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(includeNeighbors=True),
    )


def atom_token(key: tuple[int, int, int, int, int]) -> str:
    """Stable text token for an atom key."""
    return "{}|{}:{}:{}:{}:{}".format(_ATOM_PREFIX, *key)


def bond_token(begin: tuple[int, int, int, int], end: tuple[int, int, int, int], order: int) -> str:
    """Stable text token for a bond key.

    Endpoints are sorted so a bond yields the same token whichever way it is
    walked. Tokens are text rather than hashes so an index stays inspectable
    and portable — nothing depends on Python's salted ``hash``.
    """
    low, high = sorted((begin, end))
    return "{}|{}:{}:{}:{}:{}|{}:{}:{}:{}:{}|{}".format(_BOND_PREFIX, *low, *high, order)


def mol_keys(mol: Any) -> set[str]:
    """Every distinct atom and bond key in *mol*."""
    keys = {atom_token(atom_key(atom)) for atom in mol.GetAtoms()}
    for bond in mol.GetBonds():
        keys.add(
            bond_token(
                atom_key(bond.GetBeginAtom()),
                atom_key(bond.GetEndAtom()),
                int(bond.GetBondType()),
            )
        )
    return keys


def parse_token(token: str) -> tuple[str, tuple[Any, ...]]:
    """Inverse of :func:`atom_token` / :func:`bond_token`."""
    kind, _, rest = token.partition("|")
    if kind == _ATOM_PREFIX:
        return kind, (tuple(int(p) for p in rest.split(":")),)
    low, high, order = rest.split("|")
    return kind, (
        tuple(int(p) for p in low.split(":")),
        tuple(int(p) for p in high.split(":")),
        int(order),
    )


def _probe_atom(
    element: int, isotope: int, aromatic: int, total_degree: int, n_hydrogens: int
) -> Any | None:
    """A single atom carrying the stored invariants, for ``QueryAtom.Match``.

    ``total_degree`` counts hydrogens, so the probe needs
    ``total_degree - n_hydrogens`` heavy neighbours alongside its explicit
    hydrogens for RDKit to perceive the right degree.
    """
    from rdkit import Chem

    heavy = total_degree - n_hydrogens
    if heavy < 0:
        return None
    rw = Chem.RWMol()
    atom = Chem.Atom(element)
    if isotope:
        atom.SetIsotope(isotope)
    atom.SetIsAromatic(bool(aromatic))
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(n_hydrogens)
    index = rw.AddAtom(atom)
    for _ in range(heavy):
        neighbour = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(index, neighbour, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    probe = mol.GetAtomWithIdx(index)
    probe.SetIsAromatic(bool(aromatic))
    return probe


def _is_probe_safe(query_atom: Any) -> bool:
    """Whether an isolated probe can decide this query atom."""
    description = query_atom.DescribeQuery()
    return not any(primitive in description for primitive in _ENVIRONMENT_PRIMITIVES)


def _matches(query_atom: Any, key: tuple[int, int, int, int, int]) -> bool:
    probe = _probe_atom(*key)
    if probe is None:
        return False
    return bool(query_atom.Match(probe))


def _bond_type_allows(query_bond: Any, order: int) -> bool:
    """Whether *query_bond* can match a stored bond of this order.

    A query bond with a wildcard or compound expression is treated as
    permissive: being generous costs candidates, being strict would lose
    matches.
    """
    from rdkit import Chem

    try:
        description = query_bond.DescribeQuery()
    except Exception:  # noqa: BLE001 — a non-query bond has no description
        return True
    if "BondOr" in description or "BondNull" in description:
        return True
    bond_type = query_bond.GetBondType()
    if bond_type == Chem.BondType.UNSPECIFIED:
        return True
    return int(bond_type) == order


def compile_query(pattern: Any, known_keys: list[str]) -> list[list[str]] | None:
    """Per query bond (or atom), the stored keys any real match must contain.

    Returns one entry per screenable feature of *pattern*; a molecule survives
    when, for every entry, it holds at least one of those keys. Returns None
    when nothing about the query can be screened this way — an unparseable
    pattern, or one built entirely from environment-dependent primitives — and
    the caller then falls back to the pattern fingerprint.

    A returned ``[[]]`` means the opposite of None: the query is screenable
    and nothing in the index can satisfy it, so the result is empty.

    *known_keys* is the index's own key vocabulary. It is small — 325 bond
    keys plus a few dozen atom keys across 881,757 molecules — so testing
    every key against every query feature costs about a millisecond.
    """
    if pattern is None or pattern.GetNumAtoms() == 0:
        return None

    atom_keys: list[tuple[str, tuple[int, int, int, int, int]]] = []
    bond_keys: list[
        tuple[str, tuple[int, int, int, int, int], tuple[int, int, int, int, int], int]
    ] = []
    for token in known_keys:
        kind, parts = parse_token(token)
        if kind == _ATOM_PREFIX:
            atom_keys.append((token, parts[0]))
        else:
            bond_keys.append((token, *parts))  # type: ignore[arg-type]

    if pattern.GetNumBonds() == 0:
        return _compile_single_atom(pattern, atom_keys)

    per_feature: list[list[str]] = []
    for bond in pattern.GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (_is_probe_safe(begin) and _is_probe_safe(end)):
            continue

        compatible = [
            token
            for token, low, high, order in bond_keys
            if _bond_type_allows(bond, order)
            and (
                # The stored key is unordered, so either assignment of its
                # endpoints to the query's two atoms is a valid match.
                (_matches(begin, low) and _matches(end, high))
                or (_matches(begin, high) and _matches(end, low))
            )
        ]
        if not compatible:
            return [[]]
        # A feature every key satisfies constrains nothing; dropping it saves
        # an intersection over the whole index.
        if len(compatible) < len(bond_keys):
            per_feature.append(compatible)

    return per_feature or None


def _compile_single_atom(
    pattern: Any, atom_keys: list[tuple[str, tuple[int, int, int, int, int]]]
) -> list[list[str]] | None:
    """Screen a bondless pattern (``[S]``, ``[nH]``) on atom keys alone."""
    if pattern.GetNumAtoms() != 1 or not atom_keys:
        return None
    query_atom = pattern.GetAtomWithIdx(0)
    if not _is_probe_safe(query_atom):
        return None
    compatible = [token for token, key in atom_keys if _matches(query_atom, key)]
    if not compatible:
        return [[]]
    if len(compatible) == len(atom_keys):
        return None
    return [compatible]


__all__ = [
    "atom_key",
    "atom_token",
    "bond_token",
    "compile_query",
    "mol_keys",
    "parse_token",
]
