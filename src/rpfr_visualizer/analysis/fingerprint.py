"""Local SMILES fingerprint comparison across sites and molecules."""

from __future__ import annotations

from dataclasses import dataclass

from rpfr_visualizer.io.loader import Molecule


@dataclass(frozen=True)
class FingerprintMatch:
    """A matched site pair based on local_smiles shells."""

    molecule_a_id: int
    atom_a: int
    molecule_b_id: int
    atom_b: int
    shell: int
    fingerprint: str
    match_type: str


def shells(site) -> tuple[str, ...]:
    """Return R1-R5 local_smiles shells for a site."""
    return tuple(site.local_smiles)


def exact_shell_match(fp_a: tuple[str, ...], fp_b: tuple[str, ...]) -> list[int]:
    """Return shell indices (0-4) where fingerprints match exactly and are non-empty."""
    matches: list[int] = []
    for idx, (a, b) in enumerate(zip(fp_a, fp_b, strict=True)):
        if a and b and a == b:
            matches.append(idx)
    return matches


def jaccard_similarity(fp_a: tuple[str, ...], fp_b: tuple[str, ...]) -> float:
    """Simple Jaccard similarity over non-empty shell strings."""
    set_a = {s for s in fp_a if s}
    set_b = {s for s in fp_b if s}
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def find_exact_matches(
    molecules: list[Molecule],
    *,
    shell: int | None = None,
) -> list[FingerprintMatch]:
    """Find exact local_smiles matches across molecules."""
    matches: list[FingerprintMatch] = []
    indexed: list[tuple[int, int, tuple[str, ...]]] = []
    for mol in molecules:
        for site in mol.sites:
            indexed.append((mol.molecule_id, site.atom_index, shells(site)))

    for i, (mol_a, atom_a, fp_a) in enumerate(indexed):
        for mol_b, atom_b, fp_b in indexed[i + 1 :]:
            shell_indices = exact_shell_match(fp_a, fp_b)
            if shell is not None:
                shell_indices = [s for s in shell_indices if s == shell]
            for shell_idx in shell_indices:
                matches.append(
                    FingerprintMatch(
                        molecule_a_id=mol_a,
                        atom_a=atom_a,
                        molecule_b_id=mol_b,
                        atom_b=atom_b,
                        shell=shell_idx,
                        fingerprint=fp_a[shell_idx],
                        match_type="exact",
                    )
                )
    return matches


def find_similar_sites(
    molecules: list[Molecule],
    *,
    min_similarity: float = 0.5,
) -> list[FingerprintMatch]:
    """Find sites with Jaccard similarity above a threshold."""
    matches: list[FingerprintMatch] = []
    indexed: list[tuple[int, int, tuple[str, ...]]] = []
    for mol in molecules:
        for site in mol.sites:
            indexed.append((mol.molecule_id, site.atom_index, shells(site)))

    for i, (mol_a, atom_a, fp_a) in enumerate(indexed):
        for mol_b, atom_b, fp_b in indexed[i + 1 :]:
            sim = jaccard_similarity(fp_a, fp_b)
            if sim >= min_similarity:
                label = fp_a[0] if fp_a and fp_a[0] else "(mixed)"
                matches.append(
                    FingerprintMatch(
                        molecule_a_id=mol_a,
                        atom_a=atom_a,
                        molecule_b_id=mol_b,
                        atom_b=atom_b,
                        shell=-1,
                        fingerprint=label,
                        match_type=f"similar:{sim:.2f}",
                    )
                )
    return matches


__all__ = [
    "FingerprintMatch",
    "exact_shell_match",
    "find_exact_matches",
    "find_similar_sites",
    "jaccard_similarity",
    "shells",
]
