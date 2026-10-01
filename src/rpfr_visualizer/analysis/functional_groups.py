"""Functional-group filters and matching.

Curated SMARTS presets, observed R1 shells from a result set, and free-text
filters (exact R1, SMILES substring, or ``smarts:`` queries). Matching logic
only — 2D depiction of these patterns lives in :mod:`rpfr_visualizer.viz.depict`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rpfr_visualizer.analysis.fingerprint_explorer import SiteRank


@dataclass(frozen=True)
class FunctionalGroupPreset:
    """Named student-facing functional-group filter."""

    name: str
    description: str
    element: str | None = None
    r1: str | None = None
    smarts: str | None = None
    smiles_substring: str | None = None
    example_smiles: str | None = None


# Curated catalog aligned with common organic FG charts (QM9s = C/H/N/O only;
# halogen and sulfur groups are omitted because they are absent from QM9s).
FUNCTIONAL_GROUP_PRESETS: tuple[FunctionalGroupPreset, ...] = (
    # --- Hydrocarbons ---
    FunctionalGroupPreset(
        name="alkene",
        description="Alkene C=C (ene)",
        element="C",
        smarts="[CX3]=[CX3]",
        smiles_substring="C=C",
        example_smiles="C=CC",
    ),
    FunctionalGroupPreset(
        name="alkyne",
        description="Alkyne C≡C (yne)",
        element="C",
        smarts="[CX2]#[CX2]",
        smiles_substring="C#C",
        example_smiles="C#CC",
    ),
    FunctionalGroupPreset(
        name="arene",
        description="Aromatic ring (arene / benzene)",
        element="C",
        smarts="c1ccccc1",
        smiles_substring="c1ccccc1",
        example_smiles="Cc1ccccc1",
    ),
    # --- Oxygen ---
    FunctionalGroupPreset(
        name="alcohol",
        description="Alcohol R–OH (ol)",
        element="O",
        smarts="[CX4][OX2H]",
        smiles_substring="O",
        example_smiles="CCO",
    ),
    FunctionalGroupPreset(
        name="ether",
        description="Ether R–O–R (oxy)",
        element="O",
        smarts="[OD2]([#6])[#6]",
        smiles_substring="O",
        example_smiles="COC",
    ),
    FunctionalGroupPreset(
        name="epoxide",
        description="Epoxide (oxirane)",
        element="O",
        smarts="C1OC1",
        smiles_substring="C1OC1",
        example_smiles="C1OC1",
    ),
    # --- Carbonyls ---
    FunctionalGroupPreset(
        name="aldehyde",
        description="Aldehyde R–CHO (al)",
        element="C",
        smarts="[CX3H1](=O)[#6]",
        smiles_substring="C=O",
        example_smiles="CC=O",
    ),
    FunctionalGroupPreset(
        name="ketone",
        description="Ketone R–CO–R (one)",
        element="C",
        smarts="[#6][CX3](=O)[#6]",
        smiles_substring="C(=O)",
        example_smiles="CC(=O)C",
    ),
    FunctionalGroupPreset(
        name="carboxylic_acid",
        description="Carboxylic acid R–COOH (oic acid)",
        element="C",
        smarts="[CX3](=O)[OX2H1]",
        smiles_substring="C(=O)O",
        example_smiles="CC(=O)O",
    ),
    FunctionalGroupPreset(
        name="ester",
        description="Ester R–COOR (oate)",
        element="C",
        smarts="[CX3](=O)[OX2][#6]",
        smiles_substring="C(=O)O",
        example_smiles="CC(=O)OC",
    ),
    FunctionalGroupPreset(
        name="anhydride",
        description="Acid anhydride R–CO–O–CO–R",
        element="C",
        smarts="[CX3](=O)[OX2][CX3]=O",
        smiles_substring="C(=O)OC(=O)",
        example_smiles="CC(=O)OC(=O)C",
    ),
    FunctionalGroupPreset(
        name="amide",
        description="Amide R–CON (amide)",
        element="C",
        smarts="[CX3](=O)[NX3]",
        smiles_substring="C(=O)N",
        example_smiles="CC(=O)N",
    ),
    # Keep a broad carbonyl catch-all for teaching
    FunctionalGroupPreset(
        name="carbonyl",
        description="Any carbonyl carbon C=O",
        element="C",
        smarts="[CX3]=O",
        smiles_substring="C=O",
        example_smiles="CC(=O)C",
    ),
    # --- Nitrogen ---
    FunctionalGroupPreset(
        name="amine",
        description="Amine R–NH2 / R2NH / R3N (excludes amides)",
        element="N",
        # Avoid comma lists inside SMARTS; search OR-splitting is comma-based.
        smarts="[NX3;!$(NC=O)]",
        smiles_substring="N",
        example_smiles="CCN",
    ),
    FunctionalGroupPreset(
        name="nitrile",
        description="Nitrile R–C≡N",
        element="C",
        smarts="[CX2]#[NX1]",
        smiles_substring="C#N",
        example_smiles="CC#N",
    ),
    FunctionalGroupPreset(
        name="imine",
        description="Imine C=N",
        element="C",
        smarts="[CX3]=[NX2]",
        smiles_substring="C=N",
        example_smiles="CC=NC",
    ),
    FunctionalGroupPreset(
        name="isocyanate",
        description="Isocyanate R–N=C=O",
        element="N",
        smarts="[NX2]=C=O",
        smiles_substring="N=C=O",
        example_smiles="CN=C=O",
    ),
    FunctionalGroupPreset(
        name="azo",
        description="Azo R–N=N–R",
        element="N",
        smarts="[NX2]=[NX2]",
        smiles_substring="N=N",
        example_smiles="CN=NC",
    ),
)

# Older names kept resolvable for notebooks / saved sessions.
_FG_ALIASES: dict[str, str] = {
    "carboxyl_C": "carboxylic_acid",
    "alcohol_O": "alcohol",
    "amine_N": "amine",
    "carbonyl_C": "carbonyl",
    "alkene_C": "alkene",
    "ether_O": "ether",
}


def preset_by_name(name: str) -> FunctionalGroupPreset:
    """Return a curated preset by name (including legacy aliases)."""
    resolved = _FG_ALIASES.get(name, name)
    for preset in FUNCTIONAL_GROUP_PRESETS:
        if preset.name == resolved or preset.name == name:
            return preset
    available = ", ".join(p.name for p in FUNCTIONAL_GROUP_PRESETS)
    raise KeyError(f"Unknown functional-group preset {name!r}. Available: {available}")


def primary_fg_presets() -> tuple[FunctionalGroupPreset, ...]:
    """Presets shown in the student UI (excludes legacy alias names)."""
    return FUNCTIONAL_GROUP_PRESETS


def observed_r1_shells(ranks: Sequence[SiteRank], *, element: str | None = None) -> list[str]:
    """Return sorted unique non-empty R1 fingerprints from ranked sites."""
    found: set[str] = set()
    for rank in ranks:
        if element is not None and rank.element != element:
            continue
        if rank.local_smiles and rank.local_smiles[0]:
            found.add(rank.local_smiles[0])
    return sorted(found)


def _smarts_matches_smiles(smarts: str, smiles: str) -> bool:
    try:
        from rdkit import Chem
    except ImportError:
        return False
    query = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if query is None or mol is None:
        return False
    return bool(mol.HasSubstructMatch(query))


def site_matches_fg(
    *,
    element: str,
    r1: str,
    molecule_smiles: str,
    preset: FunctionalGroupPreset | None = None,
    r1_exact: str | None = None,
    smiles_substring: str | None = None,
    smarts: str | None = None,
) -> bool:
    """Return True if a site matches the active functional-group filter."""
    if preset is not None:
        if preset.element is not None and element != preset.element:
            return False
        if preset.r1 is not None and r1 != preset.r1:
            return False
        if preset.smiles_substring and preset.smiles_substring not in (r1 or molecule_smiles):
            # Prefer R1 neighborhood, fall back to full SMILES
            if preset.smiles_substring not in r1 and preset.smiles_substring not in molecule_smiles:
                return False
        if preset.smarts:
            haystack = r1 or molecule_smiles
            if not _smarts_matches_smiles(preset.smarts, haystack):
                # Also try full molecule SMILES
                if not _smarts_matches_smiles(preset.smarts, molecule_smiles):
                    return False
        return True

    if r1_exact is not None and r1 != r1_exact:
        return False
    if smiles_substring:
        if smiles_substring not in r1 and smiles_substring not in molecule_smiles:
            return False
    if smarts:
        if not (
            _smarts_matches_smiles(smarts, r1) or _smarts_matches_smiles(smarts, molecule_smiles)
        ):
            return False
    return True


def filter_ranks_by_fg(
    ranks: Sequence[SiteRank],
    *,
    preset: FunctionalGroupPreset | None = None,
    r1_exact: str | None = None,
    smiles_substring: str | None = None,
    smarts: str | None = None,
) -> list[SiteRank]:
    """Filter ranked sites by functional-group criteria."""
    return [
        rank
        for rank in ranks
        if site_matches_fg(
            element=rank.element,
            r1=rank.local_smiles[0] if rank.local_smiles else "",
            molecule_smiles=rank.molecule_smiles,
            preset=preset,
            r1_exact=r1_exact,
            smiles_substring=smiles_substring,
            smarts=smarts,
        )
    ]


def parse_free_text_filter(text: str) -> dict[str, str]:
    """Parse free-text FG filter into ``r1_exact`` / ``smiles_substring`` / ``smarts``."""
    raw = text.strip()
    if not raw:
        return {}
    lower = raw.lower()
    if lower.startswith("smarts:"):
        return {"smarts": raw.split(":", 1)[1].strip()}
    if lower.startswith("r1:"):
        return {"r1_exact": raw.split(":", 1)[1].strip()}
    return {"smiles_substring": raw}


def index_query_for_fg(
    *,
    preset: FunctionalGroupPreset | None = None,
    fg_text: str = "",
) -> str:
    """Build a molecule-index search query for dataset-wide FG matching.

    Prefers SMARTS (``smarts:…``), then SMILES substring. R1-exact filters are
    site-level and cannot be expressed as a molecule-index query alone.
    """
    free = parse_free_text_filter(fg_text)
    if free.get("smarts"):
        return f"smarts:{free['smarts']}"
    if free.get("smiles_substring"):
        return free["smiles_substring"]
    if preset is not None:
        if preset.smarts:
            return f"smarts:{preset.smarts}"
        if preset.smiles_substring:
            return preset.smiles_substring
        if preset.r1:
            return preset.r1
    return ""


def search_molecules_by_fg(
    index: Any,
    *,
    preset: FunctionalGroupPreset | None = None,
    fg_text: str = "",
    datasets: tuple[str, ...] | None = None,
    max_hits: int | None = None,
) -> list[Any]:
    """Search a :class:`~rpfr_visualizer.io.molecule_index.MoleculeIndex` for FG matches.

    Returns
    -------
    list
        :class:`~rpfr_visualizer.io.molecule_index.MoleculeSearchHit` objects.
    """
    query = index_query_for_fg(preset=preset, fg_text=fg_text)
    if not query:
        return []
    hits = index.search(query, datasets=datasets)
    if max_hits is not None:
        hits = hits[: max(0, int(max_hits))]
    return hits


__all__ = [
    "FUNCTIONAL_GROUP_PRESETS",
    "FunctionalGroupPreset",
    "filter_ranks_by_fg",
    "index_query_for_fg",
    "observed_r1_shells",
    "parse_free_text_filter",
    "preset_by_name",
    "primary_fg_presets",
    "search_molecules_by_fg",
    "site_matches_fg",
]
