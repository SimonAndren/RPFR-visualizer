"""Isotopic fingerprint explorer for ranking enrichment across sites.

Student-facing helpers: rank sites by |β|/|ε| under chosen conditions,
group by local chemical environment, and find similar sites.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rpfr_visualizer.analysis.analyze import run_molecule_analysis
from rpfr_visualizer.analysis.fingerprint import find_exact_matches, find_similar_sites, shells
from rpfr_visualizer.analysis.models import AnalysisKind, AnalysisRequest, MoleculeAnalysisResult
from rpfr_visualizer.analysis.reference import (
    ElementMeanReference,
    SelfReference,
    StandardReference,
)
from rpfr_visualizer.io.loader import Molecule
from rpfr_visualizer.io.providers import MoleculeArrays


@dataclass(frozen=True)
class SiteRank:
    """One ranked isotopic fingerprint site."""

    dataset: str
    key: str
    atom_index: int
    element: str
    value: float
    local_smiles: tuple[str, ...]
    molecule_smiles: str
    abs_value: float


@dataclass(frozen=True)
class EnvironmentGroup:
    """Sites sharing the same R1 local_smiles shell."""

    fingerprint: str
    shell: int
    sites: tuple[SiteRank, ...]
    mean_abs_value: float


# Student-friendly analysis presets
CONDITION_PRESETS: dict[str, dict[str, Any]] = {
    "room_temp_epsilon": {
        "kind": AnalysisKind.EPSILON,
        "temperature": 298.15,
        "reference": "element_mean",
        "description": (
            "ε at 298.15 K vs within-molecule same-element mean (relative; no absolute standard)"
        ),
    },
    "cold_beta": {
        "kind": AnalysisKind.BETA,
        "temperature": 100.0,
        "description": "Absolute β at 100 K (no reference / standard)",
    },
    "vs_standard": {
        "kind": AnalysisKind.EPSILON,
        "temperature": 300.0,
        "reference": "standard",
        "description": (
            "ε at 300 K vs dataset median β(T) per element "
            "(pass stats= to request_from_preset; optional constant overrides)"
        ),
    },
}


def request_from_preset(
    preset: str,
    *,
    anchor_atom_index: int = 0,
    element: str | None = None,
    stats=None,
    standard_overrides: dict[str, float] | None = None,
) -> AnalysisRequest:
    """Build an :class:`AnalysisRequest` from a named student preset.

    Parameters
    ----------
    stats:
        Optional :class:`~rpfr_visualizer.analysis.element_stats.ElementStatsCache`
        used by the ``vs_standard`` preset (dataset median β(T)).
    standard_overrides:
        Optional constant β overrides per element (editable session standards).
    """
    if preset not in CONDITION_PRESETS:
        available = ", ".join(sorted(CONDITION_PRESETS))
        raise KeyError(f"Unknown preset {preset!r}. Available: {available}")
    cfg = CONDITION_PRESETS[preset]
    kind = cfg["kind"]
    temperature = float(cfg["temperature"])
    reference = None
    if kind in {AnalysisKind.ALPHA, AnalysisKind.EPSILON, AnalysisKind.LN_ALPHA}:
        ref_mode = cfg.get("reference", "self")
        if ref_mode == "standard":
            overrides = dict(standard_overrides or cfg.get("standard_betas") or {})
            reference = StandardReference(beta_by_element=overrides, stats=stats)
        elif ref_mode == "element_mean":
            reference = ElementMeanReference()
        else:
            reference = SelfReference(anchor_atom_index=anchor_atom_index)
    return AnalysisRequest(
        kind=kind,
        temperature=temperature,
        reference=reference,
        element=element,
    )


def rank_sites(
    molecules: list[tuple[str, str, Molecule]],
    request: AnalysisRequest,
    *,
    top_n: int = 20,
    element: str | None = None,
    arrays_by_key: dict[tuple[str, str], MoleculeArrays] | None = None,
    by_abs: bool = True,
) -> list[SiteRank]:
    """Rank sites by enrichment magnitude across loaded molecules.

    Parameters
    ----------
    molecules:
        List of ``(dataset, key, Molecule)`` triples (typically search hits).
    request:
        Analysis to run (beta / alpha / epsilon / …).
    top_n:
        Maximum number of ranked sites to return.
    element:
        Optional element filter (overrides ``request.element`` when set).
    arrays_by_key:
        Optional ``(dataset, key) → MoleculeArrays`` for clumping/mass-law.
    by_abs:
        If True, rank by absolute value; else by signed value descending.
    """
    elem_filter = element if element is not None else request.element
    ranks: list[SiteRank] = []

    for dataset, key, molecule in molecules:
        arrays = None
        if arrays_by_key is not None:
            arrays = arrays_by_key.get((dataset, key))
        effective = (
            request
            if elem_filter == request.element
            else AnalysisRequest(
                kind=request.kind,
                temperature=request.temperature,
                reference=request.reference,
                element=elem_filter,
            )
        )
        result = run_molecule_analysis(molecule, effective, arrays=arrays)
        ranks.extend(_result_to_ranks(dataset, key, molecule, result))

    if by_abs:
        ranks.sort(key=lambda r: r.abs_value, reverse=True)
    else:
        ranks.sort(key=lambda r: r.value, reverse=True)
    return ranks[:top_n]


def _result_to_ranks(
    dataset: str,
    key: str,
    molecule: Molecule,
    result: MoleculeAnalysisResult,
) -> list[SiteRank]:
    ranks: list[SiteRank] = []
    site_by_idx = {site.atom_index: site for site in molecule.sites}
    for site_result in result.sites:
        if site_result.value is None:
            continue
        site = site_by_idx.get(site_result.atom_index)
        local = shells(site) if site is not None else ()
        value = float(site_result.value)
        ranks.append(
            SiteRank(
                dataset=dataset,
                key=key,
                atom_index=site_result.atom_index,
                element=site_result.element,
                value=value,
                local_smiles=local,
                molecule_smiles=molecule.smiles or "",
                abs_value=abs(value),
            )
        )
    return ranks


def group_by_environment(
    ranks: list[SiteRank],
    *,
    shell: int = 0,
) -> list[EnvironmentGroup]:
    """Group ranked sites by local_smiles shell (default R1)."""
    buckets: dict[str, list[SiteRank]] = defaultdict(list)
    for rank in ranks:
        fp = ""
        if 0 <= shell < len(rank.local_smiles):
            fp = rank.local_smiles[shell] or ""
        if not fp:
            fp = "(empty)"
        buckets[fp].append(rank)

    groups: list[EnvironmentGroup] = []
    for fingerprint, sites in buckets.items():
        mean_abs = sum(s.abs_value for s in sites) / len(sites)
        groups.append(
            EnvironmentGroup(
                fingerprint=fingerprint,
                shell=shell,
                sites=tuple(sites),
                mean_abs_value=mean_abs,
            )
        )
    groups.sort(key=lambda g: g.mean_abs_value, reverse=True)
    return groups


def similar_environments(
    molecules: list[Molecule],
    *,
    min_similarity: float = 0.5,
    exact_shell: int | None = 0,
) -> list:
    """Find similar or exact local_smiles matches among loaded molecules."""
    if exact_shell is not None:
        return find_exact_matches(molecules, shell=exact_shell)
    return find_similar_sites(molecules, min_similarity=min_similarity)


def ranks_to_frame(ranks: list[SiteRank]):
    """Convert ranked sites to a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "dataset": r.dataset,
                "key": r.key,
                "atom_index": r.atom_index,
                "element": r.element,
                "value": r.value,
                "abs_value": r.abs_value,
                "r1": r.local_smiles[0] if r.local_smiles else "",
                "molecule_smiles": r.molecule_smiles,
            }
            for r in ranks
        ]
    )


__all__ = [
    "CONDITION_PRESETS",
    "EnvironmentGroup",
    "SiteRank",
    "group_by_environment",
    "ranks_to_frame",
    "rank_sites",
    "request_from_preset",
    "similar_environments",
]
