"""Tests for functional-group filters."""

from __future__ import annotations

from rpfr_visualizer.analysis.fingerprint_explorer import SiteRank
from rpfr_visualizer.analysis.functional_groups import (
    filter_ranks_by_fg,
    parse_free_text_filter,
    preset_by_name,
)


def _rank(r1: str, element: str = "C", smiles: str = "CC(=O)O") -> SiteRank:
    return SiteRank(
        dataset="qm9s",
        key="1",
        atom_index=0,
        element=element,
        value=1.0,
        local_smiles=(r1, "", "", "", ""),
        molecule_smiles=smiles,
        abs_value=1.0,
    )


def test_parse_free_text_filter() -> None:
    assert parse_free_text_filter("smarts:C=O") == {"smarts": "C=O"}
    assert parse_free_text_filter("r1:CC") == {"r1_exact": "CC"}
    assert parse_free_text_filter("C(=O)O") == {"smiles_substring": "C(=O)O"}


def test_filter_by_r1_exact() -> None:
    ranks = [_rank("CC(=O)O"), _rank("CCO")]
    filtered = filter_ranks_by_fg(ranks, r1_exact="CC(=O)O")
    assert len(filtered) == 1
    assert filtered[0].local_smiles[0] == "CC(=O)O"


def test_filter_by_preset_element() -> None:
    preset = preset_by_name("carbonyl")
    ranks = [_rank("C=O", element="C"), _rank("O", element="O")]
    filtered = filter_ranks_by_fg(ranks, preset=preset)
    assert all(r.element == "C" for r in filtered)


def test_index_query_for_carboxyl_preset() -> None:
    from rpfr_visualizer.analysis.functional_groups import index_query_for_fg

    preset = preset_by_name("carboxylic_acid")
    query = index_query_for_fg(preset=preset)
    assert query.startswith("smarts:")


def test_legacy_alias_resolves() -> None:
    assert preset_by_name("carboxyl_C").name == "carboxylic_acid"
    assert preset_by_name("alcohol_O").name == "alcohol"


def test_index_query_prefers_free_text_smarts() -> None:
    from rpfr_visualizer.analysis.functional_groups import index_query_for_fg

    preset = preset_by_name("carboxylic_acid")
    query = index_query_for_fg(preset=preset, fg_text="smarts:[OH]")
    assert query == "smarts:[OH]"


def test_amine_preset_excludes_ketone_without_nitrogen() -> None:
    from rpfr_visualizer.analysis.functional_groups import site_matches_fg

    preset = preset_by_name("amine")
    assert "," not in (preset.smarts or "")
    assert not site_matches_fg(
        element="C",
        r1="CCC1=CCC(=O)C1C",
        molecule_smiles="CCC1=CCC(=O)C1C",
        preset=preset,
    )
    assert site_matches_fg(
        element="N",
        r1="CCN",
        molecule_smiles="CCN",
        preset=preset,
    )
