"""Unit tests for molecule search index."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.io.molecule_index import (
    DEFAULT_DRIVE_ROOT,
    DRIVE_ROOT_ENV,
    INDEX_VERSION,
    MoleculeIndex,
    MoleculeIndexEntry,
    canonical_smiles,
    default_drive_root,
    looks_like_formula,
    looks_like_smiles,
    normalize_formula,
    parse_element_counts,
    resolve_name_hybrid,
    smiles_descriptors,
    split_search_query,
    synonym_seed,
)
from rpfr_visualizer.io.providers import DatasetCatalog

pytest.importorskip("rdkit")


def test_canonical_smiles_roundtrip() -> None:
    raw = "CC(=O)O"
    canon = canonical_smiles(raw)
    assert canon is not None
    assert canonical_smiles(canon) == canon


def test_looks_like_formula() -> None:
    assert looks_like_formula("C3H7NO2")
    assert looks_like_formula("C6H6")
    assert looks_like_formula("H2O")
    assert not looks_like_formula("CC(=O)O")
    assert not looks_like_formula("alanine")
    assert not looks_like_formula("42")


def test_normalize_formula_hill_order() -> None:
    assert normalize_formula("H2O") == "H2O"
    assert normalize_formula("OH2") == "H2O"
    assert normalize_formula("C2H6O") == "C2H6O"
    counts = parse_element_counts("C3H7NO2")
    assert counts["C"] == 3
    assert counts["H"] == 7
    assert counts["N"] == 1
    assert counts["O"] == 2


@pytest.mark.parametrize(
    "written",
    ["C6H14O", "C6OH14", "OC6H14", "H14OC6", "C6H14O1"],
)
def test_a_formula_is_read_in_any_element_order(written: str) -> None:
    """Hill order is what the index stores, not what anyone has to type.

    C6H14O and C6OH14 are the same molecule, and a search that found one but
    not the other would look like missing data rather than a parsing rule.
    """
    assert looks_like_formula(written)
    assert normalize_formula(written) == "C6H14O"


@pytest.mark.parametrize("written", ["C6 H14 O", " C6H14O ", "C6H14 O"])
def test_whitespace_inside_a_formula_is_ignored(written: str) -> None:
    assert looks_like_formula(written)
    assert normalize_formula(written) == "C6H14O"


def test_unicode_subscripts_are_read_as_digits() -> None:
    """Pasting a formula out of a paper used to change which molecule it was.

    The element regex matches ASCII digits only, so every count was dropped
    silently: C6H14O became CHO, which is a different formula that the index
    will happily look up and report as a match.
    """
    assert normalize_formula("C₆H₁₄O") == "C6H14O"
    assert looks_like_formula("C₆H₁₄O")


def test_a_condensed_structural_formula_aggregates() -> None:
    """CH3CH2OH is how a chemist writes ethanol; it is still C2H6O."""
    assert normalize_formula("CH3CH2OH") == "C2H6O"
    assert parse_element_counts("CH3CH2OH") == {"C": 2, "H": 6, "O": 1}


@pytest.mark.parametrize(
    "written",
    ["CCO", "c1ccccc1", "CC(=O)O", "C[C@H](N)C(=O)O", "O=C=O"],
)
def test_tolerating_formulas_did_not_start_swallowing_smiles(written: str) -> None:
    """looks_like_smiles defers to looks_like_formula, so a formula rule that
    grew too permissive would quietly reclassify structures as formulas."""
    assert not looks_like_formula(written)
    assert looks_like_smiles(written)


@pytest.mark.parametrize("written", ["ethanol", "Vitamin C", "Methyl Ethyl Ketone"])
def test_names_are_not_mistaken_for_formulas(written: str) -> None:
    """Allowing spaces inside a formula is the risky half of that change."""
    assert not looks_like_formula(written)


def test_lowercase_is_not_treated_as_a_formula() -> None:
    """Deliberate. A formula is case-sensitive -- CO is carbon monoxide, Co is
    cobalt -- and lowercasing would collide with aromatic SMILES."""
    assert not looks_like_formula("c6h14o")


def test_smiles_descriptors() -> None:
    desc = smiles_descriptors("CCO")
    assert desc["canonical_smiles"]
    assert desc["formula"] in {"C2H6O", "C2H5OH"} or desc["formula"].startswith("C2H")
    assert "C" in desc["element_counts"]


def test_index_search_by_id(synthetic_h5_path: Path) -> None:
    catalog = DatasetCatalog()
    catalog.register("qm9s", synthetic_h5_path)
    index = MoleculeIndex.build_from_catalog(catalog)
    hits = index.search("1")
    assert hits
    assert hits[0].entry.key == "1"
    assert hits[0].entry.dataset == "qm9s"


def test_index_search_by_smiles(synthetic_h5_path: Path) -> None:
    catalog = DatasetCatalog()
    catalog.register("qm9s", synthetic_h5_path)
    index = MoleculeIndex.build_from_catalog(catalog)
    provider = catalog.get_provider("qm9s")
    mol = provider.get_molecule(1)
    if not mol.smiles:
        pytest.skip("Synthetic fixture has no SMILES attr")
    hits = index.search(mol.smiles)
    assert any(hit.entry.key == "1" for hit in hits)


def test_index_search_by_formula() -> None:
    entry = MoleculeIndexEntry(
        dataset="test",
        key="1",
        smiles="CCO",
        canonical_smiles="CCO",
        n_atoms=9,
        label="test:1",
        formula="C2H6O",
        element_counts={"C": 2, "H": 6, "O": 1},
    )
    index = MoleculeIndex([entry], allow_pubchem=False)
    hits = index.search("C2H6O")
    assert hits
    assert hits[0].match_type == "formula"
    assert hits[0].entry.key == "1"


def test_index_search_smarts() -> None:
    entry = MoleculeIndexEntry(
        dataset="test",
        key="1",
        smiles="CC(=O)O",
        canonical_smiles="CC(=O)O",
        n_atoms=8,
        label="test:1",
        formula="C2H4O2",
    )
    index = MoleculeIndex([entry], allow_pubchem=False)
    hits = index.search("smarts:C(=O)O")
    assert hits
    assert hits[0].match_type == "smarts"


def test_split_search_query_preserves_smarts_commas() -> None:
    smarts = "smarts:[NX3;H2,H1,H0;!$(NC=O)]"
    assert split_search_query(smarts) == [smarts]
    assert split_search_query("alanine, glycine") == ["alanine", "glycine"]


def test_index_search_amine_smarts_with_commas() -> None:
    """Regression: comma-splitting must not break amine-like SMARTS."""
    amine = MoleculeIndexEntry(
        dataset="test",
        key="amine",
        smiles="CCN",
        canonical_smiles="CCN",
        n_atoms=9,
        label="test:amine",
        formula="C2H7N",
        element_counts={"C": 2, "H": 7, "N": 1},
    )
    ketone = MoleculeIndexEntry(
        dataset="test",
        key="ketone",
        smiles="CCC1=CCC(=O)C1C",
        canonical_smiles="CCC1=CCC(=O)C1C",
        n_atoms=20,
        label="test:ketone",
        formula="C8H12O",
        element_counts={"C": 8, "H": 12, "O": 1},
    )
    index = MoleculeIndex([amine, ketone], allow_pubchem=False)
    hits = index.search("smarts:[NX3;H2,H1,H0;!$(NC=O)]")
    keys = {h.entry.key for h in hits}
    assert keys == {"amine"}


def test_looks_like_smiles() -> None:
    assert looks_like_smiles("CC(=O)O")
    assert not looks_like_smiles("alanine")
    assert not looks_like_smiles("42")
    assert not looks_like_smiles("C3H7NO2")


def test_smiles_search_is_the_exact_molecule() -> None:
    """A SMILES query matches that graph only, written any way around.

    ``OCC`` is ethanol, not every alcohol. Propanol contains that fragment
    as a substructure, but a SMILES search must not return it.
    """
    ethanol = MoleculeIndexEntry(
        dataset="test",
        key="ethanol",
        smiles="CCO",
        canonical_smiles="CCO",
        n_atoms=9,
        label="test:ethanol",
        formula="C2H6O",
        element_counts={"C": 2, "H": 6, "O": 1},
    )
    propanol = MoleculeIndexEntry(
        dataset="test",
        key="propanol",
        smiles="CCCO",
        canonical_smiles="CCCO",
        n_atoms=12,
        label="test:propanol",
        formula="C3H8O",
        element_counts={"C": 3, "H": 8, "O": 1},
    )
    index = MoleculeIndex([ethanol, propanol], allow_pubchem=False)

    occ = index.search("OCC")
    assert {h.entry.key for h in occ} == {"ethanol"}
    assert occ[0].match_type == "canonical_smiles"

    assert {h.entry.key for h in index.search("CCO")} == {"ethanol"}
    assert index.search("C=C") == []


def test_unprefixed_smarts_uses_the_same_path() -> None:
    acid = MoleculeIndexEntry(
        dataset="test",
        key="acid",
        smiles="CC(=O)O",
        canonical_smiles="CC(=O)O",
        n_atoms=8,
        label="test:acid",
        formula="C2H4O2",
    )
    index = MoleculeIndex([acid], allow_pubchem=False)
    hits = index.search("[CX3](=O)[OX2H1]")
    assert {h.entry.key for h in hits} == {"acid"}
    assert hits[0].match_type == "substructure"


def test_synonym_seed_has_alanine() -> None:
    seed = synonym_seed()
    assert "alanine" in seed
    assert canonical_smiles(seed["alanine"]) is not None


def test_name_search_uses_local_synonym(monkeypatch) -> None:
    alanine_smiles = synonym_seed()["alanine"]
    canon = canonical_smiles(alanine_smiles)
    assert canon
    entry = MoleculeIndexEntry(
        dataset="test",
        key="1",
        smiles=alanine_smiles,
        canonical_smiles=canon,
        n_atoms=13,
        label="test:1",
        formula="C3H7NO2",
        names=("alanine",),
    )
    index = MoleculeIndex([entry], allow_pubchem=False)

    def fail_pubchem(_name: str) -> str | None:
        raise AssertionError("PubChem should not be called for seeded names")

    monkeypatch.setattr("rpfr_visualizer.io.search.names.resolve_common_name", fail_pubchem)
    hits = index.search("alanine")
    assert hits
    assert hits[0].match_type.startswith("name:")


def test_name_resolve_cache_roundtrip(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "name_resolve.json"

    def fake_pubchem(name: str) -> str | None:
        assert name.lower() == "foobarene"
        return "C"

    monkeypatch.setattr("rpfr_visualizer.io.search.names.resolve_common_name", fake_pubchem)
    smiles, source = resolve_name_hybrid(
        "foobarene",
        resolve_cache_path=cache_path,
        allow_pubchem=True,
    )
    assert smiles == "C"
    assert source == "pubchem"
    assert cache_path.exists()

    smiles2, source2 = resolve_name_hybrid(
        "foobarene",
        resolve_cache_path=cache_path,
        allow_pubchem=False,
    )
    assert smiles2 == "C"
    assert source2 == "cache"


def test_name_search_uses_pubchem_not_smiles_parser(monkeypatch) -> None:
    entry = MoleculeIndexEntry(
        dataset="test",
        key="1",
        smiles="C",
        canonical_smiles="C",
        n_atoms=1,
        label="test:1",
    )
    index = MoleculeIndex([entry], allow_pubchem=True)

    def fake_resolve(name: str) -> str | None:
        assert name == "alanine"
        return "C"

    monkeypatch.setattr("rpfr_visualizer.io.search.names.resolve_common_name", fake_resolve)
    # Clear seed temporarily so we hit PubChem path for this test
    monkeypatch.setattr(
        "rpfr_visualizer.io.search.names.synonym_seed",
        lambda: {},
    )
    monkeypatch.setattr(
        "rpfr_visualizer.io.search.names.load_name_resolve_cache",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "rpfr_visualizer.io.search.names.save_name_resolve_cache",
        lambda *_a, **_k: None,
    )
    hits = index.search("alanine")
    assert hits
    assert "name" in hits[0].match_type


def test_cache_roundtrip_preserves_formula(tmp_path: Path) -> None:
    entry = MoleculeIndexEntry(
        dataset="test",
        key="1",
        smiles="CCO",
        canonical_smiles="CCO",
        n_atoms=9,
        label="test:1",
        formula="C2H6O",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        element_counts={"C": 2, "H": 6, "O": 1},
        names=("ethanol",),
    )
    index = MoleculeIndex([entry])
    path = tmp_path / "molecule_index.json"
    index.save_cache(path)
    loaded = MoleculeIndex.load_cache(path)
    assert loaded.entries[0].formula == "C2H6O"
    assert loaded.entries[0].canonical_smiles == "CCO"
    assert "ethanol" in loaded.entries[0].names
    payload = path.read_text()
    assert f'"index_version": {INDEX_VERSION}' in payload


def test_index_manual_entry_search() -> None:
    entry = MoleculeIndexEntry(
        dataset="test",
        key="rxn1_r",
        smiles="C",
        canonical_smiles="C",
        n_atoms=1,
        label="test:rxn1_r",
    )
    index = MoleculeIndex([entry], allow_pubchem=False)
    hits = index.search("rxn1")
    assert hits
    assert hits[0].match_type == "key_substring"


def test_default_drive_root_falls_back_to_the_bundled_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DRIVE_ROOT_ENV, raising=False)
    assert default_drive_root() == DEFAULT_DRIVE_ROOT


def test_default_drive_root_honours_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DRIVE_ROOT_ENV, str(tmp_path))
    assert default_drive_root() == tmp_path


def test_default_drive_root_ignores_an_empty_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported-but-empty variable is not a path; it means "unset"."""
    monkeypatch.setenv(DRIVE_ROOT_ENV, "")
    assert default_drive_root() == DEFAULT_DRIVE_ROOT
