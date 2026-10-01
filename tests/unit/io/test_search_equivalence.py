"""The two search backends must be interchangeable.

The SQLite backend is a performance replacement for the in-memory one, not a
behaviour change: a query answered against a prebuilt index has to return what
the same query returns against a freshly scanned entry list. This is the safety
net for the migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.io.molecule_index import MoleculeIndex
from rpfr_visualizer.io.search import SqliteSearchBackend
from rpfr_visualizer.io.search.memory_backend import MemorySearchBackend

pytest.importorskip("rdkit")

QUERIES: tuple[str, ...] = (
    "1",
    "C2H6O",
    "H6C2O",
    "CCO",
    "CC#N",
    "c1ccccc1",
    "smarts:[CX3](=O)[OX2H1]",
    "smarts:c1ccccc1",
    "smarts:[NX3;!$(NC=O)]",
    "fg:nitrile",
    "fg:alcohol",
    "ethanol",
    "nothing-matches-this",
    "",
    # Size tokens: alone, ANDed onto a preset, and ANDed onto a SMARTS whose
    # own body contains commas.
    "C:1-2",
    "nC>1",
    "heavy<=3",
    "atoms<=9",
    "fg:alcohol, C:1-2",
    "smarts:[CX3](=O)[OX2H1], C:2",
    "smarts:[NX3;!$(NC=O)], nC>=1",
    # QC tokens: alone, ANDed onto a preset, a formula and a comma-bearing
    # SMARTS. The two backends derive the verdict from different places -- one
    # from a stored column, one by rereading the HDF5 attrs -- so agreeing here
    # is not automatic.
    "qc:pass",
    "qc:fail",
    "fg:alcohol, qc:pass",
    "fg:alcohol, qc:fail",
    "C2H6O, qc:pass",
    "qc:pass, C:1-2",
    "smarts:[NX3;!$(NC=O)], qc:pass",
    "smarts:[CX3](=O)[OX2H1], C:2, qc:fail",
)


@pytest.fixture
def both_indexes(search_catalog, search_index_path: Path):
    """The same data, once scanned in memory and once as a built index."""
    memory = MoleculeIndex.build_from_catalog(search_catalog)
    memory.allow_pubchem = False
    sqlite = MoleculeIndex.open(search_index_path, allow_pubchem=False)
    return memory, sqlite


def _identity(hits) -> set[tuple[str, str]]:
    return {(h.entry.dataset, h.entry.key) for h in hits}


@pytest.mark.parametrize("query", QUERIES)
def test_search_agrees_across_backends(both_indexes, query: str) -> None:
    memory, sqlite = both_indexes
    assert _identity(memory.search(query)) == _identity(sqlite.search(query))


@pytest.mark.parametrize("query", QUERIES)
def test_search_agrees_when_filtered_by_dataset(both_indexes, query: str) -> None:
    memory, sqlite = both_indexes
    assert _identity(memory.search(query, datasets=("beta",))) == _identity(
        sqlite.search(query, datasets=("beta",))
    )


def test_inventory_agrees(both_indexes) -> None:
    memory, sqlite = both_indexes
    assert memory.datasets == sqlite.datasets
    assert len(memory) == len(sqlite)


def test_get_and_get_many_agree(both_indexes) -> None:
    memory, sqlite = both_indexes
    pairs = [("alpha", "1"), ("beta", "2"), ("alpha", "absent")]
    assert set(memory.get_many(pairs)) == set(sqlite.get_many(pairs))

    a = memory.get("alpha", "1")
    b = sqlite.get("alpha", "1")
    assert a is not None and b is not None
    assert (a.smiles, a.formula, a.inchikey) == (b.smiles, b.formula, b.inchikey)
    assert memory.get("alpha", "absent") is sqlite.get("alpha", "absent") is None


def test_element_filtering_agrees(both_indexes) -> None:
    memory, sqlite = both_indexes
    for kwargs in (
        {"must_have": ["N"]},
        {"must_not_have": ["O"]},
        {"counts": {"C": (2, 2)}},
        {"must_have": ["C"], "must_not_have": ["N"], "datasets": ("alpha",)},
    ):
        assert {(e.dataset, e.key) for e in memory.filter_elements(**kwargs)} == {
            (e.dataset, e.key) for e in sqlite.filter_elements(**kwargs)
        }


def test_limit_is_respected_by_both(both_indexes) -> None:
    memory, sqlite = both_indexes
    assert len(memory.search("smarts:[#6]", limit=3)) == 3
    assert len(sqlite.search("smarts:[#6]", limit=3)) == 3


def test_functional_group_search_agrees(both_indexes) -> None:
    """SQLite reads precomputed membership; memory runs the SMARTS live."""
    memory, sqlite = both_indexes
    for preset in ("nitrile", "alcohol", "amide", "arene"):
        assert _identity(memory.search_functional_group(preset)) == _identity(
            sqlite.search_functional_group(preset)
        )


def test_entries_property_warns_only_when_large(search_index_path: Path, monkeypatch) -> None:
    """Materializing every entry is fine when small, and warns when it is not."""
    import warnings

    index = MoleculeIndex.open(search_index_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert len(index.entries) == len(index)

    monkeypatch.setattr("rpfr_visualizer.io.molecule_index.ENTRIES_WARN_THRESHOLD", 1)
    with pytest.warns(RuntimeWarning, match="iter_entries"):
        assert index.entries


def test_memory_backend_can_be_passed_directly(search_catalog) -> None:
    scanned = MoleculeIndex.build_from_catalog(search_catalog)
    wrapped = MoleculeIndex(backend=MemorySearchBackend(list(scanned.iter_entries())))
    assert len(wrapped) == len(scanned)
    assert _identity(wrapped.search("CCO")) == _identity(scanned.search("CCO"))


def test_sqlite_backend_can_be_passed_directly(search_index_path: Path) -> None:
    index = MoleculeIndex(backend=SqliteSearchBackend(search_index_path))
    assert len(index) > 0


def test_entry_contents_agree_not_just_identities(both_indexes) -> None:
    """Compare whole entries, not just (dataset, key).

    Identity-only comparison let a real bug through: the SQLite backend was
    returning entries with an empty ``names`` tuple because the joined synonym
    column was not selected.
    """
    memory, sqlite = both_indexes
    for entry in memory.iter_entries():
        other = sqlite.get(entry.dataset, entry.key)
        assert other is not None
        assert other.smiles == entry.smiles
        assert other.canonical_smiles == entry.canonical_smiles
        assert other.formula == entry.formula
        assert other.inchikey == entry.inchikey
        assert other.element_counts == entry.element_counts
        assert other.names == entry.names


def test_synonyms_survive_every_query_path(both_indexes) -> None:
    """A molecule with a known name reports it however it was found."""
    memory, sqlite = both_indexes
    named = [e for e in memory.iter_entries() if e.names]
    if not named:
        pytest.skip("no synonym in this fixture matched a molecule")
    entry = named[0]
    name = entry.names[0]

    for found in (
        sqlite.get(entry.dataset, entry.key),
        sqlite.backend.by_name(name)[0],
        sqlite.backend.search_names(name[:4])[0],
        sqlite.backend.by_canonical_smiles(entry.canonical_smiles)[0],
    ):
        assert name in found.names


def test_functional_group_counts_agree(both_indexes) -> None:
    """The aggregate must return what counting the hits returned.

    SQLite answers this from one GROUP BY over the precomputed membership table
    rather than materialising an entry per row. That is a 23x win on the full
    index, so it has to be exactly equivalent, not approximately.
    """
    memory, sqlite = both_indexes
    assert sqlite.functional_group_counts() == memory.functional_group_counts()

    for datasets in (("alpha",), ("beta",), ("alpha", "beta")):
        assert sqlite.functional_group_counts(datasets=datasets) == memory.functional_group_counts(
            datasets=datasets
        )


def test_functional_group_counts_match_counting_the_hits(both_indexes) -> None:
    """Pin the aggregate against the path it replaced."""
    from rpfr_visualizer.analysis.functional_groups import primary_fg_presets

    _memory, sqlite = both_indexes
    datasets = ("alpha",)
    counts = sqlite.functional_group_counts(datasets=datasets)
    by_hits = {
        preset.name: len(sqlite.search_functional_group(preset.name, datasets=datasets))
        for preset in primary_fg_presets()
    }
    assert counts == {name: n for name, n in by_hits.items() if n}


def test_dataset_inventory_reports_structure_coverage(both_indexes) -> None:
    """Structure coverage is what makes a zero-hit structural search legible."""
    memory, sqlite = both_indexes
    inventory = sqlite.dataset_inventory()

    assert set(inventory) == {"alpha", "beta"}
    assert inventory["alpha"]["n"] == 10
    assert inventory["beta"]["n"] == 4
    # Every fixture molecule carries a SMILES, so coverage is complete.
    for row in inventory.values():
        assert row["n_with_structure"] == row["n"]
        assert row["n_qc_pass"] == row["n"]

    # Both backends agree on the counts they can both derive.
    mem_inventory = memory.dataset_inventory()
    for dataset, row in inventory.items():
        assert mem_inventory[dataset]["n"] == row["n"]
        assert mem_inventory[dataset]["n_with_structure"] == row["n_with_structure"]


def test_inventory_distinguishes_no_structures_from_no_matches(
    mixed_coverage_index_path: Path,
) -> None:
    """The whole point of the coverage number.

    A dataset that stores no structures cannot answer a functional-group query,
    and without this a zero-hit result is indistinguishable from "there are
    none here".
    """
    index = MoleculeIndex.open(mixed_coverage_index_path, allow_pubchem=False)
    inventory = index.dataset_inventory()

    assert inventory["withsmiles"]["n_with_structure"] == inventory["withsmiles"]["n"] == 10
    assert inventory["nosmiles"]["n"] == 3
    assert inventory["nosmiles"]["n_with_structure"] == 0

    # And the structural query really does come back empty for that dataset.
    assert index.search_functional_group("alcohol", datasets=("nosmiles",)) == []
    assert index.functional_group_counts(datasets=("nosmiles",)) == {}
    assert index.functional_group_counts(datasets=("withsmiles",))["alcohol"] > 0
