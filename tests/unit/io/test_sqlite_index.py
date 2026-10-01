"""Tests for building and querying the SQLite search index."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from rpfr_visualizer.io.search import (
    INDEX_ENV_VAR,
    MissingIndexError,
    SqliteSearchBackend,
    build_index,
    find_index,
    open_index,
)
from rpfr_visualizer.io.search.schema import INDEX_VERSION
from rpfr_visualizer.io.search.sqlite_backend import IndexVersionError

pytest.importorskip("rdkit")


def test_build_reports_contents(tmp_path: Path, search_catalog) -> None:
    report = build_index(search_catalog, tmp_path / "i.sqlite", progress=False)
    assert report.n_molecules == 14  # 10 in alpha + 4 in beta
    assert report.n_datasets == 2
    assert report.per_dataset == {"alpha": 10, "beta": 4}
    assert report.n_fg_links > 0
    assert "molecules" in report.summary()


def test_meta_and_version(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    meta = backend.meta()
    assert meta["index_version"] == str(INDEX_VERSION)
    assert meta["fp_kind"] == "rdkit-pattern"
    assert backend.rdkit_version()
    assert len(backend.source_signature()) == 2

    with sqlite3.connect(search_index_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == INDEX_VERSION
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_version_mismatch_is_a_hard_error(search_index_path: Path) -> None:
    with sqlite3.connect(search_index_path) as conn:
        conn.execute("UPDATE meta SET value = '999' WHERE key = 'index_version'")
    with pytest.raises(IndexVersionError, match="rebuild it with|Rebuild it with"):
        SqliteSearchBackend(search_index_path)


def test_inventory(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert backend.datasets() == ("alpha", "beta")
    assert backend.count() == 14
    assert backend.count(datasets=["beta"]) == 4
    assert len(list(backend.iter_entries(datasets=["beta"]))) == 4


def test_get_and_get_many(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    entry = backend.get("alpha", "1")
    assert entry is not None
    assert entry.smiles == "CCO"
    assert entry.formula == "C2H6O"
    assert entry.element_counts == {"C": 2, "H": 6, "O": 1}
    assert backend.get("alpha", "nope") is None

    found = backend.get_many([("alpha", "1"), ("beta", "3"), ("alpha", "missing")])
    assert set(found) == {("alpha", "1"), ("beta", "3")}


def test_lookups_by_identifier(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert {e.dataset for e in backend.by_key("1")} == {"alpha", "beta"}
    assert {e.dataset for e in backend.by_key("1", datasets=["alpha"])} == {"alpha"}
    # C2H6O is ethanol *and* dimethyl ether in alpha, plus ethanol in beta.
    assert len(backend.by_formula("C2H6O")) == 3
    assert len(backend.by_canonical_smiles("CCO")) == 2
    assert len(backend.by_smiles_substring("C#N")) == 2

    ethanol = backend.get("alpha", "1")
    assert ethanol is not None
    assert backend.by_inchikey(ethanol.inchikey)
    assert backend.by_inchikey(ethanol.inchikey.split("-")[0])


def test_formula_lookup_normalizes_hill_order(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert len(backend.by_formula("H6C2O")) == len(backend.by_formula("C2H6O")) == 3


def test_element_filtering(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert {e.smiles for e in backend.by_elements(must_have=["N"], datasets=["alpha"])} == {
        "CC#N",
        "CCN",
        "CC(=O)N",
    }
    no_oxygen = {e.smiles for e in backend.by_elements(must_not_have=["O"], datasets=["alpha"])}
    assert "CCO" not in no_oxygen
    assert "c1ccccc1" in no_oxygen
    ranged = backend.by_elements(counts={"C": (4, 4)}, datasets=["alpha"])
    assert {e.smiles for e in ranged} == {"CCOCC"}
    assert backend.by_elements() == []


def test_functional_group_membership(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert backend.has_functional_group("nitrile")
    assert not backend.has_functional_group("not_a_group")
    assert {e.smiles for e in backend.by_functional_group("nitrile", datasets=["alpha"])} == {
        "CC#N"
    }
    assert {
        e.smiles for e in backend.by_functional_group("carboxylic_acid", datasets=["alpha"])
    } == {"CC(=O)O"}
    assert len(backend.by_functional_group("nitrile", limit=1)) == 1


def test_name_lookup_and_prefix_search(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    assert {e.smiles for e in backend.by_name("ethanol")} == {"CCO"}
    assert {e.smiles for e in backend.search_names("ethano")} == {"CCO"}
    assert backend.search_names("") == []
    assert backend.search_names("zzzz") == []


def test_name_search_falls_back_without_fts(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    with_fts = {e.smiles for e in backend.search_names("ethano")}
    backend._has_fts = False  # force the LIKE branch
    assert {e.smiles for e in backend.search_names("ethano")} == with_fts


def test_index_is_portable_after_moving(search_index_path: Path, tmp_path: Path) -> None:
    """A published index must work wherever the user puts it."""
    moved = tmp_path / "elsewhere" / "renamed.sqlite"
    moved.parent.mkdir()
    shutil.copy(search_index_path, moved)
    backend = SqliteSearchBackend(moved)
    assert backend.count() == 14
    # Signatures record basenames, never absolute paths.
    assert all(not Path(row[1]).is_absolute() for row in backend.source_signature())


def test_index_is_opened_read_only(search_index_path: Path) -> None:
    backend = SqliteSearchBackend(search_index_path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        backend._conn.execute("DELETE FROM molecules")


def test_discovery_prefers_explicit_then_env(
    search_index_path: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(INDEX_ENV_VAR, raising=False)
    assert find_index(search_index_path) == search_index_path
    assert find_index(tmp_path / "nothing.sqlite") is None

    monkeypatch.setenv(INDEX_ENV_VAR, str(search_index_path))
    assert find_index() == search_index_path
    assert open_index().count() == 14


def test_missing_index_explains_how_to_get_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(INDEX_ENV_VAR, raising=False)
    with pytest.raises(MissingIndexError, match="rpfr-visualizer index build"):
        open_index(data_dir=tmp_path)
