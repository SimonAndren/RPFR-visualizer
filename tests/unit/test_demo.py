"""The synthetic dataset behind ``rpfr-visualizer gui --demo``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rpfr_visualizer.demo import (
    DEMO_DATASET,
    DEMO_INDEX_NAME,
    build_demo_dataset,
    ensure_demo_data,
)
from rpfr_visualizer.io.molecule_index import dataset_name_for

pytest.importorskip("rdkit")


def test_demo_file_name_and_dataset_name_agree() -> None:
    """ensure_demo_data names the file after the dataset, and the catalog derives
    the dataset back from the filename. If those two ever disagree the demo opens
    under a name the UI did not ask for."""
    assert dataset_name_for(f"{DEMO_DATASET}.h5") == DEMO_DATASET


def test_build_demo_dataset_is_readable_as_v2(tmp_path: Path) -> None:
    from rpfr_visualizer import QM9sDataset

    path = build_demo_dataset(tmp_path / "demo.h5")

    ds = QM9sDataset(path)
    assert len(ds) == 5
    mol = ds.get_molecule(ds.molecule_ids[0])
    assert mol.smiles == "C"
    assert mol.sites


def test_build_demo_dataset_covers_the_qc_states_the_ui_renders(tmp_path: Path) -> None:
    """The demo is only useful if the Data quality panel has something to show."""
    from rpfr_visualizer import QM9sDataset

    ds = QM9sDataset(build_demo_dataset(tmp_path / "demo.h5"))
    verdicts = {m: ds.get_molecule(m).qc_pass for m in ds.molecule_ids}
    assert True in verdicts.values()
    assert False in verdicts.values()


def test_ensure_demo_data_builds_data_and_index(tmp_path: Path) -> None:
    demo = ensure_demo_data(tmp_path)

    assert demo.root == tmp_path
    assert demo.h5_path.exists()
    assert demo.index_path.exists()
    assert demo.index_path.name == DEMO_INDEX_NAME
    assert demo.h5_path.stem == DEMO_DATASET


def test_ensure_demo_data_is_idempotent(tmp_path: Path) -> None:
    """The second call must not rebuild: cheap restart is the whole point."""
    first = ensure_demo_data(tmp_path)
    stamp = first.index_path.stat().st_mtime_ns

    second = ensure_demo_data(tmp_path)

    assert second == first
    assert second.index_path.stat().st_mtime_ns == stamp


def test_ensure_demo_data_rebuilds_the_index_when_the_data_is_gone(tmp_path: Path) -> None:
    """An index that outlives its data answers queries about molecules nothing
    can open. Losing the HDF5 must invalidate the index, not just replace the
    file underneath it."""
    demo = ensure_demo_data(tmp_path)
    demo.h5_path.unlink()
    demo.index_path.write_bytes(b"not a database")

    rebuilt = ensure_demo_data(tmp_path)

    assert rebuilt.h5_path.exists()
    with sqlite3.connect(rebuilt.index_path) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert names, "index was left as the corrupt sentinel"
