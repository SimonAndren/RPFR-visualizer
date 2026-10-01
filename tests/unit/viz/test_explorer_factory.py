"""Tests for the controller factories.

These are what a non-widget front end calls, so they have to resolve data and
index paths without relying on the process working directory and without ever
starting an index build.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rpfr_visualizer.io.providers import DatasetCatalog
from rpfr_visualizer.viz.explorer.factory import (
    PREFERRED_DATASET,
    controller_for_data_dir,
    controller_for_drive,
    preferred_dataset,
)

pytest.importorskip("rdkit")


@pytest.fixture
def data_dir(synthetic_h5_path: Path, tmp_path: Path) -> Path:
    """A data directory laid out the way register_subsample_catalog expects."""
    raw = tmp_path / "raw"
    raw.mkdir()
    # The catalog maps filenames to dataset names, so the name has to match.
    shutil.copy(synthetic_h5_path, raw / "qm9s_tr_project_1pct.h5")
    shutil.copy(synthetic_h5_path, raw / "qm9s_tr_skip_1pct.h5")
    return tmp_path


def test_prefers_the_current_release_over_the_first_registered_dataset() -> None:
    """An older QM9s file sorting first must not be what the app opens on."""
    catalog = DatasetCatalog()
    for name in ("qm9s", "qm9s_skip", "qm9s_v2.2"):
        catalog.register_provider(name, _stub_provider(name))

    assert catalog.datasets[0] == "qm9s"
    assert preferred_dataset(catalog) == PREFERRED_DATASET == "qm9s_v2.2"
    assert preferred_dataset(catalog, "qm9s_skip") == "qm9s_skip"


def test_falls_back_to_the_first_dataset_when_the_preferred_one_is_absent() -> None:
    catalog = DatasetCatalog()
    catalog.register_provider("qm9s_skip", _stub_provider("qm9s_skip"))
    assert preferred_dataset(catalog) == "qm9s_skip"


def test_unknown_requested_dataset_is_an_error() -> None:
    catalog = DatasetCatalog()
    catalog.register_provider("qm9s", _stub_provider("qm9s"))
    with pytest.raises(KeyError, match="not in catalog"):
        preferred_dataset(catalog, "nope")


def test_empty_catalog_is_an_error() -> None:
    with pytest.raises(ValueError, match="no registered datasets"):
        preferred_dataset(DatasetCatalog())


def test_controller_for_data_dir_resolves_paths_absolutely(data_dir: Path) -> None:
    controller = controller_for_data_dir(data_dir, max_stats_molecules=2)

    assert controller.dataset == "qm9s"
    assert controller.data_dir == data_dir
    # No index was built for this fixture, so search is simply unavailable.
    assert controller._index is None
    # Caches land beside the data, not beside the process.
    assert controller._stats_cache_path("qm9s").is_absolute()


def test_controller_for_data_dir_does_not_build_an_index(data_dir: Path, monkeypatch) -> None:
    """A missing index must never trigger a build.

    Over the full datasets that is a tens-of-minutes surprise, so the only
    acceptable behaviour is to run without search.
    """
    from rpfr_visualizer.io import molecule_index

    def explode(*args, **kwargs):
        raise AssertionError("the factory must not build an index")

    monkeypatch.setattr(molecule_index, "build_or_load_index", explode)
    controller = controller_for_data_dir(data_dir, max_stats_molecules=2)
    assert controller._index is None


def test_controller_for_data_dir_reports_a_missing_data_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No QM9s-RPFR datasets found"):
        controller_for_data_dir(tmp_path / "nothing-here")


def test_controller_for_drive_without_an_index_names_the_build_command(
    tmp_path: Path, monkeypatch
) -> None:
    """The error has to carry the command that fixes it."""
    from rpfr_visualizer.viz.explorer import factory

    catalog = DatasetCatalog()
    catalog.register_provider("qm9s", _stub_provider("qm9s"))
    monkeypatch.setattr(factory, "register_drive_catalog", lambda root: catalog)

    with pytest.raises(FileNotFoundError) as excinfo:
        controller_for_drive(drive_root=tmp_path, data_dir=tmp_path)

    message = str(excinfo.value)
    assert "rpfr-visualizer index build" in message
    assert "--drive" in message


def _stub_provider(name: str):
    """A provider stand-in for tests that never touch the file."""

    class _Provider:
        dataset = name
        path = Path(f"/nonexistent/{name}.h5")
        temperatures: tuple[float, ...] = (300.0,)

    return _Provider()


def test_mismatched_index_is_detected(data_dir: Path, monkeypatch) -> None:
    """An index outliving its data files must be reported, not silently used.

    A SQLite index is self-contained, so it keeps answering searches after the
    HDF5 files it was built from are gone — every hit then fails to open, which
    reads as a broken app rather than as missing data.
    """
    from rpfr_visualizer.io.molecule_index import MoleculeIndex, register_subsample_catalog
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index
    from rpfr_visualizer.viz.explorer.factory import index_data_mismatch

    # Build an index over a file whose keys are absent from the real one.
    other = data_dir / "other.h5"
    _write_disjoint_dataset(other)
    other_catalog = DatasetCatalog()
    other_catalog.register("qm9s", other)
    index_path = data_dir / "mismatched.sqlite"
    build_index(other_catalog, index_path, progress=False)

    catalog = register_subsample_catalog(data_dir / "raw")
    index = MoleculeIndex.open(index_path, allow_pubchem=False)

    message = index_data_mismatch(catalog, index, "qm9s")
    assert "cannot be opened" in message
    assert "qm9s_tr_project_1pct.h5" in message


def test_matching_index_reports_no_mismatch(data_dir: Path) -> None:
    from rpfr_visualizer.io.molecule_index import MoleculeIndex, register_subsample_catalog
    from rpfr_visualizer.io.search import build_index
    from rpfr_visualizer.viz.explorer.factory import index_data_mismatch

    catalog = register_subsample_catalog(data_dir / "raw")
    index_path = data_dir / "matching.sqlite"
    build_index(catalog, index_path, progress=False)
    index = MoleculeIndex.open(index_path, allow_pubchem=False)

    assert index_data_mismatch(catalog, index, "qm9s") == ""


def test_no_index_means_nothing_to_compare(data_dir: Path) -> None:
    from rpfr_visualizer.io.molecule_index import register_subsample_catalog
    from rpfr_visualizer.viz.explorer.factory import index_data_mismatch

    catalog = register_subsample_catalog(data_dir / "raw")
    assert index_data_mismatch(catalog, None, "qm9s") == ""


def _write_disjoint_dataset(path: Path) -> None:
    """A v2.0 file whose molecule keys cannot collide with the fixture's."""
    import json

    import h5py
    import numpy as np

    from rpfr_visualizer.io.schema import TEMPERATURES_K

    n_temps = len(TEMPERATURES_K)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.0"
        h5.attrs["dataset"] = "qm9s"
        h5.attrs["temperatures_K"] = json.dumps(list(TEMPERATURES_K))
        for key in ("900001", "900002", "900003", "900004", "900005"):
            group = h5.create_group(key)
            group.attrs["mol_id"] = key
            group.attrs["smiles"] = "CCO"
            group.attrs["n_atoms"] = 3
            group.attrs["rdkit_ok"] = True
            group.create_dataset("rpfr_single", data=np.ones((3, n_temps)))
            group.create_dataset("atom_symbol", data=np.array(["C"] * 3, dtype="S2"))


def test_a_single_h5_file_can_be_opened(data_dir: Path) -> None:
    """The layout a person with one big file actually has.

    controller_for_data_dir requires <dir>/raw/<hardcoded-name>.h5, so pointing
    at a file — or at an external drive, which nests one directory per dataset —
    used to fail with a message about a /raw directory that does not exist.
    """
    from rpfr_visualizer.viz.explorer.factory import catalog_for_path, controller_for_path

    h5 = data_dir / "raw" / "qm9s_tr_project_1pct.h5"
    catalog, what = catalog_for_path(h5)
    assert catalog.datasets == ("qm9s",)
    assert "qm9s_tr_project_1pct.h5" in what

    controller, notes = controller_for_path(h5, max_stats_molecules=2)
    assert controller.dataset == "qm9s"
    assert any("Opened" in n for n in notes)


def test_a_drive_style_layout_can_be_opened(tmp_path: Path, synthetic_h5_path: Path) -> None:
    """One directory per dataset, the external-drive shape."""
    import shutil

    from rpfr_visualizer.viz.explorer.factory import catalog_for_path

    root = tmp_path / "drive"
    root.mkdir()
    for name in ("qm9s", "qm9s_skip"):
        (root / name).mkdir()
        shutil.copy(synthetic_h5_path, root / name / f"{name}_tr_project.h5")

    catalog, what = catalog_for_path(root)
    assert set(catalog.datasets) == {"qm9s", "qm9s_skip"}
    assert "2 datasets" in what


def test_flat_h5_files_can_be_opened(tmp_path: Path, synthetic_h5_path: Path) -> None:
    import shutil

    from rpfr_visualizer.viz.explorer.factory import catalog_for_path

    # Its own directory: the synthetic fixture writes into tmp_path too.
    flat = tmp_path / "flat"
    flat.mkdir()
    shutil.copy(synthetic_h5_path, flat / "qm9s_tr_project.h5")
    catalog, _ = catalog_for_path(flat)
    assert catalog.datasets == ("qm9s",)


def test_applestore_sidecars_are_ignored(tmp_path: Path, synthetic_h5_path: Path) -> None:
    """A macOS-formatted drive is full of ._name.h5 metadata forks."""
    import shutil

    from rpfr_visualizer.viz.explorer.factory import catalog_for_path

    flat = tmp_path / "flat"
    flat.mkdir()
    shutil.copy(synthetic_h5_path, flat / "qm9s_tr_project.h5")
    (flat / "._qm9s_tr_project.h5").write_bytes(b"not hdf5")
    catalog, _ = catalog_for_path(flat)
    assert catalog.datasets == ("qm9s",)


def test_an_unrecognised_path_names_every_layout_it_tried(tmp_path: Path) -> None:
    from rpfr_visualizer.viz.explorer.factory import catalog_for_path

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        catalog_for_path(empty)
    message = str(excinfo.value)
    assert "raw" in message and "subdirectories" in message


def test_the_larger_matching_index_wins(data_dir: Path, tmp_path: Path) -> None:
    """Choosing by filename picks the subsample index over the full one.

    find_index knows exactly one name, so on a machine holding both it returns
    the subsample — which against the full data searches 1% of it while looking
    entirely healthy.
    """
    from rpfr_visualizer.io.molecule_index import register_subsample_catalog
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index
    from rpfr_visualizer.viz.explorer.factory import best_index_for

    catalog = register_subsample_catalog(data_dir / "raw")

    full = data_dir / "full.sqlite"
    build_index(catalog, full, progress=False)

    # An index over a disjoint file: bigger is irrelevant if it does not match.
    other = data_dir / "other.h5"
    _write_disjoint_dataset(other)
    other_catalog = DatasetCatalog()
    other_catalog.register("qm9s", other)
    wrong = data_dir / "wrong.sqlite"
    build_index(other_catalog, wrong, progress=False)

    index, why = best_index_for(catalog, "qm9s", [wrong, full])
    assert index is not None
    assert "full.sqlite" in why
    assert "skipped 1" in why


def test_no_matching_index_says_why(data_dir: Path) -> None:
    from rpfr_visualizer.io.molecule_index import register_subsample_catalog
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index
    from rpfr_visualizer.viz.explorer.factory import best_index_for

    other = data_dir / "other.h5"
    _write_disjoint_dataset(other)
    other_catalog = DatasetCatalog()
    other_catalog.register("qm9s", other)
    wrong = data_dir / "wrong.sqlite"
    build_index(other_catalog, wrong, progress=False)

    index, why = best_index_for(register_subsample_catalog(data_dir / "raw"), "qm9s", [wrong])
    assert index is None
    assert "different molecules" in why


def test_index_covering_no_molecules_of_this_dataset_is_rejected(data_dir: Path) -> None:
    """A zero-coverage index used to win by default, and search went silent.

    ``best_index_for`` only replaced its candidate on ``covered > best[0]``, which
    never runs while ``best`` is None, and the sample-based mismatch check had
    nothing to compare so it reported agreement. The index attached, the banner
    showed its full molecule count, and every query returned nothing.
    """
    from rpfr_visualizer.io.molecule_index import MoleculeIndex, register_subsample_catalog
    from rpfr_visualizer.io.search import build_index
    from rpfr_visualizer.viz.explorer.factory import best_index_for, index_data_mismatch

    catalog = register_subsample_catalog(data_dir / "raw")
    index_path = data_dir / "qm9s_only.sqlite"
    build_index(catalog, index_path, datasets=("qm9s",), progress=False)

    # 'qm9s_skip' is in the data but absent from the index.
    index, why = best_index_for(catalog, "qm9s_skip", [index_path])
    assert index is None
    assert "covers qm9s" in why
    assert "qm9s_skip" in why

    opened = MoleculeIndex.open(index_path, allow_pubchem=False)
    message = index_data_mismatch(catalog, opened, "qm9s_skip")
    assert "none for 'qm9s_skip'" in message
    assert "qm9s" in message


def test_dataset_names_ignore_the_subsample_marker() -> None:
    """A 1% file and the full file must share a name.

    Otherwise a published index built from the full data covers 'qm9s' while a
    subsample registers as 'qm9s_1pct', and the two can never be compared.
    """
    from rpfr_visualizer.io.molecule_index import dataset_name_for

    assert dataset_name_for("qm9s_tr_project.h5") == "qm9s"
    assert dataset_name_for("qm9s_tr_project_1pct.h5") == "qm9s"
    assert dataset_name_for("qm9s_tr_skip_1pct.h5") == "qm9s_skip"
    assert dataset_name_for("my_data.h5") == "my_data"

    # Schema 2.2 dropped the tr_ naming and spells the subsample marker _1pt.
    # Both files have to land on one name, or the release tier and the full
    # index become two datasets that never meet.
    assert dataset_name_for("qm9s_rpfr_v2.2.h5") == "qm9s_v2.2"
    assert dataset_name_for("qm9s_v2.2_1pt.h5") == "qm9s_v2.2"

    # _1pt must not be stripped out of _1pct and leave a stray "c" behind.
    assert dataset_name_for("qm9s_v2.2_1pct.h5") == "qm9s_v2.2"


def test_loose_h5_files_open_with_the_same_names_as_the_repo_layout(
    data_dir: Path, tmp_path: Path
) -> None:
    """Pointing at a folder of .h5 files must not rename the datasets.

    Pointing at 'data/raw' rather than 'data' used to register 'qm9s_1pct',
    which no index covers, so the app opened the wrong dataset and searched
    nothing.
    """
    from rpfr_visualizer.viz.explorer.factory import catalog_for_path

    nested, _ = catalog_for_path(data_dir)
    flat, _ = catalog_for_path(data_dir / "raw")
    assert flat.datasets == nested.datasets == ("qm9s", "qm9s_skip")


def test_a_single_file_is_named_from_its_filename(data_dir: Path) -> None:
    """The HDF5 'dataset' attribute is not unique, so it cannot be the key.

    Both qm9s_tr_* files carry "qm9s", so naming from the attribute loses one
    file of the pair.
    """
    from rpfr_visualizer.viz.explorer.factory import catalog_for_path

    catalog, described = catalog_for_path(data_dir / "raw" / "qm9s_tr_skip_1pct.h5")
    assert catalog.datasets == ("qm9s_skip",)
    assert "qm9s_skip" in described


def test_colliding_names_are_kept_rather_than_dropped(data_dir: Path, tmp_path: Path) -> None:
    """Two files deriving one name used to mean the second vanished silently."""
    import shutil

    from rpfr_visualizer.io.molecule_index import register_all
    from rpfr_visualizer.io.providers import DatasetCatalog

    folder = tmp_path / "collide"
    folder.mkdir()
    first = folder / "qm9s_tr_project.h5"
    second = folder / "qm9s_tr_project_1pct.h5"
    shutil.copy(data_dir / "raw" / "qm9s_tr_project_1pct.h5", first)
    shutil.copy(data_dir / "raw" / "qm9s_tr_project_1pct.h5", second)

    catalog = DatasetCatalog()
    notes = register_all(catalog, [first, second])
    assert catalog.datasets == ("qm9s", "qm9s_2")
    assert notes and "qm9s_2" in notes[0]
