"""Tests for GUI data discovery and settings persistence.

Every test sets ``$RPFR_VISUALIZER_GUI_CONFIG`` so the suite can never read or write a
real user's settings file.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rpfr_visualizer.gui.config import (
    CONFIG_ENV_VAR,
    CONFIG_VERSION,
    DATA_ENV_VAR,
    GuiConfig,
    config_path,
    discover,
    load_config,
    resolve,
)
from rpfr_visualizer.io.search import INDEX_ENV_VAR


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch) -> Path:
    """Point the config at a throwaway file and clear the env we resolve from."""
    target = tmp_path / "config" / "gui.json"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(target))
    monkeypatch.delenv(INDEX_ENV_VAR, raising=False)
    monkeypatch.delenv(DATA_ENV_VAR, raising=False)
    return target


@pytest.fixture
def data_tree(synthetic_h5_path: Path, tmp_path: Path) -> Path:
    """A ``data/`` directory holding a raw HDF5 file and a cache dir."""
    data = tmp_path / "project" / "data"
    (data / "raw").mkdir(parents=True)
    (data / "cache").mkdir()
    shutil.copy(synthetic_h5_path, data / "raw" / "qm9s_tr_project_1pct.h5")
    return data


# ------------------------------------------------------------------ config path


def test_config_path_follows_xdg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_path() == tmp_path / "xdg" / "rpfr-visualizer" / "gui.json"


def test_config_path_env_override_wins(isolated_config: Path) -> None:
    assert config_path() == isolated_config


# ---------------------------------------------------------------- load and save


def test_round_trip(isolated_config: Path) -> None:
    saved = GuiConfig(data_dir="/data", dataset="qm9s_skip", port=9000)
    saved.save()

    loaded, problems = load_config()
    assert problems == []
    assert loaded == saved


def test_missing_config_is_not_a_problem() -> None:
    loaded, problems = load_config()
    assert loaded == GuiConfig()
    assert problems == []


def test_malformed_config_is_ignored_but_reported(isolated_config: Path) -> None:
    """Losing settings must never stop the app, but must not be silent either."""
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text("{not json", encoding="utf-8")

    loaded, problems = load_config()
    assert loaded == GuiConfig()
    assert len(problems) == 1
    assert "unreadable settings" in problems[0]


def test_unknown_version_is_ignored_but_reported(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(json.dumps({"version": 99, "data_dir": "/x"}), encoding="utf-8")

    loaded, problems = load_config()
    assert loaded == GuiConfig()
    assert f"expects {CONFIG_VERSION}" in problems[0]


def test_unknown_keys_are_dropped(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({"version": CONFIG_VERSION, "data_dir": "/x", "future_option": 1}),
        encoding="utf-8",
    )

    loaded, problems = load_config()
    assert loaded.data_dir == "/x"
    assert problems == []


def test_saved_config_omits_unset_fields(isolated_config: Path) -> None:
    GuiConfig(dataset="qm9s").save()
    written = json.loads(isolated_config.read_text(encoding="utf-8"))
    assert written == {"version": CONFIG_VERSION, "dataset": "qm9s"}


# -------------------------------------------------------------------- discovery


def test_discovery_walks_up_to_a_data_directory(data_tree: Path) -> None:
    """The REPO_ROOT walk-up, promoted out of a notebook cell."""
    nested = data_tree.parent / "docs" / "deeper"
    nested.mkdir(parents=True)

    found = discover(start=nested)
    assert str(data_tree) in found.data_dirs


def test_discovery_lists_every_index_candidate(data_tree: Path) -> None:
    """Two plausible index files is a question, not a heuristic.

    A machine can hold both a small subsample index and the full one; picking
    silently would be a guess where asking is cheap.
    """
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    catalog = DatasetCatalog()
    catalog.register("qm9s", data_tree / "raw" / "qm9s_tr_project_1pct.h5")
    build_index(catalog, data_tree / "cache" / "qm9s_rpfr_index.sqlite", progress=False)
    build_index(catalog, data_tree / "cache" / "qm9s_rpfr_full_index.sqlite", progress=False)

    found = discover(start=data_tree.parent)
    paths = {Path(c.path).name for c in found.index_candidates}
    assert paths == {"qm9s_rpfr_index.sqlite", "qm9s_rpfr_full_index.sqlite"}
    for candidate in found.index_candidates:
        assert candidate.molecules == 5  # the synthetic fixture's molecule count
        assert candidate.size_mb > 0
        assert candidate.error == ""


def test_discovery_describes_an_unusable_index(data_tree: Path) -> None:
    (data_tree / "cache" / "broken.sqlite").write_text("not a database", encoding="utf-8")

    found = discover(start=data_tree.parent)
    broken = [c for c in found.index_candidates if c.path.endswith("broken.sqlite")]
    assert broken and broken[0].error
    assert broken[0].molecules is None


def test_discovery_reports_the_index_env_var(data_tree: Path, monkeypatch) -> None:
    monkeypatch.setenv(INDEX_ENV_VAR, "/somewhere/custom.sqlite")
    assert discover(start=data_tree.parent).index_env == "/somewhere/custom.sqlite"


def test_discovery_finds_nothing_in_an_empty_tree(tmp_path: Path) -> None:
    found = discover(start=tmp_path)
    assert found.data_dirs == []
    assert found.index_candidates == []


# ------------------------------------------------------------------- resolution


def test_arguments_beat_the_remembered_config(data_tree: Path) -> None:
    config = GuiConfig(data_dir="/from/config", dataset="from_config")
    resolution = resolve(
        data_dir=data_tree,
        dataset="from_args",
        config=config,
        start=data_tree.parent,
    )
    assert resolution.data_dir == data_tree
    assert resolution.dataset == "from_args"


def test_config_beats_discovery(data_tree: Path, tmp_path: Path) -> None:
    other = tmp_path / "other" / "data"
    (other / "raw").mkdir(parents=True)
    shutil.copy(
        data_tree / "raw" / "qm9s_tr_project_1pct.h5", other / "raw" / "qm9s_tr_project_1pct.h5"
    )

    resolution = resolve(config=GuiConfig(data_dir=str(other)), start=data_tree.parent)
    assert resolution.data_dir == other.resolve()


def test_discovery_is_the_last_resort(data_tree: Path) -> None:
    resolution = resolve(start=data_tree.parent)
    assert resolution.data_dir == data_tree
    assert resolution.ready


def test_resolved_paths_are_absolute(data_tree: Path, monkeypatch) -> None:
    """The regression guard for the cwd bug.

    A relative data_dir would resolve against wherever the process happens to
    be, which is how the explorer used to miss an existing index entirely.
    """
    monkeypatch.chdir(data_tree.parent)
    resolution = resolve(data_dir="data", start=data_tree.parent)
    assert resolution.data_dir is not None
    assert resolution.data_dir.is_absolute()
    assert resolution.data_dir == data_tree


def test_missing_data_path_is_reported_not_raised(tmp_path: Path) -> None:
    resolution = resolve(data_dir=tmp_path / "nope", start=tmp_path)
    assert resolution.data_dir is None
    assert not resolution.ready
    assert any("No such file or directory" in p for p in resolution.problems)


def test_a_single_h5_file_is_an_acceptable_target(data_tree: Path) -> None:
    """Someone with one big .h5 should not have to build a directory layout."""
    h5 = data_tree / "raw" / "qm9s_tr_project_1pct.h5"
    resolution = resolve(data_dir=h5, start=data_tree.parent)
    assert resolution.data_dir == h5
    assert resolution.ready
    assert resolution.problems == []


def test_a_non_h5_file_is_rejected(tmp_path: Path) -> None:
    stray = tmp_path / "notes.txt"
    stray.write_text("hello", encoding="utf-8")
    resolution = resolve(data_dir=stray, start=tmp_path)
    assert resolution.data_dir is None
    assert any("Not an HDF5" in p for p in resolution.problems)


def test_unmounted_drive_is_reported_not_raised(tmp_path: Path) -> None:
    resolution = resolve(drive_root=tmp_path / "not-mounted", start=tmp_path)
    assert resolution.drive_root is None
    assert any("not mounted" in p for p in resolution.problems)


def test_index_env_var_is_honoured(data_tree: Path, monkeypatch) -> None:
    """Resolution must defer to find_index, not reimplement its ladder."""
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    catalog = DatasetCatalog()
    catalog.register("qm9s", data_tree / "raw" / "qm9s_tr_project_1pct.h5")
    elsewhere = data_tree.parent / "elsewhere.sqlite"
    build_index(catalog, elsewhere, progress=False)

    monkeypatch.setenv(INDEX_ENV_VAR, str(elsewhere))
    resolution = resolve(start=data_tree.parent)
    assert resolution.index_path == elsewhere


def test_missing_explicit_index_is_reported(data_tree: Path) -> None:
    resolution = resolve(
        data_dir=data_tree, index_path=data_tree / "nope.sqlite", start=data_tree.parent
    )
    assert resolution.index_path is None
    assert any("index not found" in p for p in resolution.problems)
    # No data problem, so setup is not required just because search is missing.
    assert resolution.ready


def test_config_problems_travel_into_the_resolution(data_tree: Path) -> None:
    resolution = resolve(
        data_dir=data_tree,
        config_problems=["Ignoring unreadable settings at /x"],
        start=data_tree.parent,
    )
    assert resolution.problems == ["Ignoring unreadable settings at /x"]


def test_max_stats_molecules_precedence(data_tree: Path) -> None:
    assert resolve(data_dir=data_tree, start=data_tree.parent).max_stats_molecules == 200
    assert (
        resolve(
            data_dir=data_tree, config=GuiConfig(max_stats_molecules=50), start=data_tree.parent
        ).max_stats_molecules
        == 50
    )
    assert (
        resolve(
            data_dir=data_tree,
            max_stats_molecules=10,
            config=GuiConfig(max_stats_molecules=50),
            start=data_tree.parent,
        ).max_stats_molecules
        == 10
    )


def test_discovery_offers_a_folder_of_loose_h5_files(
    synthetic_h5_path: Path, tmp_path: Path, isolated_config: Path
) -> None:
    """Dropping .h5 files into a folder is what people actually do.

    ``catalog_for_path`` has always opened that layout, but discovery required
    ``raw/`` or ``cache/``, so the files sat there while setup said none had
    been found.
    """
    import shutil

    folder = tmp_path / "rpfr-visualizer-data"
    folder.mkdir()
    shutil.copy(synthetic_h5_path, folder / "qm9s_tr_project.h5")

    found = discover(start=folder)
    assert str(folder) in found.data_dirs


def test_discovery_ignores_a_folder_of_only_appledouble_sidecars(
    tmp_path: Path, isolated_config: Path
) -> None:
    """``._name.h5`` files are macOS metadata forks and cannot be opened.

    Offering such a folder would hand the user a data directory that fails the
    moment it is opened.
    """
    folder = tmp_path / "sidecars"
    folder.mkdir()
    (folder / "._qm9s_tr_project.h5").write_bytes(b"")

    found = discover(start=folder)
    assert str(folder) not in found.data_dirs
