"""Switching the explorer's β between the dataset and a trained model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rpfr_visualizer.empirical import split_hdf5, train_all
from rpfr_visualizer.io.providers import HDF5Provider
from rpfr_visualizer.viz.explorer import ExplorerController

pytest.importorskip("rdkit")


@pytest.fixture
def models_dir(synthetic_h5_path: Path, tmp_path: Path) -> Path:
    out = tmp_path / "models"
    split = split_hdf5(synthetic_h5_path, n_train=1, n_test=1)
    train_all(["cluster_mean"], synthetic_h5_path, split, out, log=lambda _m: None)
    return out


@pytest.fixture
def controller(synthetic_h5_path: Path, models_dir: Path) -> ExplorerController:
    return ExplorerController(
        HDF5Provider(synthetic_h5_path), max_stats_molecules=2, models_dir=models_dir
    )


def _betas(controller: ExplorerController) -> np.ndarray:
    assert controller.state.mol_a is not None
    return np.array([s.beta for s in controller.state.mol_a.molecule.sites])


def test_dataset_is_the_default_and_models_are_listed(controller: ExplorerController) -> None:
    assert controller.state.beta_source == "dataset"
    assert controller.available_beta_sources() == ("dataset", "cluster_mean")


def test_no_models_directory_means_dataset_only(synthetic_h5_path: Path, tmp_path: Path) -> None:
    controller = ExplorerController(
        HDF5Provider(synthetic_h5_path), max_stats_molecules=2, models_dir=tmp_path / "none"
    )
    assert controller.available_beta_sources() == ("dataset",)


def test_switching_reloads_what_is_loaded(controller: ExplorerController, models_dir) -> None:
    from rpfr_visualizer.empirical import DatasetSplit

    # A test molecule: a training molecule's cluster mean can be its own value.
    key = DatasetSplit.load(models_dir / "split.json").test[0]
    controller.load(key)
    controller.load(key, slot="b")
    controller.load_pool([key])
    dataset_beta = _betas(controller)

    controller.use_beta_source("cluster_mean")
    for loaded in (controller.state.mol_a, controller.state.mol_b, *controller.match_pool):
        assert loaded is not None
        assert loaded.molecule.source == "model:cluster_mean"
        assert loaded.arrays.clumping is None
    assert not np.allclose(_betas(controller), dataset_beta)

    controller.use_beta_source("dataset")
    np.testing.assert_allclose(_betas(controller), dataset_beta)
    assert controller.state.mol_a.arrays.clumping is not None


def test_model_source_is_announced_and_derived_panels_say_why(
    controller: ExplorerController,
) -> None:
    controller.load(controller.provider.list_keys()[0])
    assert not any("predicted" in w for w in controller.warnings())
    controller.use_beta_source("cluster_mean")
    assert any("predicted by the Cluster mean model" in w for w in controller.warnings())
    assert "Models predict β only" in controller._missing_array_note("clumping")


def test_unknown_source_is_refused(controller: ExplorerController) -> None:
    with pytest.raises(KeyError, match="Available"):
        controller.use_beta_source("mpnn")
    assert controller.state.beta_source == "dataset"


def test_split_membership_is_reported(controller: ExplorerController, models_dir: Path) -> None:
    from rpfr_visualizer.empirical import DatasetSplit

    split = DatasetSplit.load(models_dir / "split.json")
    assert controller.split_of(split.test[0]) == "test"
    controller.use_beta_source("cluster_mean")
    assert controller.split_of(split.train[0]) == "train"
    assert controller.split_of(next(iter(split.excluded))) is None


def test_gui_route_switches_source(synthetic_h5_path: Path, tmp_path: Path, models_dir) -> None:
    from rpfr_visualizer.gui.app import dispatch
    from rpfr_visualizer.gui.config import Resolution
    from rpfr_visualizer.gui.session import ExplorerSession

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy(synthetic_h5_path, raw / "qm9s_tr_project_1pct.h5")
    shutil.copytree(models_dir, tmp_path / "data" / "models")
    session = ExplorerSession(Resolution(data_dir=tmp_path / "data", max_stats_molecules=2))
    session.open()

    def call(method: str, path: str, body: dict | None = None):
        raw = json.dumps(body).encode() if body is not None else None
        response = dispatch(session, method, path, "", raw)
        return response.status, json.loads(response.body)

    _status, state = call("GET", "/api/state")
    assert [o["value"] for o in state["beta_sources"]] == ["dataset", "cluster_mean"]
    status, state = call("POST", "/api/source", {"name": "cluster_mean"})
    assert (status, state["beta_source"]) == (200, "cluster_mean")
    status, _payload = call("POST", "/api/state", {"beta_source": "dataset"})
    assert status >= 400
