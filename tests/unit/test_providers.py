"""Unit tests for molecule providers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpfr_visualizer.io.providers import DatasetCatalog, HDF5Provider
from rpfr_visualizer.io.schema import TEMPERATURES_K


def test_hdf5_provider_loads_molecule(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    assert provider.dataset == "qm9s"
    assert provider.source == "dft"
    assert 1 in provider.list_ids()
    assert "1" in provider.list_keys()
    mol = provider.get_molecule(1)
    assert mol.n_atoms == 4
    assert mol.sites[0].element == "C"


def test_hdf5_provider_get_by_string_key(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule_by_key("1")
    assert mol.n_atoms == 4
    arrays = provider.load_arrays_by_key("1")
    assert arrays.clumping is not None


def test_hdf5_provider_load_arrays(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    arrays = provider.load_arrays(1)
    assert arrays.clumping is not None
    assert arrays.mass_law is not None
    assert arrays.clumping.shape[0] == 4
    assert arrays.clumping.shape[2] == len(TEMPERATURES_K)


def test_dataset_catalog_multi_dataset(synthetic_h5_path: Path, tmp_path: Path) -> None:
    second_path = tmp_path / "qm9s_skip.h5"
    import shutil

    shutil.copy(synthetic_h5_path, second_path)

    import h5py

    with h5py.File(second_path, "r+") as h5:
        h5.attrs["dataset"] = "qm9s_skip"

    catalog = DatasetCatalog()
    catalog.register("qm9s", synthetic_h5_path)
    catalog.register("qm9s_skip", second_path)

    assert catalog.datasets == ("qm9s", "qm9s_skip")
    assert catalog.has("qm9s", 1)
    mol = catalog.get_molecule("qm9s_skip", 2)
    assert mol.molecule_id == 2


def test_catalog_unknown_dataset_raises(synthetic_h5_path: Path) -> None:
    catalog = DatasetCatalog()
    catalog.register("qm9s", synthetic_h5_path)
    with pytest.raises(KeyError, match="not registered"):
        catalog.get_provider("missing")
