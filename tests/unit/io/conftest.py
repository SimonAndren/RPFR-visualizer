"""Shared fixtures for search-index tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from rpfr_visualizer.io.schema import TEMPERATURES_K

# A handful of molecules spanning the functional groups the tests query for.
SAMPLE_MOLECULES: tuple[tuple[str, str], ...] = (
    ("1", "CCO"),  # ethanol — alcohol
    ("2", "CC(=O)O"),  # acetic acid — carboxylic acid
    ("3", "c1ccccc1"),  # benzene — arene
    ("4", "CC#N"),  # acetonitrile — nitrile
    ("5", "CCN"),  # ethylamine — amine
    ("6", "CC=O"),  # acetaldehyde — aldehyde
    ("7", "COC"),  # dimethyl ether
    ("8", "C1OC1"),  # oxirane — epoxide
    ("9", "CC(=O)N"),  # acetamide — amide
    ("10", "CCOCC"),  # diethyl ether
)


def _write_dataset(path: Path, molecules: tuple[tuple[str, str], ...], name: str) -> Path:
    """Write a minimal v2.0 file carrying just what the index reads."""
    n_temps = len(TEMPERATURES_K)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.0"
        h5.attrs["dataset"] = name
        h5.attrs["temperatures_K"] = json.dumps(list(TEMPERATURES_K))
        for key, smiles in molecules:
            group = h5.create_group(key)
            group.attrs["mol_id"] = key
            group.attrs["smiles"] = smiles
            group.attrs["n_atoms"] = len(smiles)
            group.attrs["freq_qc_ok"] = True
            group.attrs["rpfr_qc_ok"] = True
            group.attrs["tr_converged"] = True
            group.attrs["rdkit_ok"] = True
            group.attrs["smiles_order_match"] = True
            group.attrs["n_imag_vib"] = 0
            group.create_dataset("rpfr_single", data=np.ones((len(smiles), n_temps)))
            group.create_dataset("atom_symbol", data=np.array(["C"] * len(smiles), dtype="S2"))
    return path


@pytest.fixture
def search_catalog(tmp_path: Path):
    """A two-dataset catalog so dataset filtering is exercised."""
    from rpfr_visualizer.io.providers import DatasetCatalog

    catalog = DatasetCatalog()
    catalog.register("alpha", _write_dataset(tmp_path / "alpha.h5", SAMPLE_MOLECULES, "alpha"))
    catalog.register("beta", _write_dataset(tmp_path / "beta.h5", SAMPLE_MOLECULES[:4], "beta"))
    return catalog


@pytest.fixture
def search_index_path(tmp_path: Path, search_catalog) -> Path:
    """A built SQLite index over :func:`search_catalog`."""
    from rpfr_visualizer.io.search import build_index

    out = tmp_path / "index.sqlite"
    build_index(search_catalog, out, progress=False)
    return out


def _write_structureless_dataset(path: Path, keys: tuple[str, ...], name: str) -> Path:
    """A dataset whose groups carry no ``smiles`` attribute at all.

    Search must report such molecules as unreachable by structure rather
    than as absent.
    """
    n_temps = len(TEMPERATURES_K)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.0"
        h5.attrs["dataset"] = name
        h5.attrs["temperatures_K"] = json.dumps(list(TEMPERATURES_K))
        for key in keys:
            group = h5.create_group(key)
            group.attrs["mol_id"] = key
            group.attrs["rdkit_status"] = "na"  # no SMILES to parse
            group.attrs["freq_qc_ok"] = True
            group.attrs["rpfr_qc_ok"] = True
            group.attrs["tr_converged"] = True
            group.attrs["n_imag_vib"] = 0
            group.create_dataset("rpfr_single", data=np.ones((3, n_temps)))
            group.create_dataset("atom_symbol", data=np.array(["C"] * 3, dtype="S2"))
    return path


@pytest.fixture
def mixed_coverage_index_path(tmp_path: Path) -> Path:
    """An index over one dataset with structures and one with none."""
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    catalog = DatasetCatalog()
    catalog.register(
        "withsmiles", _write_dataset(tmp_path / "withsmiles.h5", SAMPLE_MOLECULES, "withsmiles")
    )
    catalog.register(
        "nosmiles",
        _write_structureless_dataset(tmp_path / "nosmiles.h5", ("a", "b", "c"), "nosmiles"),
    )
    out = tmp_path / "mixed.sqlite"
    build_index(catalog, out, progress=False)
    return out
