"""A tiny synthetic dataset, so the explorer runs before anything is downloaded.

**The numbers here are invented.** The file is a valid v2.0 HDF5 carrying the
real temperature grid, and its five molecules are chosen to cover the states the
interface has to render — a clean molecule, a QC failure, a translation/rotation
warning, a site-label mismatch — but no β value in it was computed from
anything. The demo shows that the software works; it says nothing about the
chemistry. Real values come from the QM9s-RPFR data files, which are a separate
download.

The dataset registers under the name ``demo`` so that label is visible in the
dataset picker for as long as the demo is open, rather than a user having to
remember which data they are looking at.

This is also the builder behind the pytest fixtures and ``scripts/dev_gui.py
--fixture``, so one definition of a valid v2.0 layout serves all three.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from rpfr_visualizer.io.schema import TEMPERATURES_K

#: Catalog name the demo registers under. Derived from the filename by
#: :func:`~rpfr_visualizer.io.molecule_index.dataset_name_for`, so the two must agree.
DEMO_DATASET = "demo"

#: Index filename the GUI looks for under ``<data dir>/cache``.
DEMO_INDEX_NAME = "qm9s_rpfr_index.sqlite"


@dataclass(frozen=True)
class DemoData:
    """Where :func:`ensure_demo_data` put things.

    Attributes:
        root: Directory to point ``--data`` at.
        h5_path: The synthetic HDF5 file.
        index_path: The search index built over it.
    """

    root: Path
    h5_path: Path
    index_path: Path


def _build_clumping(n_atoms: int, n_temps: int) -> np.ndarray:
    """Build symmetric clumping factors C near unity (Delta = ln C)."""
    clumping = np.ones((n_atoms, n_atoms, n_temps), dtype=np.float64)
    for temp_idx in range(n_temps):
        np.fill_diagonal(clumping[:, :, temp_idx], 0.0)
    return clumping


def _write_molecule_group(
    h5_file: h5py.File,
    mol_id: int,
    *,
    smiles: str,
    atom_symbols: list[str],
    rpfr_values: np.ndarray,
    qc_pass: bool = True,
    max_tr_rot_abs: float = 0.1,
    smiles_order_match: bool | None = None,
    harmonic_frequencies: np.ndarray | None = None,
    all_frequencies: np.ndarray | None = None,
) -> None:
    """Write a minimal v2.0 molecule group."""
    n_atoms = len(atom_symbols)
    group = h5_file.create_group(str(mol_id))
    group.attrs["mol_id"] = str(mol_id)
    group.attrs["smiles"] = smiles
    group.attrs["n_atoms"] = n_atoms
    group.attrs["is_linear"] = False
    group.attrs["n_imag_vib"] = 0 if qc_pass else 1
    group.attrs["freq_qc_ok"] = qc_pass
    group.attrs["rpfr_qc_ok"] = qc_pass
    group.attrs["tr_converged"] = qc_pass
    group.attrs["rdkit_ok"] = qc_pass
    group.attrs["rdkit_status"] = "ok"
    order_match = qc_pass if smiles_order_match is None else smiles_order_match
    group.attrs["smiles_order_match"] = order_match
    group.attrs["max_tr_rot_abs"] = max_tr_rot_abs

    n_temps = len(TEMPERATURES_K)
    if rpfr_values.ndim == 1:
        rpfr_values = np.tile(rpfr_values, (n_atoms, 1))

    group.create_dataset("atom_symbol", data=np.array(atom_symbols, dtype="S"))
    group.create_dataset("atom_degree", data=np.ones(n_atoms, dtype=np.int64))
    group.create_dataset("atom_valence", data=np.ones(n_atoms, dtype=np.int64))
    group.create_dataset("atom_aromaticity", data=np.zeros(n_atoms, dtype=bool))
    group.create_dataset(
        "atom_hybridization",
        data=np.array(["SP3"] * n_atoms, dtype="S"),
    )
    group.create_dataset(
        "bond_order",
        data=np.array(['["C-H"]'] * n_atoms, dtype="S"),
    )
    group.create_dataset("coordinates", data=np.zeros((n_atoms, 3)))
    group.create_dataset("rpfr_single", data=rpfr_values.astype(np.float64))
    group.create_dataset("mass_law", data=np.zeros((n_atoms, n_temps)))
    group.create_dataset("clumping", data=_build_clumping(n_atoms, n_temps))
    group.create_dataset(
        "local_smiles",
        data=np.array([[b"C", b"CC", b"", b"", b""]] * n_atoms, dtype="S"),
    )
    group.create_dataset("topo_dist", data=np.eye(n_atoms, dtype=np.int64))
    default_all = np.full(3 * n_atoms, 100.0)
    group.create_dataset(
        "all_frequencies",
        data=all_frequencies if all_frequencies is not None else default_all,
    )
    default_harmonic = np.full(max(1, 3 * n_atoms - 6), 100.0)
    group.create_dataset(
        "harmonic_frequencies",
        data=harmonic_frequencies if harmonic_frequencies is not None else default_harmonic,
    )


def build_demo_dataset(path: Path, *, dataset: str = DEMO_DATASET) -> Path:
    """Write the five-molecule synthetic v2.0 HDF5 file at *path*.

    Args:
        path: Where to write. Overwritten if it exists.
        dataset: Value for the file's ``dataset`` attribute.

    Returns:
        *path*, for chaining.
    """
    n_temps = len(TEMPERATURES_K)

    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.0"
        h5.attrs["dataset"] = dataset
        h5.attrs["temperatures_K"] = json.dumps(list(TEMPERATURES_K))
        h5.attrs["coordinate_units"] = "Angstrom"
        h5.attrs["tr_removal"] = "project"

        # Methane-like: 1 C + 3 H
        beta_methane = np.array([1.0] * n_temps)
        beta_methane[7] = 1.12  # 300 K index
        beta_h = np.linspace(5.0, 1.5, n_temps)
        rpfr_1 = np.vstack([beta_methane, beta_h, beta_h, beta_h])

        _write_molecule_group(
            h5,
            1,
            smiles="C",
            atom_symbols=["C", "H", "H", "H"],
            rpfr_values=rpfr_1,
            qc_pass=True,
        )

        # Ethane-like: 2 C + 2 H
        beta_c = np.full(n_temps, 1.15)
        beta_c[7] = 1.15
        rpfr_2 = np.vstack([beta_c, beta_c, beta_h[:n_temps], beta_h[:n_temps]])

        _write_molecule_group(
            h5,
            2,
            smiles="CC",
            atom_symbols=["C", "C", "H", "H"],
            rpfr_values=rpfr_2,
            qc_pass=True,
        )

        # Failed QC molecule
        _write_molecule_group(
            h5,
            3,
            smiles="N",
            atom_symbols=["N"],
            rpfr_values=np.full((1, n_temps), 1.05),
            qc_pass=False,
        )

        # TR warn band molecule
        beta_o = np.full(n_temps, 1.08)
        _write_molecule_group(
            h5,
            4,
            smiles="O",
            atom_symbols=["O", "H", "H"],
            rpfr_values=np.vstack([beta_o, beta_h[:n_temps], beta_h[:n_temps]]),
            qc_pass=True,
            max_tr_rot_abs=23.0,
        )

        # Site label mismatch with otherwise good physics
        _write_molecule_group(
            h5,
            5,
            smiles="CO",
            atom_symbols=["C", "O"],
            rpfr_values=np.vstack([beta_methane, beta_o]),
            qc_pass=True,
            smiles_order_match=False,
        )

    return path


def ensure_demo_data(root: Path, *, dataset: str = DEMO_DATASET) -> DemoData:
    """Build the demo dataset and its search index under *root*, once.

    Reuses whatever is already there, so a second call is a pair of ``stat``
    calls: the point of this path is that starting the app is cheap.

    Args:
        root: Directory to build under. Created if missing.
        dataset: Catalog name, which is also the HDF5 filename stem.

    Returns:
        A :class:`DemoData` naming the directory, the file and the index.

    Raises:
        ImportError: If RDKit is missing. Building the index needs it; install
            the ``gui`` extra.
    """
    raw = root / "raw"
    cache = root / "cache"
    h5_path = raw / f"{dataset}.h5"
    index_path = cache / DEMO_INDEX_NAME
    built = DemoData(root=root, h5_path=h5_path, index_path=index_path)
    if h5_path.exists() and index_path.exists():
        return built

    raw.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Imported here, not at module scope: building the index needs RDKit, and
    # importing this module to call build_demo_dataset() alone must not.
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    rebuilt_data = not h5_path.exists()
    if rebuilt_data:
        build_demo_dataset(h5_path, dataset=dataset)

    # Rebuild the index whenever the data was rebuilt, not merely when the index
    # is absent. An index outlives the file it describes easily -- and a search
    # index that answers queries about molecules nothing can open is the exact
    # failure mode factory.index_data_mismatch() exists to catch.
    if rebuilt_data or not index_path.exists():
        index_path.unlink(missing_ok=True)
        catalog = DatasetCatalog()
        catalog.register(dataset, h5_path)
        build_index(catalog, index_path, progress=False)

    return built


__all__ = [
    "DEMO_DATASET",
    "DEMO_INDEX_NAME",
    "DemoData",
    "build_demo_dataset",
    "ensure_demo_data",
]
