"""HDF5 loader for QM9s-RPFR datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rpfr_visualizer.io.schema import (
    DEFAULT_SOURCE,
    TEMPERATURES_K,
    MoleculeRecord,
    SiteRecord,
)
from rpfr_visualizer.io.validate import derive_qc_pass

# Type aliases for public API (spec uses these names).
Site = SiteRecord
Molecule = MoleculeRecord


def _decode_scalar(value: Any) -> Any:
    """Decode HDF5 scalar values to Python types."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return _decode_scalar(value.item())
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    return value


def _decode_str_array(raw: np.ndarray) -> tuple[str, ...]:
    """Decode an object or bytes string array to a tuple of strings."""
    decoded: list[str] = []
    for item in raw.flat:
        val = _decode_scalar(item)
        decoded.append("" if val is None else str(val))
    return tuple(decoded)


def _decode_bond_order(raw: Any) -> list[str]:
    """Parse bond_order JSON string into a list of bond labels."""
    text = _decode_scalar(raw)
    if not text:
        return []
    if isinstance(text, list):
        return [str(v) for v in text]
    try:
        parsed = json.loads(str(text))
    except json.JSONDecodeError:
        return [str(text)]
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return [str(parsed)]


_ROOT_NON_MOLECULE_KEYS = frozenset({"temperatures"})


def _is_molecule_group(h5_file: h5py.File, key: str) -> bool:
    """Return True if *key* is a per-molecule HDF5 group (not a root dataset)."""
    if key in _ROOT_NON_MOLECULE_KEYS:
        return False
    item = h5_file[key]
    if not isinstance(item, h5py.Group):
        return False
    return "mol_id" in item.attrs or "atom_symbol" in item or "rpfr_single" in item


def _molecule_keys(h5_file: h5py.File) -> list[str]:
    """List molecule group names in stable sorted order."""
    return sorted(
        (key for key in h5_file.keys() if _is_molecule_group(h5_file, key)),
        key=str,
    )


def _is_molecule_id(key: str) -> bool:
    """Return True if *key* is a numeric QM9s-style molecule group name."""
    return key.isdigit()


def _temperatures_from_file(h5_file: h5py.File) -> tuple[float, ...]:
    """Read the temperature grid from the file, however that file stores it.

    Two layouts, because the schema moved: 2.2 writes a root *dataset*
    ``/temperatures``; 2.0 and 2.1 wrote a JSON root *attribute*
    ``temperatures_K``. The dataset is preferred when both are present.

    Falling straight through to the package constant is what made a v2.2 file
    read as if it were a v2.0 one. Its fifteen ascending temperatures came back
    as the old thirteen in the old order, so every column from the eighth on
    was labelled with the wrong temperature -- 273.15 K reported as 300 K --
    and 460 K and 500 K did not exist at all. Nothing raised; the numbers were
    simply attached to the wrong temperature.
    """
    grid = h5_file.get("temperatures")
    if isinstance(grid, h5py.Dataset):
        values = np.asarray(grid[()], dtype=float).ravel()
        if values.size:
            return tuple(float(v) for v in values)

    raw = h5_file.attrs.get("temperatures_K")
    if raw is None:
        return TEMPERATURES_K
    if isinstance(raw, (bytes, str)):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        values = json.loads(text)
        return tuple(float(v) for v in values)
    if isinstance(raw, np.ndarray):
        return tuple(float(v) for v in raw.tolist())
    return TEMPERATURES_K


class QM9sDataset:
    """Access QM9s-RPFR data from a local HDF5 file.

    Parameters
    ----------
    path:
        Path to a QM9s-RPFR HDF5 file (v2.0 schema).
    source:
        Provenance tag applied to all loaded records. Default is ``dft``.
    """

    def __init__(self, path: Path | str, *, source: str = DEFAULT_SOURCE) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")
        self.source = source
        self._temperatures: tuple[float, ...] | None = None
        self._molecule_ids: list[int] | None = None

    @property
    def temperatures(self) -> tuple[float, ...]:
        """Temperature grid in Kelvin."""
        if self._temperatures is None:
            with h5py.File(self.path, "r") as h5:
                self._temperatures = _temperatures_from_file(h5)
        return self._temperatures

    @property
    def molecule_ids(self) -> list[int]:
        """Sorted list of molecule IDs in the file."""
        if self._molecule_ids is None:
            with h5py.File(self.path, "r") as h5:
                self._molecule_ids = sorted(int(k) for k in h5.keys() if _is_molecule_id(k))
        return self._molecule_ids

    def __len__(self) -> int:
        return len(self.molecule_ids)

    def __iter__(self) -> Iterator[Molecule]:
        for mol_id in self.molecule_ids:
            yield self.get_molecule(mol_id)

    def __contains__(self, mol_id: int) -> bool:
        return mol_id in self.molecule_ids

    def get_molecule(self, mol_id: int) -> Molecule:
        """Load a single molecule and its per-site SSI records."""
        key = str(mol_id)
        with h5py.File(self.path, "r") as h5:
            if key not in h5:
                raise KeyError(f"Molecule {mol_id} not found in {self.path}")
            return _load_molecule_group(h5[key], mol_id, self.temperatures, self.source)

    def validate(self):
        """Run legacy QC validation and return a report."""
        from rpfr_visualizer.io.validate import validate_dataset

        return validate_dataset(self)

    def qc(self):
        """Run structured QC checks and return a dataset-level report."""
        from rpfr_visualizer.qc.engine import run_dataset_qc

        return run_dataset_qc(self.path)

    def stats(self, **kwargs):
        """Compute descriptive dataset statistics."""
        from rpfr_visualizer.analysis.stats import dataset_stats

        return dataset_stats(self, **kwargs)


def _load_molecule_group(
    group: h5py.Group,
    mol_id: int,
    temperatures: tuple[float, ...],
    source: str,
    *,
    hdf5_key: str | None = None,
) -> Molecule:
    """Build a Molecule from an HDF5 molecule group."""
    attrs = group.attrs
    smiles = str(_decode_scalar(attrs.get("smiles", "")))
    n_atoms = int(_decode_scalar(attrs.get("n_atoms", len(group["atom_symbol"]))))

    freq_qc_ok = bool(_decode_scalar(attrs.get("freq_qc_ok", True)))
    rpfr_qc_ok = bool(_decode_scalar(attrs.get("rpfr_qc_ok", True)))
    tr_converged = bool(_decode_scalar(attrs.get("tr_converged", True)))
    rdkit_ok = bool(_decode_scalar(attrs.get("rdkit_ok", True)))
    smiles_order_match = bool(_decode_scalar(attrs.get("smiles_order_match", True)))
    # Absent in schema 2.0/2.1, where it was never computed; True is the right
    # reading there, since those files' SMILES were not known to disagree.
    geometry_match = bool(_decode_scalar(attrs.get("geometry_match", True)))
    n_imag_vib = int(_decode_scalar(attrs.get("n_imag_vib", 0)))
    is_linear = bool(_decode_scalar(attrs.get("is_linear", False)))
    rdkit_status = str(_decode_scalar(attrs.get("rdkit_status", "ok")))
    max_tr_rot_abs_raw = attrs.get("max_tr_rot_abs")
    max_tr_rot_abs = (
        float(_decode_scalar(max_tr_rot_abs_raw)) if max_tr_rot_abs_raw is not None else None
    )

    qc_pass = derive_qc_pass(
        freq_qc_ok=freq_qc_ok,
        rpfr_qc_ok=rpfr_qc_ok,
        tr_converged=tr_converged,
        rdkit_ok=rdkit_ok,
        smiles_order_match=smiles_order_match,
        n_imag_vib=n_imag_vib,
    )

    atom_symbols = _decode_str_array(group["atom_symbol"][()])
    n_atoms = len(atom_symbols)

    atom_degrees = (
        group["atom_degree"][()] if "atom_degree" in group else np.zeros(n_atoms, dtype=np.int64)
    )
    atom_valences = (
        group["atom_valence"][()] if "atom_valence" in group else np.zeros(n_atoms, dtype=np.int64)
    )
    atom_aromaticities = (
        group["atom_aromaticity"][()]
        if "atom_aromaticity" in group
        else np.zeros(n_atoms, dtype=bool)
    )
    atom_hybridizations = (
        _decode_str_array(group["atom_hybridization"][()])
        if "atom_hybridization" in group
        else ("UNSPECIFIED",) * n_atoms
    )
    bond_orders = (
        group["bond_order"][()] if "bond_order" in group else np.array([b""] * n_atoms, dtype="S")
    )
    coordinates = group["coordinates"][()] if "coordinates" in group else np.zeros((n_atoms, 3))
    rpfr_single = group["rpfr_single"][()]
    local_smiles_raw = (
        group["local_smiles"][()]
        if "local_smiles" in group
        else np.array([[b""] * 5] * n_atoms, dtype="S")
    )

    sites: list[Site] = []
    for idx in range(n_atoms):
        element = atom_symbols[idx]
        fp_row = local_smiles_raw[idx]
        local_smiles = _decode_str_array(
            fp_row if isinstance(fp_row, np.ndarray) else np.array([fp_row])
        )
        beta = tuple(float(v) for v in rpfr_single[idx])
        coord = tuple(float(v) for v in coordinates[idx])

        sites.append(
            Site(
                molecule_id=mol_id,
                atom_index=idx,
                element=element,
                beta=beta,
                temperatures=temperatures,
                local_smiles=local_smiles,
                source=source,
                atom_degree=int(atom_degrees[idx]),
                atom_valence=int(atom_valences[idx]),
                atom_aromaticity=bool(atom_aromaticities[idx]),
                atom_hybridization=atom_hybridizations[idx],
                bond_order=_decode_bond_order(bond_orders[idx]),
                coordinates=coord,
            )
        )

    return Molecule(
        molecule_id=mol_id,
        smiles=smiles,
        n_atoms=n_atoms,
        sites=sites,
        qc_pass=qc_pass,
        geometry_match=geometry_match,
        is_linear=is_linear,
        n_imag_vib=n_imag_vib,
        freq_qc_ok=freq_qc_ok,
        rpfr_qc_ok=rpfr_qc_ok,
        tr_converged=tr_converged,
        rdkit_ok=rdkit_ok,
        rdkit_status=rdkit_status,
        smiles_order_match=smiles_order_match,
        max_tr_rot_abs=max_tr_rot_abs,
        source=source,
    )


# Re-export schema constants for convenience.
__all__ = [
    "Molecule",
    "QM9sDataset",
    "Site",
    "TEMPERATURES_K",
]
