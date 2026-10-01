"""Pluggable molecule data providers for the visualizer.

Built on the QM9s-RPFR HDF5 schema (2.0 to 2.2). A catalog can hold several
files, such as a subsample and the full dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from rpfr_visualizer.io.loader import Molecule, QM9sDataset, _load_molecule_group


@dataclass(frozen=True)
class MoleculeArrays:
    """Optional per-molecule array data not stored on :class:`Molecule`."""

    clumping: np.ndarray | None = None
    mass_law: np.ndarray | None = None
    imag_freq_ssi: np.ndarray | None = None


class MoleculeProvider(ABC):
    """Abstract source of :class:`Molecule` records for the visualizer."""

    @property
    @abstractmethod
    def dataset(self) -> str:
        """Human-readable dataset name (e.g. ``qm9s_v2.2``)."""

    @property
    @abstractmethod
    def source(self) -> str:
        """Provenance tag applied to loaded records."""

    @abstractmethod
    def list_ids(self) -> list[int]:
        """Return sorted molecule identifiers available from this provider."""

    @abstractmethod
    def has(self, mol_id: int) -> bool:
        """Return True if *mol_id* exists in this provider."""

    @abstractmethod
    def get_molecule(self, mol_id: int) -> Molecule:
        """Load a single molecule record."""

    def load_arrays(self, mol_id: int) -> MoleculeArrays:
        """Load optional array datasets (clumping, mass_law) for a molecule."""
        return MoleculeArrays()


class HDF5Provider(MoleculeProvider):
    """Provider for one QM9s-RPFR HDF5 file."""

    def __init__(self, path: Path | str, *, source: str = "dft") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")
        self._source = source
        self._dataset = QM9sDataset(self.path, source=source)
        self._dataset_name = self._read_dataset_name()

    def _read_dataset_name(self) -> str:
        with h5py.File(self.path, "r") as h5:
            raw = h5.attrs.get("dataset", self.path.stem)
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return str(raw)

    @property
    def dataset(self) -> str:
        return self._dataset_name

    @property
    def source(self) -> str:
        return self._source

    @property
    def temperatures(self) -> tuple[float, ...]:
        return self._dataset.temperatures

    @property
    def subsample_fraction(self) -> float | None:
        """The fraction of the source this file holds, or None if it is complete.

        Written by ``scripts/make_subsample.py`` as a root attribute. Read once
        and cached: the header asks for it on every state change, and the
        answer cannot change while the file is open.
        """
        if not hasattr(self, "_subsample_fraction"):
            with h5py.File(self.path, "r") as h5:
                raw = h5.attrs.get("subsample_fraction")
            self._subsample_fraction: float | None = (
                float(raw) if raw is not None and 0 < float(raw) < 1 else None
            )
        return self._subsample_fraction

    def list_keys(self) -> list[str]:
        """Return all molecule group keys in the file (any v2.0 schema)."""
        with h5py.File(self.path, "r") as h5:
            return sorted(
                key
                for key in h5.keys()
                if key != "temperatures" and isinstance(h5[key], h5py.Group)
            )

    def list_ids(self) -> list[int]:
        """Return numeric molecule IDs when keys are numeric (qm9s-style)."""
        return sorted(int(key) for key in self.list_keys() if str(key).isdigit())

    def has_key(self, key: str) -> bool:
        with h5py.File(self.path, "r") as h5:
            return key in h5 and isinstance(h5[key], h5py.Group)

    def has(self, mol_id: int) -> bool:
        return self.has_key(str(mol_id))

    @staticmethod
    def _synthetic_mol_id(key: str) -> int:
        if key.isdigit():
            return int(key)
        return abs(hash(key)) % 900_000_000 + 100_000_000

    def get_molecule_by_key(self, key: str) -> Molecule:
        """Load a molecule by HDF5 group key."""
        with h5py.File(self.path, "r") as h5:
            if key not in h5:
                raise KeyError(f"Molecule {key!r} not found in {self.path}")
            mol_id = self._synthetic_mol_id(key)
            return _load_molecule_group(
                h5[key], mol_id, self.temperatures, self._source, hdf5_key=key
            )

    def get_molecule(self, mol_id: int) -> Molecule:
        return self.get_molecule_by_key(str(mol_id))

    def load_arrays_by_key(self, key: str) -> MoleculeArrays:
        with h5py.File(self.path, "r") as h5:
            if key not in h5:
                raise KeyError(f"Molecule {key!r} not found in {self.path}")
            group = h5[key]
            clumping = group["clumping"][()] if "clumping" in group else None
            mass_law = group["mass_law"][()] if "mass_law" in group else None
            imag_freq_ssi = group["imag_freq_ssi"][()] if "imag_freq_ssi" in group else None
        return MoleculeArrays(clumping=clumping, mass_law=mass_law, imag_freq_ssi=imag_freq_ssi)

    def load_arrays(self, mol_id: int) -> MoleculeArrays:
        return self.load_arrays_by_key(str(mol_id))


class DatasetCatalog:
    """Register multiple v2.0 HDF5 files and resolve molecules by (dataset, id)."""

    def __init__(self) -> None:
        self._providers: dict[str, HDF5Provider] = {}

    def register(self, name: str, path: Path | str, *, source: str = "dft") -> HDF5Provider:
        """Register an HDF5 file under *name* (overwrites existing name)."""
        provider = HDF5Provider(path, source=source)
        self._providers[name] = provider
        return provider

    def register_provider(self, name: str, provider: HDF5Provider) -> None:
        """Register an existing :class:`HDF5Provider` under *name*."""
        self._providers[name] = provider

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def get_provider(self, name: str) -> HDF5Provider:
        if name not in self._providers:
            available = ", ".join(self.datasets) or "(none)"
            raise KeyError(f"Dataset {name!r} not registered. Available: {available}")
        return self._providers[name]

    def list_ids(self, dataset: str) -> list[int]:
        return self.get_provider(dataset).list_ids()

    def has(self, dataset: str, mol_id: int) -> bool:
        return self.get_provider(dataset).has(mol_id)

    def get_molecule(self, dataset: str, mol_id: int) -> Molecule:
        return self.get_provider(dataset).get_molecule(mol_id)

    def list_keys(self, dataset: str) -> list[str]:
        return self.get_provider(dataset).list_keys()

    def has_key(self, dataset: str, key: str) -> bool:
        return self.get_provider(dataset).has_key(key)

    def get_molecule_by_key(self, dataset: str, key: str) -> Molecule:
        return self.get_provider(dataset).get_molecule_by_key(key)

    def load_arrays(self, dataset: str, mol_id: int) -> MoleculeArrays:
        return self.get_provider(dataset).load_arrays(mol_id)

    def load_arrays_by_key(self, dataset: str, key: str) -> MoleculeArrays:
        return self.get_provider(dataset).load_arrays_by_key(key)


__all__ = [
    "DatasetCatalog",
    "HDF5Provider",
    "MoleculeArrays",
    "MoleculeProvider",
]
