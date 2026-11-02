"""What every empirical model is, and how one is stored.

A model predicts ``1000 ln β`` for each site of a molecule at each temperature
of the grid it was trained on. The explorer only ever calls
:meth:`EmpiricalModel.predict_molecule`; how the model reads the molecule --
bond paths, local environments, a graph, orbital features -- is its own
business, which is what lets a message-passing network sit behind the same
switch as a lookup table.

Models are saved as one ``.npz`` file: numeric arrays plus a JSON header
(``__meta__``) holding the kind, the grid, the split fingerprint, the chosen
hyperparameters and the validation and test scores. Strings are packed into
byte arrays, so loading never needs ``allow_pickle`` -- a model file is data,
and it should not be able to run code.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from rpfr_visualizer.empirical.sites import SiteTable
from rpfr_visualizer.io.schema import MoleculeRecord

MODEL_FORMAT = "rpfr-visualizer/empirical-model"
MODEL_SUFFIX = ".npz"

#: Every registered kind, by name. Filled by :func:`register`.
MODEL_KINDS: dict[str, type[EmpiricalModel]] = {}


def register(cls: type[EmpiricalModel]) -> type[EmpiricalModel]:
    """Class decorator: make a model kind loadable by name."""
    MODEL_KINDS[cls.kind] = cls
    return cls


def pack_strings(items: Sequence[str]) -> np.ndarray:
    """Encode strings as one ``uint8`` array, newline-separated."""
    if any("\n" in s for s in items):
        raise ValueError("Strings containing newlines cannot be packed")
    packed: np.ndarray = np.frombuffer("\n".join(items).encode("utf-8"), dtype=np.uint8)
    return np.array(packed, copy=True)


def unpack_strings(raw: np.ndarray, count: int) -> list[str]:
    """Invert :func:`pack_strings`; *count* disambiguates an empty list."""
    if count == 0:
        return []
    return bytes(np.asarray(raw, dtype=np.uint8)).decode("utf-8").split("\n")


class EmpiricalModel(ABC):
    """A fitted predictor of per-site ``1000 ln β``.

    Attributes:
        kind: Registry name, also the default file name.
        label: Human name for menus.
        meta: The JSON header: grid, split, hyperparameters, scores.
    """

    kind: ClassVar[str] = ""
    label: ClassVar[str] = ""

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or self.kind
        self.meta: dict[str, Any] = {}

    @property
    def temperatures(self) -> tuple[float, ...]:
        return tuple(float(t) for t in self.meta.get("temperatures", ()))

    @abstractmethod
    def predict_molecule(self, molecule: MoleculeRecord) -> np.ndarray:
        """``1000 ln β`` for every site, ``(n_atoms, n_temperatures)``; NaN if unknown."""

    # ----------------------------------------------------------- persistence

    @abstractmethod
    def _arrays(self) -> dict[str, np.ndarray]:
        """Numeric state to save, keyed by array name."""

    @abstractmethod
    def _restore(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Rebuild numeric state from what :meth:`_arrays` saved."""

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        if target.suffix != MODEL_SUFFIX:
            target = target.with_suffix(MODEL_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = {"format": MODEL_FORMAT, "kind": self.kind, "name": self.name, **self.meta}
        arrays = self._arrays()
        arrays["__meta__"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(target, **arrays)  # type: ignore[arg-type]
        return target


class SiteTableModel(EmpiricalModel):
    """A model that reads only the per-site descriptors of a :class:`SiteTable`."""

    @abstractmethod
    def fit(self, table: SiteTable, train: np.ndarray, val: np.ndarray) -> None:
        """Fit on rows *train*, choosing hyperparameters on rows *val*.

        Implementations record the search in ``meta["search"]`` and the chosen
        values in ``meta["hyperparameters"]``.
        """

    @abstractmethod
    def predict_table(self, table: SiteTable, rows: np.ndarray) -> np.ndarray:
        """``1000 ln β`` for *rows* of *table*, ``(len(rows), n_temperatures)``."""

    def predict_molecule(self, molecule: MoleculeRecord) -> np.ndarray:
        table = SiteTable.from_molecules([molecule])
        out = np.full((len(table), len(self.temperatures)), np.nan)
        rows = np.flatnonzero(table.has_graph)
        if rows.size:
            out[rows] = self.predict_table(table, rows)
        return out


def read_meta(path: Path | str) -> dict[str, Any]:
    """Read a model file's header without loading its arrays."""
    with np.load(path, allow_pickle=False) as data:
        if "__meta__" not in data.files:
            raise ValueError(f"{path} is not an empirical model file")
        meta: dict[str, Any] = json.loads(bytes(data["__meta__"]).decode("utf-8"))
    if meta.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} is not an empirical model file")
    return meta


def load_model(path: Path | str) -> EmpiricalModel:
    """Load any registered model kind from *path*."""
    # Importing the package registers the built-in kinds.
    import rpfr_visualizer.empirical  # noqa: F401

    meta = read_meta(path)
    kind = meta.get("kind", "")
    if kind not in MODEL_KINDS:
        known = ", ".join(sorted(MODEL_KINDS)) or "(none)"
        raise ValueError(f"{path}: unknown model kind {kind!r}. Known: {known}")
    model = MODEL_KINDS[kind](name=meta.get("name"))
    model.meta = {k: v for k, v in meta.items() if k not in ("format", "kind", "name")}
    with np.load(path, allow_pickle=False) as data:
        model._restore({name: data[name] for name in data.files if name != "__meta__"})
    return model


__all__ = [
    "MODEL_KINDS",
    "EmpiricalModel",
    "SiteTableModel",
    "load_model",
    "pack_strings",
    "read_meta",
    "register",
    "unpack_strings",
]
