"""Find saved models, and swap a molecule's β for a model's prediction.

A directory of model files (``data/models`` by default) is a
:class:`ModelStore`. The explorer asks it for the available models and, when
the user picks one, for a copy of each molecule whose β come from that model
instead of the dataset. Everything else on the molecule -- structure, QC flags,
descriptors -- is the dataset's, unchanged.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rpfr_visualizer.empirical.base import MODEL_SUFFIX, EmpiricalModel, load_model, read_meta
from rpfr_visualizer.empirical.sites import from_target
from rpfr_visualizer.empirical.split import SPLIT_FILENAME, DatasetSplit, SplitName
from rpfr_visualizer.io.schema import MoleculeRecord


def model_source_tag(name: str) -> str:
    """The ``source`` a model's predictions carry, e.g. ``model:cluster_mean``."""
    return f"model:{name}"


def apply_model(molecule: MoleculeRecord, model: EmpiricalModel) -> MoleculeRecord:
    """A copy of *molecule* whose site β are *model*'s predictions.

    Raises:
        ValueError: If the model was trained on a different temperature grid,
            since its columns would then be attached to the wrong temperatures.
    """
    grid = tuple(molecule.sites[0].temperatures) if molecule.sites else model.temperatures
    if not np.allclose(grid, model.temperatures):
        raise ValueError(
            f"Model {model.name!r} predicts on a {len(model.temperatures)}-point grid; "
            f"this dataset stores {len(grid)}. Retrain it on this dataset's file."
        )
    beta = from_target(model.predict_molecule(molecule))
    tag = model_source_tag(model.name)
    sites = [
        dataclasses.replace(site, beta=tuple(float(v) for v in beta[i]), source=tag)
        for i, site in enumerate(molecule.sites)
    ]
    return dataclasses.replace(molecule, sites=sites, source=tag)


@dataclass(frozen=True)
class ModelInfo:
    """What a menu needs to know about one saved model, without loading it."""

    name: str
    kind: str
    label: str
    path: Path
    meta: dict[str, Any]

    def test_mae(self) -> dict[str, float]:
        """Test MAE at the reference temperature, per element."""
        return {
            element: float(entry["mae_ref"])
            for element, entry in self.meta.get("test", {}).items()
            if "mae_ref" in entry
        }


class ModelStore:
    """The model files in one directory, loaded on first use."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self._models: dict[str, EmpiricalModel] = {}
        self._infos: dict[str, ModelInfo] | None = None
        self._split: DatasetSplit | None = None
        self._membership: dict[str, SplitName] | None = None

    def refresh(self) -> None:
        """Forget what was found, so a newly trained model shows up."""
        self._infos = None
        self._models.clear()
        self._split = None
        self._membership = None

    def infos(self) -> dict[str, ModelInfo]:
        if self._infos is None:
            found: dict[str, ModelInfo] = {}
            if self.directory.is_dir():
                for path in sorted(self.directory.glob(f"*{MODEL_SUFFIX}")):
                    try:
                        meta = read_meta(path)
                    except (ValueError, OSError, KeyError):
                        continue
                    name = str(meta.get("name") or path.stem)
                    found[name] = ModelInfo(
                        name=name,
                        kind=str(meta.get("kind", "")),
                        label=str(meta.get("label") or name),
                        path=path,
                        meta=meta,
                    )
            self._infos = found
        return self._infos

    def names(self) -> tuple[str, ...]:
        return tuple(self.infos())

    def get(self, name: str) -> EmpiricalModel:
        if name not in self._models:
            infos = self.infos()
            if name not in infos:
                available = ", ".join(infos) or "(none)"
                raise KeyError(f"No model {name!r} in {self.directory}. Available: {available}")
            self._models[name] = load_model(infos[name].path)
        return self._models[name]

    @property
    def split(self) -> DatasetSplit | None:
        """The split the models were trained against, if it was saved with them."""
        if self._split is None:
            path = self.directory / SPLIT_FILENAME
            if path.is_file():
                try:
                    self._split = DatasetSplit.load(path)
                except (ValueError, OSError, KeyError):
                    self._split = None
        return self._split

    def split_of(self, key: str, name: str | None = None) -> SplitName | None:
        """Which split molecule *key* was in, for the model *name*.

        None when the molecule was not split or when the model was trained on
        a different split from the one saved here -- in which case nobody can
        say whether the molecule was seen in training.
        """
        split = self.split
        if split is None:
            return None
        if name is not None:
            info = self.infos().get(name)
            trained_on = (info.meta.get("split") or {}).get("fingerprint") if info else None
            if trained_on != split.fingerprint:
                return None
        if self._membership is None:
            self._membership = split.membership()
        return self._membership.get(key)


__all__ = ["ModelInfo", "ModelStore", "apply_model", "model_source_tag"]
