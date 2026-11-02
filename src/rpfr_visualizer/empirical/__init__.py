"""Empirical models of per-site ``1000 ln β``, trained on the dataset itself.

Each model is fitted on a fixed training split, tuned on a validation split,
and scored once on a held-out test split (:mod:`.split`), so the models can be
ranked against each other and the explorer can show any of them in place of
the dataset's own values.

Built-in kinds:

* ``bond_additivity`` -- per-element ridge regression on bond-path counts.
* ``cluster_mean`` -- mean of training sites with the same local environment.
"""

from rpfr_visualizer.empirical.base import (
    MODEL_KINDS,
    EmpiricalModel,
    SiteTableModel,
    load_model,
    read_meta,
)
from rpfr_visualizer.empirical.bond_additivity import BondAdditivityModel
from rpfr_visualizer.empirical.cluster_mean import ClusterMeanModel
from rpfr_visualizer.empirical.sites import SiteTable, from_target, to_target
from rpfr_visualizer.empirical.split import (
    DEFAULT_N_TEST,
    DEFAULT_N_TRAIN,
    SPLIT_FILENAME,
    DatasetSplit,
    make_split,
    split_hdf5,
)
from rpfr_visualizer.empirical.store import ModelInfo, ModelStore, apply_model, model_source_tag
from rpfr_visualizer.empirical.train import format_scores, train_all, train_model

__all__ = [
    "DEFAULT_N_TEST",
    "DEFAULT_N_TRAIN",
    "MODEL_KINDS",
    "SPLIT_FILENAME",
    "BondAdditivityModel",
    "ClusterMeanModel",
    "DatasetSplit",
    "EmpiricalModel",
    "ModelInfo",
    "ModelStore",
    "SiteTable",
    "SiteTableModel",
    "apply_model",
    "format_scores",
    "from_target",
    "load_model",
    "make_split",
    "model_source_tag",
    "read_meta",
    "split_hdf5",
    "to_target",
    "train_all",
    "train_model",
]
