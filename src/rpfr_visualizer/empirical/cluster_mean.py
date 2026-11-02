"""Cluster mean: predict a site by the mean of training sites that look like it.

Sites are clustered by their local-environment SMILES -- the circular
substructures R1 to R5 stored as ``local_smiles``. A site is predicted from the
widest radius whose cluster is well populated in training, falling back to
narrower radii and finally to the element mean::

    R_max cluster, if it has >= min_count training sites
    else R_max-1 cluster ...
    else R1 cluster
    else the element's mean

It is the nearest-neighbour baseline in chemical-environment space: no fitted
parameters beyond the means, so whatever a learned model adds on top of it is
what the learning bought.

Validation chooses ``R_max`` and ``min_count`` per element.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rpfr_visualizer.empirical.base import SiteTableModel, pack_strings, register, unpack_strings
from rpfr_visualizer.empirical.metrics import selection_score
from rpfr_visualizer.empirical.sites import N_RADII, SiteTable

RADII: tuple[int, ...] = tuple(range(1, N_RADII + 1))
MIN_COUNTS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


class _ElementClusters:
    """One element's cluster means, radius by radius."""

    def __init__(
        self,
        mean: np.ndarray,
        keys: Sequence[Sequence[str]],
        means: Sequence[np.ndarray],
        r_max: int,
        min_count: int,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.keys = [list(k) for k in keys]
        self.means = [np.asarray(m, dtype=np.float64) for m in means]
        self.r_max = int(r_max)
        self.min_count = int(min_count)
        self.lookup = [{k: i for i, k in enumerate(ks)} for ks in self.keys]

    def predict(self, table: SiteTable, rows: np.ndarray) -> np.ndarray:
        out = np.tile(self.mean, (rows.size, 1))
        resolved = np.zeros(rows.size, dtype=bool)
        for radius in range(self.r_max, 0, -1):
            lookup = self.lookup[radius - 1]
            vocab_row = np.array([lookup.get(s, -1) for s in table.env_vocab], dtype=np.int64)
            ids = table.env_ids[rows, radius - 1]
            hit = np.where(ids >= 0, vocab_row[np.maximum(ids, 0)], -1)
            use = ~resolved & (hit >= 0)
            out[use] = self.means[radius - 1][hit[use]]
            resolved |= use
        return out


def _cluster_stats(table: SiteTable, rows: np.ndarray, radius: int):
    """Sum, count and table-vocab ids of the populated clusters at *radius*."""
    ids = table.env_ids[rows, radius - 1]
    present = ids >= 0
    ids, y = ids[present], table.y[rows[present]]
    width = max(len(table.env_vocab), 1)
    counts = np.bincount(ids, minlength=width)
    sums = np.stack(
        [np.bincount(ids, weights=y[:, t], minlength=width) for t in range(y.shape[1])], axis=1
    )
    return sums, counts


def _predict_from_stats(table, rows, stats, mean, r_max: int, min_count: int) -> np.ndarray:
    out = np.tile(mean, (rows.size, 1))
    resolved = np.zeros(rows.size, dtype=bool)
    for radius in range(r_max, 0, -1):
        sums, counts = stats[radius - 1]
        ids = table.env_ids[rows, radius - 1]
        safe = np.maximum(ids, 0)
        use = ~resolved & (ids >= 0) & (counts[safe] >= min_count)
        out[use] = sums[safe[use]] / counts[safe[use], None]
        resolved |= use
    return out


@register
class ClusterMeanModel(SiteTableModel):
    """Mean ``1000 ln β`` of training sites sharing a local environment."""

    kind = "cluster_mean"
    label = "Cluster mean"

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.fits: dict[str, _ElementClusters] = {}

    def fit(
        self,
        table: SiteTable,
        train: np.ndarray,
        val: np.ndarray,
        *,
        radii: Sequence[int] = RADII,
        min_counts: Sequence[int] = MIN_COUNTS,
    ) -> None:
        self.meta["temperatures"] = list(table.temperatures)
        search: dict[str, list[dict[str, Any]]] = {}
        chosen: dict[str, dict[str, Any]] = {}
        for element in sorted(set(table.element[train].tolist())):
            tr = train[table.element[train] == element]
            va = val[table.element[val] == element]
            mean = table.y[tr].mean(axis=0)
            stats = [_cluster_stats(table, tr, r) for r in range(1, N_RADII + 1)]
            target_rows = va if va.size else tr
            best: tuple[float, dict[str, Any]] | None = None
            trials: list[dict[str, Any]] = []
            for r_max in radii:
                for min_count in min_counts:
                    pred = _predict_from_stats(table, target_rows, stats, mean, r_max, min_count)
                    score = selection_score(table.y[target_rows], pred)
                    trial = {"r_max": r_max, "min_count": min_count, "val_mae": score}
                    trials.append(trial)
                    if best is None or score < best[0]:
                        best = (score, trial)
            assert best is not None
            r_max, min_count = best[1]["r_max"], best[1]["min_count"]
            keys, means = [], []
            for radius in range(1, N_RADII + 1):
                sums, counts = stats[radius - 1]
                kept = np.flatnonzero(counts >= min_count) if radius <= r_max else np.zeros(0, int)
                keys.append([table.env_vocab[i] for i in kept])
                means.append(sums[kept] / counts[kept, None])
            self.fits[element] = _ElementClusters(mean, keys, means, r_max, min_count)
            search[element] = trials
            chosen[element] = {
                "r_max": r_max,
                "min_count": min_count,
                "n_clusters": sum(len(k) for k in keys),
            }
        self.meta["search"] = search
        self.meta["hyperparameters"] = chosen

    def predict_table(self, table: SiteTable, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.full((rows.size, len(self.temperatures)), np.nan)
        for element, fit in self.fits.items():
            mask = table.element[rows] == element
            if mask.any():
                out[mask] = fit.predict(table, rows[mask])
        return out

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for element, fit in self.fits.items():
            arrays[f"{element}/mean"] = fit.mean
            arrays[f"{element}/settings"] = np.array([fit.r_max, fit.min_count])
            for radius in range(1, N_RADII + 1):
                arrays[f"{element}/r{radius}/keys"] = pack_strings(fit.keys[radius - 1])
                # float32 is ample: 1e-4 per mil at H-site magnitudes, half the file.
                arrays[f"{element}/r{radius}/means"] = fit.means[radius - 1].astype(np.float32)
        return arrays

    def _restore(self, arrays: Mapping[str, np.ndarray]) -> None:
        elements = {name.split("/")[0] for name in arrays}
        for element in sorted(elements):
            r_max, min_count = (int(v) for v in arrays[f"{element}/settings"])
            keys, means = [], []
            for radius in range(1, N_RADII + 1):
                m = arrays[f"{element}/r{radius}/means"]
                means.append(m)
                keys.append(unpack_strings(arrays[f"{element}/r{radius}/keys"], m.shape[0]))
            self.fits[element] = _ElementClusters(
                arrays[f"{element}/mean"], keys, means, r_max, min_count
            )


__all__ = ["MIN_COUNTS", "RADII", "ClusterMeanModel"]
