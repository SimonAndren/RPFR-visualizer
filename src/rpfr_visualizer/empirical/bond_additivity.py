"""Bond additivity: a site's ``1000 ln β`` as a sum of bond-path increments.

The site's isotope sensitivity is written as an element baseline plus one
increment per bond path leaving the site::

    1000 ln β(site, T) = b_e(T) + Σ_p  n_p · w_p(T)

where *p* runs over the paths in the site's ``bond_order`` entry (``C-H``,
``C-C=O``, ``C-N-C-H`` ...) and *n_p* is how often it occurs. Depth 1 is the
classic bond-additivity scheme -- only the bonds the atom itself makes; depths 2
and 3 add the neighbours' bonds, the next-nearest corrections of a group
additivity scheme. Each element has its own baseline and increments, fitted by
ridge regression at every temperature at once.

Validation chooses the depth and the ridge strength per element. Paths never
seen in training contribute nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rpfr_visualizer.empirical.base import SiteTableModel, pack_strings, register, unpack_strings
from rpfr_visualizer.empirical.metrics import selection_score
from rpfr_visualizer.empirical.sites import SiteTable

DEPTHS: tuple[int, ...] = (1, 2, 3)
#: Ridge strengths, relative to the per-site normal equations (so independent
#: of how many sites there are).
ALPHAS: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

#: Rows handled per block when forming the normal equations.
_BLOCK_ROWS = 100_000


def _normal_equations(
    rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, y: np.ndarray, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``XᵀX``, ``Xᵀy`` and the column sums of a sparse design matrix.

    *rows* must be sorted. Built block by block with :func:`numpy.bincount`,
    which is what keeps a million-site fit off the order of the number of
    sites times the number of features.
    """
    gram = np.zeros(n_features * n_features)
    n_rows = y.shape[0]
    bounds = np.searchsorted(rows, np.arange(0, n_rows + _BLOCK_ROWS, _BLOCK_ROWS))
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=False):
        if hi <= lo:
            continue
        r, c, v = rows[lo:hi], cols[lo:hi], vals[lo:hi]
        base = r[0]
        lens = np.bincount(r - base)
        starts = np.cumsum(lens) - lens
        per_entry = lens[r - base]
        left = np.repeat(np.arange(r.size), per_entry)
        offsets = np.arange(left.size) - np.repeat(np.cumsum(per_entry) - per_entry, per_entry)
        right = np.repeat(starts[r - base], per_entry) + offsets
        gram += np.bincount(
            c[left] * n_features + c[right],
            weights=v[left] * v[right],
            minlength=n_features * n_features,
        )
    xty = np.stack(
        [
            np.bincount(cols, weights=vals * y[rows, t], minlength=n_features)
            for t in range(y.shape[1])
        ],
        axis=1,
    )
    sums = np.bincount(cols, weights=vals, minlength=n_features)
    return gram.reshape(n_features, n_features), xty, sums


def _solve(gram, xty, sums, ybar, n_rows: int, alpha: float):
    """Centred ridge solution: weights ``(F, T)`` and intercept ``(T,)``."""
    xbar = sums / n_rows
    a = gram / n_rows - np.outer(xbar, xbar)
    a[np.diag_indices_from(a)] += alpha
    b = xty / n_rows - np.outer(xbar, ybar)
    try:
        weights = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(a, b, rcond=None)[0]
    return weights, ybar - xbar @ weights


def _apply(
    rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, weights, intercept, n_rows: int
) -> np.ndarray:
    out = np.tile(intercept, (n_rows, 1))
    for t in range(weights.shape[1]):
        out[:, t] += np.bincount(rows, weights=vals * weights[cols, t], minlength=n_rows)
    return out


class _ElementFit:
    """One element's increments."""

    def __init__(self, paths: Sequence[str], weights, intercept, depth: int) -> None:
        self.paths = list(paths)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.depth = int(depth)
        self.column = {p: i for i, p in enumerate(self.paths)}

    def predict(self, table: SiteTable, rows: np.ndarray) -> np.ndarray:
        local, ids, counts = table.paths_coo(rows, max_depth=self.depth)
        lookup = np.array([self.column.get(p, -1) for p in table.path_vocab], dtype=np.int64)
        cols = lookup[ids] if ids.size else ids
        known = cols >= 0
        return _apply(
            local[known], cols[known], counts[known], self.weights, self.intercept, rows.size
        )


@register
class BondAdditivityModel(SiteTableModel):
    """Per-element ridge regression on bond-path counts."""

    kind = "bond_additivity"
    label = "Bond additivity"

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.fits: dict[str, _ElementFit] = {}

    def fit(
        self,
        table: SiteTable,
        train: np.ndarray,
        val: np.ndarray,
        *,
        depths: Sequence[int] = DEPTHS,
        alphas: Sequence[float] = ALPHAS,
    ) -> None:
        self.meta["temperatures"] = list(table.temperatures)
        search: dict[str, list[dict[str, Any]]] = {}
        chosen: dict[str, dict[str, Any]] = {}
        for element in sorted(set(table.element[train].tolist())):
            tr = train[table.element[train] == element]
            va = val[table.element[val] == element]
            best: tuple[float, _ElementFit, dict[str, Any]] | None = None
            trials: list[dict[str, Any]] = []
            for depth in depths:
                r, ids, v = table.paths_coo(tr, max_depth=depth)
                features = np.unique(ids)
                column = np.full(len(table.path_vocab), -1, dtype=np.int64)
                column[features] = np.arange(features.size)
                y = table.y[tr]
                gram, xty, sums = _normal_equations(r, column[ids], v, y, features.size)
                for alpha in alphas:
                    weights, intercept = _solve(gram, xty, sums, y.mean(axis=0), tr.size, alpha)
                    fit = _ElementFit(
                        [table.path_vocab[i] for i in features], weights, intercept, depth
                    )
                    score = (
                        selection_score(table.y[va], fit.predict(table, va))
                        if va.size
                        else selection_score(y, fit.predict(table, tr))
                    )
                    trial = {"depth": depth, "alpha": alpha, "val_mae": score}
                    trials.append(trial)
                    if best is None or score < best[0]:
                        best = (score, fit, trial)
            assert best is not None
            self.fits[element] = best[1]
            search[element] = trials
            chosen[element] = {
                "depth": best[2]["depth"],
                "alpha": best[2]["alpha"],
                "n_features": len(best[1].paths),
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
            arrays[f"{element}/paths"] = pack_strings(fit.paths)
            arrays[f"{element}/weights"] = fit.weights
            arrays[f"{element}/intercept"] = fit.intercept
            arrays[f"{element}/depth"] = np.array(fit.depth)
        return arrays

    def _restore(self, arrays: Mapping[str, np.ndarray]) -> None:
        elements = {name.split("/")[0] for name in arrays}
        for element in sorted(elements):
            weights = arrays[f"{element}/weights"]
            self.fits[element] = _ElementFit(
                unpack_strings(arrays[f"{element}/paths"], weights.shape[0]),
                weights,
                arrays[f"{element}/intercept"],
                int(arrays[f"{element}/depth"]),
            )


__all__ = ["ALPHAS", "DEPTHS", "BondAdditivityModel"]
