"""Error statistics for per-site ``1000 ln β`` predictions.

Scores are per element, because the elements live on different scales: an
H-site error of 5 ‰ is good and a C-site error of 5 ‰ is poor, and one pooled
number would be dominated by hydrogen. Within an element every temperature is
scored; the headline is the mean absolute error at the reference temperature.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: The temperature a single headline number is quoted at.
REFERENCE_TEMPERATURE = 298.15


def reference_column(temperatures: Sequence[float]) -> int:
    """Index of the grid temperature closest to :data:`REFERENCE_TEMPERATURE`."""
    grid = np.asarray(temperatures, dtype=float)
    return int(np.argmin(np.abs(grid - REFERENCE_TEMPERATURE)))


def site_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    elements: np.ndarray,
    temperatures: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Score predictions per element.

    Args:
        y_true: Reference ``1000 ln β``, ``(n_sites, n_temperatures)``.
        y_pred: Predictions of the same shape; NaN where the model had no answer.
        elements: Element symbol per site.
        temperatures: The grid the columns refer to.

    Returns:
        ``{element: {...}}`` with the site count, coverage (fraction of sites
        predicted), MAE and RMSE per temperature, and at the reference
        temperature MAE, RMSE and R².
    """
    col = reference_column(temperatures)
    out: dict[str, dict[str, Any]] = {}
    for element in sorted(set(elements.tolist())):
        mask = elements == element
        truth, pred = y_true[mask], y_pred[mask]
        answered = np.isfinite(pred).all(axis=1)
        entry: dict[str, Any] = {
            "n_sites": int(mask.sum()),
            "coverage": float(answered.mean()) if answered.size else 0.0,
        }
        if answered.any():
            err = pred[answered] - truth[answered]
            mae = np.abs(err).mean(axis=0)
            rmse = np.sqrt((err**2).mean(axis=0))
            ref_truth = truth[answered, col]
            ss_tot = float(((ref_truth - ref_truth.mean()) ** 2).sum())
            ss_res = float((err[:, col] ** 2).sum())
            entry.update(
                {
                    "mae": [float(v) for v in mae],
                    "rmse": [float(v) for v in rmse],
                    "mae_ref": float(mae[col]),
                    "rmse_ref": float(rmse[col]),
                    "r2_ref": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                    "mae_mean": float(mae.mean()),
                }
            )
        out[element] = entry
    return out


def selection_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """What hyperparameter search minimises: MAE over all temperatures.

    A site with no prediction counts as an error of its own magnitude, so a
    setting cannot win by declining to answer the hard cases.
    """
    filled = np.where(np.isfinite(y_pred), y_pred, 0.0)
    return float(np.abs(filled - y_true).mean()) if y_true.size else float("inf")


__all__ = ["REFERENCE_TEMPERATURE", "reference_column", "selection_score", "site_metrics"]
