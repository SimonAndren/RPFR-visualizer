"""Descriptive statistics for QM9s-RPFR datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from rpfr_visualizer.io.loader import QM9sDataset


def dataset_stats(
    dataset: QM9sDataset,
    *,
    temperature: float = 300.0,
) -> dict[str, Any]:
    """Compute per-dataset descriptive statistics.

    Parameters
    ----------
    dataset:
        Loaded QM9sDataset instance.
    temperature:
        Temperature in Kelvin for beta summaries.

    Returns
    -------
    dict
        Summary statistics including molecule/site counts, element breakdown,
        heavy-atom size distribution, beta ranges, and QC pass rate.
    """
    element_counts: dict[str, int] = {}
    heavy_atom_counts: list[int] = []
    beta_by_element: dict[str, list[float]] = {}
    qc_pass_count = 0
    n_sites = 0

    for mol in dataset:
        if mol.qc_pass:
            qc_pass_count += 1
        heavy_atom_counts.append(mol.heavy_atom_count)
        n_sites += len(mol.sites)

        for site in mol.sites:
            element_counts[site.element] = element_counts.get(site.element, 0) + 1
            beta_val = site.beta_at(temperature)
            beta_by_element.setdefault(site.element, []).append(beta_val)

    n_molecules = len(dataset)
    beta_ranges = {}
    for element, values in beta_by_element.items():
        arr = np.array(values)
        beta_ranges[element] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "count": len(values),
        }

    heavy_arr = np.array(heavy_atom_counts) if heavy_atom_counts else np.array([0])
    size_distribution = (
        pd.Series(heavy_atom_counts, name="heavy_atom_count").value_counts().sort_index().to_dict()
        if heavy_atom_counts
        else {}
    )

    return {
        "n_molecules": n_molecules,
        "n_sites": n_sites,
        "qc_pass_count": qc_pass_count,
        "qc_pass_rate": qc_pass_count / n_molecules if n_molecules else 0.0,
        "element_counts": element_counts,
        "heavy_atom_count": {
            "min": int(heavy_arr.min()),
            "max": int(heavy_arr.max()),
            "mean": float(heavy_arr.mean()),
            "std": float(heavy_arr.std()),
        },
        "heavy_atom_size_distribution": size_distribution,
        "beta_at_temperature": temperature,
        "beta_ranges_by_element": beta_ranges,
        "temperatures_K": list(dataset.temperatures),
        "source": dataset.source,
    }
