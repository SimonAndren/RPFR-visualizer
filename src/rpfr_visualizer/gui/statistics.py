"""What is actually in this data.

Built from cheap SQL aggregates plus the cached per-element percentiles, not from
:func:`~rpfr_visualizer.analysis.stats.dataset_stats` — that iterates the whole dataset,
which is unusable behind an HTTP request on an 881,757-molecule file, and it takes
a ``QM9sDataset`` where the explorer holds an ``HDF5Provider``.

The percentiles come from a *sample*, and this module says so. A page headed
"dataset statistics" that quietly reports sample percentiles as population ones
would be misleading.
"""

from __future__ import annotations

from typing import Any

from rpfr_visualizer.gui.payloads import jsonable
from rpfr_visualizer.viz.explorer import ExplorerController
from rpfr_visualizer.viz.explorer.state import ELEMENTS


def statistics_payload(controller: ExplorerController) -> dict[str, Any]:
    """Per-dataset inventory, the temperature grid, percentiles and group counts."""
    state = controller.state
    return {
        "dataset": controller.dataset,
        "temperature": float(state.temperature),
        "temperatures": [float(t) for t in controller.temperatures_for_display],
        "datasets": _dataset_rows(controller),
        "elements": _element_rows(controller),
        "groups": _group_rows(controller),
        "provenance": _provenance(controller),
        "sampled_molecules": controller._max_stats_molecules,
    }


def _dataset_rows(controller: ExplorerController) -> list[dict[str, Any]]:
    """One row per dataset, with the structure coverage that makes it legible.

    ``n_with_structure`` is how many molecules a name, formula, structure or
    functional-group search can reach at all; the rest are found only by ID.
    """
    try:
        inventory = controller.dataset_inventory()
    except ValueError:
        return []

    rows = []
    for name in sorted(inventory):
        row = inventory[name]
        total = row["n"] or 1
        rows.append(
            {
                "dataset": name,
                "active": name == controller.dataset,
                "available": name in controller.available_datasets(),
                "n": row["n"],
                "n_with_structure": row["n_with_structure"],
                "structure_pct": round(100.0 * row["n_with_structure"] / total, 1),
                "n_qc_pass": row["n_qc_pass"],
                "n_qc_fail": row["n_qc_fail"],
                "n_qc_unknown": row["n_qc_unknown"],
                "note": _coverage_note(name, row),
            }
        )
    return rows


def _coverage_note(dataset: str, row: dict[str, int]) -> str:
    """Explain a structure gap in the terms that caused it."""
    missing = row["n"] - row["n_with_structure"]
    if not missing:
        return ""
    if row["n_with_structure"] == 0:
        return f"{dataset} stores no structures, so it can only be searched by molecule ID."
    return f"{missing:,} of these have no stored structure and can only be found by ID."


def _element_rows(controller: ExplorerController) -> list[dict[str, Any]]:
    """Per-element β percentiles at the chosen temperature.

    These are exactly the numbers the colour scale normalises against, which is
    why they belong on this page rather than only in a caption.
    """
    try:
        stats = controller.stats
    except (OSError, ValueError):
        return []

    temperature = float(controller.state.temperature)
    rows = []
    for element in ELEMENTS:
        if element not in stats.elements():
            continue
        percentiles = stats.percentiles(element, temperature, kind="beta")
        if percentiles is None:
            continue
        rows.append(
            {
                "element": element,
                "p5": jsonable(percentiles.p5),
                "p50": jsonable(percentiles.p50),
                "p95": jsonable(percentiles.p95),
            }
        )
    return rows


def _group_rows(controller: ExplorerController) -> list[dict[str, Any]]:
    """Functional-group census for the active dataset.

    Groups with no members are kept, listed as zero: a chemist looking for
    anhydrides should learn there are none rather than wonder where the row went.
    """
    try:
        counts = controller.functional_group_counts()
    except ValueError:
        return []
    return [
        {"name": name, "description": description, "count": count}
        for name, description, count in counts
    ]


def _provenance(controller: ExplorerController) -> dict[str, Any]:
    """Index build metadata.

    RDKit version drift is a live hazard — canonical SMILES, InChIKeys and the
    screening fingerprints all depend on it — so it is worth showing somewhere.
    """
    index = controller._index
    if index is None:
        return {"index": None}
    backend = index.backend
    meta = backend.meta() if hasattr(backend, "meta") else {}
    running = ""
    try:
        import rdkit

        running = rdkit.__version__
    except ImportError:
        pass
    built_with = meta.get("rdkit_version", "")
    return {
        "index": str(getattr(backend, "path", "")),
        "molecules": len(index),
        "index_version": meta.get("index_version", ""),
        "rdkit_built_with": built_with,
        "rdkit_running": running,
        "rdkit_drift": bool(built_with and running and built_with != running),
    }


__all__ = ["statistics_payload"]
