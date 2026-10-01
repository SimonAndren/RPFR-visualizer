"""Multi-molecule grid visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rpfr_visualizer.viz.electron_cloud import EnrichmentView

from rpfr_visualizer.viz._deps import require_widgets
from rpfr_visualizer.viz.py3dmol_local import display_view


@dataclass(frozen=True)
class GridCell:
    """One cell in a molecule grid."""

    label: str
    viewer: object


def molecule_grid(
    views: list[EnrichmentView],
    *,
    labels: list[str] | None = None,
    width: int = 400,
    height: int = 300,
    show_labels: bool = False,
) -> list[GridCell]:
    """Render multiple enrichment views as a grid of py3Dmol viewers."""
    if not views:
        return []

    cells: list[GridCell] = []
    for idx, view in enumerate(views):
        label = labels[idx] if labels and idx < len(labels) else view.title
        viewer = view.build_viewer(width=width, height=height, show_labels=show_labels)
        cells.append(GridCell(label=label, viewer=viewer))
    return cells


def display_grid(cells: list[GridCell], *, cols: int = 2) -> None:
    """Display grid cells in a Jupyter notebook using ipywidgets."""
    widgets, display, _clear_output, _HTML = require_widgets("grid display")

    if not cells:
        display(widgets.HTML("<i>No molecules to display</i>"))
        return

    rows: list[widgets.HBox] = []
    row: list[widgets.VBox] = []
    for idx, cell in enumerate(cells):
        viewer_out = widgets.Output()
        with viewer_out:
            display_view(cell.viewer)
        box = widgets.VBox(
            [
                widgets.HTML(f"<b>{cell.label}</b>"),
                viewer_out,
            ]
        )
        row.append(box)
        if len(row) == cols or idx == len(cells) - 1:
            rows.append(widgets.HBox(row))
            row = []

    display(widgets.VBox(rows))


__all__ = ["GridCell", "display_grid", "molecule_grid"]
