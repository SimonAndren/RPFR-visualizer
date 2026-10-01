"""The published molecule RPFR explorer.

Split into state (what is selected), controller (what it does), and view (how
it looks) so the analysis path can be used and tested without a notebook::

    from rpfr_visualizer.viz.explorer import ExplorerController, open_explorer

    open_explorer(catalog, dataset="qm9s").show()      # interactive panel

    controller = ExplorerController(catalog)           # headless
    controller.load("100506")
    controller.rank()
"""

from __future__ import annotations

from rpfr_visualizer.viz.explorer.controller import ExplorerController, RenderPayload
from rpfr_visualizer.viz.explorer.factory import (
    PREFERRED_DATASET,
    controller_for_data_dir,
    controller_for_drive,
    preferred_dataset,
)
from rpfr_visualizer.viz.explorer.state import (
    ELEMENTS,
    ExplorerState,
    LoadedMolecule,
    ReferenceMode,
)
from rpfr_visualizer.viz.explorer.view import (
    ExplorerView,
    open_explorer,
    open_full_dataset_explorer,
)

__all__ = [
    "ELEMENTS",
    "PREFERRED_DATASET",
    "ExplorerController",
    "ExplorerState",
    "ExplorerView",
    "LoadedMolecule",
    "ReferenceMode",
    "RenderPayload",
    "controller_for_data_dir",
    "controller_for_drive",
    "open_explorer",
    "open_full_dataset_explorer",
    "preferred_dataset",
]
