"""Visualization for the QM9s-RPFR data product.

The published surface: the molecule RPFR explorer and the pieces it is built
from. Everything here renders per-site isotope data for a molecule — 3D
enrichment views, classic 2D depictions, multi-molecule grids, and QC strips.

Open the explorer with :func:`open_explorer` (a catalog, provider or HDF5 path)
or :func:`open_full_dataset_explorer` (the full datasets on an external drive).
:class:`~rpfr_visualizer.viz.explorer.ExplorerController` drives the same analysis
without any widgets, for scripts that want figures instead of a panel.
"""

from rpfr_visualizer.viz.colors import cpk_color, cpk_fingerprint_style, hex_to_rgb01, range_caption
from rpfr_visualizer.viz.depict import (
    EnrichmentDepiction,
    depict_enrichment,
    depict_enrichment_svg,
    depict_fg_pattern,
    depict_with_highlight,
    format_site_value_table,
    site_anchor_options,
)
from rpfr_visualizer.viz.electron_cloud import EnrichmentView
from rpfr_visualizer.viz.explorer import (
    ExplorerController,
    ExplorerState,
    ExplorerView,
    LoadedMolecule,
    RenderPayload,
    open_explorer,
    open_full_dataset_explorer,
)
from rpfr_visualizer.viz.grid import GridCell, display_grid, molecule_grid
from rpfr_visualizer.viz.py3dmol_local import create_view, display_view
from rpfr_visualizer.viz.qc_strip import format_molecule_qc, qc_panel_text, stored_qc_flags
from rpfr_visualizer.viz.structure import molecule_to_xyz

__all__ = [
    # The explorer
    "ExplorerController",
    "ExplorerState",
    "ExplorerView",
    "LoadedMolecule",
    "RenderPayload",
    "open_explorer",
    "open_full_dataset_explorer",
    # Rendering
    "EnrichmentView",
    "GridCell",
    "create_view",
    "display_grid",
    "display_view",
    "molecule_grid",
    # 2D depiction
    "EnrichmentDepiction",
    "depict_enrichment",
    "depict_enrichment_svg",
    "depict_fg_pattern",
    "depict_with_highlight",
    "format_site_value_table",
    "site_anchor_options",
    # Colour + geometry helpers
    "cpk_color",
    "cpk_fingerprint_style",
    "hex_to_rgb01",
    "molecule_to_xyz",
    "range_caption",
    # QC
    "format_molecule_qc",
    "qc_panel_text",
    "stored_qc_flags",
]
