"""The standalone isotope-fingerprint explorer.

A small local web application over
:class:`~rpfr_visualizer.viz.explorer.ExplorerController`. Start it from a terminal::

    rpfr-visualizer gui

This package is a sibling of :mod:`rpfr_visualizer.viz`, not part of it: the
dependency runs one way (``gui`` imports ``viz``), and ``viz/__init__`` imports
its whole surface eagerly, so server code living under it would load on every
``import rpfr_visualizer.viz``.
"""

from __future__ import annotations

from rpfr_visualizer.gui.config import GuiConfig, Resolution, config_path, discover, resolve
from rpfr_visualizer.gui.server import start_explorer

__all__ = [
    "GuiConfig",
    "Resolution",
    "config_path",
    "discover",
    "resolve",
    "start_explorer",
]
