"""Import guards for the optional visualization dependencies.

Everything in :mod:`rpfr_visualizer.viz` is reachable without the ``viz`` extra
installed; the heavy imports happen inside these helpers so a missing package
raises one actionable message at call time instead of an ImportError at import
time. *feature* names the thing the caller is building, so the message points at
what the user was actually trying to do.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = "Install with: uv sync --extra viz"


def _missing(package: str, feature: str) -> ImportError:
    return ImportError(f"{package} is required for {feature}. {_INSTALL_HINT}")


def require_matplotlib(feature: str = "static plots") -> Any:
    """Return ``matplotlib.pyplot``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise _missing("matplotlib", feature) from exc
    return plt


def require_py3dmol(feature: str = "3D visualization") -> Any:
    """Return the ``py3Dmol`` module."""
    try:
        import py3Dmol
    except ImportError as exc:
        raise _missing("py3Dmol", feature) from exc
    return py3Dmol


def require_rdkit(feature: str = "structure visualization") -> Any:
    """Return ``rdkit.Chem``."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise _missing("RDKit", feature) from exc
    return Chem


def require_rdkit_draw(feature: str = "2D depiction") -> tuple[Any, Any]:
    """Return ``(rdkit.Chem, rdkit.Chem.Draw)``."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError as exc:
        raise _missing("RDKit", feature) from exc
    return Chem, Draw


def require_widgets(feature: str = "the control panel") -> tuple[Any, Any, Any, Any]:
    """Return ``(ipywidgets, display, clear_output, HTML)``."""
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display
    except ImportError as exc:
        raise _missing("ipywidgets", feature) from exc
    return widgets, display, clear_output, HTML


__all__ = [
    "require_matplotlib",
    "require_py3dmol",
    "require_rdkit",
    "require_rdkit_draw",
    "require_widgets",
]
