"""XYZ helpers: turn stored Cartesian coordinates into 3Dmol input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rpfr_visualizer.io.loader import Molecule


def molecule_to_xyz(molecule: Molecule) -> str:
    """Build an XYZ block from the molecule's stored Cartesian coordinates."""
    symbols = [site.element for site in molecule.sites]
    coords = np.array(
        [site.coordinates or (0.0, 0.0, 0.0) for site in molecule.sites],
        dtype=np.float64,
    )
    lines = [str(len(symbols)), ""]
    for sym, (x, y, z) in zip(symbols, coords, strict=True):
        lines.append(f"{sym:2s}  {float(x):12.6f}  {float(y):12.6f}  {float(z):12.6f}")
    return "\n".join(lines)


__all__ = ["molecule_to_xyz"]
