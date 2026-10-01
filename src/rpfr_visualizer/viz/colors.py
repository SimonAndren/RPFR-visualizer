"""Shared element colours and quantity symbols for the 2D and 3D views."""

from __future__ import annotations

import math
from collections.abc import Sequence

# Jmol/CPK-style palette used consistently across 3D viewers and matplotlib.
ELEMENT_CPK: dict[str, str] = {
    "H": "#FFFFFF",
    "C": "#000000",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
    "B": "#FFB5B5",
    "Si": "#F0C8A0",
}

FALLBACK_CPK = "#C8C8C8"
# Floor is high on purpose: a weak site must still read as its CPK element
# (blue nitrogen, red oxygen), not as a ghost. Magnitude is an opacity nudge
# plus the numeric label, not a fade to white.
FINGERPRINT_EMPTY_OPACITY = 0.78
FINGERPRINT_FULL_OPACITY = 1.00


def cpk_color(element: str) -> str:
    """Return the standard CPK hex color for *element*."""
    return ELEMENT_CPK.get(element, FALLBACK_CPK)


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """Convert a ``#rrggbb`` string to floats in 0–1 (RDKit's colour range)."""
    text = hex_color.lstrip("#")
    return (
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
    )


def normalize_fingerprint(
    value: float,
    vmin: float,
    vmax: float,
    *,
    symmetric: bool = True,
) -> float:
    """Map a fingerprint value to [0, 1] fill intensity."""
    if not math.isfinite(value):
        return 0.0
    if symmetric:
        bound = max(abs(vmin), abs(vmax), 1e-12)
        return float(min(abs(value) / bound, 1.0))
    if vmax == vmin:
        return 0.5
    return float(min(max((value - vmin) / (vmax - vmin), 0.0), 1.0))


def cpk_fingerprint_style(
    element: str,
    value: float | None,
    vmin: float,
    vmax: float,
    *,
    symmetric: bool = True,
) -> tuple[str, float]:
    """Return the canonical CPK colour, with fingerprint strength as opacity.

    Hue never leaves the CPK palette: nitrogen stays blue, oxygen stays red,
    carbon stays black. A missing value is a structure atom (full colour);
    a numeric value fades only the opacity, so a weak site is still recognisably
    the same element.
    """
    base = cpk_color(element)
    if value is None or not math.isfinite(value):
        return base, FINGERPRINT_FULL_OPACITY

    fill = normalize_fingerprint(value, vmin, vmax, symmetric=symmetric)
    opacity = FINGERPRINT_EMPTY_OPACITY + fill * (
        FINGERPRINT_FULL_OPACITY - FINGERPRINT_EMPTY_OPACITY
    )
    return base, opacity


def cpk_with_alpha(element: str, *, alpha: float = 1.0) -> str:
    """Return an rgba string for matplotlib scatter plots."""
    hex_color = cpk_color(element)
    r, g, b = hex_to_rgb01(hex_color)
    return f"rgba({r:.3f}, {g:.3f}, {b:.3f}, {alpha:.3f})"


def range_caption(
    element: str,
    *,
    kind: str,
    temperature: float,
    vmin: float,
    vmax: float,
) -> str:
    """What the opacity means, for one element and the quantity on screen."""
    return opacity_caption(kind=kind, temperature=temperature, ranges=[(element, vmin, vmax)])


def opacity_caption(
    *,
    kind: str,
    temperature: float,
    ranges: Sequence[tuple[str, float, float]],
) -> str:
    """One sentence saying opacity is magnitude, with each element's scale.

    Hue is the element, so opacity is the only channel carrying the value,
    and it is scaled per element: an opaque oxygen and an opaque carbon are
    each strong *for their element*, not equal.
    """
    if not ranges:
        return ""
    symbol = quantity_symbol(kind)
    scale = f"scaled to the dataset's 5th–95th percentile at {temperature:g} K"
    if len(ranges) == 1:
        el, lo, hi = ranges[0]
        return f"Opacity shows the magnitude of {symbol} for {el}, {scale} ({lo:.4g} to {hi:.4g})."
    scales = " · ".join(f"{el} {lo:.4g} to {hi:.4g}" for el, lo, hi in ranges)
    return f"Opacity shows the magnitude of {symbol} for each element, {scale}: {scales}."


#: The symbol each quantity is written with. One table, so the caption, the
#: site table and the ranked list cannot name the same quantity differently.
QUANTITY_SYMBOLS: dict[str, str] = {
    "beta": "β",
    "alpha": "α",
    "ln_alpha": "1000 ln α",
    "epsilon": "ε",
}


def quantity_symbol(kind: str) -> str:
    """The symbol for *kind*, or *kind* itself if it has none."""
    return QUANTITY_SYMBOLS.get(kind, kind)


__all__ = [
    "hex_to_rgb01",
    "ELEMENT_CPK",
    "FALLBACK_CPK",
    "FINGERPRINT_EMPTY_OPACITY",
    "FINGERPRINT_FULL_OPACITY",
    "QUANTITY_SYMBOLS",
    "cpk_color",
    "cpk_fingerprint_style",
    "cpk_with_alpha",
    "normalize_fingerprint",
    "opacity_caption",
    "quantity_symbol",
    "range_caption",
]
