"""Unit tests for shared CPK colors."""

from __future__ import annotations

from rpfr_visualizer.viz.colors import cpk_color, cpk_fingerprint_style


def test_cpk_color_known_elements() -> None:
    assert cpk_color("C") == "#000000"
    assert cpk_color("N") == "#3050F8"
    assert cpk_color("O") == "#FF0D0D"


def test_cpk_color_unknown_element() -> None:
    assert cpk_color("Xx").startswith("#")


def test_fingerprint_fill_increases_opacity() -> None:
    _, low = cpk_fingerprint_style("O", 0.1, -1.0, 1.0)
    _, high = cpk_fingerprint_style("O", 0.9, -1.0, 1.0)
    assert high > low


def test_fingerprint_keeps_canonical_cpk_hue() -> None:
    """Nitrogen stays blue at every fill; strength is opacity only."""
    low_color, _ = cpk_fingerprint_style("N", 0.1, 1.0, 1.2, symmetric=False)
    high_color, _ = cpk_fingerprint_style("N", 1.2, 1.0, 1.2, symmetric=False)
    empty_color, _ = cpk_fingerprint_style("N", None, 1.0, 1.2)
    assert low_color == high_color == empty_color == cpk_color("N")


def test_range_caption() -> None:
    from rpfr_visualizer.viz.colors import range_caption

    text = range_caption("C", kind="epsilon", temperature=298.15, vmin=-5.0, vmax=5.0)
    assert "C" in text and "ε" in text and "298.15" in text


def test_opacity_caption_names_every_element_scale() -> None:
    """Hue is the element, so the caption must say opacity carries the value."""
    from rpfr_visualizer.viz.colors import opacity_caption

    text = opacity_caption(
        kind="beta", temperature=298.15, ranges=[("C", 1.02, 1.2), ("O", 1.04, 1.1)]
    )
    assert text.startswith("Opacity shows the magnitude of β for each element")
    assert "C 1.02 to 1.2" in text and "O 1.04 to 1.1" in text
    assert opacity_caption(kind="beta", temperature=298.15, ranges=[]) == ""
