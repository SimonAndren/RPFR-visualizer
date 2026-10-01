"""Tests for EnrichmentView fill bounds and H toggle."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.analysis.element_stats import ElementStatsCache
from rpfr_visualizer.analysis.fractionation import compute_fractionation
from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.io.providers import HDF5Provider
from rpfr_visualizer.viz.electron_cloud import EnrichmentView


def test_enrichment_view_uses_stats_bounds(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    cache = ElementStatsCache.from_molecules([mol, provider.get_molecule(2)])
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result, stats=cache)
    element = next(s.element for s in mol.sites if s.element != "H")
    caption = view.range_caption_for(element)
    assert element in caption
    assert "5th–95th percentile" in caption
    vmin, vmax = view._fill_bounds(element, view._visible_indices(element))
    expected = cache.quantity_range(element, 300.0, kind="beta")
    assert expected is not None
    assert (vmin, vmax) == expected


def test_caption_is_available_without_building_a_viewer(synthetic_h5_path: Path) -> None:
    """The colour-scale caption must not require a 3D viewer to exist.

    ``caption`` is only assigned while spheres are added, so anything reading it
    before a render got an empty string.
    """
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result)
    element = next(s.element for s in mol.sites if s.element != "H")

    assert view.caption == ""  # nothing rendered yet
    caption = view.caption_for(element)
    assert element in caption
    assert "5th–95th percentile" in caption


def test_caption_for_all_elements_joins_every_focus_element(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result)

    caption = view.caption_for(None)
    elements = {s.element for s in mol.sites}
    for element in elements:
        assert f"{element} " in caption
    # One sentence, one scale per element -- not one sentence per element.
    assert caption.count("Opacity") == 1
    assert caption.count(" · ") == len(elements) - 1


def test_caption_for_unrepresented_element_is_empty(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result)
    assert "Xe" not in {s.element for s in mol.sites}
    assert view.caption_for("Xe") == ""


def test_hide_hydrogens_filters_indices(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result, show_hydrogens=False)
    indices = view._visible_indices(None)
    assert indices
    assert all(view.symbols[i] != "H" for i in indices)


def test_3d_labels_carry_the_value_as_well_as_the_identity(synthetic_h5_path: Path) -> None:
    """Unlike the 2D skeleton, the 3D scene has room for the number.

    Opacity ranks the sites but never states one, so without this a reader has
    to cross-reference the site table to find out what they are looking at.
    The 2D depiction cannot afford the same treatment: there the number sits
    beside a bare vertex and the structure is what gets crowded out.
    """
    import re

    pytest.importorskip("py3Dmol")

    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result)

    viewer = view.build_viewer(show_labels=True)
    # The bundled 3Dmol source also contains the string "addLabel", so match the
    # generated call specifically rather than the word.
    calls = re.findall(r'\.addLabel\("([^"]*)"', getattr(viewer, "startjs", "") or "")

    assert calls, "no labels were drawn"
    for text in calls:
        assert re.fullmatch(r"[A-Z][a-z]?\d+(\\n-?[0-9.eE+-]+)?", text), (
            f"label {text!r} is neither an identity nor an identity plus a value"
        )
    assert any("\\n" in text for text in calls), "no label carried a value"


def test_a_3d_site_with_no_value_still_gets_its_name(synthetic_h5_path: Path) -> None:
    """\"Which atom is this?\" is a question a valueless site still has."""
    import re

    import numpy as np

    pytest.importorskip("py3Dmol")

    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result)
    view.values = np.full_like(view.values, np.nan)

    viewer = view.build_viewer(show_labels=True)
    calls = re.findall(r'\.addLabel\("([^"]*)"', getattr(viewer, "startjs", "") or "")

    assert calls, "no labels were drawn"
    assert all(re.fullmatch(r"[A-Z][a-z]?\d+", text) for text in calls), calls


def test_3d_labels_can_still_be_turned_off(synthetic_h5_path: Path) -> None:
    import re

    pytest.importorskip("py3Dmol")

    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(mol, kind=AnalysisKind.BETA, temperature=300.0)
    view = EnrichmentView.from_analysis(mol, result)

    viewer = view.build_viewer(show_labels=False)
    calls = re.findall(r'\.addLabel\("([^"]*)"', getattr(viewer, "startjs", "") or "")

    assert calls == []


def test_a_1000_ln_alpha_view_is_labelled_as_itself(synthetic_h5_path: Path) -> None:
    """1000 ln α borrows epsilon's colour scale, and was named after it.

    Picking 1000 ln α put "ε" in the caption and "epsilon" over the site table
    -- the quantity the dropdown no longer even offers.
    """
    import numpy as np

    from rpfr_visualizer.analysis.reference import ElementMeanReference

    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    result = compute_fractionation(
        mol,
        kind=AnalysisKind.LN_ALPHA,
        temperature=300.0,
        reference=ElementMeanReference(),
    )
    view = EnrichmentView.from_analysis(mol, result)
    assert view.display_kind == "ln_alpha"
    assert view.quantity_kind == "epsilon"  # the scale it borrows is unchanged

    caption = view.caption_for("C")
    assert "1000 ln α" in caption
    assert "ε" not in caption

    from rpfr_visualizer.viz.depict import format_site_value_table

    table = format_site_value_table(mol, view.values, quantity_kind=view.display_kind)
    assert "1000 ln α" in table.splitlines()[0]
    assert "epsilon" not in table
    assert np.isfinite(view.values).any()


def test_1000_ln_alpha_percentiles_are_converted_not_borrowed() -> None:
    """The cache stores epsilon's percentiles. 1000 ln(1 + ε/1000) is
    monotonic, so each one converts exactly -- stating epsilon's numbers as
    1000 ln α's would have been wrong by the difference between the two."""
    import numpy as np

    class _Stats:
        def quantity_range(self, element, temperature, *, kind, warn, method=None):
            return (-15.89, 24.02)

    class _Mol:
        molecule_id = 1
        n_atoms = 0
        sites = []

    view = EnrichmentView(
        _Mol(), [], stats=_Stats(), quantity_kind="epsilon", display_kind="ln_alpha"
    )
    low, high = view._fill_bounds("C", [])
    assert low == pytest.approx(1000 * np.log1p(-15.89 / 1000))
    assert high == pytest.approx(1000 * np.log1p(24.02 / 1000))
    assert (low, high) != (-15.89, 24.02)
