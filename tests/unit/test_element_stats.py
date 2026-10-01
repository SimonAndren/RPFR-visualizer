"""Tests for ElementStatsCache and dataset-median standards."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.analysis.element_stats import ElementStatsCache
from rpfr_visualizer.analysis.fingerprint_explorer import request_from_preset
from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.analysis.reference import StandardReference
from rpfr_visualizer.io.providers import HDF5Provider
from rpfr_visualizer.io.schema import TEMPERATURES_K


def test_stats_from_molecules(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [provider.get_molecule(mid) for mid in provider.list_ids()]
    cache = ElementStatsCache.from_molecules(mols, temperatures=TEMPERATURES_K)
    assert cache.n_sites > 0
    assert "C" in cache.elements() or "H" in cache.elements()
    element = cache.elements()[0]
    med = cache.median_beta(element, 300.0)
    assert med is not None and med > 0
    rng = cache.quantity_range(element, 300.0, kind="beta")
    assert rng is not None
    lo, hi = rng
    assert lo <= med <= hi


def test_stats_epsilon_range(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [provider.get_molecule(mid) for mid in provider.list_ids()]
    cache = ElementStatsCache.from_molecules(mols)
    element = cache.elements()[0]
    pct = cache.percentiles(element, 300.0, kind="epsilon")
    assert pct is not None
    assert pct.p5 <= pct.p50 <= pct.p95
    # Median site vs median → ε ≈ 0
    assert abs(pct.p50) < 1e-6


def test_stats_off_grid_median(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [provider.get_molecule(mid) for mid in provider.list_ids()]
    cache = ElementStatsCache.from_molecules(mols)
    element = cache.elements()[0]
    med = cache.median_beta(element, 250.0, warn=False)
    assert med is not None and med > 0


def test_stats_from_hdf5(synthetic_h5_path: Path) -> None:
    cache = ElementStatsCache.from_hdf5(synthetic_h5_path)
    assert cache.n_sites > 0
    assert cache.elements()


def test_standard_reference_uses_median(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [provider.get_molecule(mid) for mid in provider.list_ids()]
    cache = ElementStatsCache.from_molecules(mols)
    element = cache.elements()[0]
    ref = StandardReference(stats=cache)
    mol = mols[0]
    resolved = ref.resolve_beta(
        target_molecule=mol,
        target_atom_index=0,
        element=element,
        temperature=300.0,
    )
    assert resolved == pytest.approx(cache.median_beta(element, 300.0))  # type: ignore[arg-type]


def test_standard_reference_override(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [provider.get_molecule(mid) for mid in provider.list_ids()]
    cache = ElementStatsCache.from_molecules(mols)
    element = cache.elements()[0]
    ref = StandardReference(stats=cache).with_override(element, 1.234)
    mol = mols[0]
    assert (
        ref.resolve_beta(
            target_molecule=mol,
            target_atom_index=0,
            element=element,
            temperature=300.0,
        )
        == 1.234
    )
    cleared = ref.without_override(element)
    assert cleared.resolve_beta(
        target_molecule=mol,
        target_atom_index=0,
        element=element,
        temperature=300.0,
    ) == pytest.approx(cache.median_beta(element, 300.0))  # type: ignore[arg-type]


def test_vs_standard_preset_with_stats(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [provider.get_molecule(mid) for mid in provider.list_ids()]
    cache = ElementStatsCache.from_molecules(mols)
    request = request_from_preset("vs_standard", stats=cache, element="C")
    assert request.kind == AnalysisKind.EPSILON
    assert isinstance(request.reference, StandardReference)
    assert request.reference.stats is cache
