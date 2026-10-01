"""Tests for temperature-grid beta interpolation."""

from __future__ import annotations

import math
import warnings

import pytest

from rpfr_visualizer.analysis.temperature import (
    T_MAX_K,
    T_MIN_K,
    beta_at_temperature,
    interpolate_beta,
    is_on_grid,
)
from rpfr_visualizer.io.schema import TEMPERATURES_K, SiteRecord


def _synthetic_betas(temperatures: tuple[float, ...]) -> tuple[float, ...]:
    """Smooth, positive β(T) suitable for ln(β) vs 1/T fitting."""
    return tuple(math.exp(0.05 + 30.0 / t) for t in temperatures)


def test_on_grid_returns_exact_value():
    temps = TEMPERATURES_K
    betas = _synthetic_betas(temps)
    for t, b in zip(temps, betas, strict=True):
        result = interpolate_beta(temps, betas, t)
        assert result.interpolated is False
        assert result.value == pytest.approx(b)
        assert is_on_grid(t, temps)


def test_off_grid_interpolates_between_neighbors():
    temps = TEMPERATURES_K
    betas = _synthetic_betas(temps)
    t_mid = 250.0
    result = interpolate_beta(temps, betas, t_mid)
    assert result.interpolated is True
    assert result.out_of_range is False
    # Synthetic truth: exp(0.05 + 30/T)
    expected = math.exp(0.05 + 30.0 / t_mid)
    assert result.value == pytest.approx(expected, rel=1e-3)


def test_out_of_range_warns_and_flags():
    temps = TEMPERATURES_K
    betas = _synthetic_betas(temps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = interpolate_beta(temps, betas, 10.0, warn=True)
    assert result.out_of_range is True
    assert result.interpolated is True
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert T_MIN_K == 20.0
    assert T_MAX_K == 420.0


def test_out_of_range_can_suppress_warning():
    temps = TEMPERATURES_K
    betas = _synthetic_betas(temps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beta_at_temperature(temps, betas, 500.0, warn=False)
    assert caught == []


def test_site_record_beta_at_off_grid():
    betas = _synthetic_betas(TEMPERATURES_K)
    site = SiteRecord(
        molecule_id=1,
        atom_index=0,
        element="C",
        beta=betas,
        temperatures=TEMPERATURES_K,
    )
    value = site.beta_at(250.0, warn=False)
    assert value == pytest.approx(math.exp(0.05 + 30.0 / 250.0), rel=1e-3)


def test_site_record_beta_at_on_grid_unchanged():
    site = SiteRecord(
        molecule_id=1,
        atom_index=0,
        element="C",
        beta=tuple(1.0 + 0.01 * i for i in range(len(TEMPERATURES_K))),
        temperatures=TEMPERATURES_K,
    )
    idx = list(TEMPERATURES_K).index(300.0)
    assert site.beta_at(300.0) == site.beta[idx]


def test_polyfit_handles_unsorted_grid():
    """273.15 / 298.15 sit after 420 K in the canonical grid."""
    temps = TEMPERATURES_K
    betas = _synthetic_betas(temps)
    # Ensure grid is not sorted by T
    assert temps[-1] < temps[-3]
    value = beta_at_temperature(temps, betas, 280.0, warn=False)
    assert value == pytest.approx(math.exp(0.05 + 30.0 / 280.0), rel=1e-3)
    # Fit should still recover on-grid points exactly via lookup
    assert beta_at_temperature(temps, betas, 298.15) == betas[temps.index(298.15)]


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="length mismatch"):
        interpolate_beta((20.0, 300.0), (1.1,), 100.0)
