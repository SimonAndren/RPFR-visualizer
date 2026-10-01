"""Unit tests for isotopic fingerprint explorer ranking."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.analysis.fingerprint_explorer import (
    CONDITION_PRESETS,
    group_by_environment,
    rank_sites,
    request_from_preset,
)
from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.io.providers import HDF5Provider


def test_request_from_preset_room_temp() -> None:
    request = request_from_preset("room_temp_epsilon")
    assert request.kind == AnalysisKind.EPSILON
    assert abs(request.temperature - 298.15) < 1e-6
    assert request.reference is not None


def test_request_from_preset_unknown() -> None:
    with pytest.raises(KeyError, match="Unknown preset"):
        request_from_preset("not_a_preset")


def test_rank_sites_orders_by_abs(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [
        ("qm9s", "1", provider.get_molecule(1)),
        ("qm9s", "2", provider.get_molecule(2)),
    ]
    request = request_from_preset("cold_beta")
    ranks = rank_sites(mols, request, top_n=10)
    assert ranks
    abs_values = [r.abs_value for r in ranks]
    assert abs_values == sorted(abs_values, reverse=True)


def test_rank_sites_element_filter(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [("qm9s", "1", provider.get_molecule(1))]
    request = request_from_preset("cold_beta", element="C")
    ranks = rank_sites(mols, request, top_n=20, element="C")
    assert ranks
    assert all(r.element == "C" for r in ranks)


def test_group_by_environment(synthetic_h5_path: Path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [("qm9s", "1", provider.get_molecule(1))]
    request = request_from_preset("cold_beta")
    ranks = rank_sites(mols, request, top_n=50)
    groups = group_by_environment(ranks, shell=0)
    assert groups
    assert groups[0].mean_abs_value >= groups[-1].mean_abs_value


def test_presets_defined() -> None:
    assert "room_temp_epsilon" in CONDITION_PRESETS
    assert "cold_beta" in CONDITION_PRESETS
    assert "vs_standard" in CONDITION_PRESETS
