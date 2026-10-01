"""Tests for qc/fallback.py."""

import h5py
import numpy as np

from rpfr_visualizer.qc.fallback import (
    count_soft_modes,
    source_regression_metrics,
    worst_imag_vib_cm1,
)
from rpfr_visualizer.qc.thresholds import QcThresholds


def test_worst_imag_vib_cm1_computed(tmp_path):
    path = tmp_path / "freq.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.create_dataset("all_frequencies", data=np.array([100.0, -12.5, 50.0]))

    with h5py.File(path, "r") as h5:
        assert worst_imag_vib_cm1(h5["1"]) == -12.5


def test_count_soft_modes_computed(tmp_path):
    path = tmp_path / "soft.h5"
    thresholds = QcThresholds.from_dict({})
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.create_dataset("harmonic_frequencies", data=np.array([100.0, 25.0, 5.0, 60.0]))

    with h5py.File(path, "r") as h5:
        assert count_soft_modes(h5["1"], thresholds=thresholds) == 2


def test_source_regression_unavailable(tmp_path):
    path = tmp_path / "qm9s.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.create_dataset("harmonic_frequencies", data=np.array([100.0, 200.0]))

    with h5py.File(path, "r") as h5:
        mae, max_err, lengths_match = source_regression_metrics(h5["1"])
        assert mae is None
        assert max_err is None
        assert lengths_match is True
