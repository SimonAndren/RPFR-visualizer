"""Tests for qc/thresholds.py."""

import json

import h5py

from rpfr_visualizer.qc.thresholds import DEFAULT_THRESHOLDS, QcThresholds, load_thresholds


def test_default_thresholds():
    thresholds = QcThresholds.from_dict(DEFAULT_THRESHOLDS)
    assert thresholds.tr_residual_pass == 10.0
    assert thresholds.tr_residual_fail == 50.0
    assert thresholds.qc_spec_version == "1.0"


def test_load_thresholds_from_hdf5(tmp_path):
    path = tmp_path / "thresholds.h5"
    override = {
        "tr_residual_cm1": {"pass": 5, "fail": 25},
        "qc_spec_version": "1.1",
    }
    with h5py.File(path, "w") as h5:
        h5.attrs["qc_thresholds"] = json.dumps(override)

    with h5py.File(path, "r") as h5:
        thresholds = load_thresholds(h5)

    assert thresholds.tr_residual_pass == 5.0
    assert thresholds.tr_residual_fail == 25.0
    assert thresholds.qc_spec_version == "1.1"
    assert thresholds.source_mae_pass == 1.0
