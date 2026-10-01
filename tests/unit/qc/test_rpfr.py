"""Tests for qc/checks/rpfr.py."""

import h5py
import numpy as np

from rpfr_visualizer.qc.checks.rpfr import run_rpfr_checks
from rpfr_visualizer.qc.thresholds import QcThresholds


def test_clumping_factor_passes_near_unity(tmp_path):
    path = tmp_path / "clump.h5"
    thresholds = QcThresholds.from_dict({})
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.attrs["rpfr_qc_ok"] = True
        clumping = np.ones((3, 3, 2), dtype=np.float64)
        clumping[0, 1, :] = 1.001
        clumping[1, 0, :] = 1.001
        np.fill_diagonal(clumping[:, :, 0], 0.0)
        np.fill_diagonal(clumping[:, :, 1], 0.0)
        group.create_dataset("clumping", data=clumping)

    with h5py.File(path, "r") as h5:
        checks = run_rpfr_checks(h5["1"], thresholds=thresholds, temperatures=(300.0,))

    factor = next(check for check in checks if check.check_id == "rpfr.clumping_factor")
    assert factor.status == "pass"


def test_clumping_factor_passes_elevated_values(tmp_path):
    path = tmp_path / "clump_elevated.h5"
    thresholds = QcThresholds.from_dict({})
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.attrs["rpfr_qc_ok"] = True
        clumping = np.ones((3, 3, 1), dtype=np.float64)
        clumping[0, 1, 0] = 2.0
        clumping[1, 0, 0] = 2.0
        np.fill_diagonal(clumping[:, :, 0], 0.0)
        group.create_dataset("clumping", data=clumping)

    with h5py.File(path, "r") as h5:
        checks = run_rpfr_checks(h5["1"], thresholds=thresholds, temperatures=(300.0,))

    factor = next(check for check in checks if check.check_id == "rpfr.clumping_factor")
    assert factor.status == "pass"


def test_clumping_factor_warns_on_large_values(tmp_path):
    path = tmp_path / "clump_warn.h5"
    thresholds = QcThresholds.from_dict({})
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.attrs["rpfr_qc_ok"] = True
        clumping = np.ones((3, 3, 1), dtype=np.float64)
        clumping[0, 1, 0] = 4.0
        clumping[1, 0, 0] = 4.0
        np.fill_diagonal(clumping[:, :, 0], 0.0)
        group.create_dataset("clumping", data=clumping)

    with h5py.File(path, "r") as h5:
        checks = run_rpfr_checks(h5["1"], thresholds=thresholds, temperatures=(300.0,))

    factor = next(check for check in checks if check.check_id == "rpfr.clumping_factor")
    assert factor.status == "warn"


def test_clumping_factor_tolerates_float_noise_below_one(tmp_path):
    path = tmp_path / "clump_noise.h5"
    thresholds = QcThresholds.from_dict({})
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.attrs["rpfr_qc_ok"] = True
        clumping = np.ones((3, 3, 1), dtype=np.float64)
        clumping[0, 1, 0] = 0.99999995
        clumping[1, 0, 0] = 0.99999995
        np.fill_diagonal(clumping[:, :, 0], 0.0)
        group.create_dataset("clumping", data=clumping)

    with h5py.File(path, "r") as h5:
        checks = run_rpfr_checks(h5["1"], thresholds=thresholds, temperatures=(300.0,))

    factor = next(check for check in checks if check.check_id == "rpfr.clumping_factor")
    assert factor.status == "pass"


def test_clumping_factor_ignores_low_temperature_noise(tmp_path):
    path = tmp_path / "clump_low_t_noise.h5"
    thresholds = QcThresholds.from_dict({})
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.attrs["rpfr_qc_ok"] = True
        clumping = np.ones((3, 3, 2), dtype=np.float64)
        clumping[0, 1, 0] = 0.999997
        clumping[1, 0, 0] = 0.999997
        clumping[0, 1, 1] = 1.001
        clumping[1, 0, 1] = 1.001
        np.fill_diagonal(clumping[:, :, 0], 0.0)
        np.fill_diagonal(clumping[:, :, 1], 0.0)
        group.create_dataset("clumping", data=clumping)

    with h5py.File(path, "r") as h5:
        checks = run_rpfr_checks(
            h5["1"],
            thresholds=thresholds,
            temperatures=(100.0, 300.0),
        )

    factor = next(check for check in checks if check.check_id == "rpfr.clumping_factor")
    assert factor.status == "pass"
    assert factor.value["temperature_K"] == 300.0


def test_clumping_factor_fails_below_one(tmp_path):
    path = tmp_path / "clump_bad.h5"
    thresholds = QcThresholds.from_dict({})
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1")
        group.attrs["rpfr_qc_ok"] = True
        clumping = np.ones((3, 3, 1), dtype=np.float64)
        clumping[0, 1, 0] = 0.99
        clumping[1, 0, 0] = 0.99
        np.fill_diagonal(clumping[:, :, 0], 0.0)
        group.create_dataset("clumping", data=clumping)

    with h5py.File(path, "r") as h5:
        checks = run_rpfr_checks(h5["1"], thresholds=thresholds, temperatures=(300.0,))

    factor = next(check for check in checks if check.check_id == "rpfr.clumping_factor")
    assert factor.status == "fail"
