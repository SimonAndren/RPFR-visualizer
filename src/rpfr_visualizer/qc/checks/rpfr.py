"""RPFR-related QC checks."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from rpfr_visualizer.io.schema import TEMPERATURES_K
from rpfr_visualizer.qc.models import CheckResult
from rpfr_visualizer.qc.thresholds import QcThresholds


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _read_attr(group: h5py.Group, name: str, default: Any = None) -> Any:
    if name not in group.attrs:
        return default
    return _decode_scalar(group.attrs[name])


def _reference_temp_index(temperatures: tuple[float, ...], reference_k: float) -> int:
    """Return the grid index for a reference temperature in Kelvin."""
    for idx, temp in enumerate(temperatures):
        if abs(temp - reference_k) < 1e-6:
            return idx
    available = ", ".join(str(t) for t in temperatures)
    raise ValueError(
        f"Reference temperature {reference_k} K is not on the grid. Available: [{available}]"
    )


def _off_diagonal_clumping_at_temp(clumping: np.ndarray, temp_index: int) -> np.ndarray:
    """Return off-diagonal clumping factors at one temperature index."""
    n_atoms = clumping.shape[0]
    if n_atoms < 2:
        return np.array([], dtype=np.float64)
    mask = ~np.eye(n_atoms, dtype=bool)
    return clumping[mask, temp_index]


def _clumping_factor_status(
    off_diag: np.ndarray,
    *,
    thresholds: QcThresholds,
    temperature_k: float,
) -> tuple[str, float, float, float, str]:
    """Evaluate multiplicative clumping factors C (Delta = ln C) at one temperature."""
    if off_diag.size == 0:
        return "info", 1.0, 1.0, 0.0, f"No off-diagonal clumping pairs at {temperature_k:g} K"

    min_factor = float(np.min(off_diag))
    max_factor = float(np.max(off_diag))
    max_delta = float(np.log(max_factor))
    min_allowed = thresholds.clumping_min_factor - thresholds.clumping_below_one_tol

    if min_factor < min_allowed:
        return (
            "fail",
            min_factor,
            max_factor,
            max_delta,
            (
                f"Clumping factor below 1 at {temperature_k:g} K "
                f"(min C={min_factor:.8g}); Delta=ln(C) undefined"
            ),
        )
    if max_factor > thresholds.clumping_fail_max_factor:
        return (
            "fail",
            min_factor,
            max_factor,
            max_delta,
            (
                f"Extreme clumping factor at {temperature_k:g} K "
                f"max C={max_factor:.4f} (Delta={max_delta:.4f})"
            ),
        )
    if max_factor > thresholds.clumping_warn_max_factor:
        return (
            "warn",
            min_factor,
            max_factor,
            max_delta,
            (
                f"Large clumping factor at {temperature_k:g} K "
                f"max C={max_factor:.4f} (Delta={max_delta:.4f})"
            ),
        )
    return (
        "pass",
        min_factor,
        max_factor,
        max_delta,
        (
            f"Clumping factors physical at {temperature_k:g} K "
            f"(min C={min_factor:.6g}, max C={max_factor:.4f}, Delta={max_delta:.4f})"
        ),
    )


def run_rpfr_checks(
    group: h5py.Group,
    *,
    thresholds: QcThresholds,
    temperatures: tuple[float, ...] = TEMPERATURES_K,
) -> list[CheckResult]:
    """Run all rpfr.* checks."""
    checks: list[CheckResult] = []

    rpfr_qc_ok = bool(_read_attr(group, "rpfr_qc_ok", True))
    checks.append(
        CheckResult(
            check_id="rpfr.finite_positive",
            category="rpfr",
            status="pass" if rpfr_qc_ok else "fail",
            value=rpfr_qc_ok,
            threshold="True",
            message="RPFR values finite and positive" if rpfr_qc_ok else "RPFR QC failed",
            source="hdf5_attr",
        )
    )

    if "clumping" not in group:
        checks.append(
            CheckResult(
                check_id="rpfr.clumping_symmetry",
                category="rpfr",
                status="info",
                value=None,
                threshold="symmetric, zero diagonal",
                message="Clumping array unavailable",
                source="hdf5_attr",
            )
        )
        checks.append(
            CheckResult(
                check_id="rpfr.clumping_factor",
                category="rpfr",
                status="info",
                value=None,
                threshold="off-diagonal C >= 1; Delta=ln(C)",
                message="Clumping array unavailable",
                source="hdf5_attr",
            )
        )
        return checks

    clumping = np.asarray(group["clumping"][()], dtype=np.float64)
    n_atoms = clumping.shape[0]
    n_temps = clumping.shape[2]
    grid_temps = temperatures[:n_temps]
    ref_k = thresholds.clumping_reference_temperature_k
    ref_index = _reference_temp_index(grid_temps, ref_k)

    diag_max = float(np.max(np.abs(np.diagonal(clumping[:, :, 0])))) if n_atoms else 0.0
    symmetry_max = float(np.max(np.abs(clumping - np.transpose(clumping, (1, 0, 2)))))
    symmetry_ok = diag_max == 0.0 and symmetry_max < 1e-8
    checks.append(
        CheckResult(
            check_id="rpfr.clumping_symmetry",
            category="rpfr",
            status="pass" if symmetry_ok else "fail",
            value={"diag_max": diag_max, "symmetry_max": symmetry_max},
            threshold="symmetric, zero diagonal",
            message="Clumping matrix is symmetric with zero diagonal"
            if symmetry_ok
            else "Clumping matrix symmetry or diagonal check failed",
            source="computed",
        )
    )

    off_diag = _off_diagonal_clumping_at_temp(clumping, ref_index)
    status, min_factor, max_factor, max_delta, message = _clumping_factor_status(
        off_diag,
        thresholds=thresholds,
        temperature_k=ref_k,
    )
    checks.append(
        CheckResult(
            check_id="rpfr.clumping_factor",
            category="rpfr",
            status=status,
            value={
                "temperature_K": ref_k,
                "min_factor": min_factor,
                "max_factor": max_factor,
                "max_delta": max_delta,
            },
            threshold=(
                f"at {ref_k:g} K: C >= {thresholds.clumping_min_factor}; "
                f"warn max C > {thresholds.clumping_warn_max_factor}, "
                f"fail max C > {thresholds.clumping_fail_max_factor}"
            ),
            message=message,
            source="computed",
        )
    )

    return checks
