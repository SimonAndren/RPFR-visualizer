"""Compute QC metrics from HDF5 arrays when v2.1 attrs are missing."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from rpfr_visualizer.qc.thresholds import QcThresholds


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return _decode_scalar(value.item())
    return value


def _read_attr(group: h5py.Group | h5py.File, name: str, default: Any = None) -> Any:
    if name not in group.attrs:
        return default
    return _decode_scalar(group.attrs[name])


def worst_imag_vib_cm1(group: h5py.Group) -> float | None:
    """Return the most negative frequency in all_frequencies, or None if none."""
    persisted = _read_attr(group, "worst_imag_vib_cm1")
    if persisted is not None:
        return float(persisted)

    if "all_frequencies" not in group:
        return None

    freqs = np.asarray(group["all_frequencies"][()], dtype=np.float64)
    negative = freqs[freqs < 0]
    if negative.size == 0:
        return None
    return float(np.min(negative))


def count_soft_modes(group: h5py.Group, *, thresholds: QcThresholds) -> int | None:
    """Count harmonic modes with 0 < nu <= soft_modes_upper."""
    persisted = _read_attr(group, "n_soft_modes")
    if persisted is not None:
        return int(persisted)

    if "harmonic_frequencies" not in group:
        return None

    freqs = np.asarray(group["harmonic_frequencies"][()], dtype=np.float64)
    upper = thresholds.soft_modes_upper
    return int(np.sum((freqs > 0) & (freqs <= upper)))


def source_regression_metrics(group: h5py.Group) -> tuple[float | None, float | None, bool]:
    """Return (mae, max_error, lengths_match) for source regression checks."""
    mae = _read_attr(group, "source_mae_cm1")
    max_err = _read_attr(group, "source_max_cm1")
    if mae is not None and max_err is not None:
        return float(mae), float(max_err), True

    if "source_nsi_frequencies" not in group or "harmonic_frequencies" not in group:
        return None, None, True

    source = np.asarray(group["source_nsi_frequencies"][()], dtype=np.float64)
    harmonic = np.asarray(group["harmonic_frequencies"][()], dtype=np.float64)
    if source.shape != harmonic.shape:
        return None, None, False

    errors = np.abs(source - harmonic)
    return float(np.mean(errors)), float(np.max(errors)), True
