"""Default and file-overridden QC thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import h5py

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "qc_spec_version": "1.0",
    "tr_residual_cm1": {"pass": 10, "fail": 50},
    "imaginary_vib_cm1": {"pass": 1.0, "info": 10, "warn": 50},
    "soft_modes_cm1": {
        "upper": 50,
        "warn_count": 1,
        "fail_count": 3,
        "fail_single": 10,
    },
    "source_regression_cm1": {
        "mae_pass": 1.0,
        "mae_warn": 3.0,
        "max_pass": 5.0,
        "max_warn": 20.0,
    },
    "rpfr_clumping": {
        "min_factor": 1.0,
        "below_one_tol": 1e-4,
        "reference_temperature_K": 300.0,
        "warn_max_factor": 3.0,
        "fail_max_factor": 10.0,
    },
}


@dataclass(frozen=True)
class QcThresholds:
    """Resolved QC thresholds for a dataset scan."""

    qc_spec_version: str = "1.0"
    tr_residual_pass: float = 10.0
    tr_residual_fail: float = 50.0
    imaginary_vib_pass: float = 1.0
    imaginary_vib_info: float = 10.0
    imaginary_vib_warn: float = 50.0
    soft_modes_upper: float = 50.0
    soft_modes_warn_count: int = 1
    soft_modes_fail_count: int = 3
    soft_modes_fail_single: float = 10.0
    source_mae_pass: float = 1.0
    source_mae_warn: float = 3.0
    source_max_pass: float = 5.0
    source_max_warn: float = 20.0
    clumping_min_factor: float = 1.0
    clumping_below_one_tol: float = 1e-4
    clumping_reference_temperature_k: float = 300.0
    clumping_warn_max_factor: float = 3.0
    clumping_fail_max_factor: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical threshold JSON shape."""
        return {
            "qc_spec_version": self.qc_spec_version,
            "tr_residual_cm1": {"pass": self.tr_residual_pass, "fail": self.tr_residual_fail},
            "imaginary_vib_cm1": {
                "pass": self.imaginary_vib_pass,
                "info": self.imaginary_vib_info,
                "warn": self.imaginary_vib_warn,
            },
            "soft_modes_cm1": {
                "upper": self.soft_modes_upper,
                "warn_count": self.soft_modes_warn_count,
                "fail_count": self.soft_modes_fail_count,
                "fail_single": self.soft_modes_fail_single,
            },
            "source_regression_cm1": {
                "mae_pass": self.source_mae_pass,
                "mae_warn": self.source_mae_warn,
                "max_pass": self.source_max_pass,
                "max_warn": self.source_max_warn,
            },
            "rpfr_clumping": {
                "min_factor": self.clumping_min_factor,
                "below_one_tol": self.clumping_below_one_tol,
                "reference_temperature_K": self.clumping_reference_temperature_k,
                "warn_max_factor": self.clumping_warn_max_factor,
                "fail_max_factor": self.clumping_fail_max_factor,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QcThresholds:
        """Build thresholds from a merged dictionary."""
        merged = _deep_merge(DEFAULT_THRESHOLDS, data)
        tr = merged["tr_residual_cm1"]
        imag = merged["imaginary_vib_cm1"]
        soft = merged["soft_modes_cm1"]
        src = merged["source_regression_cm1"]
        clump = merged["rpfr_clumping"]
        return cls(
            qc_spec_version=str(merged["qc_spec_version"]),
            tr_residual_pass=float(tr["pass"]),
            tr_residual_fail=float(tr["fail"]),
            imaginary_vib_pass=float(imag.get("pass", 1.0)),
            imaginary_vib_info=float(imag["info"]),
            imaginary_vib_warn=float(imag["warn"]),
            soft_modes_upper=float(soft["upper"]),
            soft_modes_warn_count=int(soft["warn_count"]),
            soft_modes_fail_count=int(soft["fail_count"]),
            soft_modes_fail_single=float(soft["fail_single"]),
            source_mae_pass=float(src["mae_pass"]),
            source_mae_warn=float(src["mae_warn"]),
            source_max_pass=float(src["max_pass"]),
            source_max_warn=float(src["max_warn"]),
            clumping_min_factor=float(clump["min_factor"]),
            clumping_below_one_tol=float(clump["below_one_tol"]),
            clumping_reference_temperature_k=float(clump.get("reference_temperature_K", 300.0)),
            clumping_warn_max_factor=float(clump["warn_max_factor"]),
            clumping_fail_max_factor=float(clump["fail_max_factor"]),
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _decode_attr(value: Any) -> Any:
    """Decode HDF5 attribute values."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def load_thresholds(h5_file: h5py.File) -> QcThresholds:
    """Load thresholds from HDF5 root attrs or fall back to defaults."""
    raw = h5_file.attrs.get("qc_thresholds")
    if raw is None:
        return QcThresholds.from_dict(DEFAULT_THRESHOLDS)
    text = _decode_attr(raw)
    if isinstance(text, str):
        data = json.loads(text)
    else:
        data = dict(text)
    return QcThresholds.from_dict(data)
