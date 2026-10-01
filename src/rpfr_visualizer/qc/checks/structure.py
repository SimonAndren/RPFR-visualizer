"""Structure-related QC checks."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from rpfr_visualizer.qc.models import CheckResult
from rpfr_visualizer.qc.thresholds import QcThresholds

REQUIRED_DATASETS = (
    "rpfr_single",
    "clumping",
    "mass_law",
    "harmonic_frequencies",
    "atom_symbol",
    "coordinates",
)


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


def run_structure_checks(
    group: h5py.Group,
    *,
    root: h5py.File,
    n_temps: int,
    thresholds: QcThresholds,
) -> list[CheckResult]:
    """Run all structure.* checks."""
    del thresholds
    checks: list[CheckResult] = []

    schema_version = str(_read_attr(root, "schema_version", "0.0"))
    try:
        major = float(schema_version.split(".", maxsplit=1)[0])
        schema_ok = major >= 2.0
    except ValueError:
        schema_ok = False
    checks.append(
        CheckResult(
            check_id="structure.schema_version",
            category="structure",
            status="pass" if schema_ok else "fail",
            value=schema_version,
            threshold=">= 2.0",
            message="Schema version meets v2.0 requirement"
            if schema_ok
            else "Schema version below 2.0",
            source="hdf5_attr",
        )
    )

    missing = [name for name in REQUIRED_DATASETS if name not in group]
    checks.append(
        CheckResult(
            check_id="structure.required_datasets",
            category="structure",
            status="pass" if not missing else "fail",
            value=",".join(missing) if missing else "all present",
            threshold="core arrays present",
            message="All required datasets present"
            if not missing
            else f"Missing datasets: {', '.join(missing)}",
            source="hdf5_attr",
        )
    )

    shape_issues: list[str] = []
    n_atoms = int(_read_attr(group, "n_atoms", 0))
    if n_atoms <= 0 and "atom_symbol" in group:
        n_atoms = len(group["atom_symbol"])

    if "rpfr_single" in group:
        rpfr = group["rpfr_single"]
        if rpfr.shape != (n_atoms, n_temps):
            shape_issues.append(f"rpfr_single shape {rpfr.shape} != ({n_atoms}, {n_temps})")
    if "clumping" in group:
        clump = group["clumping"]
        if clump.shape[:2] != (n_atoms, n_atoms) or clump.shape[2] != n_temps:
            shape_issues.append(f"clumping shape {clump.shape} inconsistent with N={n_atoms}")
    if "harmonic_frequencies" in group:
        harmonic = group["harmonic_frequencies"]
        is_linear = bool(_read_attr(group, "is_linear", False))
        expected = 3 * n_atoms - (5 if is_linear else 6)
        if len(harmonic) != expected:
            shape_issues.append(
                f"harmonic_frequencies length {len(harmonic)} != expected {expected}"
            )

    checks.append(
        CheckResult(
            check_id="structure.shape_consistency",
            category="structure",
            status="pass" if not shape_issues else "fail",
            value="; ".join(shape_issues) if shape_issues else "consistent",
            threshold="N/T dimensions match",
            message="Array shapes are consistent" if not shape_issues else "; ".join(shape_issues),
            source="computed",
        )
    )

    non_finite: list[str] = []
    for name in ("rpfr_single", "clumping", "mass_law", "coordinates"):
        if name not in group:
            continue
        arr = np.asarray(group[name][()])
        if not np.all(np.isfinite(arr)):
            non_finite.append(name)

    checks.append(
        CheckResult(
            check_id="structure.finite_arrays",
            category="structure",
            status="pass" if not non_finite else "fail",
            value=",".join(non_finite) if non_finite else "all finite",
            threshold="no NaN/Inf",
            message="All core arrays are finite"
            if not non_finite
            else f"Non-finite values in: {', '.join(non_finite)}",
            source="computed",
        )
    )

    return checks
