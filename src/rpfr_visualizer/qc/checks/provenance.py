"""Provenance-related QC checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py

from rpfr_visualizer.qc.models import CheckResult


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _read_attr(group: h5py.Group | h5py.File, name: str, default: Any = None) -> Any:
    if name not in group.attrs:
        return default
    return _decode_scalar(group.attrs[name])


def _load_errors_sidecar(path: Path) -> dict[str, Any] | None:
    errors_path = path.with_suffix(".errors.json")
    if not errors_path.exists():
        return None
    with errors_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_provenance_checks(
    group: h5py.Group,
    *,
    root: h5py.File,
    dataset_path: Path,
    thresholds_version: str,
) -> list[CheckResult]:
    """Run all provenance.* checks."""
    del group
    checks: list[CheckResult] = []

    tr_removal = _read_attr(root, "tr_removal")
    checks.append(
        CheckResult(
            check_id="provenance.tr_method",
            category="provenance",
            status="pass" if tr_removal else "warn",
            value=str(tr_removal) if tr_removal else None,
            threshold="project or skip",
            message=f"TR removal method: {tr_removal}"
            if tr_removal
            else "TR removal method missing",
            source="hdf5_attr",
        )
    )

    qc_spec_version = _read_attr(root, "qc_spec_version", thresholds_version)
    version_match = str(qc_spec_version) == thresholds_version
    checks.append(
        CheckResult(
            check_id="provenance.qc_spec_version",
            category="provenance",
            status="pass" if version_match else "info",
            value=str(qc_spec_version),
            threshold=thresholds_version,
            message="QC spec version matches applied thresholds"
            if version_match
            else f"Using local thresholds ({thresholds_version}); file has {qc_spec_version}",
            source="hdf5_attr" if _read_attr(root, "qc_spec_version") else "computed",
        )
    )

    errors = _load_errors_sidecar(dataset_path)
    if errors is None:
        checks.append(
            CheckResult(
                check_id="provenance.generation_errors",
                category="provenance",
                status="info",
                value=None,
                threshold="sidecar present",
                message="No generation errors sidecar found",
                source="sidecar",
            )
        )
    else:
        skipped = errors.get("skipped_molecules") or errors.get("errors") or []
        count = len(skipped) if isinstance(skipped, list) else 0
        checks.append(
            CheckResult(
                check_id="provenance.generation_errors",
                category="provenance",
                status="info" if count == 0 else "warn",
                value=count,
                threshold="zero skipped molecules",
                message="No generation errors recorded"
                if count == 0
                else f"{count} molecules skipped during generation",
                source="sidecar",
            )
        )

    return checks
