"""Assignment-related QC checks."""

from __future__ import annotations

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


def run_assignment_checks(group: h5py.Group, *, root: h5py.File) -> list[CheckResult]:
    """Run all assignment.* checks."""
    checks: list[CheckResult] = []

    smiles_order_match = bool(_read_attr(group, "smiles_order_match", True))
    checks.append(
        CheckResult(
            check_id="assignment.smiles_order_match",
            category="assignment",
            status="pass" if smiles_order_match else "fail",
            value=smiles_order_match,
            threshold="True",
            message="Atom order matches SMILES labels"
            if smiles_order_match
            else "Atom index may not match chemical site labels",
            source="hdf5_attr",
        )
    )

    element_source = _read_attr(group, "element_source")
    if element_source is None:
        checks.append(
            CheckResult(
                check_id="assignment.element_source",
                category="assignment",
                status="info",
                value="unknown",
                threshold="atomic numbers preferred",
                message="Element source not recorded; assuming dataset default",
                source="hdf5_attr",
            )
        )
    else:
        source_text = str(element_source)
        trusted = source_text.lower() in {"z", "atomic_number", "qm9s"}
        checks.append(
            CheckResult(
                check_id="assignment.element_source",
                category="assignment",
                status="pass" if trusted else "warn",
                value=source_text,
                threshold="trusted atomic-number source",
                message=f"Element source: {source_text}",
                source="hdf5_attr",
            )
        )

    isotope_table = _read_attr(root, "isotope_table")
    checks.append(
        CheckResult(
            check_id="assignment.isotope_table",
            category="assignment",
            status="pass" if isotope_table else "warn",
            value=str(isotope_table) if isotope_table else None,
            threshold="present at file root",
            message="Isotope table recorded at file root"
            if isotope_table
            else "Isotope table missing from file root",
            source="hdf5_attr",
        )
    )

    return checks
