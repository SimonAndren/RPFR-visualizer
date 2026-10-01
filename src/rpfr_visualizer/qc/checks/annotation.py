"""Annotation-related QC checks."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from rpfr_visualizer.qc.models import CheckResult


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _read_attr(group: h5py.Group, name: str, default: Any = None) -> Any:
    if name not in group.attrs:
        return default
    return _decode_scalar(group.attrs[name])


def _decode_str(value: Any) -> str:
    decoded = _decode_scalar(value)
    return "" if decoded is None else str(decoded)


def run_annotation_checks(group: h5py.Group) -> list[CheckResult]:
    """Run all annotation.* checks."""
    checks: list[CheckResult] = []

    rdkit_ok = bool(_read_attr(group, "rdkit_ok", True))
    rdkit_status = str(_read_attr(group, "rdkit_status", "ok"))
    if rdkit_ok:
        status = "pass" if rdkit_status == "ok" else "warn"
        message = "RDKit parse succeeded"
        if rdkit_status != "ok":
            message = f"RDKit parse succeeded with warnings ({rdkit_status})"
    else:
        status = "fail"
        message = f"RDKit parse failed ({rdkit_status})"

    checks.append(
        CheckResult(
            check_id="annotation.rdkit_parse",
            category="annotation",
            status=status,
            value=rdkit_status,
            threshold="ok",
            message=message,
            source="hdf5_attr",
        )
    )

    rdkit_order = _read_attr(group, "rdkit_order_ok")
    if rdkit_order is not None:
        order_ok = bool(rdkit_order)
        checks.append(
            CheckResult(
                check_id="annotation.rdkit_order",
                category="annotation",
                status="pass" if order_ok else "fail",
                value=order_ok,
                threshold="True",
                message="RDKit atom order matches" if order_ok else "RDKit atom order mismatch",
                source="hdf5_attr",
            )
        )

    if "local_smiles" not in group:
        checks.append(
            CheckResult(
                check_id="annotation.local_smiles_present",
                category="annotation",
                status="fail",
                value="missing",
                threshold="shells populated",
                message="local_smiles dataset missing",
                source="hdf5_attr",
            )
        )
    else:
        raw = group["local_smiles"][()]
        n_atoms = len(raw)
        populated = 0
        for row in raw:
            shells = [_decode_str(item) for item in np.asarray(row).flat[:5]]
            if any(shells):
                populated += 1
        if populated == n_atoms:
            status = "pass"
            message = "local_smiles populated for all sites"
        elif populated == 0:
            status = "fail"
            message = "local_smiles empty for all sites"
        else:
            status = "warn"
            message = f"local_smiles partially populated ({populated}/{n_atoms} sites)"
        checks.append(
            CheckResult(
                check_id="annotation.local_smiles_present",
                category="annotation",
                status=status,
                value=f"{populated}/{n_atoms}",
                threshold="shells populated",
                message=message,
                source="computed",
            )
        )

    return checks
