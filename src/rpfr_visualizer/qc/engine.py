"""QC engine for scanning HDF5 datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from rpfr_visualizer.io.loader import _molecule_keys, _temperatures_from_file
from rpfr_visualizer.qc.checks import (
    run_annotation_checks,
    run_assignment_checks,
    run_frequency_checks,
    run_provenance_checks,
    run_rpfr_checks,
    run_structure_checks,
)
from rpfr_visualizer.qc.models import DatasetQC, MoleculeQC
from rpfr_visualizer.qc.thresholds import QcThresholds, load_thresholds
from rpfr_visualizer.qc.tiers import finalize_molecule_qc


@dataclass(frozen=True)
class QcContext:
    """Shared context for molecule-level QC scans."""

    dataset_path: Path
    thresholds: QcThresholds
    n_temps: int
    temperatures: tuple[float, ...]
    root_attrs: dict[str, Any]


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _collect_root_attrs(h5_file: h5py.File) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in h5_file.attrs:
        raw = h5_file.attrs[key]
        if key == "temperatures_K" and isinstance(raw, (bytes, str)):
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            attrs[key] = json.loads(text)
        else:
            attrs[key] = _decode_scalar(raw)
    return attrs


def run_molecule_qc(group: h5py.Group, *, root: h5py.File, ctx: QcContext) -> MoleculeQC:
    """Run all QC checks for one molecule group."""
    mol_id = str(_decode_scalar(group.attrs.get("mol_id", group.name.strip("/"))))
    checks: list = []
    checks.extend(
        run_structure_checks(
            group,
            root=root,
            n_temps=ctx.n_temps,
            thresholds=ctx.thresholds,
        )
    )
    checks.extend(run_frequency_checks(group, thresholds=ctx.thresholds))
    checks.extend(run_assignment_checks(group, root=root))
    checks.extend(run_annotation_checks(group))
    checks.extend(run_rpfr_checks(group, thresholds=ctx.thresholds, temperatures=ctx.temperatures))
    checks.extend(
        run_provenance_checks(
            group,
            root=root,
            dataset_path=ctx.dataset_path,
            thresholds_version=ctx.thresholds.qc_spec_version,
        )
    )
    return finalize_molecule_qc(mol_id, checks)


def run_dataset_qc(path: Path | str, *, thresholds: QcThresholds | None = None) -> DatasetQC:
    """Run QC checks for every molecule in an HDF5 dataset."""
    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as h5:
        resolved_thresholds = thresholds or load_thresholds(h5)
        temperatures = _temperatures_from_file(h5)
        ctx = QcContext(
            dataset_path=dataset_path,
            thresholds=resolved_thresholds,
            n_temps=len(temperatures),
            temperatures=temperatures,
            root_attrs=_collect_root_attrs(h5),
        )

        molecules: dict[str, MoleculeQC] = {}
        tier_histogram: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        failure_counts: dict[str, int] = {}

        molecule_keys = _molecule_keys(h5)
        for key in molecule_keys:
            mol_qc = run_molecule_qc(h5[key], root=h5, ctx=ctx)
            molecules[mol_qc.mol_id] = mol_qc
            tier_histogram[mol_qc.overall_tier] = tier_histogram.get(mol_qc.overall_tier, 0) + 1
            for check in mol_qc.checks:
                if check.status in {"warn", "fail"}:
                    failure_counts[check.check_id] = failure_counts.get(check.check_id, 0) + 1

    return DatasetQC(
        path=str(dataset_path),
        n_molecules=len(molecules),
        molecules=molecules,
        tier_histogram=tier_histogram,
        failure_counts=failure_counts,
        qc_spec_version=resolved_thresholds.qc_spec_version,
    )
