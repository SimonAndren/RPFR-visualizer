"""QC summary helpers for the student visualizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py

from rpfr_visualizer.io.loader import Molecule, _temperatures_from_file
from rpfr_visualizer.qc.engine import QcContext, _collect_root_attrs, run_molecule_qc
from rpfr_visualizer.qc.models import MoleculeQC
from rpfr_visualizer.qc.thresholds import load_thresholds


@dataclass(frozen=True)
class StoredQcFlags:
    """QC attributes already present on a loaded :class:`Molecule`."""

    qc_pass: bool
    freq_qc_ok: bool
    rpfr_qc_ok: bool
    tr_converged: bool
    rdkit_ok: bool
    smiles_order_match: bool
    n_imag_vib: int
    rdkit_status: str = ""
    #: Not part of the stored verdict, but the one flag a reader of the 2D
    #: drawing most needs: whether its SMILES still describes the geometry.
    geometry_match: bool = True

    def format_lines(self) -> list[str]:
        """One line per check, in words, with a mark for how it went.

        This used to print the attributes as stored --
        ``freq_qc_ok=True  rpfr_qc_ok=True`` -- which is the file's vocabulary
        rather than the reader's, and said "FAIL" where the panel above it said
        "some checks need attention".
        """

        def line(ok: bool, passed: str, failed: str) -> str:
            return f"{'✓' if ok else '✗'}  {passed if ok else failed}"

        imag = self.n_imag_vib
        lines = [
            "Stored verdict: " + ("pass" if self.qc_pass else "needs checking"),
            "",
            line(
                self.freq_qc_ok,
                "Frequency calculation converged",
                "Frequency calculation did not converge",
            ),
            line(self.rpfr_qc_ok, "β values are physical", "Some β values are unphysical"),
            line(
                self.tr_converged,
                "Translations and rotations removed cleanly",
                "Translation and rotation removal did not converge",
            ),
            line(self.rdkit_ok, "RDKit read the structure", self._rdkit_failure()),
            line(
                self.smiles_order_match,
                "Atom order matches the SMILES",
                "Atom order differs from the SMILES",
            ),
            line(
                imag == 0,
                "No imaginary vibrational modes",
                f"{imag} imaginary vibrational mode{'' if imag == 1 else 's'}",
            ),
        ]
        # A geometry mismatch normally surfaces as the RDKit failure above, and
        # counts against the verdict through it. Say it separately only where it
        # did not, rather than twice.
        if not self.geometry_match and self.rdkit_ok:
            lines += [
                "",
                "Note: the stored SMILES does not match the geometry; the 2D may be wrong.",
            ]
        return lines

    def _rdkit_failure(self) -> str:
        """Why RDKit failed, in words rather than as the stored status code."""
        if self.rdkit_status == "geometry_mismatch" or not self.geometry_match:
            return "The stored SMILES does not match the geometry; the 2D may be wrong"
        reason = self.rdkit_status.replace("_", " ").strip()
        if reason:
            return f"RDKit could not read the structure ({reason})"
        return "RDKit could not read the structure"


def stored_qc_flags(molecule: Molecule) -> StoredQcFlags:
    """Extract stored QC flags from a loaded molecule."""
    return StoredQcFlags(
        qc_pass=bool(molecule.qc_pass),
        freq_qc_ok=bool(molecule.freq_qc_ok),
        rpfr_qc_ok=bool(molecule.rpfr_qc_ok),
        tr_converged=bool(molecule.tr_converged),
        rdkit_ok=bool(molecule.rdkit_ok),
        smiles_order_match=bool(molecule.smiles_order_match),
        n_imag_vib=int(molecule.n_imag_vib),
        rdkit_status=str(molecule.rdkit_status),
        geometry_match=bool(getattr(molecule, "geometry_match", True)),
    )


def format_molecule_qc(report: MoleculeQC) -> list[str]:
    """Format a structured :class:`MoleculeQC` report for the UI."""
    lines = [
        f"Structured QC: tier {report.overall_tier} — {report.summary}",
        f"  usable_for={list(report.usable_for)}",
        f"  exclude_from={list(report.exclude_from)}",
    ]
    failing = [c for c in report.checks if c.status in {"warn", "fail"}]
    if not failing:
        lines.append("  No warn/fail checks.")
        return lines
    lines.append("  Warn/fail checks:")
    for check in failing:
        lines.append(f"    [{check.status}] {check.check_id}: {check.message}")
    return lines


def run_qc_for_key(path: Path | str, key: str) -> MoleculeQC:
    """Run the structured QC engine for one molecule group in an HDF5 file."""
    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as h5:
        if key not in h5:
            raise KeyError(f"Molecule key {key!r} not found in {dataset_path}")
        thresholds = load_thresholds(h5)
        temperatures = _temperatures_from_file(h5)
        ctx = QcContext(
            dataset_path=dataset_path,
            thresholds=thresholds,
            n_temps=len(temperatures),
            temperatures=temperatures,
            root_attrs=_collect_root_attrs(h5),
        )
        return run_molecule_qc(h5[key], root=h5, ctx=ctx)


def qc_panel_text(molecule: Molecule, *, structured: MoleculeQC | None = None) -> str:
    """Combine stored flags and optional structured QC into one text block."""
    lines = stored_qc_flags(molecule).format_lines()
    if structured is not None:
        lines.append("")
        lines.extend(format_molecule_qc(structured))
    return "\n".join(lines)


__all__ = [
    "StoredQcFlags",
    "format_molecule_qc",
    "qc_panel_text",
    "run_qc_for_key",
    "stored_qc_flags",
]
