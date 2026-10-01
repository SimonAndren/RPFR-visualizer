"""QC data models for molecule- and dataset-level assurance reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CheckStatus = str  # pass | info | warn | fail
CheckSource = str  # hdf5_attr | computed | sidecar
OverallTier = str  # A | B | C | D


@dataclass
class CheckResult:
    """Result of a single QC check for one molecule."""

    check_id: str
    category: str
    status: CheckStatus
    value: float | int | str | bool | None
    threshold: str | None
    message: str
    source: CheckSource

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckResult:
        """Deserialize from a dictionary."""
        return cls(
            check_id=str(data["check_id"]),
            category=str(data["category"]),
            status=str(data["status"]),
            value=data.get("value"),
            threshold=data.get("threshold"),
            message=str(data["message"]),
            source=str(data["source"]),
        )


@dataclass
class MoleculeQC:
    """Aggregated QC report for a single molecule."""

    mol_id: str
    overall_tier: OverallTier
    checks: list[CheckResult]
    usable_for: list[str]
    exclude_from: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "mol_id": self.mol_id,
            "overall_tier": self.overall_tier,
            "checks": [check.to_dict() for check in self.checks],
            "usable_for": list(self.usable_for),
            "exclude_from": list(self.exclude_from),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoleculeQC:
        """Deserialize from a dictionary."""
        return cls(
            mol_id=str(data["mol_id"]),
            overall_tier=str(data["overall_tier"]),
            checks=[CheckResult.from_dict(item) for item in data["checks"]],
            usable_for=list(data.get("usable_for", [])),
            exclude_from=list(data.get("exclude_from", [])),
            summary=str(data["summary"]),
        )

    def check_by_id(self, check_id: str) -> CheckResult | None:
        """Return the first check with the given ID."""
        for check in self.checks:
            if check.check_id == check_id:
                return check
        return None


@dataclass
class DatasetQC:
    """Aggregated QC report for an entire dataset."""

    path: str
    n_molecules: int
    molecules: dict[str, MoleculeQC]
    tier_histogram: dict[str, int] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    qc_spec_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "path": self.path,
            "n_molecules": self.n_molecules,
            "qc_spec_version": self.qc_spec_version,
            "tier_histogram": dict(self.tier_histogram),
            "failure_counts": dict(self.failure_counts),
            "molecules": {mol_id: mol_qc.to_dict() for mol_id, mol_qc in self.molecules.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetQC:
        """Deserialize from a dictionary."""
        molecules = {
            mol_id: MoleculeQC.from_dict(mol_data)
            for mol_id, mol_data in data.get("molecules", {}).items()
        }
        return cls(
            path=str(data["path"]),
            n_molecules=int(data["n_molecules"]),
            molecules=molecules,
            tier_histogram=dict(data.get("tier_histogram", {})),
            failure_counts=dict(data.get("failure_counts", {})),
            qc_spec_version=str(data.get("qc_spec_version", "1.0")),
        )

    def summary_row(self, mol_id: str) -> dict[str, Any]:
        """Return a flat summary row for CSV export."""
        mol_qc = self.molecules[mol_id]
        row: dict[str, Any] = {
            "mol_id": mol_id,
            "overall_tier": mol_qc.overall_tier,
            "usable_for": ",".join(mol_qc.usable_for),
            "exclude_from": ",".join(mol_qc.exclude_from),
            "summary": mol_qc.summary,
        }
        for check in mol_qc.checks:
            row[f"{check.check_id}_status"] = check.status
        return row
