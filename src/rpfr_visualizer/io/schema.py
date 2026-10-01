"""Field definitions and units for QM9s-RPFR HDF5 schema v2.0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "2.0"

# Canonical temperature grid (13 values). Matches file root attr ``temperatures_K``.
TEMPERATURES_K: tuple[float, ...] = (
    20.0,
    60.0,
    100.0,
    140.0,
    180.0,
    220.0,
    260.0,
    300.0,
    340.0,
    380.0,
    420.0,
    273.15,
    298.15,
)

# Root-level file attributes expected in v2.0 files.
ROOT_ATTRS: dict[str, str] = {
    "schema_version": "Schema version string (e.g. '2.0').",
    "dataset": "Source dataset name (e.g. 'qm9s').",
    "temperatures_K": "JSON-encoded temperature grid in Kelvin.",
    "coordinate_units": "Units for Cartesian coordinates (Angstrom).",
    "hessian_units": "Units for Hessian data when present (eV/Angstrom^2).",
    "isotope_table": "Reference isotope mass table identifier.",
    "tr_removal": "Translation/rotation removal method ('project' or 'skip').",
    "generated_at_utc": "ISO timestamp of file generation.",
    "git_commit": "Git commit hash of the processing pipeline.",
    "source_path": "Original raw data path used to build this file.",
}

# Per-molecule group attributes.
MOLECULE_ATTRS: dict[str, str] = {
    "mol_id": "QM9s molecule identifier (string).",
    "smiles": "Full-molecule canonical SMILES.",
    "n_atoms": "Total atom count.",
    "is_linear": "Whether the molecule is linear.",
    "n_imag_vib": "Number of imaginary vibrational modes.",
    "freq_qc_ok": "Frequency calculation passed QC.",
    "rpfr_qc_ok": "RPFR calculation passed QC.",
    "tr_converged": "Translation/rotation removal converged.",
    "rdkit_ok": "RDKit structure parsing succeeded.",
    "rdkit_status": "RDKit validation status string.",
    "smiles_order_match": "Atom order matches RDKit expectation.",
    "max_tr_rot_abs": "Maximum absolute TR residual (diagnostic).",
}

# Per-molecule datasets (keys inside each molecule group).
MOLECULE_DATASETS: dict[str, str] = {
    "atom_symbol": "Element symbol per atom (N,).",
    "atom_degree": "Graph degree per atom (N,).",
    "atom_valence": "Valence per atom (N,).",
    "atom_aromaticity": "Aromaticity flag per atom (N,).",
    "atom_hybridization": "Hybridization label per atom (N,).",
    "bond_order": "JSON-encoded bond-order strings per atom (N,).",
    "coordinates": "Cartesian coordinates per atom (N, 3) in Angstrom.",
    "rpfr_single": "SSI RPFR beta per site and temperature (N, 13), dimensionless.",
    "mass_law": "Triple-oxygen mass-law exponent lambda per site (N, 13).",
    "clumping": "Pairwise clumping factor C (N, N, 13); Delta = ln(C), C=1 => Delta=0.",
    "local_smiles": "Circular substructure fingerprints R1-R5 per site (N, 5).",
    "topo_dist": "Topological distance matrix (N, N).",
    "all_frequencies": "All vibrational frequencies (3N,).",
    "harmonic_frequencies": "Harmonic frequencies after Eckart TR projection (3N-6 or 3N-5).",
}

DEFAULT_SOURCE = "dft"

# Elements present in QM9s-RPFR (no S, P in this dataset).
VALID_ELEMENTS = frozenset({"H", "C", "N", "O"})

# Heavy elements for beta sanity checks (beta should be >= 1 for heavy isotopes).
HEAVY_ELEMENTS = frozenset({"C", "N", "O"})


@dataclass(frozen=True)
class SiteRecord:
    """In-memory representation of a singly substituted isotopologue site."""

    molecule_id: int
    atom_index: int
    element: str
    beta: tuple[float, ...]
    temperatures: tuple[float, ...] = TEMPERATURES_K
    local_smiles: tuple[str, ...] = field(default_factory=tuple)
    source: str = DEFAULT_SOURCE
    atom_degree: int | None = None
    atom_valence: int | None = None
    atom_aromaticity: bool | None = None
    atom_hybridization: str | None = None
    bond_order: list[str] | None = None
    coordinates: tuple[float, float, float] | None = None

    def beta_at(
        self,
        temperature: float,
        *,
        warn: bool = True,
        method: str | None = None,
    ) -> float:
        """Return beta at *temperature*.

        Exact grid temperatures return the stored value whatever *method* is.
        Off-grid temperatures interpolate ``ln(β)`` against ``1/T``; *method*
        names which interpolant, defaulting to the local quadratic (see
        :mod:`rpfr_visualizer.analysis.temperature` for what the choice costs).
        """
        from rpfr_visualizer.analysis.temperature import beta_at_temperature

        return beta_at_temperature(
            self.temperatures,
            self.beta,
            temperature,
            warn=warn,
            method=method,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary."""
        return {
            "molecule_id": self.molecule_id,
            "atom_index": self.atom_index,
            "element": self.element,
            "beta": list(self.beta),
            "temperatures": list(self.temperatures),
            "local_smiles": list(self.local_smiles),
            "source": self.source,
            "atom_degree": self.atom_degree,
            "atom_valence": self.atom_valence,
            "atom_aromaticity": self.atom_aromaticity,
            "atom_hybridization": self.atom_hybridization,
            "bond_order": self.bond_order,
            "coordinates": self.coordinates,
        }


@dataclass
class MoleculeRecord:
    """In-memory representation of a molecule and its sites."""

    molecule_id: int
    smiles: str
    n_atoms: int
    sites: list[SiteRecord]
    qc_pass: bool = True
    is_linear: bool = False
    n_imag_vib: int = 0
    freq_qc_ok: bool = True
    rpfr_qc_ok: bool = True
    tr_converged: bool = True
    rdkit_ok: bool = True
    rdkit_status: str = "ok"
    smiles_order_match: bool = True
    #: Whether the DFT geometry still matches the stored SMILES.
    #:
    #: False for roughly 0.3% of schema 2.2 molecules -- mostly strained rings
    #: that relaxed into something the SMILES no longer describes. The molecule
    #: is flagged, never dropped, and it keeps its SMILES, so anything drawing
    #: a structure from that SMILES draws a different molecule from the one the
    #: beta values were computed on.
    geometry_match: bool = True
    max_tr_rot_abs: float | None = None
    source: str = DEFAULT_SOURCE

    @property
    def heavy_atom_count(self) -> int:
        """Count non-hydrogen atoms."""
        return sum(1 for site in self.sites if site.element != "H")

    def beta_at(self, atom_index: int, temperature: float) -> float:
        """Return beta for a site at a given temperature."""
        return self.sites[atom_index].beta_at(temperature)

    def to_dict(self) -> dict[str, Any]:
        """Serialize molecule metadata (excluding full site payloads)."""
        return {
            "molecule_id": self.molecule_id,
            "smiles": self.smiles,
            "n_atoms": self.n_atoms,
            "heavy_atom_count": self.heavy_atom_count,
            "qc_pass": self.qc_pass,
            "is_linear": self.is_linear,
            "n_imag_vib": self.n_imag_vib,
            "source": self.source,
        }
