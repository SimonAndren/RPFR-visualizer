# QM9s-RPFR HDF5 Schema (2.2)

Authoritative field documentation for the QM9s-RPFR processed HDF5 format.

## Overview

Each HDF5 file contains one group per molecule, keyed by numeric molecule ID
(e.g. `"1"`, `"100"`, `"129817"`). Root-level attributes describe the file
provenance; the root dataset `/temperatures` holds the temperature grid.
Schema 2.0 and 2.1 files, with a 13-point grid, are still read.

## Root attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `schema_version` | str | Schema version (currently `"2.2"`) |
| `dataset` | str | Source dataset name (`qm9s`) |
| `temperatures_K` | JSON str | Temperature grid in Kelvin (2.2 also stores it as the `/temperatures` dataset) |
| `coordinate_units` | str | Cartesian coordinate units (`Angstrom`) |
| `hessian_units` | str | Hessian units when present |
| `isotope_table` | str | Isotope mass table identifier |
| `tr_removal` | str | TR removal method (`project` or `skip`) |
| `generated_at_utc` | str | File generation timestamp (UTC) |
| `git_commit` | str | Processing pipeline git commit |
| `source_path` | str | Original raw data path |
| `isotope_masses_amu` | JSON str | Parent, substituted and (for O) mass-law isotope masses |
| `isotope_convention` | str | What each array holds per element, and what NaN means |
| `subsample_fraction` | float | Present on subsamples only, with `subsample_seed` and `subsample_n_molecules` |

## Temperature grid

15 temperatures in Kelvin, ascending (schema 2.2):

```
20, 60, 100, 140, 180, 220, 260, 273.15, 298.15, 300, 340, 380, 420, 460, 500
```

Schema 2.0 and 2.1 stored 13 (no 460 or 500 K), with 273.15 and 298.15 last.

## Molecule group attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mol_id` | str | QM9s molecule identifier |
| `smiles` | str | Full-molecule SMILES |
| `n_atoms` | int | Total atom count |
| `is_linear` | bool | Linear molecule flag |
| `n_imag_vib` | int | Imaginary vibrational mode count |
| `freq_qc_ok` | bool | Frequency QC passed |
| `rpfr_qc_ok` | bool | RPFR QC passed |
| `tr_converged` | bool | TR removal converged |
| `rdkit_ok` | bool | RDKit parsing succeeded |
| `rdkit_status` | str | RDKit status string |
| `smiles_order_match` | bool | Atom order matches RDKit |
| `geometry_match` | bool | Stored SMILES still describes the geometry (2.2; absent reads as true) |
| `max_tr_rot_abs` | float | Max TR residual (diagnostic) |

## Molecule datasets

| Dataset | Shape | Description | Units |
|---------|-------|-------------|-------|
| `atom_symbol` | (N,) | Element symbol per atom | — |
| `atom_degree` | (N,) | Graph degree | — |
| `atom_valence` | (N,) | Atom valence | — |
| `atom_aromaticity` | (N,) | Aromaticity flag | — |
| `atom_hybridization` | (N,) | Hybridization label | — |
| `bond_order` | (N,) | JSON bond-order strings | — |
| `coordinates` | (N, 3) | Cartesian coordinates | Angstrom |
| `rpfr_single` | (N, T) | SSI RPFR beta | dimensionless |
| `mass_law` | (N, T) | Triple-O mass-law exponent λ = ln β₁₇ / ln β₁₈; NaN off oxygen | dimensionless |
| `clumping` | (N, N, T) | Pairwise clumping factor C = βᵢⱼ / βᵢβⱼ; Δ = 1000 ln C in per mil; NaN diagonal | dimensionless |
| `local_smiles` | (N, 5) | Circular substructure fingerprints R1-R5 | — |
| `topo_dist` | (N, N) | Topological distance matrix | — |
| `all_frequencies` | (3N,) | All vibrational frequencies (skip spectrum) | cm^-1 |
| `harmonic_frequencies` | (3N-6,) or (3N-5,) | Eckart-projected harmonic frequencies (TR removed) | cm^-1 |

T is the number of stored temperatures: 15 in schema 2.2, 13 before.

## In-memory record model (Phase 1)

### Per-site record (SSI)

| Field | Type | Description |
|-------|------|-------------|
| `molecule_id` | int | QM9s molecule index |
| `atom_index` | int | Substituted atom position |
| `element` | str | H, C, N, or O |
| `beta` | float[T] | RPFR at each temperature |
| `temperatures` | float[T] | Temperature grid |
| `local_smiles` | str[5] | Site fingerprints R1-R5 |
| `source` | str | `dft` or `model_<version>` |

### Per-molecule record

| Field | Type | Description |
|-------|------|-------------|
| `molecule_id` | int | QM9s molecule index |
| `smiles` | str | Full molecule SMILES |
| `heavy_atom_count` | int | Non-hydrogen atom count |
| `qc_pass` | bool | Derived QC pass flag |

## QC pass derivation

A molecule passes QC when all of the following hold:

- `freq_qc_ok == True`
- `rpfr_qc_ok == True`
- `tr_converged == True`
- `rdkit_ok == True`
- `smiles_order_match == True`
- `n_imag_vib == 0`

Schema 2.2 also records `geometry_match`. It is shown as a warning on the 2D
structure, and counts against the verdict through `rdkit_ok` when RDKit flagged
it.

Additionally, beta values are scanned for NaN/Inf and unphysical values
(beta < 1 for C/N/O; beta <= 0 for H).

## Structured QC

See [qc.md](./qc.md) for the full check catalog, tier model, frequency-spectrum
semantics (skip vs Eckart), and clumping-factor thresholds.

## Data files

| File | Description |
|------|-------------|
| `qm9s_rpfr_v2.2.h5` | Full QM9s, schema 2.2, Eckart-projected |
| `qm9s_v2.2_1pt.h5` | 1% subsample of it (1,298 molecules), the published tier |
| `qm9s_tr_project.h5`, `qm9s_tr_skip.h5` | Schema 2.0 files, still readable |

Generate the subsample:

```bash
uv run python scripts/make_subsample.py
```
