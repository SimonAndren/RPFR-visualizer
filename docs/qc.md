# QM9s-RPFR Quality Control Reference

Structured QC is implemented in `rpfr_visualizer.qc`. Each molecule receives a
`MoleculeQC` report with per-check results, an overall tier (A–D), and
`usable_for` / `exclude_from` tags.

```python
from rpfr_visualizer import QM9sDataset

report = QM9sDataset("data/raw/qm9s_v2.2_1pt.h5").qc()
mol = report.molecules["567"]
print(mol.overall_tier, mol.usable_for, mol.summary)
```

CLI:

```bash
uv run python scripts/validate_dataset.py data/raw/qm9s_v2.2_1pt.h5 \
  --output-dir reports/
```

---

## Two frequency spectra (do not mix them)

The HDF5 file stores **two different** vibrational arrays. QC uses each for a
distinct purpose:

| Dataset | TR handling | Used for |
|---------|-------------|----------|
| `all_frequencies` | **Skip** spectrum (TR modes retained in raw block) | Imaginary-mode detection, TR residual diagnostics (`max_tr_rot_abs`) |
| `harmonic_frequencies` | **Eckart** projection (TR modes removed) | Soft-mode count, RPFR physics, source regression |

**Soft modes** (`frequencies.soft_modes`) are counted from
`harmonic_frequencies` **after Eckart TR projection**: modes with
`0 < ν ≤ 50 cm⁻¹`. These are the frequencies that feed the RPFR Hessian
pipeline, not the raw skip-spectrum TR bands.

**TR residuals** (`frequencies.tr_residual`) come from `max_tr_rot_abs` on the
**skip** spectrum. Eckart projection hides TR convergence problems — never infer
TR quality from `harmonic_frequencies` alone.

Root attribute `tr_removal` (`project` or `skip`) records which method was used
when building the file (`provenance.tr_method`).

---

## Clumping factors vs SSI betas

| Dataset | Meaning | Typical values |
|---------|---------|----------------|
| `rpfr_single` | SSI reduced partition function ratio β per site | H: ≫1; C/N/O: ≥1 |
| `clumping` | Pairwise **multiplicative clumping factor** C | Off-diagonal: ≥1; often ≈1.000–1.01, pairs up to ~2+ |

The clumping **anomaly** is:

```
Delta = ln(C)
```

- `C = 1` → `Delta = 0` (no pairwise clumping anomaly)
- Physical pair factors are **≥ 1**; QC flags off-diagonal values below 1
- Most QM9s pairs sit very close to 1, but **elevated clumping is normal** for some
  pairs (typical max C per molecule ≈ 1.5–2; higher values occur)

`clumping` is **not** the product `β_i × β_j`. The check `rpfr.clumping_factor`
validates:

| Condition | Status |
|-----------|--------|
| Any off-diagonal `C < 1 − 10⁻⁴` at **300 K** | **fail** (unphysical; allows float noise) |
| `max C > 10` at 300 K | **fail** (extreme) |
| `max C > 3` at 300 K | **warn** (large anomaly — review pair) |
| Otherwise | **pass** |

Reported metrics at the reference temperature (default **300 K**):
`min_factor`, `max_factor`, and `max_delta = ln(max C)`.

Symmetry: `rpfr.clumping_symmetry` requires a symmetric matrix with zero
diagonal.

---

## Overall tiers

| Tier | Label | Typical meaning |
|------|-------|-----------------|
| A | High confidence | All checks pass |
| B | Good physics, label caveats | Assignment/annotation info; site-label issues |
| C | Physics caveats | TR warn band, soft modes, moderate clumping anomaly |
| D | Major issues | Large imaginary modes, failed RPFR, C < 1 |

Tier overrides:

- `smiles_order_match == false` → excludes `site_labeled_analysis`, `local_env`, `ml_features`
- Worst imaginary ν < −50 cm⁻¹ → tier D for site-level RPFR and clumping use cases

---

## Check catalog

Severity levels: **pass | info | warn | fail**.

### Structure (`structure.*`)

| check_id | Rule |
|----------|------|
| `structure.schema_version` | Root `schema_version` ≥ 2.0 |
| `structure.required_datasets` | Core arrays present |
| `structure.shape_consistency` | (N, T) shapes match atom count and 13 temperatures |
| `structure.finite_arrays` | No NaN/Inf in core arrays |

### Frequencies (`frequencies.*`)

| check_id | Source | Notes |
|----------|--------|-------|
| `frequencies.tr_converged` | attr | Pipeline TR convergence flag |
| `frequencies.tr_residual` | `max_tr_rot_abs` | Skip spectrum; pass <10, warn 10–50, fail >50 cm⁻¹ |
| `frequencies.imaginary_vib_count` | `n_imag_vib` | From skip spectrum |
| `frequencies.imaginary_vib_worst` | attr or `all_frequencies` | pass if \|ν\| &lt; 1 cm⁻¹ (numerical); info 1–10; warn 10–50; fail &gt; 50 |
| `frequencies.soft_modes` | attr or `harmonic_frequencies` | **Eckart** spectrum; 0 < ν ≤ 50 cm⁻¹ |
| `frequencies.source_regression_*` | optional `source_nsi_frequencies` | N/A for QM9s-only files |

### Assignment (`assignment.*`)

| check_id | Rule |
|----------|------|
| `assignment.smiles_order_match` | Atom index aligns with chemical site labels |
| `assignment.element_source` | Trusted element provenance when recorded |
| `assignment.isotope_table` | Root isotope table present |

### Annotation (`annotation.*`)

| check_id | Rule |
|----------|------|
| `annotation.rdkit_parse` | `rdkit_ok` / `rdkit_status` |
| `annotation.rdkit_order` | Optional `rdkit_order_ok` attr |
| `annotation.local_smiles_present` | Circular fingerprint shells populated |

### RPFR (`rpfr.*`)

| check_id | Rule |
|----------|------|
| `rpfr.finite_positive` | `rpfr_qc_ok` attr |
| `rpfr.clumping_symmetry` | Symmetric, zero diagonal |
| `rpfr.clumping_factor` | Off-diagonal C ≥ 1; warn if max C > 3; fail if max C > 10 |

### Provenance (`provenance.*`)

| check_id | Source |
|----------|--------|
| `provenance.tr_method` | Root `tr_removal` |
| `provenance.qc_spec_version` | Root `qc_spec_version` vs applied thresholds |
| `provenance.generation_errors` | Adjacent `*.errors.json` sidecar |

---

## Default thresholds

Loaded from root HDF5 attr `qc_thresholds` when present; otherwise:

```json
{
  "qc_spec_version": "1.0",
  "tr_residual_cm1": { "pass": 10, "fail": 50 },
  "imaginary_vib_cm1": { "pass": 1.0, "info": 10, "warn": 50 },
  "soft_modes_cm1": {
    "upper": 50,
    "warn_count": 1,
    "fail_count": 3,
    "fail_single": 10
  },
  "source_regression_cm1": {
    "mae_pass": 1.0,
    "mae_warn": 3.0,
    "max_pass": 5.0,
    "max_warn": 20.0
  },
  "rpfr_clumping": {
    "min_factor": 1.0,
    "below_one_tol": 1e-4,
    "reference_temperature_K": 300.0,
    "warn_max_factor": 3.0,
    "fail_max_factor": 10.0
  }
}
```

---

## Usable-for tags

| Tag | Meaning |
|-----|---------|
| `rpfr_bulk` | Corpus-level RPFR distributions |
| `rpfr_site` | Trust per-atom β values |
| `clumping` | Pair clumping factors |
| `local_env` | `local_smiles`, `topo_dist`, bond metadata |
| `ml_features` | ML featurization from subgraphs |
| `site_labeled_analysis` | RPFR index aligned with chemical labels |

Policies:

- **Flag, don't delete** — failed checks lower tier and set `exclude_from`; molecules remain in the file and exports unless the user filters explicitly.
- **Pipeline `qc_pass`** — binary flag on `MoleculeRecord`; what the explorer's `qc:pass` filter reads.
- **Structured QC** — superset with tiers and use-case tags via `ds.qc()`.

---

## Export bundle

`export_with_qc()` writes:

| File | Content |
|------|---------|
| `qc_report.json` | Full `DatasetQC` |
| `qc_summary.csv` | Flat per-molecule summary |
| `qc_thresholds.json` | Thresholds applied |

See [dataset_qc_examples.ipynb](./dataset_qc_examples.ipynb) for a walkthrough.
