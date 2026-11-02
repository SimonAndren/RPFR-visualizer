# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Empirical models of per-site `1000 ln β`, trained on the dataset and shown in
  place of the DFT values from a **β from** menu in the explorer. Two baselines:
  bond additivity (ridge regression on bond-path counts) and cluster mean (the
  mean over training sites with the same local environment). See
  [docs/models.md](docs/models.md)
- `rpfr-visualizer models split | train | list`: one shared split by molecule
  (100k train / 24k test / the rest for validation), validation tuning, and
  scoring on the test set
- The explorer shows whether the open molecule is in the models' training,
  validation or test set. The CSV download has a `beta_source` column
- Schema 2.6: the 72-point grid (10–1000 K) is read and drawn; any
  `qm9s_rpfr_vX.Y.h5` registers as `qm9s_vX.Y`; the newest release on a drive
  is the one opened and preferred; a drive mounted under `/Volumes/<disk>/` is
  found without configuration

### Fixed

- `scripts/make_subsample.py` names its output after the source's version and
  refuses to overwrite an existing file without `--force`. Its fixed default
  output was the published v2.2 subsample

## [0.1.0] - 2026-10-01

First release of RPFR-visualizer, a browser explorer and Python package for the
QM9s-RPFR database.

### Added

- `rpfr-visualizer gui`: a local web app. Search by name, formula, SMILES,
  molecule ID, functional group (`fg:alcohol`, `fg:a&b`) or SMARTS; show β, α
  or 1000 ln α on the 3D structure and a 2D skeleton; rank sites; inspect data
  quality; download CSVs
- Temperatures: the 15 stored points (20–500 K) are read directly; anything
  else is fitted in ln β against 1/T, by default with a local quadratic through
  the four nearest stored points. Monotone spline and polynomials of degree
  2–5 are offered
- Derived panel: β against temperature, the triple-oxygen mass-law exponent λ,
  and clumped Δᵢⱼ = 1000 ln(βᵢⱼ / βᵢβⱼ) in per mil
- In 3D, colour is the element and opacity is the magnitude of the quantity
  for that element, scaled to the dataset's 5th–95th percentile
- CSV export of one molecule or every search match, at one or several
  temperatures
- A SQLite search index per data release, with `rpfr-visualizer index
  build|info|verify`
- `rpfr-visualizer data fetch`: the published 1% tier of QM9s v2.2 (1,298
  molecules) and its index, checksummed on arrival
- `QM9sDataset` loader for schema 2.0–2.2, structured QC reports, and
  per-site analysis from Python
- Tutorial notebook `docs/explorer_tutorial.ipynb`
