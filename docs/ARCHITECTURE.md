# Architecture

`rpfr_visualizer` reads the QM9s-RPFR HDF5 files, indexes them for search, and
serves a local browser explorer over the same Python API.

## Layers

```
 gui/ (rpfr-visualizer gui)      viz/explorer (notebook)      scripts / API
                 \                       |                       /
                  ----------  ExplorerController (no UI)  --------
                                         |
         analysis/  ·  qc/  ·  io/search/ (SQLite index)  ·  viz/ (2D, 3D)
                                         |
                              io/ (QM9sDataset, providers)
                                         |
                               local HDF5, schema 2.2
```

The dependency runs one way: `gui/` uses `viz/`, `viz/` uses `analysis/` and
`io/`, and nothing below imports anything above.

## I/O (`io/`)

| Module | Role |
|--------|------|
| `loader.py` | `QM9sDataset`: open a file, list molecules, load `Site` records |
| `schema.py` | Field definitions, `MoleculeRecord`, `SiteRecord`, the temperature grid |
| `validate.py` | The pipeline QC flags as a `ValidationReport` |
| `providers.py` | `HDF5Provider` and `DatasetCatalog`: one file or several under one interface |
| `molecule_index.py` | `MoleculeIndex` over interchangeable SQLite and in-memory backends |
| `search/` | Index build, query grammar, substructure screening. See [search.md](search.md) |

## QC (`qc/`)

Per-molecule `MoleculeQC` reports, with checks grouped by category (structure,
frequencies, assignment, annotation, RPFR, provenance). See [qc.md](qc.md).

## Analysis (`analysis/`)

| Module | Role |
|--------|------|
| `reference.py`, `analyze.py` | Reference frames and the α / 1000 ln α quantities |
| `fractionation.py`, `clumping.py`, `mass_law.py` | Per-site fractionation, clumped Δ, the triple-oxygen mass law |
| `temperature.py` | Off-grid β from ln β against 1/T: local quadratic (default), monotone spline, or polynomial |
| `element_stats.py` | Cached per-element percentiles that scale the opacity |
| `functional_groups.py` | Curated SMARTS presets and site-level group matching |
| `fingerprint_explorer.py` | Site ranking and local-environment grouping |
| `stats.py` | Dataset statistics |

## Visualization (`viz/`)

3D ball-and-stick (py3Dmol, bundled locally) and 2D skeletal structures (RDKit
SVG), QC text and multi-molecule grids. `viz/explorer/` is split into state,
controller and view, so the analysis can be driven with no UI attached.

| Module | Role |
|--------|------|
| `explorer/` | `ExplorerState`, `ExplorerController`, the ipywidgets view, and the factories |
| `electron_cloud.py` | `EnrichmentView`: CPK colour per element, opacity for magnitude |
| `depict.py` | The 2D structure, and why a drawing is incomplete when it is |
| `py3dmol_local.py` | The vendored 3Dmol.js loader |
| `qc_strip.py`, `colors.py`, `grid.py` | QC text, colours and symbols, grids |

## GUI (`gui/`)

`rpfr-visualizer gui` serves a local web app over `ExplorerController`, using the
standard-library HTTP server and no framework. Server state is authoritative;
the browser renders what it is sent. See [gui.md](gui.md).

## Dependencies

Core: `h5py`, `pandas`, `numpy`, `tqdm`. Extras: `gui` (RDKit, py3Dmol), `viz`
(adds matplotlib and ipywidgets), `dev`, `e2e` (Playwright).
