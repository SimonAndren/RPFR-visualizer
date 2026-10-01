# RPFR-visualizer

[![Tests](https://github.com/SimonAndren/RPFR-visualizer/actions/workflows/test.yml/badge.svg)](https://github.com/SimonAndren/RPFR-visualizer/actions/workflows/test.yml)
[![Lint](https://github.com/SimonAndren/RPFR-visualizer/actions/workflows/lint.yml/badge.svg)](https://github.com/SimonAndren/RPFR-visualizer/actions/workflows/lint.yml)
[![Coverage](https://github.com/SimonAndren/RPFR-visualizer/actions/workflows/coverage.yml/badge.svg)](https://github.com/SimonAndren/RPFR-visualizer/actions/workflows/coverage.yml)

A browser explorer for the **QM9s-RPFR** database: reduced partition function
ratios (β) for every atom of the QM9s molecules, from DFT harmonic frequencies,
at 15 temperatures from 20 to 500 K.

Search by name, formula, SMILES or functional group. See β, α or 1000 ln α on
the 3D structure and a 2D skeleton, rank sites, follow β with temperature,
inspect clumped-isotope Δ and the triple-oxygen mass law, check data quality,
and download CSVs. Everything runs locally.

---

## Install

```bash
pip install 'rpfr-visualizer[gui]'
```

`[all]` adds the notebook stack (matplotlib, ipywidgets). Neither installs the
test or lint toolchain.

## Get the data

```bash
rpfr-visualizer data fetch --tier qm9s-1pct --dest data
```

About 92 MB: a 1% subsample of QM9s v2.2 (1,298 molecules, schema 2.2,
Eckart-projected) and the search index built over exactly that file. Each file
is checksummed on arrival. `rpfr-visualizer data list` shows what is available.

## Open it

```bash
rpfr-visualizer gui --data data
```

No data yet? `rpfr-visualizer gui --demo` opens five synthetic molecules. Their
numbers are invented; they show the interface, not the chemistry.

The full QM9s v2.2 file works too. Point `--drive` (or `$RPFR_VISUALIZER_DRIVE`)
at the folder holding `qm9s/qm9s_rpfr_v2.2.h5`, and build an index once:

```bash
rpfr-visualizer index build --drive /path/to/Processed-Data --out data/cache/qm9s_rpfr_full_index.sqlite
rpfr-visualizer gui --drive /path/to/Processed-Data --index data/cache/qm9s_rpfr_full_index.sqlite
```

---

## Documentation

| | |
|---|---|
| [docs/gui.md](docs/gui.md) | The explorer: quantities, temperatures and fits, pictures, downloads |
| [docs/explorer_tutorial.ipynb](docs/explorer_tutorial.ipynb) | A runnable tour of the Python API behind the app |
| [docs/quickstart.ipynb](docs/quickstart.ipynb) | Load a file, read β, compute statistics |
| [docs/search.md](docs/search.md) | Query grammar and the search index |
| [docs/schema.md](docs/schema.md) | HDF5 field reference (schema 2.2) |
| [docs/qc.md](docs/qc.md) | QC checks, tiers and thresholds |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the package is layered |

## From Python

```python
from pathlib import Path

from rpfr_visualizer import QM9sDataset

ds = QM9sDataset(Path("data/raw/qm9s_v2.2_1pt.h5"))
mol = ds.get_molecule(ds.molecule_ids[0])
print(mol.smiles, mol.sites[0].beta_at(298.15))
```

Off-grid temperatures are fitted in ln β against 1/T, by default with a local
quadratic through the four nearest stored points.

---

## Development

```bash
git clone https://github.com/SimonAndren/RPFR-visualizer.git
cd RPFR-visualizer
uv sync --all-extras
uv run pytest tests/
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

The browser tests drive the Chrome already installed. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow.

## Citation

See [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button reads it.

```bibtex
@software{rpfr_visualizer,
  author  = {Andren, Simon},
  title   = {RPFR-visualizer: a browser explorer for the QM9s-RPFR isotope effects database},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/SimonAndren/RPFR-visualizer}
}
```

## License

MIT. See [LICENSE](LICENSE).
