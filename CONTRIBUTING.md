# Contributing to rpfr-visualizer

Thank you for your interest in contributing! This document covers setup, workflow, and code conventions.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git clone https://github.com/SimonAndren/RPFR-visualizer.git
cd RPFR-visualizer
uv sync --all-extras
```

Fetch the published 1% tier, or generate a subsample from the full file:

```bash
uv run rpfr-visualizer data fetch --tier qm9s-1pct --dest data
uv run python scripts/make_subsample.py
```

### Verify

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/rpfr_visualizer --ignore-missing-imports
```

---

## Branch Workflow

- **`main`** — stable, release-ready. Never commit directly.
- **`develop`** — integration branch. Never commit directly.
- **`feature/<name>`** — new features. Branch from `develop`.
- **`fix/<name>`** — bug fixes. Branch from `develop`.

Before making changes:

1. Ensure `develop` exists (create from `main` if needed).
2. Create a `feature/` or `fix/` branch from `develop`.
3. Make focused commits on that branch.

Run the full test suite and lint before every commit:

```bash
uv run pytest tests/ -v && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

---

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add QC summary panel to viz module
fix: correct beta_at for off-grid temperatures
docs: update schema.md with v2.1 attrs
test: add loader edge case for missing coordinates
refactor: extract HDF5 decode helpers
chore: update CI workflow versions
```

---

## Code Style

- **Line length:** 100 characters
- **Formatter:** Ruff (handles import sorting)
- **Type hints:** required on all public API functions
- **Docstrings:** Google style
- **Python:** 3.10+ (use `X | Y` unions, etc.)

---

## Project Structure

```
src/rpfr_visualizer/
├── io/          # Loader, schema, validation, search index
├── qc/          # Structured QC engine
├── analysis/    # Fractionation, temperature fits, clumping, mass law, statistics
├── viz/         # 2D and 3D rendering; viz/explorer/ is state/controller/view
└── gui/         # The local web app
```

### Adding a new module

1. Place it in the appropriate layer (`io/`, `qc/`, `analysis/`, `viz/`).
2. Export public symbols from the package `__init__.py` if part of the top-level API.
3. Add unit tests in `tests/unit/`.
4. Update `docs/schema.md` if the data model changes.

---

## Testing

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/` (may require local HDF5 subsample)
- Shared fixtures: `tests/conftest.py`

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=rpfr_visualizer --cov-report=term-missing

# Single file
uv run pytest tests/unit/test_loader.py -v
```

Integration tests that need the 1% subsample are skipped automatically when `data/raw/qm9s_v2.2_1pt.h5` is missing.

---

## Documentation

When changing the loader, schema, or public API:

- Update [docs/schema.md](docs/schema.md) for HDF5 field changes
- Update [docs/quickstart.ipynb](docs/quickstart.ipynb) if the basic workflow changes
- Update [README.md](README.md) for user-facing changes

Key docs to keep in sync:

| Doc | Purpose |
|-----|---------|
| `docs/schema.md` | HDF5 field reference |
| `docs/ARCHITECTURE.md` | Package layer overview |
| `docs/gui.md` | The explorer |

---

## Pull Requests

1. Ensure tests pass and code is formatted.
2. Update CHANGELOG.md under `[Unreleased]` if the change is user-facing.
3. Open a PR against `develop` with a clear description and test plan.

---

## Questions

Open a [GitHub Issue](https://github.com/SimonAndren/RPFR-visualizer/issues) or [Discussion](https://github.com/SimonAndren/RPFR-visualizer/discussions).
