#!/usr/bin/env python3
"""Rebuild the enriched molecule search index from registered subsample files."""

from __future__ import annotations

import argparse
from pathlib import Path

from rpfr_visualizer.io.molecule_index import (
    build_or_load_index,
    index_cache_path,
    register_subsample_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing *_1pct.h5 subsample files.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Output cache path (default: data/cache/molecule_index.json).",
    )
    args = parser.parse_args()

    catalog = register_subsample_catalog(args.raw)
    if not catalog.datasets:
        raise SystemExit(f"No subsample HDF5 files found under {args.raw}")

    cache = args.cache or index_cache_path(args.raw)
    index = build_or_load_index(catalog, cache, force_rebuild=True)
    n_with_formula = sum(1 for e in index.iter_entries() if e.formula)
    print(f"Wrote {len(index)} entries ({n_with_formula} with formula) to {cache}")
    print(f"Datasets: {', '.join(catalog.datasets)}")


if __name__ == "__main__":
    main()
