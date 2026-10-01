#!/usr/bin/env python3
"""Create a schema-faithful 1% subsample of a QM9s-RPFR HDF5 file.

Copies selected molecule groups verbatim (datasets + group attrs) and
preserves all root-level file attributes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from rpfr_visualizer.io.loader import _molecule_keys
from rpfr_visualizer.io.molecule_index import default_drive_root


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    """Copy all attributes from source to destination."""
    for key, value in src.items():
        dst[key] = value


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    """Recursively copy a group and its datasets."""
    _copy_attrs(src.attrs, dst.attrs)
    for key in src.keys():
        item = src[key]
        if isinstance(item, h5py.Group):
            _copy_group(item, dst.create_group(key))
        else:
            src.copy(item, dst, key)


def make_subsample(
    source_path: Path,
    output_path: Path,
    *,
    fraction: float = 0.01,
    seed: int = 42,
) -> int:
    """Write a subsampled HDF5 file and return the number of molecules copied."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as src_file:
        molecule_ids = _molecule_keys(src_file)
        if not molecule_ids:
            raise ValueError(f"No molecule groups found in {source_path}")
        n_select = max(1, int(round(len(molecule_ids) * fraction)))
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(molecule_ids, size=n_select, replace=False), key=str)

        with h5py.File(output_path, "w") as dst_file:
            _copy_attrs(src_file.attrs, dst_file.attrs)
            dst_file.attrs["subsample_fraction"] = fraction
            dst_file.attrs["subsample_seed"] = seed
            dst_file.attrs["subsample_n_molecules"] = n_select
            dst_file.attrs["subsample_source"] = str(source_path)

            # Root datasets, not just root attributes. Schema 2.2 stores the
            # temperature grid as /temperatures rather than as the JSON
            # attribute 2.0 used, so copying attributes alone produced a
            # subsample with schema_version 2.2 and no grid at all -- which a
            # reader then filled in from the hardcoded 2.0 default, silently
            # relabelling every column.
            for key in src_file.keys():
                item = src_file[key]
                if not isinstance(item, h5py.Group):
                    src_file.copy(item, dst_file, key)

            for mol_id in selected:
                _copy_group(src_file[mol_id], dst_file.create_group(mol_id))

    return n_select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=default_drive_root() / "qm9s" / "qm9s_tr_project.h5",
        help="Path to the full QM9s-RPFR HDF5 file (default: under $RPFR_VISUALIZER_DRIVE).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/qm9s_v2.2_1pt.h5"),
        help="Path for the subsampled output file.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.01,
        help="Fraction of molecules to copy (default: 0.01).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for molecule selection (default: 42).",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source file not found: {args.source}")

    n_copied = make_subsample(
        args.source,
        args.output,
        fraction=args.fraction,
        seed=args.seed,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {n_copied} molecules to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
