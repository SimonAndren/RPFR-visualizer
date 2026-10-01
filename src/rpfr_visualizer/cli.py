"""Command-line interface for rpfr-visualizer.

Open the explorer::

    rpfr-visualizer gui

Or, with no data at all, on a synthetic five-molecule dataset::

    rpfr-visualizer gui --demo

Get the data::

    rpfr-visualizer data list
    rpfr-visualizer data fetch --tier qm9s-1pct --dest data

Manage the search index::

    rpfr-visualizer index build --raw data/raw --out data/cache/qm9s_rpfr_index.sqlite
    rpfr-visualizer index info  data/cache/qm9s_rpfr_index.sqlite
    rpfr-visualizer index verify data/cache/qm9s_rpfr_index.sqlite --raw data/raw
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from rpfr_visualizer.io.search import INDEX_FILENAME, default_index_path


def _catalog_from_args(args: argparse.Namespace):
    """Build a dataset catalog from --raw or --drive."""
    from rpfr_visualizer.io.molecule_index import register_drive_catalog, register_subsample_catalog

    if args.drive:
        return register_drive_catalog(args.drive)
    catalog = register_subsample_catalog(Path(args.raw))
    if not catalog.datasets:
        raise SystemExit(
            f"No HDF5 datasets found under {args.raw}. Pass --raw <dir> or --drive <root>."
        )
    return catalog


def _cmd_index_build(args: argparse.Namespace) -> int:
    from rpfr_visualizer.io.search import build_index

    catalog = _catalog_from_args(args)
    out = Path(args.out) if args.out else default_index_path(args.raw)
    datasets = tuple(args.datasets) if args.datasets else None
    report = build_index(
        catalog,
        out,
        datasets=datasets,
        with_functional_groups=not args.no_functional_groups,
        progress=not args.quiet,
    )
    if args.quiet:
        print(report.summary())
    return 0


def _cmd_index_info(args: argparse.Namespace) -> int:
    from rpfr_visualizer.io.search import SqliteSearchBackend

    backend = SqliteSearchBackend(args.path, check_version=False)
    meta = backend.meta()
    print(f"index:   {args.path}")
    print(f"size:    {Path(args.path).stat().st_size / 1e6:.1f} MB")
    for key in sorted(meta):
        if key == "source_signature":
            continue
        print(f"{key + ':':16s} {meta[key]}")
    print(f"{'molecules:':16s} {backend.count()}")
    print("datasets:")
    for dataset, source_file, count in backend.source_signature():
        print(f"  {dataset:20s} {count:>8} molecules   ({source_file})")
    return 0


def _cmd_index_verify(args: argparse.Namespace) -> int:
    from rpfr_visualizer.io.search import SqliteSearchBackend

    problems: list[str] = []

    with sqlite3.connect(args.path) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        problems.append(f"integrity_check: {integrity}")
    else:
        print("integrity_check: ok")

    try:
        backend = SqliteSearchBackend(args.path)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"index_version:   ok ({backend.meta().get('index_version')})")
    print(f"molecules:       {backend.count()}")

    stored_rdkit = backend.rdkit_version()
    try:
        import rdkit

        if stored_rdkit and stored_rdkit != rdkit.__version__:
            problems.append(
                f"built with RDKit {stored_rdkit}, running {rdkit.__version__} — "
                "canonical SMILES and fingerprints may differ; rebuild if lookups miss"
            )
        else:
            print(f"rdkit_version:   ok ({stored_rdkit})")
    except ImportError:
        print("rdkit_version:   skipped (RDKit not installed)")

    if args.raw or args.drive:
        catalog = _catalog_from_args(args)
        stored = {row[0]: (row[1], row[2]) for row in backend.source_signature()}
        for name in catalog.datasets:
            provider = catalog.get_provider(name)
            if name not in stored:
                problems.append(f"dataset {name!r} is not in the index")
                continue
            source_file, count = stored[name]
            actual = len(provider.list_keys())
            if actual != count:
                problems.append(f"dataset {name!r}: index has {count} molecules, file has {actual}")
            elif provider.path.name != source_file:
                # Renaming is fine; the index is path-independent by design.
                print(f"note: {name!r} built from {source_file}, now {provider.path.name}")
        missing = set(stored) - set(catalog.datasets)
        if missing:
            print(f"note: index also covers {', '.join(sorted(missing))} (not in catalog)")

    if problems:
        print("\nProblems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


def demo_data_dir() -> Path:
    """Where ``--demo`` keeps its generated dataset.

    Beside the remembered GUI settings, so the demo lives with the rest of
    this tool's state rather than wherever the shell happened to be.
    """
    from rpfr_visualizer.gui.config import config_path

    return config_path().parent / "demo"


def _cmd_data_list(args: argparse.Namespace) -> int:
    from rpfr_visualizer.fetch import human_bytes, load_manifest

    for tier in load_manifest().values():
        print(f"{tier.key}  ({human_bytes(tier.total_bytes)}, {len(tier.artifacts)} files)")
        if tier.description:
            print(f"    {tier.description}")
        print(f"    from {tier.base_url}")
    return 0


def _cmd_data_fetch(args: argparse.Namespace) -> int:
    from rpfr_visualizer.fetch import FetchError, fetch_tier, human_bytes, load_manifest, verify

    tiers = load_manifest()
    tier = tiers.get(args.tier)
    if tier is None:
        raise SystemExit(f"Unknown tier {args.tier!r}. Available: {', '.join(sorted(tiers))}")

    root = Path(args.dest)

    if args.verify_only:
        bad = [a.name for a in tier.artifacts if not verify(a, root)]
        if bad:
            print(f"Missing or damaged under {root}:")
            for name in bad:
                print(f"  {name}")
            return 1
        print(f"All {len(tier.artifacts)} files present and verified under {root}.")
        return 0

    print(
        f"Fetching tier {tier.key!r}: {len(tier.artifacts)} files, "
        f"{human_bytes(tier.total_bytes)} into {root}"
    )
    try:
        for artifact, status in fetch_tier(tier, root, force=args.force):
            mark = "have" if status == "present" else "got "
            print(f"  {mark} {artifact.name}  ({human_bytes(artifact.size_bytes)})")
    except FetchError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"\nDone. Open it with:\n  rpfr-visualizer gui --data {root}")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    # Imported here so `rpfr-visualizer index build` never loads RDKit or the server.
    from rpfr_visualizer.gui.server import launch

    data_dir = args.data
    index_path = args.index
    if args.demo:
        if args.data or args.drive:
            raise SystemExit(
                "--demo generates its own dataset; it cannot be combined with --data or --drive."
            )
        from rpfr_visualizer.demo import ensure_demo_data

        target = demo_data_dir()
        print(f"Building the demo dataset in {target} ...", flush=True)
        demo = ensure_demo_data(target)
        data_dir = str(demo.root)
        index_path = str(demo.index_path)
        print(
            "\nThis is a SYNTHETIC demo. Its five molecules and every beta value\n"
            "in them are invented, and say nothing about real isotope chemistry.\n"
            "It exists so the interface can be tried without a download.\n",
            flush=True,
        )

    return launch(
        data_dir=data_dir,
        index_path=index_path,
        drive_root=args.drive,
        dataset=args.dataset,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        max_stats_molecules=args.max_stats_molecules,
        reset_config=args.reset_config,
        verbose=args.verbose,
        dev=args.dev,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpfr-visualizer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gui = sub.add_parser("gui", help="open the RPFR visualizer in a browser")
    # Default None, not "data": a hardcoded relative path resolves against the
    # process working directory, which is the bug this whole layer avoids.
    gui.add_argument(
        "--data",
        default=None,
        help="a .h5 file, a data directory, or a drive root (default: auto-discover)",
    )
    gui.add_argument("--index", default=None, help="path to a .sqlite search index")
    gui.add_argument(
        "--demo",
        action="store_true",
        help="run on a generated five-molecule dataset; needs no data files",
    )
    gui.add_argument("--drive", default=None, help="folder holding qm9s/qm9s_rpfr_v2.2.h5")
    gui.add_argument("--dataset", default=None, help="dataset to open (default: qm9s_v2.2)")
    gui.add_argument("--host", default="127.0.0.1", help="loopback address to bind")
    gui.add_argument("--port", type=int, default=8765, help="port (0 picks any free one)")
    gui.add_argument("--no-browser", action="store_true", help="do not open a browser")
    gui.add_argument(
        "--max-stats-molecules",
        type=int,
        default=None,
        help="molecules sampled for the colour-scale ranges (default 200)",
    )
    gui.add_argument("--reset-config", action="store_true", help="forget remembered settings")
    gui.add_argument("-v", "--verbose", action="store_true", help="log every request")
    gui.add_argument(
        "--dev",
        action="store_true",
        help="report browser-side errors to this terminal (implies --verbose)",
    )
    gui.set_defaults(func=_cmd_gui)

    data = sub.add_parser("data", help="download published data tiers")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    data_list = data_sub.add_parser("list", help="show the tiers this version knows about")
    data_list.set_defaults(func=_cmd_data_list)

    fetch = data_sub.add_parser("fetch", help="download a tier and verify its checksums")
    fetch.add_argument(
        "--tier", default="qm9s-1pct", help="which tier to fetch (default: qm9s-1pct)"
    )
    fetch.add_argument("--dest", default="data", help="destination root (default: data)")
    fetch.add_argument("--force", action="store_true", help="re-download files that already verify")
    fetch.add_argument(
        "--verify-only",
        action="store_true",
        help="check what is already on disk and download nothing",
    )
    fetch.set_defaults(func=_cmd_data_fetch)

    index = sub.add_parser("index", help="manage the molecule search index")
    index_sub = index.add_subparsers(dest="index_command", required=True)

    build = index_sub.add_parser("build", help="build an index from HDF5 datasets")
    build.add_argument("--raw", default="data/raw", help="directory of HDF5 subsamples")
    build.add_argument("--drive", default=None, help="folder holding qm9s/qm9s_rpfr_v2.2.h5")
    build.add_argument(
        "--out", default=None, help=f"output path (default data/cache/{INDEX_FILENAME})"
    )
    build.add_argument("--datasets", nargs="*", default=None, help="limit to these datasets")
    build.add_argument(
        "--no-functional-groups",
        action="store_true",
        help="skip precomputing functional-group membership",
    )
    build.add_argument("-q", "--quiet", action="store_true", help="only print the summary")
    build.set_defaults(func=_cmd_index_build)

    info = index_sub.add_parser("info", help="show what an index contains")
    info.add_argument("path", help="path to the .sqlite index")
    info.set_defaults(func=_cmd_index_info)

    verify = index_sub.add_parser("verify", help="check an index for damage or drift")
    verify.add_argument("path", help="path to the .sqlite index")
    verify.add_argument("--raw", default=None, help="cross-check against HDF5 files here")
    verify.add_argument("--drive", default=None, help="cross-check against the drive root")
    verify.set_defaults(func=_cmd_index_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
