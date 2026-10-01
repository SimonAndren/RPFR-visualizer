"""Ways to build a configured :class:`ExplorerController`.

Separate from :mod:`~rpfr_visualizer.viz.explorer.controller` because these functions
import the catalog and search machinery at module level, which the controller
deliberately keeps lazy; and separate from the view because a controller is what
a non-widget front end needs.

Nothing here ever *builds* a search index. Building over the full datasets takes
tens of minutes, so a missing index is reported with the command that would fix
it rather than triggered implicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from rpfr_visualizer.io.molecule_index import (
    MoleculeIndex,
    dataset_name_for,
    default_drive_root,
    register_all,
    register_drive_catalog,
    register_subsample_catalog,
)
from rpfr_visualizer.io.providers import DatasetCatalog, HDF5Provider
from rpfr_visualizer.io.search import find_index, open_index
from rpfr_visualizer.viz.explorer.controller import ExplorerController

#: Opened in preference to whatever happens to be first in a catalog: the
#: current QM9s release, ahead of any older QM9s file in the same folder.
PREFERRED_DATASET = "qm9s_v2.2"


def _datasets(n: int) -> str:
    """ "1 dataset" / "3 datasets", rather than "1 datasets"."""
    return f"{n} dataset" if n == 1 else f"{n} datasets"


def preferred_dataset(catalog: DatasetCatalog, requested: str | None = None) -> str:
    """Pick which dataset to open.

    An explicit request wins. Otherwise prefer :data:`PREFERRED_DATASET`, then
    fall back to the first registered dataset.
    """
    names = catalog.datasets
    if not names:
        raise ValueError("Catalog has no registered datasets")
    if requested:
        if requested not in names:
            raise KeyError(f"Dataset {requested!r} not in catalog. Available: {', '.join(names)}")
        return requested
    return PREFERRED_DATASET if PREFERRED_DATASET in names else names[0]


def _open_index_or_none(
    index_path: Path | str | None,
    *,
    data_dir: Path | str,
) -> MoleculeIndex | None:
    """Open an existing index, or return None when there is none to open.

    Returning None rather than raising lets a caller run without search — every
    molecule is still reachable by key — while never silently starting a build.

    An index that exists but was written by an incompatible builder is a
    different matter: :class:`~rpfr_visualizer.io.search.IndexVersionError` propagates,
    because querying columns that mean something else is worse than refusing.
    """
    resolved = find_index(index_path, data_dir=data_dir)
    if resolved is None:
        return None
    return MoleculeIndex(backend=open_index(resolved), allow_pubchem=False)


#: Keys sampled when checking that an index describes the data on hand.
_MATCH_SAMPLE = 5


def _h5_files(directory: Path) -> list[Path]:
    """HDF5 files in *directory*, ignoring macOS AppleDouble sidecars.

    An external drive formatted for macOS is full of ``._name.h5`` files, which
    are metadata forks rather than data and cannot be opened.
    """
    return sorted(p for p in directory.glob("*.h5") if not p.name.startswith("._"))


def catalog_for_path(path: Path | str) -> tuple[DatasetCatalog, str]:
    """Build a catalog from whatever the user pointed at, and say what it was.

    Four layouts are recognised, because a person with data on disk should not
    have to know which loader wants which shape:

    * a single ``.h5`` file
    * a directory holding ``raw/*.h5`` — the repo's subsample layout
    * a directory of per-dataset subdirectories, ``<name>/<name>_tr_*.h5`` — the
      external-drive layout
    * a directory with ``.h5`` files sitting directly in it

    Raises
    ------
    FileNotFoundError
        With the layouts it looked for, rather than naming only one of them.
    """
    target = Path(path).expanduser()

    if target.is_file():
        if target.suffix != ".h5":
            raise FileNotFoundError(f"{target} is not an HDF5 (.h5) file.")
        # Named from the filename like every other branch. The HDF5 ``dataset``
        # attribute would collide: both qm9s_tr_* files carry "qm9s".
        name = dataset_name_for(target)
        catalog = DatasetCatalog()
        catalog.register_provider(name, HDF5Provider(target))
        return catalog, f"the file {target.name} (dataset {name!r})"

    if not target.is_dir():
        raise FileNotFoundError(f"No such file or directory: {target}")

    # The repo layout.
    if _h5_files(target / "raw"):
        catalog = register_subsample_catalog(target / "raw")
        if catalog.datasets:
            return catalog, f"{_datasets(len(catalog.datasets))} under {target / 'raw'}"

    # The external-drive layout: one subdirectory per dataset.
    drive = DatasetCatalog()
    nested = [
        h5
        for child in sorted(
            p for p in target.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        for h5 in _h5_files(child)
    ]
    register_all(drive, nested)
    if drive.datasets:
        return drive, f"{_datasets(len(drive.datasets))} under {target}"

    # Files sitting directly in the directory.
    flat = DatasetCatalog()
    register_all(flat, _h5_files(target))
    if flat.datasets:
        return flat, f"{_datasets(len(flat.datasets))} in {target}"

    raise FileNotFoundError(
        f"No HDF5 data found at {target}. Looked for: the file itself if you gave one, "
        f"{target / 'raw'}/*.h5, per-dataset subdirectories like {target.name}/qm9s/*.h5, "
        f"and *.h5 directly in {target}."
    )


#: An index covering less than this share of the data file is treated as the
#: wrong index rather than a slightly stale one.
_COVERAGE_FLOOR = 0.5


@lru_cache(maxsize=32)
def _molecule_count(path: Path, mtime: float) -> int:
    """Number of root groups in an HDF5 file.

    About six seconds on a 9 GB file, and the coverage check would otherwise pay
    it once per candidate index. Keyed on mtime so a rebuilt file is not served a
    stale count.
    """
    import h5py

    try:
        with h5py.File(path, "r") as h5:
            return len(h5)
    except OSError:
        return 0


def _coverage_gap(provider: HDF5Provider, index: MoleculeIndex, dataset: str) -> str:
    """Report an index that covers only a fraction of the data file.

    This is what catches the case that matters: a subsample index opened against
    the full datasets, which searches 1% of them while looking entirely healthy.
    """
    indexed = index.backend.count(datasets=(dataset,))
    try:
        in_file = _molecule_count(provider.path, provider.path.stat().st_mtime)
    except OSError:
        return ""
    if not in_file or indexed >= in_file * _COVERAGE_FLOOR:
        return ""
    return (
        f"The search index covers {indexed:,} of the {in_file:,} molecules in "
        f"{provider.path.name} ({100.0 * indexed / in_file:.1f}%). Searching would miss most of "
        "this dataset. This is usually a subsample index opened against the full data — point at "
        "the index built from these files, or build one."
    )


def best_index_for(
    catalog: DatasetCatalog,
    dataset: str,
    candidates: Sequence[Path | str],
) -> tuple[MoleculeIndex | None, str]:
    """Pick the index that actually describes *catalog*, and say why.

    Choosing by filename is not good enough. ``find_index`` knows exactly one
    name, so on a machine holding both a subsample index and a full one it
    returns the subsample — and against the full data that means searching 1% of
    it while appearing to work perfectly.

    So every candidate is opened, and the one covering the *most* molecules of
    this dataset while still matching it wins. Both halves matter: the count
    alone would pick a big index built from different files, and matching alone
    would accept a 1% index over the full data, since a subsample's keys are all
    present in the file it was sampled from.

    Ranking by count costs one ``COUNT(*)`` per candidate; verifying the match
    costs a handful of ``has_key`` calls. Neither scans the HDF5.
    """
    checked: list[str] = []
    best: tuple[int, MoleculeIndex, str] | None = None

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            index = MoleculeIndex(backend=open_index(path), allow_pubchem=False)
        except Exception as exc:  # noqa: BLE001 — a bad candidate is not fatal
            checked.append(f"{path.name}: {exc}")
            continue
        covered = index.backend.count(datasets=(dataset,))
        if covered == 0:
            # Checked before the sample-based comparison, and separately from it,
            # because an index covering none of this dataset used to win by
            # default: it raised no mismatch to compare, and ``covered > best[0]``
            # never runs when ``best`` is still None.
            known = ", ".join(index.datasets) or "no datasets"
            checked.append(f"{path.name}: covers {known}, not {dataset!r}")
            continue
        if index_data_mismatch(catalog, index, dataset):
            checked.append(f"{path.name}: describes different molecules than this data")
            continue
        if best is None or covered > best[0]:
            best = (covered, index, f"{path.name} ({len(index):,} molecules)")

    if best is not None:
        note = best[2]
        if checked:
            note += f"; skipped {len(checked)}"
        return best[1], note

    joined = "; ".join(checked)
    return None, f"no matching index ({joined})" if joined else "no index found"


def index_data_mismatch(
    catalog: DatasetCatalog,
    index: MoleculeIndex | None,
    dataset: str,
) -> str:
    """Explain it when an index describes molecules the data files do not hold.

    A SQLite index is self-contained, so it keeps answering searches long after
    the HDF5 files it was built from have gone — an index over the full datasets
    still searches all 881,757 molecules with the external drive unplugged. But
    every hit is then unopenable, which reads as a broken app rather than as
    missing data. Detecting it costs a handful of ``has_key`` calls (about 0.1 ms
    each), so the app can say so up front.

    Returns an empty string when the index and the data agree, or when there is
    nothing to compare.
    """
    if index is None:
        return ""
    try:
        provider = catalog.get_provider(dataset)
    except KeyError:
        return ""

    sample = [
        entry.key
        for entry, _ in zip(
            index.iter_entries(datasets=(dataset,)), range(_MATCH_SAMPLE), strict=False
        )
    ]
    if not sample:
        # No entries at all for this dataset. That is not "nothing to compare":
        # every search would come back empty however it was spelled, while the
        # index still reports its full molecule count and looks healthy.
        covered = index.datasets
        if not covered:
            return ""
        return (
            f"The search index holds {len(index):,} molecules but none for {dataset!r}; "
            f"it covers {', '.join(covered)}. Every search in this dataset would return "
            "nothing. Point at an index built for this data, or open the data these "
            "names belong to."
        )
    if any(provider.has_key(key) for key in sample):
        # The index's molecules are in the file. That is necessary but not
        # sufficient: a 1% subsample index passes this trivially against the full
        # file it was sampled from, so also check the file has no large body of
        # molecules the index has never heard of.
        return _coverage_gap(provider, index, dataset)

    indexed = index.backend.count(datasets=(dataset,))
    return (
        f"The search index describes {indexed:,} molecules in {dataset!r}, but none of the "
        f"first {len(sample)} are present in {provider.path.name}. Searches will return "
        "molecules that cannot be opened. This usually means the index was built from the "
        "full datasets while only a subsample (or no external drive) is available — point at "
        "the matching index, or reconnect the drive the index was built from."
    )


def controller_for_path(
    path: Path | str,
    *,
    dataset: str | None = None,
    index_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    max_stats_molecules: int | None = 200,
) -> tuple[ExplorerController, list[str]]:
    """Open whatever the user pointed at, and report what was found.

    This is the forgiving entry point: a single ``.h5`` file, a repo-style data
    directory, or an external-drive root all work, because someone who has data
    on disk should not have to know which loader wants which layout.

    Returns the controller and human-readable notes about what was opened and
    which index matched — the notes belong on screen, since "opened 7 datasets
    and a 881,757-molecule index" and "opened one file with no search" are very
    different situations to be in.
    """
    catalog, what = catalog_for_path(path)
    chosen = preferred_dataset(catalog, dataset)
    notes = [f"Opened {what}."]

    # Normalise whatever was given: data_dir may itself be the .h5 file the user
    # pointed at, which is not a place to look for caches.
    caches = _cache_dir_for(data_dir if data_dir is not None else path)
    candidates: list[Path] = []
    if index_path:
        candidates.append(Path(index_path))
    else:
        # Exactly one place to look, so this cannot quietly pick up an index from
        # wherever the process happens to be running.
        candidates.extend(sorted((caches / "cache").glob("*.sqlite")))
    index, why = best_index_for(catalog, chosen, candidates)
    notes.append(f"Search index: {why}, found in {caches / 'cache'}.")
    if index is None:
        notes.append(
            "Without a matching index you can still open molecules by ID, but not search by "
            "name, formula, structure or functional group. Build one with: rpfr-visualizer index build"
        )

    controller = ExplorerController(
        catalog,
        dataset=chosen,
        index=index,
        data_dir=caches,
        max_stats_molecules=max_stats_molecules,
    )
    return controller, notes


def _cache_dir_for(path: Path | str) -> Path:
    """Where to keep caches for data at *path*.

    Never inside the data location itself: that may be a read-only or removable
    volume, and a 9 GB file's neighbour directory is a poor place for a JSON
    cache. Falls back to the working directory's ``data``.
    """
    target = Path(path).expanduser()
    if target.is_file():
        target = target.parent
    if (target / "raw").is_dir() or (target / "cache").is_dir():
        return target
    return Path("data").resolve()


def controller_for_data_dir(
    data_dir: Path | str,
    *,
    dataset: str | None = None,
    index_path: Path | str | None = None,
    max_stats_molecules: int | None = 200,
) -> ExplorerController:
    """Build a controller over the HDF5 subsamples under ``<data_dir>/raw``.

    Raises
    ------
    FileNotFoundError
        If no recognised dataset file is present.
    """
    data_dir = Path(data_dir)
    catalog = register_subsample_catalog(data_dir / "raw")
    if not catalog.datasets:
        raise FileNotFoundError(
            f"No QM9s-RPFR datasets found under {data_dir / 'raw'}. Expected files such as "
            "qm9s_v2.2_1pt.h5 — create them with scripts/make_subsample.py, or point "
            "at a different directory."
        )
    return ExplorerController(
        catalog,
        dataset=preferred_dataset(catalog, dataset),
        index=_open_index_or_none(index_path, data_dir=data_dir),
        data_dir=data_dir,
        max_stats_molecules=max_stats_molecules,
    )


def controller_for_drive(
    dataset: str | None = None,
    *,
    drive_root: Path | str | None = None,
    index_path: Path | str | None = None,
    data_dir: Path | str = "data",
    max_stats_molecules: int | None = 200,
) -> ExplorerController:
    """Build a controller over the **full** datasets on the external drive.

    The full files are tens of gigabytes, so nothing is read up front: molecules
    are loaded from HDF5 one at a time and all searching is answered by the
    SQLite index.

    Raises
    ------
    FileNotFoundError
        If the drive is not mounted, or no search index is found. Building an
        index over the full data takes tens of minutes, so it is never started
        implicitly.
    """
    catalog = register_drive_catalog(drive_root)
    chosen = preferred_dataset(catalog, dataset)

    index = _open_index_or_none(index_path, data_dir=data_dir)
    if index is None:
        raise FileNotFoundError(
            "No search index found, and the full datasets are unusable without one.\n"
            "Build it once with:\n"
            f"  rpfr-visualizer index build --drive '{drive_root or default_drive_root()}' "
            "--out data/cache/qm9s_rpfr_full_index.sqlite"
        )

    return ExplorerController(
        catalog,
        dataset=chosen,
        index=index,
        data_dir=data_dir,
        max_stats_molecules=max_stats_molecules,
    )


__all__ = [
    "PREFERRED_DATASET",
    "best_index_for",
    "catalog_for_path",
    "controller_for_data_dir",
    "controller_for_drive",
    "controller_for_path",
    "index_data_mismatch",
    "preferred_dataset",
]
