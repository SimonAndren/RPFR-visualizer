"""Molecule search index.

The published data ships with a prebuilt SQLite index, so the normal path is::

    from rpfr_visualizer.io.search import open_index
    backend = open_index()          # discovers the index file

Developers working from raw HDF5 build one first::

    rpfr-visualizer index build --raw data/raw --out data/cache/qm9s_rpfr_index.sqlite
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from rpfr_visualizer.io.search.build import BuildReport, build_index
from rpfr_visualizer.io.search.models import MoleculeIndexEntry, MoleculeSearchHit
from rpfr_visualizer.io.search.schema import INDEX_VERSION
from rpfr_visualizer.io.search.sqlite_backend import IndexVersionError, SqliteSearchBackend

if TYPE_CHECKING:
    from rpfr_visualizer.io.providers import DatasetCatalog

#: Filename of the published index artifact.
INDEX_FILENAME = "qm9s_rpfr_index.sqlite"

#: Environment variable that overrides index discovery.
INDEX_ENV_VAR = "RPFR_VISUALIZER_INDEX"

_DOWNLOAD_HINT = (
    "Download the prebuilt index published alongside the QM9s-RPFR data, or "
    "build one from your HDF5 files with:\n"
    "  rpfr-visualizer index build --raw data/raw --out data/cache/" + INDEX_FILENAME
)


class MissingIndexError(FileNotFoundError):
    """Raised when no search index could be found."""


def default_index_path(data_dir: Path | str = "data") -> Path:
    """Conventional location of the index under a data directory."""
    data = Path(data_dir)
    if data.name == "raw":
        data = data.parent
    return data / "cache" / INDEX_FILENAME


def find_index(path: Path | str | None = None, *, data_dir: Path | str = "data") -> Path | None:
    """Locate an index file: explicit path, then $RPFR_VISUALIZER_INDEX, then data dir."""
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.exists() else None
    env = os.environ.get(INDEX_ENV_VAR)
    if env:
        candidate = Path(env)
        if candidate.exists():
            return candidate
    candidate = default_index_path(data_dir)
    return candidate if candidate.exists() else None


def open_index(
    path: Path | str | None = None, *, data_dir: Path | str = "data"
) -> SqliteSearchBackend:
    """Open a prebuilt index read-only.

    Raises :class:`MissingIndexError` with download and build instructions when
    no index is found.
    """
    found = find_index(path, data_dir=data_dir)
    if found is None:
        looked = path or os.environ.get(INDEX_ENV_VAR) or default_index_path(data_dir)
        raise MissingIndexError(f"No search index at {looked}.\n{_DOWNLOAD_HINT}")
    return SqliteSearchBackend(found)


def open_or_build_index(
    catalog: DatasetCatalog,
    path: Path | str | None = None,
    *,
    data_dir: Path | str = "data",
    progress: bool = True,
) -> SqliteSearchBackend:
    """Open an index, building it from *catalog* if it does not exist yet.

    Convenience for development; the published workflow ships the index.
    """
    found = find_index(path, data_dir=data_dir)
    if found is None:
        found = Path(path) if path is not None else default_index_path(data_dir)
        build_index(catalog, found, progress=progress)
    return SqliteSearchBackend(found)


__all__ = [
    "INDEX_ENV_VAR",
    "INDEX_FILENAME",
    "INDEX_VERSION",
    "BuildReport",
    "IndexVersionError",
    "MissingIndexError",
    "MoleculeIndexEntry",
    "MoleculeSearchHit",
    "SqliteSearchBackend",
    "build_index",
    "default_index_path",
    "find_index",
    "open_index",
    "open_or_build_index",
]
