"""Finding the data, and remembering where it was.

A GUI can be launched from anywhere, by someone who has never set an
environment variable. So it discovers what it can, shows what it found, and
saves the answer — but it never guesses between two plausible index files, and
it never starts a thirty-minute index build on its own.

Discovery *proposes*; the user confirms.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rpfr_visualizer.io.search import INDEX_ENV_VAR, find_index

#: Bumped when the remembered schema changes meaning. An unrecognised version is
#: treated as no config at all rather than guessed at.
CONFIG_VERSION = 1

#: Overrides the config location. Exists mainly so tests never touch a real
#: user's file — set it rather than monkeypatching :func:`config_path`.
CONFIG_ENV_VAR = "RPFR_VISUALIZER_GUI_CONFIG"

#: Optional data-directory override, symmetric with ``$RPFR_VISUALIZER_INDEX``.
DATA_ENV_VAR = "RPFR_VISUALIZER_DATA"

#: How far up from the working directory to look for a data directory.
_WALK_UP_LIMIT = 3


def config_path() -> Path:
    """Where remembered GUI settings live.

    XDG on Unix and macOS, ``%APPDATA%`` on Windows. The platform-correct macOS
    location would be ``~/Library/Application Support``, but this is a
    terminal-resident scientific tool and XDG behaves identically on both, which
    is worth more here than platform purity — and it avoids a dependency.
    """
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "rpfr-visualizer" / "gui.json"


@dataclass
class GuiConfig:
    """Settings worth surviving a restart. Every field is optional."""

    data_dir: str | None = None
    index_path: str | None = None
    drive_root: str | None = None
    dataset: str | None = None
    port: int | None = None
    max_stats_molecules: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"version": CONFIG_VERSION}
        payload.update({k: v for k, v in asdict(self).items() if v is not None})
        return payload

    def save(self, path: Path | str | None = None) -> Path:
        """Write the config, creating its directory. Returns where it landed."""
        target = Path(path) if path is not None else config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def load_config(path: Path | str | None = None) -> tuple[GuiConfig, list[str]]:
    """Read the remembered config, plus anything wrong with it.

    A missing, malformed or unrecognised file yields an empty config rather than
    an exception — losing settings must never stop the app from starting. But the
    reason comes back in the problem list, so a user whose settings vanished
    finds out why instead of silently re-doing setup.
    """
    target = Path(path) if path is not None else config_path()
    if not target.exists():
        return GuiConfig(), []

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return GuiConfig(), [f"Ignoring unreadable settings at {target}: {exc}"]

    if not isinstance(raw, dict):
        return GuiConfig(), [f"Ignoring settings at {target}: expected a JSON object."]

    version = raw.get("version")
    if version != CONFIG_VERSION:
        return GuiConfig(), [
            f"Ignoring settings at {target}: written for version {version!r}, "
            f"this build expects {CONFIG_VERSION}."
        ]

    known = set(GuiConfig.__dataclass_fields__)
    return GuiConfig(**{k: v for k, v in raw.items() if k in known}), []


@dataclass
class IndexCandidate:
    """An index file found on disk, described well enough to choose between."""

    path: str
    size_mb: float
    molecules: int | None = None
    error: str = ""


@dataclass
class Discovery:
    """What is lying around, before anyone chooses."""

    data_dirs: list[str] = field(default_factory=list)
    index_candidates: list[IndexCandidate] = field(default_factory=list)
    drive_root: str | None = None
    index_env: str | None = None
    data_env: str | None = None


def _describe_index(path: Path) -> IndexCandidate:
    """Size and molecule count for an index file, without opening it fully."""
    size_mb = round(path.stat().st_size / 1e6, 1)
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
            molecules = int(conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0])
    except sqlite3.Error as exc:
        return IndexCandidate(str(path), size_mb, None, f"not a usable index: {exc}")
    return IndexCandidate(str(path), size_mb, molecules)


def _has_h5(directory: Path) -> bool:
    """Whether *directory* holds an openable HDF5 file, directly.

    ``._name.h5`` files are macOS AppleDouble metadata forks, not data. The same
    rule runs in :func:`rpfr_visualizer.viz.explorer.factory._h5_files`; a folder that
    holds nothing else would be offered here and then fail to open.
    """
    return any(not p.name.startswith("._") for p in directory.glob("*.h5"))


def _looks_like_data_dir(candidate: Path) -> bool:
    """Whether *candidate* is worth offering as a data directory.

    Loose ``.h5`` files count. ``catalog_for_path`` has always opened that
    layout, and dropping files into a folder is what people actually do — but
    discovery used to require ``raw/`` or ``cache/``, so those files sat there
    while the setup screen said no data had been found.
    """
    return (
        any(candidate.glob("raw/*.h5"))
        or any(candidate.glob("cache/*.sqlite"))
        or _has_h5(candidate)
    )


def discover(start: Path | str | None = None) -> Discovery:
    """Look in the usual places. Every step is cheap and none is recursive.

    Deliberately no walk of the home directory: it is slow, and it reads places
    the user never pointed us at.
    """
    found = Discovery(
        index_env=os.environ.get(INDEX_ENV_VAR) or None,
        data_env=os.environ.get(DATA_ENV_VAR) or None,
    )
    seen: set[Path] = set()

    def offer(candidate: Path) -> None:
        resolved = candidate.expanduser()
        if resolved in seen or not resolved.is_dir():
            return
        seen.add(resolved)
        if _looks_like_data_dir(resolved):
            found.data_dirs.append(str(resolved))

    # Walk up from the working directory. This is the REPO_ROOT walk-up that
    # used to live in a notebook cell, which is why it belongs here.
    here = Path(start).expanduser().resolve() if start else Path.cwd()
    for level in (here, *here.parents[:_WALK_UP_LIMIT]):
        offer(level / "data")
        offer(level)

    if found.data_env:
        offer(Path(found.data_env))
    offer(Path.home() / "rpfr-visualizer-data")

    from rpfr_visualizer.io.molecule_index import default_drive_root

    drive_root = default_drive_root()
    if drive_root.exists():  # one stat; never a mount attempt
        found.drive_root = str(drive_root)

    # Index files: list every candidate with enough detail to choose. find_index
    # knows exactly one filename, but a machine can easily hold both a small
    # subsample index and the full one, and picking by heuristic would be a
    # guess where a question is cheap.
    index_paths: list[Path] = []
    if found.index_env:
        index_paths.append(Path(found.index_env).expanduser())
    for data_dir in found.data_dirs:
        index_paths.extend(sorted(Path(data_dir).glob("cache/*.sqlite")))
    index_paths.extend(sorted(Path.home().glob("Downloads/rpfr_visualizer*.sqlite")))

    described: set[Path] = set()
    for candidate in index_paths:
        if candidate in described or not candidate.is_file():
            continue
        described.add(candidate)
        found.index_candidates.append(_describe_index(candidate))

    return found


@dataclass
class Resolution:
    """Where the app decided to look, and what it could not settle.

    ``ready`` is False when there is no data to open, which is the setup
    screen's cue — not an error.
    """

    data_dir: Path | None = None
    index_path: Path | None = None
    drive_root: Path | None = None
    dataset: str | None = None
    max_stats_molecules: int | None = 200
    problems: list[str] = field(default_factory=list)
    discovery: Discovery = field(default_factory=Discovery)

    @property
    def ready(self) -> bool:
        return self.data_dir is not None or self.drive_root is not None


def resolve(
    *,
    data_dir: Path | str | None = None,
    index_path: Path | str | None = None,
    drive_root: Path | str | None = None,
    dataset: str | None = None,
    max_stats_molecules: int | None = None,
    config: GuiConfig | None = None,
    config_problems: list[str] | None = None,
    start: Path | str | None = None,
) -> Resolution:
    """Settle on paths: arguments, then env, then remembered config, then discovery.

    Arguments here stand for command-line flags, so they win outright.
    """
    problems = list(config_problems or [])
    config = config or GuiConfig()
    found = discover(start)

    chosen_drive = drive_root or config.drive_root
    chosen_data = data_dir or config.data_dir
    if chosen_data is None and not chosen_drive and found.data_dirs:
        chosen_data = found.data_dirs[0]

    # A file is allowed: someone with one big .h5 should be able to point at it
    # rather than having to build the directory layout a loader happens to want.
    data = Path(chosen_data).expanduser().resolve() if chosen_data else None
    if data is not None and not data.exists():
        problems.append(f"No such file or directory: {data}")
        data = None
    elif data is not None and data.is_file() and data.suffix != ".h5":
        problems.append(f"Not an HDF5 (.h5) file: {data}")
        data = None

    drive = Path(chosen_drive).expanduser() if chosen_drive else None
    if drive is not None and not drive.is_dir():
        problems.append(f"Drive root is not mounted: {drive}")
        drive = None

    # Index resolution stays with find_index, which owns the
    # explicit -> $RPFR_VISUALIZER_INDEX -> <data_dir>/cache chain. Reproducing that
    # ladder here would create a second, divergent contract.
    # Only an *explicit* index is resolved here. Left to convention, find_index
    # returns the one file it knows the name of — which on a machine holding both
    # a subsample index and a full one is the subsample, and against the full
    # data that silently searches 1% of it. When nothing was asked for, leave this
    # None so the factory can rank every candidate by how much it actually covers.
    # $RPFR_VISUALIZER_INDEX counts as explicit: someone who set it meant it.
    explicit_index = index_path or config.index_path or os.environ.get(INDEX_ENV_VAR) or None
    index: Path | None = None
    if explicit_index:
        index_root = data if (data and data.is_dir()) else "data"
        index = find_index(explicit_index, data_dir=index_root)
        if index is None:
            problems.append(f"Search index not found: {explicit_index}")

    return Resolution(
        data_dir=data,
        index_path=index,
        drive_root=drive,
        dataset=dataset or config.dataset,
        max_stats_molecules=(
            max_stats_molecules
            if max_stats_molecules is not None
            else config.max_stats_molecules
            if config.max_stats_molecules is not None
            else 200
        ),
        problems=problems,
        discovery=found,
    )


__all__ = [
    "CONFIG_ENV_VAR",
    "CONFIG_VERSION",
    "DATA_ENV_VAR",
    "Discovery",
    "GuiConfig",
    "IndexCandidate",
    "Resolution",
    "config_path",
    "discover",
    "load_config",
    "resolve",
]
