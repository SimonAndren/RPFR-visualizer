"""What an installed copy of this package has to contain.

These are the failures that never show up in a source tree, because a source
tree has every file whether the wheel ships it or not. They surface only after
`pip install`, usually as a 404 in the browser or a stale version in a bug
report — so they are worth pinning from inside the test suite rather than
discovering downstream.
"""

from __future__ import annotations

import sys
from importlib import resources
from importlib.metadata import version
from pathlib import Path

import pytest

import rpfr_visualizer

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: tomllib landed in 3.11 and this package supports 3.10, so the tests that
#: read pyproject.toml skip there rather than adding a dependency for them.
needs_tomllib = pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")

#: Every non-Python file the running package reads at runtime. If one of these
#: is missing from a wheel the import still succeeds, which is what makes the
#: failure so late and so confusing.
DATA_FILES: tuple[tuple[str, str], ...] = (
    ("rpfr_visualizer.gui", "static/index.html"),
    ("rpfr_visualizer.gui", "static/app.css"),
    ("rpfr_visualizer.gui", "static/app.js"),
    ("rpfr_visualizer.viz", "static/3Dmol-min.js"),
    ("rpfr_visualizer.data", "releases.json"),
    ("rpfr_visualizer.data", "name_synonyms.json"),
)


def _config() -> dict:
    import tomllib

    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("package", "name"), DATA_FILES, ids=lambda v: str(v))
def test_the_package_can_read_its_own_data_files(package: str, name: str) -> None:
    """Reached the way the code reaches them, not by walking the source tree."""
    handle = resources.files(package) / name
    assert handle.is_file(), f"{package}:{name} is not readable from the package"
    assert handle.read_bytes(), f"{package}:{name} is empty"


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An actual wheel, built once for the session.

    The source-tree checks above cannot see a packaging mistake: the files are
    on disk whether or not the wheel ships them. Only opening a built wheel
    can. It costs well under a second, which is cheaper than finding out from
    somebody's `pip install`.
    """
    import shutil
    import subprocess

    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH, so no wheel can be built here")

    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=PYPROJECT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"wheel build failed here: {result.stderr[-400:]}")

    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_the_wheel_ships_every_runtime_asset(built_wheel: Path) -> None:
    """The failure this catches is invisible until someone installs it.

    A wheel missing app.js imports perfectly and then serves a blank page.
    """
    import zipfile

    with zipfile.ZipFile(built_wheel) as archive:
        inside = set(archive.namelist())

    missing = [
        f"{package.replace('.', '/')}/{name}"
        for package, name in DATA_FILES
        if f"{package.replace('.', '/')}/{name}" not in inside
    ]
    assert not missing, f"built wheel is missing: {missing}"


def test_the_wheel_declares_the_console_script(built_wheel: Path) -> None:
    import zipfile

    with zipfile.ZipFile(built_wheel) as archive:
        entry_points = next((n for n in archive.namelist() if n.endswith("entry_points.txt")), None)
        assert entry_points, "no entry_points.txt in the wheel"
        assert "rpfr-visualizer" in archive.read(entry_points).decode()


def test_every_static_asset_is_covered_by_this_test() -> None:
    """The list above is only a guard if it is complete.

    Assets used to be force-included into the wheel one filename at a time, so
    adding one produced a wheel that imported cleanly and then 404ed. The
    include is a directory now; this makes sure the test list keeps up too.
    """
    on_disk = set()
    for package in ("rpfr_visualizer.gui", "rpfr_visualizer.viz"):
        static = Path(str(resources.files(package))) / "static"
        if static.is_dir():
            on_disk |= {f"{package}:static/{p.name}" for p in static.iterdir() if p.is_file()}

    listed = {f"{package}:{name}" for package, name in DATA_FILES}
    missing = sorted(on_disk - listed)
    assert not missing, f"static assets nothing checks: {missing}"


def test_the_reported_version_is_the_installed_one() -> None:
    """__version__ was hardcoded and had drifted a release behind pyproject.

    A version a user quotes in a bug report is only useful if it is real.
    """
    assert rpfr_visualizer.__version__ == version("rpfr-visualizer")
    assert rpfr_visualizer.__version__ != "0.0.0+unknown", "package is not installed"


@needs_tomllib
def test_the_declared_version_matches_the_installed_one() -> None:
    assert _config()["project"]["version"] == version("rpfr-visualizer")


@needs_tomllib
def test_the_all_extra_installs_no_developer_tooling() -> None:
    """`pip install rpfr-visualizer[all]` is a user action, not a contributor one.

    It used to pull in pytest, ruff, mypy, black, jupyterlab, pre-commit and a
    browser driver, which is a lot of surprise for somebody who only wanted the
    3D viewer.
    """
    extras = _config()["project"]["optional-dependencies"]
    assert "dev" not in str(extras["all"])
    assert "e2e" not in str(extras["all"])

    # ...and the one that does exist for contributors still reaches everything.
    assert "dev" in str(extras["dev-all"])
    assert "e2e" in str(extras["dev-all"])
    assert "all" in str(extras["dev-all"])


@needs_tomllib
def test_the_console_script_points_at_a_real_entry_point() -> None:
    from rpfr_visualizer.cli import main

    target = _config()["project"]["scripts"]["rpfr-visualizer"]
    assert target == "rpfr_visualizer.cli:main"
    assert callable(main)
