"""Fixtures for the browser suite.

Everything expensive is session-scoped: one Chrome, one server, one dataset. The
server runs in this process via ``serve_in_thread`` rather than as a subprocess,
so there is no second interpreter to start, no second RDKit import, and the
controller's caches stay warm between tests — the first ``/view/render`` costs
about half a second and every one after it is free.

The suite makes itself invisible when the ``e2e`` extra is not installed, so a
fresh clone still gets a green run.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

collect_ignore_glob: list[str] = []
try:  # pragma: no cover — the gate itself is the behaviour
    import playwright.sync_api  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    collect_ignore_glob = ["test_*.py"]

pytest.importorskip("rdkit")

#: Overrides which browser the suite drives. Unset means the installed Chrome.
E2E_CHANNEL_ENV = "RPFR_VISUALIZER_E2E_CHANNEL"

#: Where screenshots go. In-repo rather than a temp dir so a path quoted in a
#: review still resolves, and gitignored so they never land in a commit.
DEFAULT_SCREENS = Path(__file__).parent / "_screens"


@pytest.fixture(scope="session")
def e2e_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory with one synthetic dataset and a real search index."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conftest import build_synthetic_h5  # type: ignore[import-not-found]
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    root = tmp_path_factory.mktemp("e2e-data")
    raw = root / "raw"
    raw.mkdir()
    (root / "cache").mkdir()

    source = build_synthetic_h5(root / "source.h5")
    target = raw / "qm9s_tr_project_1pct.h5"
    shutil.copy(source, target)

    catalog = DatasetCatalog()
    catalog.register("qm9s", target)
    # A real index, not a stub: it is a fraction of a second and it is the only
    # thing that exercises search, functional groups and statistics at all.
    build_index(catalog, root / "cache" / "qm9s_rpfr_index.sqlite", progress=False)
    return root


@pytest.fixture(scope="session")
def e2e_session(e2e_root: Path):
    """The one controller every test drives."""
    from rpfr_visualizer.gui.config import Resolution
    from rpfr_visualizer.gui.session import ExplorerSession

    session = ExplorerSession(
        Resolution(
            data_dir=e2e_root,
            index_path=e2e_root / "cache" / "qm9s_rpfr_index.sqlite",
            max_stats_molecules=2,
        )
    )
    session.open()
    return session


@pytest.fixture(scope="session")
def e2e_server(e2e_session) -> Iterator[str]:
    from rpfr_visualizer.gui.server import serve_in_thread

    server, url = serve_in_thread(e2e_session, port=0, poll_interval=0.05)
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _reset_between_tests(e2e_session) -> None:
    """Put the controller back to a known state before each test.

    The server owns the state and there is one of it, so without this a test
    that sets an off-grid temperature silently changes what the next test is
    looking at — a shared-fixture bug that reads as a flaky assertion.
    """
    controller = e2e_session.controller
    controller.clear("a")
    controller.clear("b")
    controller.clear_pool()
    controller.last_hits = []

    state = controller.state
    state.temperature = 298.15
    state.element = "C"
    state.kind = type(state.kind).BETA
    state.reference_mode = "element_mean"
    state.top_n = 15
    state.show_hydrogens = True
    state.show_labels = True
    state.use_match_pool = True
    # The 3D viewer costs a 525 KB bundle and a WebGL context per page. It is a
    # writable setting with no control of its own, so turning it off costs no
    # realism for the tests that are not about it.
    state.use_3d = False


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args: dict) -> dict:
    """Use the Chrome that is already installed.

    Avoids a 150 MB `playwright install chromium` step on every developer
    machine. SwiftShader gives the 3D viewer a WebGL context headless, which
    would otherwise be a console error in a test that is not about the viewer.

    Set ``RPFR_VISUALIZER_E2E_CHANNEL`` to pick a different one -- CI sets it to
    ``chromium`` so the run does not depend on whichever Chrome a runner image
    happens to ship, or on it being there at all.
    """
    channel = os.environ.get(E2E_CHANNEL_ENV, "chrome")
    args = {
        **browser_type_launch_args,
        "args": ["--enable-unsafe-swiftshader", "--disable-lcd-text"],
    }
    # "chromium" means Playwright's own download, which is selected by passing
    # no channel at all rather than by naming one.
    if channel and channel != "chromium":
        args["channel"] = channel
    return args


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict) -> dict:
    """A fixed viewport, so layout assertions and screenshots are comparable."""
    return {
        **browser_context_args,
        "viewport": {"width": 1440, "height": 900},
        "device_scale_factor": 1,
        "reduced_motion": "reduce",
        "color_scheme": "light",
    }


@pytest.fixture(scope="session")
def screens_dir(request: pytest.FixtureRequest) -> Path:
    configured = request.config.getoption("--screens-dir")
    target = Path(configured) if configured else DEFAULT_SCREENS
    if target.exists():
        shutil.rmtree(target)  # a stale PNG read as current is worse than none
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def ui(page, e2e_server: str, screens_dir: Path, request: pytest.FixtureRequest) -> Iterator:
    from tests.e2e.harness import Ui, attach

    harness = Ui(page=page, base_url=e2e_server, screens=screens_dir, node_name=request.node.name)
    attach(harness)  # before any navigation
    yield harness
    harness.assert_clean()
