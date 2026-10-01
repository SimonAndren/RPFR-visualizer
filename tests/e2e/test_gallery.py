"""Screenshots of the UI, for looking at rather than asserting on.

Run with ``--gallery``; skipped otherwise, because writing eight PNGs is not
something an ordinary test run should do. One page session walks the states
worth seeing and shoots each, so "show me what it looks like now" is a command
rather than a description.

Deliberately no pixel-diff baselines. The 2D pane is RDKit's SVG rasterised by
the browser and the 3D pane is WebGL, so both move with the RDKit release, the
platform font stack and the GPU. Baselines over that would be re-blessed on
every intentional change, which teaches everyone to re-bless reflexively — and
a test that is always re-blessed is worse than no test, because it launders the
regressions it was bought to catch. Layout is asserted geometrically in
test_boot.py instead, where the oracle is exact.
"""

from __future__ import annotations

import pytest

from tests.e2e.harness import Ui

pytestmark = [pytest.mark.e2e, pytest.mark.gallery]


@pytest.fixture(autouse=True)
def _needs_gallery_flag(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--gallery"):
        pytest.skip("pass --gallery to write screenshots")


def test_gallery(ui: Ui) -> None:
    """Walk the app and photograph it.

    One test rather than eight, because a page load is the expensive part and
    these are a sequence a person would actually perform.
    """
    shots = []

    ui.open()
    shots.append(ui.shot("01-empty-explorer"))

    ui.search("C")
    shots.append(ui.shot("02-search-results"))

    ui.click_result()
    shots.append(ui.shot("03-molecule-loaded"))

    ui.set_control("element", "H")
    shots.append(ui.shot("04-hydrogen-sites"))

    ui.set_control("element", "C")
    ui.set_temperature("250")
    shots.append(ui.shot("05-interpolated-temperature"))

    ui.set_temperature("298.15")
    ui.open_panel("ranked")
    shots.append(ui.shot("06-ranked-sites"))

    ui.open_panel("quality")
    shots.append(ui.shot("07-data-quality"))

    ui.open_panel("molecule")
    ui.open_view("dataset")
    shots.append(ui.shot("08-dataset-view"))

    assert all(path.exists() and path.stat().st_size > 0 for path in shots)
    print(f"\n{len(shots)} screenshots in {ui.screens}")


def test_gallery_3d(ui: Ui, e2e_session) -> None:
    """The 3D viewer, in a page of its own.

    Costs a 525 KB bundle and a WebGL context, so the rest of the suite runs
    with use_3d off; this is the one place that is not true.
    """
    e2e_session.controller.state.use_3d = True

    ui.open()
    ui.search("C")
    ui.click_result()
    # The viewer is an iframe with its own document, so give it a moment to
    # paint before the shutter rather than photographing an empty box.
    ui.page.wait_for_timeout(1200)
    path = ui.shot("09-three-d-viewer")

    assert ui.page.locator("#panes iframe").count() >= 1
    assert path.exists()
