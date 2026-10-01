"""Contracts between app.js, index.html and the routing table.

These are the front-end bugs that are cheap to catch and expensive to notice:
an id renamed on one side of the pair, a route renamed under a caller, an
innerHTML added where textContent was the rule. None of it needs a browser —
it is all statically checkable — so these run in milliseconds and go first,
which keeps most edit-and-check cycles from launching one at all.

What they cannot see is behaviour: whether a handler is bound, whether the
state document reaches the controls. That is what the browser suite is for.
"""

from __future__ import annotations

import re
from importlib import resources

STATIC = resources.files("rpfr_visualizer.gui") / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP_CSS = (STATIC / "app.css").read_text(encoding="utf-8")

#: ``$('name')`` — the only way app.js reaches for an element by id.
_BY_ID = re.compile(r"\$\('([A-Za-z0-9_-]+)'\)")

#: ``api('GET', '/path')`` — the only way it calls the server.
_API_CALL = re.compile(r"api\('(GET|POST)',\s*'([^']+)'")

#: ``id="name"`` in the shell.
_HTML_IDS = re.compile(r'id="([A-Za-z0-9_-]+)"')

#: The one id built at runtime, from a literal list of panel names.
_DYNAMIC_IDS = frozenset({"panel-molecule", "panel-ranked", "panel-quality", "panel-derived"})


def _html_ids() -> frozenset[str]:
    return frozenset(_HTML_IDS.findall(INDEX_HTML))


def test_every_element_app_js_reaches_for_exists_in_the_page() -> None:
    """The "Cannot set properties of null" class of bug, caught without a browser.

    applyState() writes to some forty elements in a row, so one renamed id there
    stops the whole render — and the failure surfaces as a blank pane rather
    than as anything naming the id.
    """
    wanted = set(_BY_ID.findall(APP_JS))
    missing = sorted(wanted - _html_ids())
    assert not missing, f"app.js reaches for ids that index.html does not define: {missing}"


def test_the_dynamic_panel_ids_all_exist() -> None:
    """``$('panel-' + name)`` is the one id app.js builds at runtime.

    Its three values come from a literal list, so the contract stays statically
    checkable — but only if this test enumerates them.
    """
    assert _DYNAMIC_IDS <= _html_ids()

    # The list is closed: every data-panel button must have a matching section.
    panels = set(re.findall(r'data-panel="([A-Za-z0-9_-]+)"', INDEX_HTML))
    assert {f"panel-{name}" for name in panels} == _DYNAMIC_IDS


def test_the_only_innerHTML_is_the_renderer_svg() -> None:  # noqa: N802 — names the JS property
    """Enforces the trust boundary that render.py documents.

    Molecule synonyms can come from PubChem and keys come from the data files,
    so "data-derived text uses textContent" is a security rule rather than a
    style one. A rule with no test is a comment.
    """
    occurrences = re.findall(r"\.innerHTML\b", APP_JS)
    assert len(occurrences) == 1, f"expected exactly one innerHTML, found {len(occurrences)}"
    assert "plate.innerHTML = pane.svg" in APP_JS


def test_every_endpoint_app_js_calls_is_a_registered_route() -> None:
    """A renamed route otherwise fails as a dead control nobody clicked.

    The 404 goes to a toast, so the app keeps working and the button simply
    stops doing anything — which is exactly the kind of fault that survives to
    a release.
    """
    from rpfr_visualizer.gui.app import _ROUTES

    called = {(method, path.split("?")[0]) for method, path in _API_CALL.findall(APP_JS)}
    unknown = sorted(call for call in called if call not in _ROUTES)
    assert not unknown, f"app.js calls routes that are not registered: {unknown}"


def test_the_page_and_the_script_agree_on_their_data_attributes() -> None:
    """data-panel, data-scope and data-view are read by JS and written by HTML."""
    for attribute in ("panel", "scope", "view"):
        in_html = set(re.findall(rf'data-{attribute}="([A-Za-z0-9_-]+)"', INDEX_HTML))
        assert in_html, f"index.html defines no data-{attribute} values"
        assert f"dataset.{attribute}" in APP_JS, f"app.js never reads data-{attribute}"


def test_the_page_references_the_assets_the_server_serves() -> None:
    """Catches a force-include line forgotten in pyproject.toml."""
    from rpfr_visualizer.gui.app import STATIC_ASSETS

    assert "/static/app.js" in INDEX_HTML
    assert "/static/app.css" in INDEX_HTML
    for name in STATIC_ASSETS:
        assert (STATIC / name).is_file(), name


def test_the_busy_and_ready_signals_the_harness_waits_on_are_set() -> None:
    """A browser test has to know when the app has settled.

    Both signals are load-bearing for the e2e suite, and both are easy to drop
    in a refactor because nothing in the UI depends on them.
    """
    assert "document.body.dataset.ready" in APP_JS
    for value in ("'explore'", "'setup'", "'error'"):
        assert f"dataset.ready = {value}" in APP_JS, value

    # Counted rather than a flag, so it cannot flicker between two requests.
    assert "pending += 1" in APP_JS
    assert "pending -= 1" in APP_JS
    assert "[data-busy" in APP_CSS or "data-busy" in APP_CSS
