"""Booting the app in a real browser.

These cover the ground the router tests cannot reach: whether the state document
actually arrives in the controls, whether every handler is bound to an element
that exists, and whether the page holds together at a real viewport size.
"""

from __future__ import annotations

import pytest

from tests.e2e.harness import Ui

pytestmark = pytest.mark.e2e


def test_the_app_boots_with_no_console_errors(ui: Ui) -> None:
    """The broadest test here, and the one most likely to catch a typo.

    applyState() writes to some forty elements in a row, so a single missing id
    throws before the page finishes rendering — and the ui fixture fails the
    test on any browser error, so this needs no assertion of its own to be
    worth running.
    """
    ui.open()
    assert ui.ready == "explore"
    assert ui.visible("explore")
    assert not ui.visible("setup")


def test_the_controls_are_populated_from_the_server(ui: Ui) -> None:
    """The state document has to reach the DOM, not just the client."""
    ui.open()

    datasets = ui.page.eval_on_selector_all("#dataset option", "nodes => nodes.map(n => n.value)")
    assert "qm9s" in datasets

    elements = ui.page.eval_on_selector_all("#element option", "nodes => nodes.map(n => n.value)")
    assert {"C", "H"} <= set(elements)

    quantities = ui.page.eval_on_selector_all("#kind option", "nodes => nodes.map(n => n.value)")
    assert "beta" in quantities

    temperatures = ui.page.eval_on_selector_all(
        "#temperature option", "nodes => nodes.map(n => n.value)"
    )
    assert "298.15" in temperatures
    assert temperatures[-1] == "custom", "no way to ask for an unstored temperature"


def test_a_setting_round_trips_through_the_server_to_the_control(ui: Ui) -> None:
    """The central claim in both app.py and app.js: the response is authoritative.

    Asserted through the DOM rather than the API, because the bug this guards
    against is a control left displaying a value the analysis is not using.
    """
    ui.open()

    ui.set_control("element", "H")
    assert ui.client_state()["settings"]["element"] == "H"
    assert ui.page.input_value("#element") == "H"

    # top_n lives in the Ranked sites panel, so it is only reachable once that
    # panel is open — which is itself worth asserting.
    ui.open_panel("ranked")
    ui.set_control("top_n", "50")
    assert ui.client_state()["settings"]["top_n"] == 50
    assert ui.page.input_value("#top_n") == "50"


def test_an_off_grid_temperature_is_announced(ui: Ui) -> None:
    """Interpolation is never silent — that is a documented promise."""
    ui.open()
    ui.set_temperature("250")

    state = ui.client_state()
    assert state["on_grid"] is False
    assert any("fitted" in warning for warning in state["warnings"])


def test_the_reference_control_appears_only_when_it_is_needed(ui: Ui) -> None:
    """Absent, not disabled: a disabled control asks why it cannot be clicked."""
    ui.open()
    ui.set_control("kind", "beta")
    assert not ui.visible("reference-field")

    ui.set_control("kind", "ln_alpha")
    assert ui.visible("reference-field")


def test_nothing_overflows_the_window(ui: Ui) -> None:
    """The CSS regression that actually happens here, and is invisible to a router test."""
    ui.open()
    ui.search("C")
    ui.click_result()

    assert ui.page.evaluate(
        "() => document.scrollingElement.scrollWidth <= window.innerWidth + 1"
    ), "the page scrolls horizontally at 1440px"
