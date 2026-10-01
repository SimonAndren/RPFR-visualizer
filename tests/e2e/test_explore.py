"""Searching, loading and reading a molecule.

The paths here exist only in the client. ``showResults`` chains three requests
and auto-loads the first hit; the panel buttons swap sections; the scope toggle
rewrites a setting and re-renders. None of it is reachable from ``dispatch()``.
"""

from __future__ import annotations

import pytest

from tests.e2e.harness import Ui

pytestmark = pytest.mark.e2e


def test_search_lists_hits_and_says_why_each_matched(ui: Ui) -> None:
    """Every hit carries its reason — a zero-relevance list has to explain itself."""
    ui.open()
    keys = ui.search("C")
    assert keys, "searching for C returned nothing"

    reasons = ui.page.eval_on_selector_all("#results li .why", "n => n.map(x => x.textContent)")
    assert len(reasons) == len(keys)
    assert all(reason.strip() for reason in reasons)
    assert "result" in ui.text("results-head")


def test_clicking_a_hit_renders_the_molecule(ui: Ui) -> None:
    """Search, click, draw — the chain a person actually performs."""
    ui.open()
    ui.search("C")
    ui.click_result()

    # The 2D depiction is RDKit's SVG, the one place innerHTML is allowed.
    assert ui.page.locator("#panes svg").count() >= 1
    assert ui.text("legend")
    assert ui.text("site-table")


def test_the_provenance_line_states_stored_or_interpolated(ui: Ui) -> None:
    """Interpolation is never silent; the line is present in both cases."""
    ui.open()
    ui.search("C")
    ui.click_result()
    assert "298.15" in ui.text("provenance")

    ui.set_temperature("250")
    assert "fitted" in ui.text("provenance").lower()


def test_the_panels_swap_and_only_one_shows(ui: Ui) -> None:
    """The bug that hid here: a class rule outranking [hidden] on the panels."""
    ui.open()
    ui.search("C")
    ui.click_result()

    for name in ("molecule", "ranked", "quality"):
        ui.open_panel(name)
        shown = [
            panel for panel in ("molecule", "ranked", "quality") if ui.visible(f"panel-{panel}")
        ]
        assert shown == [name], f"opening {name} left {shown} visible"


def test_the_ranked_panel_lists_sites(ui: Ui) -> None:
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.open_panel("ranked")

    assert ui.text("rank-caption")
    assert ui.text("ranks")


def test_the_quality_panel_reports_on_the_loaded_molecule(ui: Ui) -> None:
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.open_panel("quality")

    assert ui.text("qc-verdict")


def test_a_search_with_no_hits_says_so_rather_than_going_blank(ui: Ui) -> None:
    """Absence is information — the app must not look broken when it is empty."""
    ui.open()
    keys = ui.search("Xx9999zz")
    assert keys == []
    assert ui.text("results-head") or ui.toast()


def test_an_empty_query_does_nothing_at_all(ui: Ui) -> None:
    """Whitespace is guarded in the client, before a request exists.

    The controller's "Type something to search for" is the backstop for anyone
    calling the API directly; in the page an empty box is not an error worth
    interrupting someone over, so no request is sent and nothing is disturbed.
    """
    ui.open()
    ui.search("C")
    before = ui.result_keys()

    sent = len([line for line in ui.network if "/api/search" in line])
    ui.page.fill("#q", "   ")
    ui.page.click("#go")
    ui.idle()

    assert len([line for line in ui.network if "/api/search" in line]) == sent
    assert ui.result_keys() == before


def test_the_dataset_view_replaces_the_explorer(ui: Ui) -> None:
    ui.open()
    ui.open_view("dataset")
    assert ui.visible("dataset-view")
    assert not ui.visible("explore")

    ui.open_view("explore")
    assert ui.visible("explore")
    assert not ui.visible("dataset-view")


def test_the_download_view_previews_the_open_molecule(ui: Ui) -> None:
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.open_view("download")
    assert ui.visible("download-view")
    assert not ui.visible("explore")
    summary = ui.text("download-summary").lower()
    assert "molecule" in summary
    assert ui.text("download-filename").endswith(".csv")
    assert not ui.page.locator("#download-go").is_disabled()


def test_the_qc_checkbox_removes_a_failing_molecule(ui: Ui) -> None:
    """Molecule "N" in the fixture is the recorded QC failure, so ticking the
    box has to drop it. This path lives entirely in the client -- withFilters()
    appends the token -- so dispatch() cannot see it."""
    ui.open()
    assert ui.search("N"), "the failing fixture molecule was not found unfiltered"

    ui.page.check("#qc_pass")
    ui.idle()

    assert ui.result_keys() == []


def test_the_qc_checkbox_keeps_a_passing_molecule(ui: Ui) -> None:
    """The filter must narrow, not empty: methane passes and has to survive it."""
    ui.open()
    before = ui.search("C")
    assert before

    ui.page.check("#qc_pass")
    ui.idle()

    assert ui.result_keys() == before


def test_unticking_the_qc_checkbox_restores_the_results(ui: Ui) -> None:
    ui.open()
    everything = ui.search("N")

    ui.page.check("#qc_pass")
    ui.idle()
    ui.page.uncheck("#qc_pass")
    ui.idle()

    assert ui.result_keys() == everything


def test_a_filtered_group_search_goes_through_the_query_grammar(ui: Ui) -> None:
    """Choosing a preset normally calls /api/search/fg, which takes a group and
    nothing else. With the filter on it must route to /api/search instead, so
    the checkbox is not quietly ignored."""
    ui.open()
    ui.page.check("#qc_pass")
    ui.network.clear()

    ui.choose_group("alcohol")

    assert any(entry.startswith("POST /api/search ") for entry in ui.network), ui.network
    assert not any("/api/search/fg" in entry for entry in ui.network), ui.network


def test_the_example_keys_are_offered_and_work(ui: Ui) -> None:
    """A subsample holds whichever molecules it holds, so the placeholder's own
    suggestions ("ethanol", "CCO") can match nothing. The server computes keys
    that will actually render; this checks the client offers them and that
    clicking one finds something."""
    ui.open()

    examples = ui.page.locator("#examples .linky")
    assert examples.count() > 0, "no example molecules were offered"

    key = examples.first.text_content()
    examples.first.click()
    ui.idle()

    assert ui.page.input_value("#q") == key
    assert key in ui.result_keys()


def test_picking_two_groups_composes_an_and_query(ui: Ui) -> None:
    """The picker writes query text rather than calling a second endpoint, so
    what runs is exactly what a user could have typed."""
    ui.open()

    ui.page.click("#fg-mode button[data-mode='all']")
    ui.page.check("#preset-groups input[data-fg='alcohol']")
    ui.idle()
    ui.page.check("#preset-groups input[data-fg='amine']")
    ui.idle()

    assert ui.page.input_value("#q") == "fg:alcohol&amine"


def test_the_any_mode_composes_comma_separated_tokens(ui: Ui) -> None:
    ui.open()

    ui.page.check("#preset-groups input[data-fg='alcohol']")
    ui.idle()
    ui.page.check("#preset-groups input[data-fg='amine']")
    ui.idle()

    assert ui.page.input_value("#q") == "fg:alcohol, fg:amine"


def test_switching_mode_rewrites_the_query(ui: Ui) -> None:
    ui.open()
    ui.page.check("#preset-groups input[data-fg='alcohol']")
    ui.idle()
    ui.page.check("#preset-groups input[data-fg='amine']")
    ui.idle()
    assert "," in ui.page.input_value("#q")

    ui.page.click("#fg-mode button[data-mode='all']")
    ui.idle()

    assert "&" in ui.page.input_value("#q")


def test_groups_with_no_members_are_shown_but_not_selectable(ui: Ui) -> None:
    """Absence is information: a group that vanished would read as a missing
    feature rather than as "there are none of these here"."""
    ui.open()

    boxes = ui.page.locator("#preset-groups input[type=checkbox]")
    assert boxes.count() > 1
    disabled = ui.page.locator("#preset-groups input[type=checkbox]:disabled")
    assert disabled.count() >= 1, "the fixture has no empty group to check against"


def test_the_guide_tab_lists_every_group_with_a_picture(ui: Ui) -> None:
    """The catalog answers "what can I search for, and what does it look like".

    Pictures are <img src="/api/fg/depict?...">, so nothing is inserted as
    markup and the single-innerHTML rule is untouched.
    """
    ui.open()
    ui.page.click("#views button[data-view='guide']")
    ui.idle()

    assert ui.page.locator("#guide-view").is_visible()
    assert not ui.page.locator("#explore").is_visible()

    cards = ui.page.locator("#guide-groups .fg-card")
    assert cards.count() == 18

    images = ui.page.locator("#guide-groups img.fg-image")
    assert images.count() == 18

    # The images are lazy, so only what has been scrolled to is fetched. Bring
    # the grid into view and assert on what actually loaded: a broken <img>
    # reports naturalWidth 0, a drawn SVG does not.
    images.first.scroll_into_view_if_needed()
    ui.page.wait_for_function(
        "() => [...document.querySelectorAll('#guide-groups img.fg-image')]"
        ".filter(n => n.complete && n.naturalWidth > 0).length >= 2",
        timeout=15000,
    )


def test_searching_a_group_from_the_guide_returns_to_explore(ui: Ui) -> None:
    ui.open()
    ui.page.click("#views button[data-view='guide']")
    ui.idle()

    ui.page.locator("#guide-groups .fg-card", has_text="alcohol").first.locator("button").click()
    ui.idle()

    assert ui.page.locator("#explore").is_visible()
    assert ui.page.input_value("#q") == "fg:alcohol"
    assert ui.result_keys()


def test_a_group_with_no_molecules_cannot_be_searched_from_the_guide(ui: Ui) -> None:
    """Listed so absence is visible, but not offered as a search that returns
    nothing."""
    ui.open()
    ui.page.click("#views button[data-view='guide']")
    ui.idle()

    disabled = ui.page.locator("#guide-groups .fg-card button:disabled")
    assert disabled.count() >= 1


def test_the_derived_panel_draws_its_three_charts(ui: Ui) -> None:
    """Charts are inline SVG built in the client, so nothing below the payload
    shape is reachable from dispatch()."""
    ui.open()
    ui.search("C")
    ui.click_result()

    ui.page.click("#panels button[data-panel='derived']")
    ui.idle()

    charts = ui.page.locator("#panel-derived svg.chart")
    assert charts.count() >= 1

    beta = charts.first
    series = beta.locator("polyline.line").count()
    assert series >= 1
    assert beta.locator("circle.marker").count() >= 1

    # A legend for two or more series, none for one: three of the light-mode
    # series steps sit below 3:1 contrast, so identity must never be colour
    # alone -- but a lone series is already named by its heading.
    legend = ui.page.locator("#panel-derived .chart-legend").first.locator("span").count()
    assert legend == (series if series >= 2 else 0)


def test_the_derived_values_are_readable_without_hovering(ui: Ui) -> None:
    """A tooltip enhances; it must never be the only route to a number."""
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.page.click("#panels button[data-panel='derived']")
    ui.idle()

    details = ui.page.locator("#panel-derived details.more").first
    details.locator("summary").click()

    assert details.locator("table tr").count() > 2


def test_an_off_grid_temperature_offers_a_way_back(ui: Ui) -> None:
    """Clumping and mass law exist only on the stored grid. Saying so is not
    enough -- the panel offers the nearest stored temperatures as buttons."""
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.page.click("#panels button[data-panel='derived']")
    ui.idle()

    ui.set_temperature("305")
    ui.idle()

    body = ui.text("panel-derived")
    assert ", fitted" in body
    jumps = ui.page.locator("#panel-derived button.linky")
    assert jumps.count() >= 1

    jumps.first.click()
    ui.idle()
    # The panel redraws after the state round trip; on a slow runner that can
    # land after idle() returns, so wait for the redraw rather than racing it.
    ui.page.wait_for_function(
        "() => document.getElementById('panel-derived').textContent.includes(', stored')",
        timeout=15000,
    )
    assert ", fitted" not in ui.text("panel-derived")


def test_the_off_grid_fit_can_be_changed_and_the_chart_says_which(ui: Ui) -> None:
    """The choice is only honest if the drawing names the fit it drew.

    Off the grid the five interpolants disagree by tens of percent below 60 K,
    so a curve that does not say which one produced it is unreadable.
    """
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.open_panel("derived")

    # On the grid there is nothing to fit, so the picker is not there at all.
    assert not ui.visible("beta-method-field")
    ui.set_temperature("25")
    assert ui.visible("beta-method-field"), "asking for an unstored temperature hid the fit"

    def caption() -> str:
        return ui.page.locator("#panel-derived .derived-section p.muted").first.text_content() or ""

    def curve() -> str:
        return (
            ui.page.locator("#panel-derived svg.chart polyline.line").first.get_attribute("points")
            or ""
        )

    assert "local quadratic" in caption().lower()
    drawn_by_default = curve()

    ui.set_control("beta_method", "poly5")
    assert ui.client_state()["settings"]["beta_method"] == "poly5"
    assert "polynomial, degree 5" in caption().lower()
    assert curve() != drawn_by_default, "the chart still shows the previous fit"

    # Back on the grid, nothing is fitted and the control goes away again.
    ui.set_temperature("298.15")
    assert not ui.visible("beta-method-field")

    ui.assert_clean()


def test_a_stored_temperature_chosen_after_a_typed_one_sticks(ui: Ui) -> None:
    """The custom box keeps whatever was typed into it.

    Picking a stored temperature moves focus off that box, and the blur fired
    a second change event carrying the stale value -- so the dropdown appeared
    to do nothing. Driven through the raw controls rather than the harness
    helper, which waits for the state and would hide the bug.
    """
    ui.open()
    ui.search("C")
    ui.click_result()

    select = ui.page.locator("#temperature")
    select.select_option("custom")
    box = ui.page.locator("#temperature-custom")
    box.fill("25")
    box.dispatch_event("change")
    ui.idle()

    select.select_option("298.15")
    ui.idle()
    ui.page.wait_for_timeout(300)

    settings = ui.client_state()
    assert settings["settings"]["temperature"] == 298.15
    assert settings["on_grid"] is True
    assert not ui.visible("temperature-custom-field")

    ui.assert_clean()


def test_clumping_can_be_filtered_to_heavy_atoms(ui: Ui) -> None:
    """H pairs run an order of magnitude above heavy-atom pairs. On one shared
    scale the heavy structure vanishes, so the filter has to rescale the colour
    to whichever set is showing — the caption states which."""
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.page.click("#panels button[data-panel='derived']")
    ui.idle()

    caption = "#panel-derived .matrix-wrap p.muted"
    all_pairs = ui.page.locator(caption).first.text_content()

    ui.page.locator("#panel-derived .segmented button", has_text="no hydrogen").click()
    ui.idle()
    heavy = ui.page.locator(caption).first.text_content()

    assert all_pairs != heavy, "the scale did not change with the filter"


def test_clumping_is_labelled_in_per_mil(ui: Ui) -> None:
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.page.click("#panels button[data-panel='derived']")
    ui.idle()

    assert "Δ" in ui.text("panel-derived")
    assert "‰" in ui.text("panel-derived")


def test_the_guide_explains_the_derived_quantities(ui: Ui) -> None:
    """Including the two caveats a reader cannot infer from the charts: that λ
    is an oxygen-only quantity, and that interpolated β misbehaves near 20 K."""
    ui.open()
    ui.page.click("#views button[data-view='guide']")
    ui.idle()

    body = ui.text("guide-view")
    assert "Clumping" in body
    assert "Clumping Δ" in body
    assert "oxygen only" in body
    assert "20 K" in body


def test_a_list_of_temperatures_is_counted_and_remembered(ui: Ui) -> None:
    """Both of these were reported done and neither had been committed: the
    summary named only the first temperature of a list, and the box came back
    empty after a reload. The row count was right, being computed on the
    server, which is what hid the rest."""
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.open_view("download")

    box = ui.page.locator("#download-temperatures")
    box.fill("250, 298.15")
    box.dispatch_event("change")
    ui.idle()
    ui.page.wait_for_function(
        "() => document.getElementById('download-summary').textContent.includes('2 temperatures')"
    )

    # Leave and come back: the saved list must be in the box, not a blank.
    ui.open_view("explore")
    ui.open_view("download")
    ui.page.locator("#download-go").focus()
    ui.page.wait_for_function(
        "() => document.getElementById('download-temperatures').value === '250, 298.15'"
    )
    ui.assert_clean()


def test_leaving_the_custom_box_does_not_resend_the_temperature(ui: Ui) -> None:
    """Leaving the box re-fired change with the value already applied. The
    resulting write redrew the Derived panel between mouse-down and mouse-up
    on its own "jump to a stored temperature" button, so on a slow runner the
    click landed on a button that no longer existed and was lost."""
    ui.open()
    ui.search("C")
    ui.click_result()
    ui.set_temperature("305")

    writes: list[str] = []
    ui.page.on(
        "request",
        lambda r: writes.append(r.url) if r.method == "POST" and "/api/state" in r.url else None,
    )
    ui.page.locator("#temperature-custom").dispatch_event("change")
    ui.page.locator("#temperature-custom").blur()
    ui.page.wait_for_timeout(300)
    ui.idle()
    assert writes == []
