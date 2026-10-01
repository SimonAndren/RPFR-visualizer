"""A thin driver for the explorer, in the app's own vocabulary.

Not a page-object hierarchy: this is one screen and 550 lines of front end, so
a class per region would be more structure than there is behaviour. What it does
buy is that selectors live in one file, so a renamed id breaks here rather than
in a dozen tests.

Every browser error is collected and asserted at the end of the test. Non-2xx
HTTP is deliberately *not* an error — 400, 409 and 503 are legitimate paths in
this app — but the request timeline is printed as context when something else
does fail, because "which requests had already run" is the first question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import indent
from urllib.parse import urlparse

from playwright.sync_api import Page, expect

#: Waits fail in three seconds rather than the default five: a real failure
#: should come back while you are still looking at it.
DEFAULT_TIMEOUT_MS = 3000


def _path(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-")[:80]


@dataclass
class Ui:
    """One page, plus what the browser said while it was open."""

    page: Page
    base_url: str
    screens: Path
    node_name: str = "test"
    errors: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ navigation

    def open(self, path: str = "/") -> None:
        """Go to *path* and wait until the client says it has finished booting."""
        self.page.goto(f"{self.base_url}{path}")
        self.page.wait_for_selector("body[data-ready]", timeout=15000)

    def idle(self) -> None:
        """Wait until no request is in flight.

        data-busy is a counter, so it cannot read '0' between the two requests
        of a single setState().
        """
        self.page.wait_for_function("() => document.body.dataset.busy !== '1'", timeout=15000)

    @property
    def ready(self) -> str:
        return self.page.evaluate("() => document.body.dataset.ready || ''")

    # --------------------------------------------------------------- actions

    def search(self, text: str) -> list[str]:
        """Type a query, run it, and return the molecule keys that came back."""
        self.page.fill("#q", text)
        self.page.click("#go")
        self.idle()
        return self.result_keys()

    def result_keys(self) -> list[str]:
        return self.page.eval_on_selector_all(
            "#results li .key", "nodes => nodes.map(n => n.textContent)"
        )

    def click_result(self, n: int = 0) -> None:
        self.page.locator("#results li").nth(n).click()
        self.idle()

    def choose_group(self, preset: str) -> None:
        """Tick one group in the picker. Groups with no members are disabled."""
        self.page.check(f"#preset-groups input[data-fg='{preset}']")
        self.idle()

    def set_control(self, element_id: str, value: str) -> None:
        """Set a control the way a person would, so the change event fires."""
        locator = self.page.locator(f"#{element_id}")
        if locator.evaluate("node => node.tagName") == "SELECT":
            locator.select_option(value)
        elif locator.evaluate("node => node.type") == "checkbox":
            locator.set_checked(value in {"1", "true", "on"})
        else:
            locator.fill(value)
            locator.dispatch_event("change")
        self.idle()

    def set_temperature(self, kelvin: str) -> None:
        """Set the temperature the way a person would.

        The stored values are a dropdown; anything else means picking
        "another temperature…" first and then typing into the box that
        reveals. Tests should not have to know which of the two a given
        number is.
        """
        select = self.page.locator("#temperature")
        options = select.locator("option").evaluate_all("nodes => nodes.map(n => n.value)")
        if kelvin in options:
            select.select_option(kelvin)
        else:
            select.select_option("custom")
            box = self.page.locator("#temperature-custom")
            box.fill(kelvin)
            box.dispatch_event("change")
        self.idle()
        # idle() can return before the change event has even started its
        # request, and then the caller asserts against the previous render.
        # Wait for the authoritative document to carry the new value instead.
        self.page.wait_for_function(
            "want => state && Math.abs(state.settings.temperature - want) < 1e-9",
            arg=float(kelvin),
            timeout=15000,
        )

    def open_panel(self, name: str) -> None:
        self.page.click(f"button[data-panel='{name}']")
        self.idle()

    def open_view(self, name: str) -> None:
        self.page.click(f".bar button[data-view='{name}']")
        self.idle()

    # ---------------------------------------------------------------- reading

    def text(self, element_id: str) -> str:
        return (self.page.locator(f"#{element_id}").text_content() or "").strip()

    def visible(self, element_id: str) -> bool:
        return self.page.locator(f"#{element_id}").is_visible()

    def toast(self) -> str | None:
        node = self.page.locator("#toast")
        return node.text_content() if node.is_visible() else None

    def client_state(self) -> dict:
        """The client's copy of the authoritative state document.

        ``state`` is a top-level binding in a classic script, so this reads what
        the client actually holds — which turns "did the server's clamp reach
        the controls?" into a dict comparison instead of DOM scraping.
        """
        return self.page.evaluate("() => state")

    # ------------------------------------------------------------ screenshots

    def shot(self, label: str) -> Path:
        """Write a PNG and note where it went.

        Named from the test so the path is stable across runs, and recorded in a
        manifest because pytest swallows stdout when a test passes.
        """
        self.idle()
        self.screens.mkdir(parents=True, exist_ok=True)
        path = self.screens / f"{_slug(self.node_name)}__{_slug(label)}.png"
        self.page.screenshot(path=str(path), full_page=True)
        with (self.screens / "manifest.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{path}\n")
        return path

    # ----------------------------------------------------------- the verdict

    def assert_clean(self) -> None:
        """Fail the test if the browser reported anything."""
        kept = [
            message
            for message in self.errors
            if not any(allowed in message for allowed in self.allow)
        ]
        if not kept:
            return
        timeline = " · ".join(self.network[-14:])
        raise AssertionError(
            f"{len(kept)} browser error(s) in {self.node_name}:\n\n"
            + "\n\n".join(kept)
            + f"\n\nnetwork: {timeline}"
        )


def attach(ui: Ui) -> None:
    """Start listening. Must run before any navigation, or the first load is missed."""
    page = ui.page
    expect.set_options(timeout=DEFAULT_TIMEOUT_MS)

    def on_console(message) -> None:
        if message.type != "error":
            return
        # The browser logs every non-2xx response as a console error of its own.
        # 400, 409 and 503 are designed paths here — /view/render answers 400
        # with no molecule loaded, and the app draws an empty stage — so this
        # would fail tests for behaving correctly. The status is still in the
        # network timeline, and a genuine JS fault arrives via pageerror, which
        # is not filtered. app.js never calls console itself, so nothing else is
        # lost.
        if message.text.startswith("Failed to load resource"):
            return
        where = message.location or {}
        ui.errors.append(
            f"[console.error] {message.text}\n    at {where.get('url', '?')}:"
            f"{where.get('lineNumber', '?')}:{where.get('columnNumber', '?')}"
        )

    def on_pageerror(error) -> None:
        # No bundler and no source maps, so the stack names real app.js lines.
        ui.errors.append(f"[pageerror] {error.message}\n" + indent(error.stack or "", "    "))

    def on_requestfailed(request) -> None:
        ui.errors.append(
            f"[requestfailed] {request.method} {_path(request.url)} — {request.failure}"
        )

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    page.on("requestfailed", on_requestfailed)
    page.on(
        "response",
        lambda response: ui.network.append(
            f"{response.request.method} {_path(response.url)} {response.status}"
        ),
    )
