"""Tests for the HTTP router and the transport around it.

Almost everything goes through ``dispatch()`` directly — no socket, no client, no
event loop — because the router is a pure function of a request. Only the last
few tests bind a port, and those check the transport rather than any behaviour.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from rpfr_visualizer.gui.app import MAX_HITS, Response, dispatch
from rpfr_visualizer.gui.config import Resolution
from rpfr_visualizer.gui.session import ExplorerSession

pytest.importorskip("rdkit")


@pytest.fixture
def indexed_session(synthetic_h5_path: Path, tmp_path: Path) -> ExplorerSession:
    """A session with a real search index built over the synthetic fixture."""
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(synthetic_h5_path, raw / "qm9s_tr_project_1pct.h5")
    (tmp_path / "cache").mkdir()

    catalog = DatasetCatalog()
    catalog.register("qm9s", raw / "qm9s_tr_project_1pct.h5")
    index_path = tmp_path / "cache" / "qm9s_rpfr_index.sqlite"
    build_index(catalog, index_path, progress=False)

    session = ExplorerSession(
        Resolution(data_dir=tmp_path, index_path=index_path, max_stats_molecules=2)
    )
    session.open()
    return session


def call(
    session: ExplorerSession, method: str, path: str, query: str = "", body: dict | None = None
) -> tuple[int, object]:
    response = dispatch(
        session, method, path, query, json.dumps(body).encode() if body is not None else None
    )
    assert isinstance(response, Response)
    try:
        return response.status, json.loads(response.body)
    except json.JSONDecodeError:
        return response.status, response.body


# ------------------------------------------------------------------- the shell


def test_healthz_never_needs_a_controller() -> None:
    status, payload = call(ExplorerSession(Resolution()), "GET", "/healthz")
    assert status == 200
    assert payload == {"ok": True, "ready": False}


def test_unknown_route_is_404(indexed_session: ExplorerSession) -> None:
    status, payload = call(indexed_session, "GET", "/api/nope")
    assert status == 404
    assert payload["error"]["kind"] == "not_found"


def test_every_data_route_is_503_before_setup() -> None:
    """A client must never have to guess why a route came back empty."""
    session = ExplorerSession(Resolution())
    for method, path in [
        ("GET", "/api/state"),
        ("GET", "/view/render"),
        ("GET", "/view/ranks"),
        ("GET", "/api/statistics"),
        ("GET", "/api/download"),
        ("GET", "/api/download.csv"),
        ("POST", "/api/search"),
    ]:
        status, payload = call(session, method, path, body={"query": "x"})
        assert status == 503, path
        assert payload["error"]["kind"] == "not_ready"

    # …but setup and liveness answer in that state, which is the point.
    assert call(session, "GET", "/api/setup")[0] == 200
    assert call(session, "GET", "/healthz")[0] == 200


def test_malformed_body_is_a_user_error(indexed_session: ExplorerSession) -> None:
    response = dispatch(indexed_session, "POST", "/api/state", "", b"{not json")
    assert response.status == 400
    assert json.loads(response.body)["error"]["kind"] == "user"


# ----------------------------------------------------------------------- state


def test_state_round_trips_one_field_of_each_type(indexed_session: ExplorerSession) -> None:
    for field, given, expected in [
        ("element", "C", "C"),
        ("temperature", 273.15, 273.15),
        ("show_hydrogens", False, False),
        ("top_n", 25, 25),
        ("kind", "ln_alpha", "ln_alpha"),
    ]:
        status, payload = call(indexed_session, "POST", "/api/state", body={field: given})
        assert status == 200, field
        assert payload["settings"][field] == expected
        assert getattr(indexed_session.controller.state, field) is not None


def test_state_rejects_an_unknown_field(indexed_session: ExplorerSession) -> None:
    status, payload = call(indexed_session, "POST", "/api/state", body={"colour": "dark"})
    assert status == 400
    assert "Unknown setting" in payload["error"]["message"]


def test_state_points_dataset_changes_at_their_own_action(
    indexed_session: ExplorerSession,
) -> None:
    """use_dataset also resets the selection, so it cannot be a plain setting."""
    status, payload = call(indexed_session, "POST", "/api/state", body={"dataset": "qm9s"})
    assert status == 400
    assert "its own action" in payload["error"]["message"]


def test_setting_the_temperature_off_grid_is_flagged(indexed_session: ExplorerSession) -> None:
    status, payload = call(indexed_session, "POST", "/api/state", body={"temperature": 250.0})
    assert status == 200
    assert payload["on_grid"] is False
    assert any("fitted" in w for w in payload["warnings"])


# ------------------------------------------------------------------- selection


def test_load_then_render(indexed_session: ExplorerSession) -> None:
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200

    call(indexed_session, "POST", "/api/state", body={"use_3d": False})
    status, fragment = call(indexed_session, "GET", "/view/render")
    assert status == 200
    pane = fragment["panes"][0]
    assert pane["svg"] and "<svg" in pane["svg"].lower()
    assert "site" in pane["table"]
    assert fragment["caption"]
    assert fragment["provenance"]["level"] in {"ok", "warn", "error"}


def test_rendering_without_a_molecule_uses_the_controller_wording(
    indexed_session: ExplorerSession,
) -> None:
    """The controller's messages are the documentation; do not paraphrase them."""
    status, payload = call(indexed_session, "GET", "/view/render")
    assert status == 400
    assert payload["error"]["message"] == "Load molecule A first"


def test_examples_only_offers_renderable_molecules(indexed_session: ExplorerSession) -> None:
    """Taken from the index, not from list_keys().

    list_keys() opens the HDF5 and sorts every group name with no caching, and in
    a dataset with no structures it hands back a molecule that cannot be drawn.
    """
    status, payload = call(indexed_session, "GET", "/api/examples")
    assert status == 200
    assert payload["renderable"] is True
    assert payload["keys"]
    assert len(payload["keys"]) <= 5


# --------------------------------------------------------------------- search


def test_search_reports_truncation(indexed_session: ExplorerSession) -> None:
    status, payload = call(indexed_session, "POST", "/api/search", body={"query": "C", "limit": 1})
    assert status == 200
    assert payload["shown"] <= 1
    assert "elapsed_ms" in payload


def test_search_limit_is_capped(indexed_session: ExplorerSession) -> None:
    """No pagination, so the cap has to be enforced rather than trusted."""
    status, payload = call(
        indexed_session, "POST", "/api/search", body={"query": "C", "limit": 10_000_000}
    )
    assert status == 200
    assert payload["limit"] == MAX_HITS


def test_empty_search_is_a_user_error(indexed_session: ExplorerSession) -> None:
    status, payload = call(indexed_session, "POST", "/api/search", body={"query": "  "})
    assert status == 400
    assert "Type something" in payload["error"]["message"]


def test_download_preview_asks_for_a_molecule_until_one_is_loaded(
    indexed_session: ExplorerSession,
) -> None:
    status, payload = call(indexed_session, "GET", "/api/download", query="scope=molecule")
    assert status == 200
    assert payload["can_download"] is False
    assert "Load a molecule" in payload["reason"]

    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    status, payload = call(indexed_session, "GET", "/api/download", query="scope=molecule")
    assert status == 200
    assert payload["can_download"] is True
    assert payload["n_molecules"] == 1
    assert payload["filename"].endswith(".csv")
    assert "value" in payload["columns"]


def test_download_preview_disables_when_no_sites_match_the_element(
    indexed_session: ExplorerSession,
) -> None:
    """Carbon lock on an N-only molecule must not enable a zero-row CSV."""
    assert call(indexed_session, "POST", "/api/load", body={"key": "3"})[0] == 200
    assert call(indexed_session, "POST", "/api/state", body={"element": "C"})[0] == 200
    status, payload = call(indexed_session, "GET", "/api/download", query="scope=molecule")
    assert status == 200
    assert payload["can_download"] is False
    assert payload["n_sites_estimate"] == 0
    assert "C sites" in payload["reason"]


def csv_text(response) -> str:
    """The CSV body, whether it arrived inline or as a stream.

    The export spools to a temporary file so a large download costs bounded
    memory, so the bytes are on ``response.stream`` rather than ``response.body``.
    """
    if getattr(response, "stream", None) is not None:
        try:
            return response.stream.read().decode("utf-8")
        finally:
            response.stream.close()
    return response.body.decode()


def test_download_csv_is_one_row_per_site(indexed_session: ExplorerSession) -> None:
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert (
        call(indexed_session, "POST", "/api/state", body={"element": "", "kind": "beta"})[0] == 200
    )

    response = dispatch(indexed_session, "GET", "/api/download.csv", "scope=molecule")
    assert response.status == 200
    assert response.content_type.startswith("text/csv")
    assert "attachment" in response.headers.get("Content-Disposition", "")
    assert response.content_length is not None
    text = csv_text(response)
    assert len(text.encode("utf-8")) == response.content_length
    lines = [line for line in text.splitlines() if line]
    assert lines[0].startswith("dataset,molecule_key")
    assert len(lines) > 1


def test_the_csv_records_which_fit_produced_its_beta(indexed_session: ExplorerSession) -> None:
    """The beta column depends on a sidebar setting, so the file must say which.

    Two exports of the same query at the same temperature can hold different
    numbers. Without this column there is nothing in the file to explain why.
    """
    import csv as _csv

    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert call(indexed_session, "POST", "/api/state", body={"element": ""})[0] == 200

    def column(temperature: float, method: str) -> list[tuple[str, str]]:
        body = {"temperature": temperature, "beta_method": method}
        assert call(indexed_session, "POST", "/api/state", body=body)[0] == 200
        response = dispatch(indexed_session, "GET", "/api/download.csv", "scope=molecule")
        rows = list(_csv.DictReader(csv_text(response).splitlines()))
        assert rows
        return [(row["beta_method"], row["beta"]) for row in rows]

    off_poly = column(305.0, "poly5")
    off_spline = column(305.0, "pchip")
    assert [m for m, _b in off_poly] == ["poly5"] * len(off_poly)
    assert [m for m, _b in off_spline] == ["pchip"] * len(off_spline)

    # A stored temperature is read, never fitted: no method, and the two agree.
    on_poly = column(298.15, "poly5")
    on_spline = column(298.15, "pchip")
    assert [m for m, _b in on_poly] == [""] * len(on_poly)
    assert [b for _m, b in on_poly] == [b for _m, b in on_spline]


def test_the_3d_viewer_needs_no_lock_once_a_render_has_run(
    indexed_session: ExplorerSession,
) -> None:
    """The <iframe> is fetched by the browser, not by the client's own queue.

    It therefore lands alongside whatever the client asks for next -- a Derived
    reload, say -- and refusing it as busy put a raw JSON error inside the pane
    where the structure should be. The render that handed out the URL builds
    the document, so answering it needs no lock at all.
    """
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert dispatch(indexed_session, "GET", "/view/render").status == 200

    # Held for the whole request, and not released: nothing may wait on it.
    indexed_session._lock.acquire()
    try:
        response = dispatch(indexed_session, "GET", "/viewer/3d", "slot=a")
    finally:
        indexed_session._lock.release()

    assert response.status == 200
    assert response.content_type.startswith("text/html")
    assert b"3dmolviewer" in response.body.lower(), "served a placeholder, not the structure"


def test_the_3d_viewer_still_answers_a_direct_hit(indexed_session: ExplorerSession) -> None:
    """Opening the URL with no render behind it falls back to building one."""
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200

    response = dispatch(indexed_session, "GET", "/viewer/3d", "slot=a")
    assert response.status == 200
    assert response.content_type.startswith("text/html")


def test_a_state_change_is_still_refused_rather_than_queued(
    indexed_session: ExplorerSession,
) -> None:
    """The other half of the rule. A double-clicked control must be dropped,
    not run twice, so nothing that writes state waits for the lock."""
    indexed_session._lock.acquire()
    try:
        status, payload = call(indexed_session, "POST", "/api/state", body={"temperature": 300.0})
    finally:
        indexed_session._lock.release()

    assert status == 409
    assert payload["error"]["kind"] == "busy"


def test_the_csv_can_cover_several_temperatures(indexed_session: ExplorerSession) -> None:
    """One row per site per temperature, each row saying which it is.

    A single export is how a temperature calibration gets built; doing it by
    downloading once per temperature and stitching the files is the workflow
    this replaces.
    """
    import csv as _csv

    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert call(indexed_session, "POST", "/api/state", body={"element": ""})[0] == 200

    one = list(
        _csv.DictReader(
            csv_text(
                dispatch(indexed_session, "GET", "/api/download.csv", "scope=molecule")
            ).splitlines()
        )
    )

    body = {"download_temperatures": "250, 273.15, 298.15"}
    status, payload = call(indexed_session, "POST", "/api/state", body=body)
    assert status == 200
    assert payload["settings"]["download_temperatures"] == [250.0, 273.15, 298.15]

    many = list(
        _csv.DictReader(
            csv_text(
                dispatch(indexed_session, "GET", "/api/download.csv", "scope=molecule")
            ).splitlines()
        )
    )
    assert len(many) == 3 * len(one)
    # A site's temperature series lands contiguously, which is the shape you
    # want if you are fitting a calibration to it.
    assert [row["temperature_K"] for row in many[:3]] == ["250", "273.15", "298.15"]
    assert len({row["atom_index"] for row in many[:3]}) == 1

    # 250 K is not stored; the other two are. Each row says which it was.
    by_temperature = {row["temperature_K"]: row for row in many}
    assert by_temperature["250"]["on_grid"] == "false"
    assert by_temperature["250"]["beta_method"]
    assert by_temperature["298.15"]["on_grid"] == "true"
    assert by_temperature["298.15"]["beta_method"] == ""


def test_the_download_preview_counts_rows_across_the_temperatures(
    indexed_session: ExplorerSession,
) -> None:
    """A four-temperature export is four times the rows, and the count has to
    say so before the click rather than after it."""
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert call(indexed_session, "POST", "/api/state", body={"element": ""})[0] == 200

    _status, single = call(indexed_session, "GET", "/api/download", query="scope=molecule")
    assert (
        call(indexed_session, "POST", "/api/state", body={"download_temperatures": "250, 300"})[0]
        == 200
    )
    _status, double = call(indexed_session, "GET", "/api/download", query="scope=molecule")

    assert double["temperatures"] == [250.0, 300.0]
    assert double["n_sites_estimate"] == 2 * single["n_sites_estimate"]
    # The name cannot claim one temperature when the file holds two.
    assert double["filename"].endswith("_2T.csv")


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ("250, banana", "not a temperature"),
        ("5", "outside the stored range"),
        ("1000", "outside the stored range"),
        (", ".join(str(200 + i) for i in range(60)), "more than the"),
    ],
)
def test_a_bad_temperature_list_is_refused_in_plain_words(
    indexed_session: ExplorerSession, given: str, message: str
) -> None:
    status, payload = call(
        indexed_session, "POST", "/api/state", body={"download_temperatures": given}
    )
    assert status == 400
    assert message in payload["error"]["message"]


def test_an_empty_temperature_list_means_the_one_on_screen(
    indexed_session: ExplorerSession,
) -> None:
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert (
        call(indexed_session, "POST", "/api/state", body={"download_temperatures": "250, 300"})[0]
        == 200
    )
    status, payload = call(
        indexed_session, "POST", "/api/state", body={"download_temperatures": ""}
    )
    assert status == 200
    assert payload["settings"]["download_temperatures"] == []

    _status, preview = call(indexed_session, "GET", "/api/download", query="scope=molecule")
    assert preview["temperatures"] == [preview["temperature"]]


def test_download_search_results_uses_the_hit_list(indexed_session: ExplorerSession) -> None:
    assert call(indexed_session, "POST", "/api/search", body={"query": "C"})[0] == 200
    status, preview = call(indexed_session, "GET", "/api/download", query="scope=results")
    assert status == 200
    assert preview["can_download"] is True
    assert preview["n_molecules"] >= 1

    response = dispatch(indexed_session, "GET", "/api/download.csv", "scope=results")
    assert response.status == 200
    assert csv_text(response).count("\n") > 1


def test_download_covers_every_match_not_just_the_page(
    indexed_session: ExplorerSession,
) -> None:
    """The result list stops at the display limit; the file must not.

    Searched with a limit of 1, the page shows one molecule while the query
    matches more -- and the CSV has to contain all of them, or a partial export
    goes out with nothing in it saying so.
    """
    search = call(indexed_session, "POST", "/api/search", body={"query": "smarts:[#6]", "limit": 1})
    assert search[0] == 200

    status, preview = call(indexed_session, "GET", "/api/download", query="scope=results")
    assert status == 200
    assert preview["shown"] == 1
    assert preview["n_molecules"] > preview["shown"], "fixture has only one match"

    response = dispatch(indexed_session, "GET", "/api/download.csv", "scope=results")
    keys = {line.split(",")[1] for line in csv_text(response).splitlines()[1:] if line}
    assert len(keys) == preview["n_molecules"]


def test_download_without_results_is_a_user_error(indexed_session: ExplorerSession) -> None:
    response = dispatch(indexed_session, "GET", "/api/download.csv", "scope=results")
    assert response.status == 400
    assert "Search" in json.loads(response.body)["error"]["message"]


def test_functional_group_search_without_a_choice_uses_the_controller_wording(
    indexed_session: ExplorerSession,
) -> None:
    status, payload = call(indexed_session, "POST", "/api/search/fg", body={})
    assert status == 400
    assert "Choose a functional group" in payload["error"]["message"]


def test_presets_list_every_group_including_empty_ones(
    indexed_session: ExplorerSession,
) -> None:
    """Absence is information: "none here" beats a missing row."""
    status, payload = call(indexed_session, "GET", "/api/fg/presets")
    assert status == 200
    assert len(payload["presets"]) == 18
    assert any(p["count"] == 0 for p in payload["presets"])
    assert all(set(p) == {"name", "description", "count"} for p in payload["presets"])


# ----------------------------------------------------------------------- assets


def test_static_assets_are_served(indexed_session: ExplorerSession) -> None:
    for name, marker in [("index.html", b"<"), ("app.css", b""), ("app.js", b"")]:
        response = dispatch(indexed_session, "GET", f"/static/{name}")
        assert response.status == 200, name
        assert marker in response.body


def test_path_traversal_is_impossible(indexed_session: ExplorerSession) -> None:
    """A whitelist, not a guard: there is no path join from user input."""
    for attempt in ["../../etc/passwd", "..%2f..%2fetc%2fpasswd", "nope.js", "app.css/../app.js"]:
        response = dispatch(indexed_session, "GET", f"/static/{attempt}")
        assert response.status == 404, attempt


def test_the_bundle_is_served_patched(indexed_session: ExplorerSession) -> None:
    """Served raw, the original UMD footer would never set window.$3Dmol."""
    pytest.importorskip("py3Dmol")
    response = dispatch(indexed_session, "GET", "/static/3Dmol-min.js")
    assert response.status == 200
    assert len(response.body) > 500_000
    assert b'window["3Dmol"]=window.$3Dmol=factory();' in response.body
    assert "immutable" in response.headers["Cache-Control"]


def test_the_viewer_document_links_the_bundle(indexed_session: ExplorerSession) -> None:
    pytest.importorskip("py3Dmol")
    keys = indexed_session.controller.provider.list_keys()
    call(indexed_session, "POST", "/api/load", body={"key": keys[0]})

    response = dispatch(indexed_session, "GET", "/viewer/3d", "slot=a")
    assert response.status == 200
    assert b'<script src="/static/3Dmol-min.js"></script>' in response.body
    # The guard against silently regressing to a 537 KB inline bundle.
    assert len(response.body) < 60_000


# ------------------------------------------------------------------- busy/lock


def test_busy_is_409_and_leaves_reads_working(indexed_session: ExplorerSession) -> None:
    """Testable without racing threads: just hold the lock."""
    indexed_session._lock.acquire()
    try:
        status, payload = call(indexed_session, "GET", "/view/render")
        assert status == 409
        assert payload["error"]["kind"] == "busy"

        # The snapshot read and liveness never take the lock.
        assert call(indexed_session, "GET", "/api/state")[0] == 200
        assert call(indexed_session, "GET", "/healthz")[0] == 200
    finally:
        indexed_session._lock.release()


# ------------------------------------------------------------------- transport


@pytest.fixture
def running(indexed_session: ExplorerSession):
    """A real server on an ephemeral port."""
    from rpfr_visualizer.gui.server import serve_in_thread

    server, url = serve_in_thread(indexed_session, port=0)
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


def test_server_answers_over_a_socket(running: str) -> None:
    with urllib.request.urlopen(f"{running}/healthz", timeout=10) as response:
        assert response.status == 200
        assert json.loads(response.read())["ok"] is True


def test_content_length_matches_the_body(running: str) -> None:
    """HTTP/1.1 keep-alive is only safe with an exact Content-Length."""
    with urllib.request.urlopen(f"{running}/api/state", timeout=10) as response:
        body = response.read()
        assert int(response.headers["Content-Length"]) == len(body)


def test_two_requests_on_one_connection(running: str) -> None:
    """The HTTP/1.0-default trap: without protocol_version this hangs or resets."""
    import http.client
    from urllib.parse import urlparse

    parsed = urlparse(running)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    try:
        for _ in range(2):
            conn.request("GET", "/healthz")
            response = conn.getresponse()
            assert response.status == 200
            response.read()
    finally:
        conn.close()


def test_a_foreign_host_header_is_refused(running: str) -> None:
    """DNS rebinding: a public page must not be able to talk to this server."""
    from urllib.parse import urlparse

    parsed = urlparse(running)
    request = urllib.request.Request(f"{running}/api/state", headers={"Host": "evil.example.com"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 403
    assert parsed.port  # the fixture really did bind somewhere


def test_refuses_to_bind_a_public_address() -> None:
    from rpfr_visualizer.gui.server import launch

    with pytest.raises(SystemExit, match="Refusing to bind"):
        launch(host="0.0.0.0", open_browser=False)  # noqa: S104 — that is the point


def test_start_explorer_opens_chrome_and_answers(
    indexed_session: ExplorerSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notebook entry point must return, not block the kernel."""
    from rpfr_visualizer.gui.server import start_explorer

    opened: list[str] = []
    monkeypatch.setattr(
        "rpfr_visualizer.gui.server.open_in_chrome",
        lambda url: opened.append(url) or "Google Chrome",
    )
    server, url = start_explorer(
        data_dir=indexed_session.resolution.data_dir,
        index_path=indexed_session.resolution.index_path,
        port=0,
        open_browser=True,
    )
    try:
        with urllib.request.urlopen(f"{url}/healthz", timeout=10) as response:
            assert json.loads(response.read()) == {"ok": True, "ready": True}
        assert opened == [url]
    finally:
        server.shutdown()
        server.server_close()


def test_open_in_chrome_falls_back_to_the_system_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpfr_visualizer.gui import server as gui_server

    monkeypatch.setattr(gui_server.sys, "platform", "linux")
    monkeypatch.setattr(gui_server.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        gui_server.webbrowser,
        "get",
        lambda _name: (_ for _ in ()).throw(gui_server.webbrowser.Error("no")),
    )
    opened: list[str] = []
    monkeypatch.setattr(gui_server.webbrowser, "open", lambda url, new=2: opened.append(url))

    assert gui_server.open_in_chrome("http://127.0.0.1:8765") == "default browser"
    assert opened == ["http://127.0.0.1:8765"]


def test_every_static_asset_resolves_from_the_package(indexed_session: ExplorerSession) -> None:
    """Catches a force-include line forgotten in pyproject.toml.

    The assets load through importlib.resources, so this is what would fail in a
    built wheel rather than in the source tree.
    """
    from importlib import resources

    for name in ("index.html", "app.css", "app.js"):
        data = (resources.files("rpfr_visualizer.gui") / "static" / name).read_bytes()
        assert len(data) > 200, name


def test_the_page_references_the_assets_it_needs(indexed_session: ExplorerSession) -> None:
    response = dispatch(indexed_session, "GET", "/")
    assert response.status == 200
    body = response.body.decode()
    assert "/static/app.css" in body
    assert "/static/app.js" in body


def test_two_servers_in_one_process_keep_their_own_sessions(
    synthetic_h5_path: Path, tmp_path: Path
) -> None:
    """The handler used to hold the session as a class attribute.

    Class attributes are process-global, so starting a second server silently
    retargeted the first: it kept answering, from the wrong data. That fails as
    a wrong answer rather than an error, which is the worst way to fail.
    """
    import json
    import urllib.request

    from rpfr_visualizer.gui.server import serve_in_thread

    def build(root: Path, dataset_file: str) -> ExplorerSession:
        from rpfr_visualizer.io.providers import DatasetCatalog
        from rpfr_visualizer.io.search import build_index

        raw = root / "raw"
        raw.mkdir(parents=True)
        (root / "cache").mkdir()
        shutil.copy(synthetic_h5_path, raw / dataset_file)
        catalog = DatasetCatalog()
        catalog.register(dataset_file.split("_tr_")[0], raw / dataset_file)
        index_path = root / "cache" / "qm9s_rpfr_index.sqlite"
        build_index(catalog, index_path, progress=False)
        session = ExplorerSession(
            Resolution(data_dir=root, index_path=index_path, max_stats_molecules=2)
        )
        session.open()
        return session

    first = build(tmp_path / "one", "qm9s_tr_project_1pct.h5")
    second = build(tmp_path / "two", "qm9s_tr_skip_1pct.h5")

    server_a, url_a = serve_in_thread(first, port=0)
    server_b, url_b = serve_in_thread(second, port=0)
    try:
        with urllib.request.urlopen(f"{url_a}/api/state", timeout=10) as response:
            state_a = json.loads(response.read())
        with urllib.request.urlopen(f"{url_b}/api/state", timeout=10) as response:
            state_b = json.loads(response.read())
    finally:
        for server in (server_a, server_b):
            server.shutdown()
            server.server_close()

    assert state_a["dataset"] == "qm9s"
    assert state_b["dataset"] == "qm9s_skip"


def test_client_error_routes_are_absent_without_dev() -> None:
    """A normal install must expose no endpoint that accepts browser text.

    This is the assertion that matters: the route stores and prints whatever it
    is sent, so it exists only when the person running the server asked for it.
    """
    session = ExplorerSession(Resolution())
    assert session.dev is False

    status, _ = call(session, "POST", "/api/_client_error", body={"kind": "error", "message": "x"})
    assert status == 404
    status, _ = call(session, "GET", "/api/_client_error")
    assert status == 404
    assert session.client_errors == []


def test_dev_mode_records_and_returns_client_errors(capsys) -> None:
    """An uncaught TypeError in app.js otherwise dies in a console nobody reads."""
    session = ExplorerSession(Resolution())
    session.dev = True

    status, _ = call(
        session,
        "POST",
        "/api/_client_error",
        body={
            "kind": "error",
            "message": "Cannot set properties of null",
            "source": "http://127.0.0.1:8765/static/app.js",
            "line": 126,
            "stack": "at applyState (app.js:126:31)",
        },
    )
    assert status == 200

    printed = capsys.readouterr().err
    assert "Cannot set properties of null" in printed
    assert "applyState" in printed

    status, payload = call(session, "GET", "/api/_client_error")
    assert status == 200
    assert payload["errors"][0]["message"] == "Cannot set properties of null"
    assert payload["errors"][0]["line"] == "126"


def test_client_error_fields_are_truncated_and_bounded() -> None:
    """One bad loop in the browser must not be able to flood the server."""
    from rpfr_visualizer.gui.app import MAX_CLIENT_ERRORS, MAX_CLIENT_FIELD

    session = ExplorerSession(Resolution())
    session.dev = True

    call(session, "POST", "/api/_client_error", body={"kind": "error", "message": "x" * 9000})
    assert len(session.client_errors[0]["message"]) == MAX_CLIENT_FIELD

    for n in range(MAX_CLIENT_ERRORS + 25):
        call(session, "POST", "/api/_client_error", body={"kind": "error", "message": str(n)})
    assert len(session.client_errors) == MAX_CLIENT_ERRORS
    # The cap drops the oldest, so the most recent report survives.
    assert session.client_errors[-1]["message"] == str(MAX_CLIENT_ERRORS + 24)


def test_editable_assets_are_not_cached_but_the_bundle_is() -> None:
    """A front-end edit is live on reload; a heuristic cache would undo that."""
    session = ExplorerSession(Resolution())

    for name in ("index.html", "app.css", "app.js"):
        response = dispatch(session, "GET", f"/static/{name}")
        assert response.headers.get("Cache-Control") == "no-store", name

    bundle = dispatch(session, "GET", "/static/3Dmol-min.js")
    assert "immutable" in bundle.headers.get("Cache-Control", "")


# --- derived properties -----------------------------------------------------


def test_derived_needs_a_molecule(indexed_session: ExplorerSession) -> None:
    status, payload = call(indexed_session, "GET", "/view/derived")
    assert status == 200
    assert payload["available"] is False
    assert "Load molecule" in payload["message"]


def test_derived_reports_beta_curves_and_the_clumping_matrix(
    indexed_session: ExplorerSession,
) -> None:
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200

    status, payload = call(indexed_session, "GET", "/view/derived")

    assert status == 200
    assert payload["available"] is True
    assert payload["on_grid"] is True
    series = payload["beta"]["sites"]
    assert series
    for s in series:
        # Stored points and the fitted curve are separate: the markers are read
        # from the file, the line is the model.
        assert len(s["temperatures"]) == len(s["values"])
        assert len(s["curve_temperatures"]) > len(s["temperatures"])
        assert s["temperatures"] == sorted(s["temperatures"])
    assert payload["clumping"]["available"] is True
    assert payload["clumping"]["matrix"][0][0] is None, "the diagonal must stay off the scale"


def test_derived_off_grid_is_unavailable_not_an_error(
    indexed_session: ExplorerSession,
) -> None:
    """Typing an arbitrary temperature is normal use. Clumping and mass law have
    no interpolation model, so they go unavailable -- but a 400 would read as a
    bug rather than as an answer, and beta must still draw."""
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert call(indexed_session, "POST", "/api/state", body={"temperature": 305.0})[0] == 200

    status, payload = call(indexed_session, "GET", "/view/derived")

    assert status == 200
    assert payload["on_grid"] is False
    assert payload["clumping"]["available"] is False
    assert payload["nearest_grid"], "the user needs somewhere to go"
    assert all(t in payload["nearest_grid"] for t in payload["nearest_grid"])
    assert payload["beta"]["sites"], "beta interpolates and must still be drawn"


def test_a_clumping_pair_can_be_highlighted(indexed_session: ExplorerSession) -> None:
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200

    status, payload = call(indexed_session, "POST", "/api/state", body={"highlight_pair": [0, 1]})

    assert status == 200
    assert payload["settings"]["highlight_pair"] == [0, 1]


def test_a_pair_of_one_atom_is_rejected(indexed_session: ExplorerSession) -> None:
    status, payload = call(indexed_session, "POST", "/api/state", body={"highlight_pair": [2, 2]})
    assert status == 400
    assert "two different atoms" in payload["error"]["message"]


def test_clumping_aggregation_is_validated(indexed_session: ExplorerSession) -> None:
    assert (
        call(indexed_session, "POST", "/api/state", body={"derived_aggregation": "max"})[0] == 200
    )
    status, payload = call(
        indexed_session, "POST", "/api/state", body={"derived_aggregation": "median"}
    )
    assert status == 400
    assert "mean" in payload["error"]["message"]


def test_clumping_is_reported_in_per_mil(indexed_session: ExplorerSession) -> None:
    """The clumped-isotope Δ: 1000 ln(beta_ij / beta_i beta_j), in per mil.

    Reporting the raw ratio put every value at ~1.0, which is not an excess
    scale at all.
    """
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200

    _status, payload = call(indexed_session, "GET", "/view/derived")
    clumping = payload["clumping"]

    assert clumping["unit"] == "Δ (‰)"
    values = [v for row in clumping["matrix"] for v in row if v is not None]
    assert values
    # ln of a ratio near 1 is near 0; scaled by 1000 it is a per-mil number,
    # so nothing should sit at the ~1.0 the raw factor would give.
    assert all(abs(v) < 500 for v in values)


def test_clumping_pairs_are_classified_by_hydrogen(
    indexed_session: ExplorerSession,
) -> None:
    """H pairs run an order of magnitude above heavy-atom pairs, so the client
    has to be able to separate them or the heavy structure is invisible."""
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200

    _status, payload = call(indexed_session, "GET", "/view/derived")
    clumping = payload["clumping"]

    assert len(clumping["is_hydrogen"]) == len(clumping["labels"])
    for pair in clumping["pairs"]:
        expected = clumping["is_hydrogen"][pair["i"]] or clumping["is_hydrogen"][pair["j"]]
        assert pair["has_h"] is expected


def test_a_fitted_temperature_is_said_once_not_twice(indexed_session: ExplorerSession) -> None:
    """The provenance line and a notice both said the temperature was fitted,
    in different words, one above the other."""
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert call(indexed_session, "POST", "/api/state", body={"temperature": 210.0})[0] == 200

    _status, fragment = call(indexed_session, "GET", "/view/render")
    provenance = fragment["provenance"]["text"]
    assert "not a stored temperature" in provenance
    assert "local quadratic" in provenance
    assert not any("stored temperature" in w for w in fragment["warnings"])


def test_the_download_page_uses_the_explore_wording(indexed_session: ExplorerSession) -> None:
    keys = indexed_session.controller.provider.list_keys()
    assert call(indexed_session, "POST", "/api/load", body={"key": keys[0]})[0] == 200
    assert call(indexed_session, "POST", "/api/state", body={"temperature": 210.0})[0] == 200

    _status, preview = call(indexed_session, "GET", "/api/download", query="scope=molecule")
    _status, fragment = call(indexed_session, "GET", "/view/render")
    assert preview["temperature_note"] == fragment["provenance"]["text"]
    assert preview["quantity_symbol"] == "β"

    body = {"download_temperatures": "210, 298.15"}
    assert call(indexed_session, "POST", "/api/state", body=body)[0] == 200
    _status, preview = call(indexed_session, "GET", "/api/download", query="scope=molecule")
    assert "1 stored, 1 fitted" in preview["temperature_note"]
    assert "beta_method column" in preview["temperature_note"]
