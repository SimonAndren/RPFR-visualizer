"""Routing, as a pure function of a request.

``dispatch()`` takes a method, a path, a query and a body, and returns a
:class:`Response`. It opens no sockets and imports nothing from
:mod:`http.server`, which is what lets almost every route be tested by calling a
function rather than by standing up a server.

Handlers write settings only to ``controller.state`` and then rebuild the state
document, the same discipline the widget layer records in ``sections.py``: no
handler keeps its own copy of a setting. The web version gets one property for
free — the hits list is read from ``controller.last_hits`` on every response, so
there is no retained widget to forget to refresh, which is the bug class that
motivated that rule.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from textwrap import indent
from typing import Any
from urllib.parse import parse_qs

from rpfr_visualizer.gui import payloads
from rpfr_visualizer.gui.render import qc_fragment, render_fragment, viewer_document
from rpfr_visualizer.gui.session import Busy, ExplorerSession, NotReady

#: Static assets, whitelisted by name. Never a path join from user input, so
#: traversal is impossible rather than merely guarded against.
STATIC_ASSETS: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}

#: The vendored 3Dmol bundle, served patched (see ``bundled_3dmol_js``).
BUNDLE_ASSET = "3Dmol-min.js"

#: Results cap. There is no cursor in the index and no relevance ranking, so
#: deeper paging would order by internal id — progress-shaped, but not progress.
MAX_HITS = 1000
DEFAULT_HITS = 200


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)

    #: An open binary file to send instead of :attr:`body`. A CSV export can be
    #: hundreds of megabytes, and buffering it as one bytes object costs that
    #: much resident memory on top of the rows it was built from. The file is
    #: read and closed by the transport; ``content_length`` must match its size
    #: so the response still carries Content-Length and the browser can show a
    #: real progress bar rather than an indeterminate spinner.
    stream: Any | None = None
    content_length: int | None = None


def json_response(payload: Any, status: int = 200) -> Response:
    body = json.dumps(payloads.jsonable(payload), allow_nan=False).encode("utf-8")
    return Response(status, body)


def error_response(kind: str, message: str, status: int) -> Response:
    """One envelope for every failure, so the client has a single error path."""
    return json_response({"error": {"kind": kind, "message": message}}, status)


Handler = Callable[[ExplorerSession, dict[str, str], Any], Response]
_ROUTES: dict[tuple[str, str], Handler] = {}


def route(method: str, path: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        _ROUTES[(method, path)] = handler
        return handler

    return register


def dispatch(
    session: ExplorerSession,
    method: str,
    path: str,
    query: str = "",
    body: bytes | None = None,
) -> Response:
    """Route one request. The only entry point the transport layer needs."""
    if path.startswith("/static/"):
        return _static(path[len("/static/") :])

    handler = _ROUTES.get((method, path))
    if handler is None:
        return error_response("not_found", f"No route for {method} {path}.", 404)

    params = {k: v[0] for k, v in parse_qs(query).items()}
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        return error_response("user", f"Malformed request body: {exc}", 400)
    if not isinstance(payload, dict):
        return error_response("user", "Request body must be a JSON object.", 400)

    try:
        return handler(session, params, payload)
    except Busy as exc:
        return error_response("busy", str(exc), 409)
    except NotReady as exc:
        return error_response("not_ready", str(exc), 503)
    except (ValueError, KeyError) as exc:
        # The controller's own messages are the documentation — "Load molecule A
        # first", "Choose a functional group or enter smarts:/substring text" —
        # so they go through verbatim rather than being paraphrased here.
        return error_response("user", str(exc), 400)


# --------------------------------------------------------------------- assets


def _static(name: str) -> Response:
    if name == BUNDLE_ASSET:
        from rpfr_visualizer.viz.py3dmol_local import bundled_3dmol_js

        # The *patched* source: served raw, the original UMD footer sets only
        # root["3Dmol"] and the viewer's promise check would never pass.
        source = bundled_3dmol_js().encode("utf-8")
        return Response(
            200,
            source,
            "text/javascript; charset=utf-8",
            {"Cache-Control": "public, max-age=31536000, immutable"},
        )

    content_type = STATIC_ASSETS.get(name)
    if content_type is None:
        return error_response("not_found", f"No such asset: {name}", 404)
    try:
        data = (resources.files("rpfr_visualizer.gui") / "static" / name).read_bytes()
    except (FileNotFoundError, OSError):
        return error_response("not_found", f"Asset is missing from the install: {name}", 404)
    # Read fresh per request, so in an editable install a change to app.js is
    # live on the next reload with no restart. no-store keeps a heuristic cache
    # from quietly undoing that; the 3Dmol bundle above is immutable and keeps
    # its long max-age.
    return Response(200, data, content_type, {"Cache-Control": "no-store"})


# ---------------------------------------------------------------------- shell


@route("GET", "/")
def _index(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """One shell for both the setup and explorer states."""
    return _static("index.html")


@route("GET", "/healthz")
def _healthz(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Liveness. Never takes the lock, so it answers during a slow query."""
    return json_response({"ok": True, "ready": session.ready})


@route("GET", "/favicon.ico")
def _favicon(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Answer the request every browser makes on its own.

    There is no icon to serve, but a 404 here is a console error on every single
    page load — noise in the one place a person looks when something is actually
    wrong. 204 says "nothing to show" without raising an alarm.
    """
    return Response(204, b"", "image/x-icon")


# --------------------------------------------------------------- client errors

#: Retained client reports. Nothing prunes this but the cap.
MAX_CLIENT_ERRORS = 200

#: Each field is truncated to this before being stored or printed.
MAX_CLIENT_FIELD = 2000

_CLIENT_FIELDS = ("kind", "message", "source", "line", "column", "stack")


@route("POST", "/api/_client_error")
def _post_client_error(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Dev only: a browser-side error, reported so it reaches the terminal.

    An uncaught TypeError in app.js otherwise dies in a devtools console nobody
    is reading, which is why front-end faults were invisible to everything
    except a person who happened to be looking.

    404 unless the server was started with --dev, so a normal install exposes no
    endpoint that accepts client-supplied text at all.
    """
    if not session.dev:
        return error_response("not_found", "No route for POST /api/_client_error.", 404)

    payload = body if isinstance(body, dict) else {}
    record = {name: str(payload.get(name, ""))[:MAX_CLIENT_FIELD] for name in _CLIENT_FIELDS}
    session.client_errors.append(record)
    del session.client_errors[:-MAX_CLIENT_ERRORS]

    where = f"{record['source']}:{record['line']}" if record["source"] else "?"
    print(f"[client {record['kind']}] {record['message']} ({where})", file=sys.stderr, flush=True)
    if record["stack"]:
        print(indent(record["stack"], "    "), file=sys.stderr, flush=True)
    return json_response({"ok": True})


@route("GET", "/api/_client_error")
def _get_client_errors(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Dev only: what the browser has reported, so it can be polled not tailed."""
    if not session.dev:
        return error_response("not_found", "No route for GET /api/_client_error.", 404)
    return json_response({"errors": list(session.client_errors)})


# ---------------------------------------------------------------------- setup


def _setup_payload(session: ExplorerSession) -> dict[str, Any]:
    found = session.resolution.discovery
    return {
        "ready": session.ready,
        "problems": list(session.problems),
        "warnings": list(session.warnings),
        "resolved": {
            "data_dir": str(session.resolution.data_dir or ""),
            "index_path": str(session.resolution.index_path or ""),
            "drive_root": str(session.resolution.drive_root or ""),
            "dataset": session.resolution.dataset or "",
        },
        "discovered": {
            "data_dirs": list(found.data_dirs),
            "drive_root": found.drive_root,
            "index_env": found.index_env,
            "data_env": found.data_env,
            "index_candidates": [
                {
                    "path": c.path,
                    "size_mb": c.size_mb,
                    "molecules": c.molecules,
                    "error": c.error,
                }
                for c in found.index_candidates
            ],
        },
    }


@route("GET", "/api/setup")
def _get_setup(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    return json_response(_setup_payload(session))


@route("POST", "/api/setup")
def _post_setup(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """The only route that may construct a controller."""
    from rpfr_visualizer.gui.config import GuiConfig, resolve

    resolution = resolve(
        data_dir=body.get("data_dir") or None,
        index_path=body.get("index_path") or None,
        drive_root=body.get("drive_root") or None,
        dataset=body.get("dataset") or None,
    )
    if not resolution.ready:
        session.resolution = resolution
        session.problems = list(resolution.problems) or [
            "Could not find any QM9s-RPFR data at that location."
        ]
        return json_response(_setup_payload(session))

    try:
        session.open(resolution)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        session.problems = [str(exc)]
        return json_response(_setup_payload(session))

    session.problems = list(resolution.problems)
    with session.use():
        session.touch(payloads.state_payload)

    if body.get("remember"):
        GuiConfig(
            data_dir=str(resolution.data_dir) if resolution.data_dir else None,
            index_path=str(resolution.index_path) if resolution.index_path else None,
            drive_root=str(resolution.drive_root) if resolution.drive_root else None,
            dataset=resolution.dataset,
        ).save()

    return json_response(_setup_payload(session))


# ---------------------------------------------------------------------- state


@route("GET", "/api/state")
def _get_state(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Served from the snapshot, so it answers while the lock is held."""
    if not session.ready:
        return error_response("not_ready", "No data is open yet.", 503)
    return json_response(session.snapshot())


@route("POST", "/api/state")
def _post_state(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Apply a partial settings change and return the whole new state.

    The response *is* the authoritative document and the client re-applies all
    of it, so a coercion or clamp the server performed is immediately visible
    rather than leaving a control disagreeing with the analysis.

    Renders nothing: that keeps a slow query from firing on every checkbox.
    """
    with session.use() as controller:
        for name, value in body.items():
            setattr(
                controller.state,
                name,
                payloads.coerce_setting(name, value, grid=controller.temperatures),
            )
        session.touch(payloads.state_payload)
    return json_response(session.snapshot())


@route("POST", "/api/dataset")
def _post_dataset(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Switch datasets. Separate from settings because it resets the selection."""
    name = body.get("name")
    if not name:
        raise ValueError("Which dataset? Pass a name.")
    with session.use() as controller:
        controller.use_dataset(str(name))
        session.touch(payloads.state_payload)
    return json_response(session.snapshot())


# ------------------------------------------------------------------ selection


@route("POST", "/api/load")
def _post_load(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    key = body.get("key")
    if not key:
        raise ValueError("Which molecule? Pass a key.")
    slot = str(body.get("slot", "a"))
    with session.use() as controller:
        controller.load(str(key), slot=slot)
        session.touch(payloads.state_payload)
    return json_response(session.snapshot())


@route("POST", "/api/clear")
def _post_clear(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    with session.use() as controller:
        controller.clear(str(body.get("slot", "b")))
        session.touch(payloads.state_payload)
    return json_response(session.snapshot())


@route("GET", "/api/examples")
def _get_examples(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """A few molecules that will actually render.

    Taken from the index, not from ``provider.list_keys()``: that call opens the
    HDF5 and sorts every group name with no caching, and in a dataset with no
    structures it would hand back a molecule whose 2D pane comes back empty.
    """
    with session.use() as controller:
        index = controller.index
        if index is None:
            keys = controller.provider.list_keys()[:5]
            return json_response({"keys": list(keys), "renderable": False})
        keys = [
            entry.key
            for entry in index.iter_entries(datasets=(controller.dataset,))
            if entry.smiles
        ][:5]
    return json_response({"keys": keys, "renderable": True})


# --------------------------------------------------------------------- search


def _limit_from(body: Any) -> int:
    requested = body.get("limit")
    if requested in (None, ""):
        return DEFAULT_HITS
    return max(1, min(int(requested), MAX_HITS))


@route("POST", "/api/search")
def _post_search(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    query = str(body.get("query", "")).strip()
    if not query:
        raise ValueError("Type something to search for.")
    limit = _limit_from(body)
    with session.use() as controller:
        started = time.perf_counter()
        hits = controller.search(query, limit=limit)
        elapsed = (time.perf_counter() - started) * 1000
        session.touch(payloads.state_payload)
    return json_response(payloads.hit_payload(hits, limit=limit, elapsed_ms=elapsed))


@route("POST", "/api/search/fg")
def _post_search_fg(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Find molecules containing a group, or matching a pattern.

    The group chooses the *molecules*; the element lock chooses which *sites* get
    ranked. That is the workflow the notebook had to explain in prose.
    """
    limit = _limit_from(body)
    with session.use() as controller:
        started = time.perf_counter()
        pairs = controller.search_functional_group(
            preset_name=body.get("preset") or None,
            fg_text=str(body.get("text", "")),
            limit=limit,
        )
        elapsed = (time.perf_counter() - started) * 1000
        entries = controller.index.get_many(pairs) if controller.index else {}
        hits = [
            _as_hit(entries[pair], body.get("preset") or body.get("text", ""))
            for pair in pairs
            if pair in entries
        ]
        controller.load_hits_into_pool()
        session.touch(payloads.state_payload)
    return json_response(payloads.hit_payload(hits, limit=limit, elapsed_ms=elapsed))


def _as_hit(entry: Any, query: str) -> Any:
    from rpfr_visualizer.io.search.models import MoleculeSearchHit

    return MoleculeSearchHit(entry, str(query), "functional_group")


@route("GET", "/api/fg/presets")
def _get_presets(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """The curated groups with per-dataset counts.

    Every preset is listed, including those with no members here — absence is
    information, and it is what turns an empty search into "none in this
    dataset" rather than "the search is broken".
    """
    with session.use() as controller:
        if controller.index is None:
            from rpfr_visualizer.analysis.functional_groups import primary_fg_presets

            rows = [
                {"name": p.name, "description": p.description, "count": None}
                for p in primary_fg_presets()
            ]
        else:
            rows = [
                {"name": name, "description": description, "count": count}
                for name, description, count in controller.functional_group_counts()
            ]
    return json_response({"presets": rows})


@lru_cache(maxsize=64)
def _fg_svg(name: str) -> str | None:
    """Draw one curated group, cached for the life of the process.

    The catalog is eighteen static patterns, so rendering each at most once is
    the whole caching story. Returns None when RDKit is absent or the pattern
    has no sensible depiction.
    """
    from rpfr_visualizer.analysis.functional_groups import preset_by_name
    from rpfr_visualizer.viz.depict import depict_fg_pattern

    try:
        preset = preset_by_name(name)
    except KeyError:
        return None
    svg, _mode = depict_fg_pattern(
        smarts=preset.smarts,
        smiles=preset.smiles_substring,
        example_smiles=preset.example_smiles,
    )
    return svg


def _depictions_available() -> bool:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


@route("GET", "/api/fg/reference")
def _get_fg_reference(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """The full catalog: what each group is, its SMARTS, and how many are here.

    Richer than ``/api/fg/presets``, which only feeds the sidebar picker. This
    one carries the pattern strings and example molecules the Guide tab shows.
    """
    from rpfr_visualizer.analysis.functional_groups import primary_fg_presets

    counts: dict[str, int] = {}
    if session.ready:
        with session.use() as controller:
            if controller.index is not None:
                counts = {
                    name: count for name, _desc, count in controller.functional_group_counts()
                }

    groups = [
        {
            "name": preset.name,
            "description": preset.description,
            "element": preset.element,
            "smarts": preset.smarts,
            "example_smiles": preset.example_smiles,
            "count": counts.get(preset.name) if counts else None,
        }
        for preset in primary_fg_presets()
    ]
    return json_response({"groups": groups, "depictions_available": _depictions_available()})


@route("GET", "/api/fg/depict")
def _get_fg_depict(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """One group as an SVG image.

    Deliberately not JSON, and deliberately not behind ``session.use()``: this
    is pure computation over constants, and eighteen image requests on tab-open
    would otherwise queue behind the data lock -- which does not wait, so most
    of them would come back "still working".

    The client loads these with <img src=...>, which is why nothing here has to
    be inserted as markup and the one-innerHTML rule is untouched.
    """
    del session, body
    name = (params.get("name") or "").strip()
    if not name:
        return error_response("user", "Which group? Pass ?name=<preset>.", 400)
    svg = _fg_svg(name)
    if svg is None:
        return error_response("not_found", f"No depiction available for {name!r}.", 404)
    return Response(
        200,
        svg.encode("utf-8"),
        "image/svg+xml; charset=utf-8",
        # Static for the life of the install, and the client asks for eighteen
        # of them every time the tab opens.
        {"Cache-Control": "max-age=86400"},
    )


# ------------------------------------------------------------------ the stage


#: How long a read-only route waits for the controller before giving up.
#:
#: Sized against what it is actually waiting for. The two slowest holders are a
#: cold render (~440 ms, RDKit's first depiction) and the 3D document (~270 ms);
#: warm, both are under 2 ms. Two seconds covers either with room to spare and
#: still bounds a chain of reads well inside the browser's own patience, so a
#: genuinely stuck session answers rather than hanging the page.
#:
#: Routes that change state do not use it -- see ExplorerSession.use.
READ_WAIT_SECONDS = 2.0


@route("GET", "/view/render")
def _get_render(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Draw the current state.

    Takes no parameters on purpose: settings arrive through ``/api/state``, and a
    render route that accepted overrides would be a second source of truth.
    """
    with session.use(wait=READ_WAIT_SECONDS) as controller:
        fragment = render_fragment(controller)
        # Build the 3D documents here, while the lock is already held and the
        # analysis is already done. The iframe that will ask for them is
        # created from this very response, so this is the state it should show
        # -- and it costs less than letting /viewer/3d recompute it, which
        # meant a second full render_payload() per render.
        session.cache_viewers(
            {
                pane["slot"]: viewer_document(controller, pane["slot"])
                for pane in fragment["panes"]
                if pane.get("viewer_url")
            }
        )
        return json_response(fragment)


@route("GET", "/viewer/3d")
def _get_viewer(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """The document for one 3D pane.

    Served from what the render cached, without the lock. The browser fetches
    this itself, for an ``<iframe src>``, so it is not covered by the client's
    own request serialisation -- and refusing it as busy put a raw JSON error
    inside the pane. The fallback covers a direct hit on this URL before any
    render has run.
    """
    slot = params.get("slot", "a")
    document = session.viewer_document(slot)
    if document is None:
        with session.use(wait=READ_WAIT_SECONDS) as controller:
            document = viewer_document(controller, slot)
    return Response(200, document.encode("utf-8"), "text/html; charset=utf-8")


@route("GET", "/view/ranks")
def _get_ranks(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    with session.use(wait=READ_WAIT_SECONDS) as controller:
        controller.rank()
        return json_response(payloads.rank_payload(controller))


@route("GET", "/view/derived")
def _get_derived(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Temperature curves, mass-law values and the clumping matrix.

    Returns 200 with per-section ``available: false`` rather than an error when
    a quantity cannot be produced. Clumping and mass law exist only on the
    stored temperature grid, and typing an arbitrary temperature is normal use
    of this app -- surfacing that as a 400 would read as a bug.
    """
    with session.use(wait=READ_WAIT_SECONDS) as controller:
        return json_response(controller.derived_payload())


@route("GET", "/view/qc")
def _get_qc(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    structured = params.get("structured") in {"1", "true", "yes"}
    with session.use(wait=READ_WAIT_SECONDS) as controller:
        return json_response(qc_fragment(controller, structured=structured))


# ----------------------------------------------------------------- statistics


@route("GET", "/api/statistics")
def _get_statistics(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    from rpfr_visualizer.gui.statistics import statistics_payload

    with session.use() as controller:
        return json_response(statistics_payload(controller))


# ------------------------------------------------------------------- download


def _download_scope(params: dict[str, str], body: Any) -> str:
    scope = str(params.get("scope") or body.get("scope") or "molecule").strip()
    if scope == "search":
        scope = "results"
    if scope not in {"molecule", "results"}:
        raise ValueError("Download either this molecule or the current search results.")
    return scope


@route("GET", "/api/download")
def _get_download_preview(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """What a CSV would contain, without opening every molecule."""
    from rpfr_visualizer.gui.download import download_preview

    with session.use() as controller:
        return json_response(download_preview(controller, _download_scope(params, body)))


@route("GET", "/api/download.csv")
def _get_download_csv(session: ExplorerSession, params: dict[str, str], body: Any) -> Response:
    """Every molecule the current query matched, as a CSV attachment.

    The lock is held for the whole export. h5py is not thread-safe and the
    controller is mutable, so releasing it mid-read would risk two requests
    sharing one file handle -- and ``session.use()`` does not queue, so a
    concurrent request gets an immediate "still working" rather than hanging.
    Bounded memory is handled by spooling to disk instead.
    """
    from rpfr_visualizer.gui.download import open_download_csv

    with session.use() as controller:
        filename, handle, size = open_download_csv(controller, _download_scope(params, body))
    return Response(
        200,
        b"",
        "text/csv; charset=utf-8",
        {"Content-Disposition": f'attachment; filename="{filename}"'},
        stream=handle,
        content_length=size,
    )


__all__ = [
    "BUNDLE_ASSET",
    "DEFAULT_HITS",
    "MAX_HITS",
    "STATIC_ASSETS",
    "Response",
    "dispatch",
    "error_response",
    "json_response",
]
