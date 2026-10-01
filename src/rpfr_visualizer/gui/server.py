"""A small local HTTP server, and the ``launch()`` the CLI calls.

Deliberately the standard library. What this app needs is static files, a handful
of endpoints, and *serialized* access to the data — and a framework's async model
would manufacture the concurrency the lock then has to take away again. See
:mod:`rpfr_visualizer.gui.session`.

Threading is still on, so a stylesheet is never stuck behind the slow query it is
meant to be animating.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from rpfr_visualizer.gui.app import dispatch
from rpfr_visualizer.gui.config import load_config, resolve
from rpfr_visualizer.gui.session import ExplorerSession

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: How far to walk when the requested port is taken.
PORT_SEARCH_RANGE = 20

#: Bodies larger than this are refused outright. Nothing this API accepts is big.
MAX_BODY_BYTES = 1 << 20


class ExplorerHTTPServer(ThreadingHTTPServer):
    """One server, one session.

    The handler used to hold these as class attributes, which are process-global:
    a second server in the same process — a test suite, an embedded launch —
    silently retargeted the first one's data, and the symptom was a request
    answered from the wrong dataset rather than an error.
    """

    session: ExplorerSession
    allowed_hosts: frozenset[str] = frozenset()
    verbose: bool = False


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so connections are reused; every response carries an exact
    # Content-Length, which is what makes that safe.
    protocol_version = "HTTP/1.1"

    @property
    def _owner(self) -> ExplorerHTTPServer:
        """The server this request arrived on, and so the session to use."""
        return cast(ExplorerHTTPServer, self.server)

    @property
    def session(self) -> ExplorerSession:
        return self._owner.session

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return self._owner.allowed_hosts

    @property
    def verbose(self) -> bool:
        return self._owner.verbose

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet by default: forty asset lines scrolling past reads as a fault."""
        if self.verbose:
            super().log_message(fmt, *args)

    def _host_allowed(self) -> bool:
        """Reject a Host header we do not recognise.

        This is the DNS-rebinding defence: without it, a page on the open web
        could resolve its own hostname to 127.0.0.1 and talk to this server.
        """
        host = (self.headers.get("Host") or "").strip().lower()
        return host in self.allowed_hosts

    def _respond(self, response: Any) -> None:
        stream = getattr(response, "stream", None)
        length = (
            int(response.content_length)
            if stream is not None and response.content_length is not None
            else len(response.body)
        )
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(length))
        for name, value in response.headers.items():
            self.send_header(name, value)
        self.end_headers()
        if stream is not None:
            try:
                if self.command != "HEAD":
                    shutil.copyfileobj(stream, self.wfile)
            finally:
                stream.close()
            return
        if self.command != "HEAD":
            self.wfile.write(response.body)

    def _handle(self, method: str) -> None:
        from rpfr_visualizer.gui.app import error_response

        if not self._host_allowed():
            self._respond(
                error_response("not_found", "This server only answers on localhost.", 403)
            )
            return

        path, _, query = self.path.partition("?")
        body = b""
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._respond(error_response("user", "Request body is too large.", 413))
            return
        if length:
            body = self.rfile.read(length)

        try:
            response = dispatch(self.session, method, path, query, body)
        except Exception as exc:  # noqa: BLE001 — never take the server down
            import traceback

            traceback.print_exc()
            response = error_response("internal", f"{type(exc).__name__}: {exc}", 500)
        self._respond(response)

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")


def _bind(host: str, port: int) -> ExplorerHTTPServer:
    """Bind the requested port, or the next free one nearby.

    Not a fallback to an arbitrary free port: someone who bookmarked the URL
    deserves a predictable one, so this walks a small range and then says so.
    """
    attempts = [port] if port == 0 else range(port, port + PORT_SEARCH_RANGE + 1)
    last: OSError | None = None
    for candidate in attempts:
        try:
            return ExplorerHTTPServer((host, candidate), _Handler)
        except OSError as exc:
            last = exc
    raise SystemExit(
        f"Ports {port}-{port + PORT_SEARCH_RANGE} are all in use on {host} "
        f"({last}). Pass --port to choose another."
    )


def _banner(session: ExplorerSession, url: str, *, open_browser: bool) -> str:
    resolved = session.resolution
    lines = ["rpfr-visualizer explorer"]

    if resolved.data_dir:
        controller = session.controller
        count = len(controller.available_datasets()) if controller else 0
        lines.append(f"  data      {resolved.data_dir}  ({count} datasets)")
    if resolved.drive_root:
        lines.append(f"  drive     {resolved.drive_root}")
    # Read the index off the controller, not off the resolution: which index is
    # used is decided by coverage at open time, not by the path resolution.
    index = session.controller._index if session.controller is not None else None
    if index is not None:
        path = getattr(index.backend, "path", resolved.index_path or "")
        lines.append(f"  index     {path}  ({len(index):,} molecules)")
    else:
        lines.append("  index     none — searching is unavailable")
    if session.controller is not None:
        lines.append(f"  dataset   {session.controller.dataset}")
    lines.append(f"  config    {_display_path(_config_display())}")

    for problem in session.problems:
        lines.append(f"  ! {problem}")
    for warning in session.warnings:
        lines.append(f"  ! {warning}")

    lines.append("")
    opening = "   (opening your browser…)" if open_browser else ""
    lines.append(f"  Open {url}{opening}")
    lines.append("  Press Ctrl-C to stop.")
    return "\n".join(lines)


def _config_display() -> Path:
    from rpfr_visualizer.gui.config import config_path

    return config_path()


def _display_path(path: Path) -> str:
    """Shorten a home-relative path, the way a shell prompt would."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _prepare_session(
    *,
    data_dir: Path | str | None = None,
    index_path: Path | str | None = None,
    drive_root: Path | str | None = None,
    dataset: str | None = None,
    host: str = DEFAULT_HOST,
    max_stats_molecules: int | None = None,
    reset_config: bool = False,
    dev: bool = False,
) -> ExplorerSession:
    """Build a session the way ``launch`` and ``start_explorer`` both need."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(
            f"Refusing to bind {host!r}. The explorer has no authentication and reads local "
            "files, so it only serves loopback."
        )

    from rpfr_visualizer.gui.config import config_path

    if reset_config:
        config_path().unlink(missing_ok=True)

    config, problems = load_config()
    resolution = resolve(
        data_dir=data_dir,
        index_path=index_path,
        drive_root=drive_root,
        dataset=dataset,
        max_stats_molecules=max_stats_molecules,
        config=config,
        config_problems=problems,
    )

    session = ExplorerSession(resolution)
    if resolution.ready:
        try:
            session.open()
        except (FileNotFoundError, ValueError, KeyError) as exc:
            # Not fatal: the setup screen is a better place to explain this than
            # a traceback in a terminal the user may not be watching.
            session.problems.append(str(exc))
    session.dev = dev
    return session


def open_in_chrome(url: str) -> str:
    """Open *url*, preferring Google Chrome. Returns how it was opened.

    A shipping tutorial should land in Chrome when it is installed. The system
    default is the fallback, not the first try.
    """
    if sys.platform == "darwin":
        chrome_app = Path("/Applications/Google Chrome.app")
        if chrome_app.exists():
            subprocess.Popen(["open", "-a", "Google Chrome", url])  # noqa: S603
            return "Google Chrome"

    for name in ("google-chrome", "chromium", "chrome", "chromium-browser"):
        binary = shutil.which(name)
        if binary:
            subprocess.Popen([binary, url])  # noqa: S603
            return name

    for controller in ("chrome", "google-chrome", "chromium"):
        try:
            webbrowser.get(controller).open(url, new=2)
            return controller
        except webbrowser.Error:
            continue

    webbrowser.open(url, new=2)
    return "default browser"


def launch(
    *,
    data_dir: Path | str | None = None,
    index_path: Path | str | None = None,
    drive_root: Path | str | None = None,
    dataset: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    max_stats_molecules: int | None = None,
    reset_config: bool = False,
    verbose: bool = False,
    dev: bool = False,
) -> int:
    """Start the explorer and serve until interrupted.

    *dev* enables the client-error routes and turns logging on, so a browser
    fault reaches this terminal instead of a devtools console nobody is reading.
    """
    session = _prepare_session(
        data_dir=data_dir,
        index_path=index_path,
        drive_root=drive_root,
        dataset=dataset,
        host=host,
        max_stats_molecules=max_stats_molecules,
        reset_config=reset_config,
        dev=dev,
    )

    server = _bind(host, port)  # bound and listening once this returns
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"

    server.session = session
    server.verbose = verbose or dev
    server.allowed_hosts = frozenset(
        {
            f"127.0.0.1:{actual_port}",
            f"localhost:{actual_port}",
            f"[::1]:{actual_port}",
        }
    )

    # flush: stdout is block-buffered when redirected, so without this the
    # banner would not appear until the process exits.
    print(_banner(session, url, open_browser=open_browser), flush=True)
    if open_browser:
        # No race: TCPServer.__init__ already called listen(), so a connection
        # made before serve_forever() waits in the backlog.
        open_in_chrome(url)

    try:
        server.serve_forever()  # pragma: no cover — the blocking loop
    except KeyboardInterrupt:  # pragma: no cover
        print("\nStopped.")
    finally:
        server.shutdown()
        server.server_close()
    return 0


def start_explorer(
    *,
    data_dir: Path | str | None = None,
    index_path: Path | str | None = None,
    drive_root: Path | str | None = None,
    dataset: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    max_stats_molecules: int | None = None,
    reset_config: bool = False,
) -> tuple[ExplorerHTTPServer, str]:
    """Start the explorer on a background thread and open Chrome.

    This is the notebook entry point: ``launch()`` blocks the kernel, so a
    tutorial cell would never return. The returned server can be shut down
    from a later cell.
    """
    session = _prepare_session(
        data_dir=data_dir,
        index_path=index_path,
        drive_root=drive_root,
        dataset=dataset,
        host=host,
        max_stats_molecules=max_stats_molecules,
        reset_config=reset_config,
    )
    server, url = serve_in_thread(session, host=host, port=port)
    print(_banner(session, url, open_browser=open_browser), flush=True)
    if open_browser:
        opened = open_in_chrome(url)
        print(f"  opened in {opened}", flush=True)
    return server, url


def serve_in_thread(
    session: ExplorerSession,
    *,
    host: str = DEFAULT_HOST,
    port: int = 0,
    poll_interval: float = 0.05,
) -> tuple[ExplorerHTTPServer, str]:
    """Start a server on a background thread. For tests and embedding.

    *poll_interval* is how often ``serve_forever`` checks whether it has been
    asked to stop, so it is also the cost of ``shutdown()``. The stdlib default
    of 0.5 s is half a second of dead time per server in a test run.
    """
    server = _bind(host, port)
    actual_port = server.server_address[1]
    server.session = session
    server.verbose = False
    server.allowed_hosts = frozenset(
        {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}", f"[::1]:{actual_port}"}
    )
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": poll_interval}, daemon=True
    )
    thread.start()
    return server, f"http://{host}:{actual_port}"


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ExplorerHTTPServer",
    "launch",
    "open_in_chrome",
    "serve_in_thread",
    "start_explorer",
]
