"""The one controller the server owns, and the lock that keeps it consistent.

The server holds a single :class:`~rpfr_visualizer.viz.explorer.ExplorerController`
and the client holds nothing. The controller already *is* the state container,
and it owns things that cannot cross the wire — h5py-backed molecules, a
fingerprint matrix, an open SQLite connection. Mirroring its state in JavaScript
would reintroduce exactly the drift the state dataclass was built to prevent.

Access is serialized. ``h5py`` is not thread-safe, the controller carries mutable
state, and ``SqliteSearchBackend`` opens its connection with
``check_same_thread=False`` — so cross-thread use will *not* raise, which makes
this lock the only thing preventing concurrent use of one connection.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from rpfr_visualizer.gui.config import Resolution
from rpfr_visualizer.viz.explorer import ExplorerController
from rpfr_visualizer.viz.explorer.factory import controller_for_path


class Busy(RuntimeError):
    """Raised when the controller is already in use.

    Deliberately not a queue. A user who double-clicks Search should not wait
    twice, and a browser with six open connections should not be able to stack
    six requests behind one slow query with no way to cancel.

    Read-only routes pass a small :meth:`ExplorerSession.use` *wait* anyway --
    see the note there. The distinction is between refusing to repeat work and
    refusing to take turns.
    """


class ExplorerSession:
    """Owns the controller, the lock, and the snapshot clients read from."""

    def __init__(self, resolution: Resolution | None = None) -> None:
        self.resolution = resolution or Resolution()
        self.controller: ExplorerController | None = None
        self.problems: list[str] = list(self.resolution.problems)
        self.warnings: list[str] = []

        #: Started with --dev. Gates the client-error routes, which are the only
        #: ones that accept text from the browser, so a normal install exposes
        #: no such endpoint at all.
        self.dev = False
        #: What the browser has reported. Bounded by MAX_CLIENT_ERRORS.
        self.client_errors: list[dict[str, str]] = []

        self._lock = threading.Lock()
        self._busy = False
        self._revision = 0
        self._snapshot: dict[str, Any] = {"ready": False, "revision": 0, "busy": False}
        #: Slot -> the 3D document that slot's <iframe> will ask for, built by
        #: the render that handed out its URL. See :meth:`viewer_document`.
        self._viewers: dict[str, str] = {}

    # ------------------------------------------------------------------ status

    @property
    def ready(self) -> bool:
        return self.controller is not None

    @property
    def revision(self) -> int:
        return self._revision

    def snapshot(self) -> dict[str, Any]:
        """The last built state document, readable without taking the lock.

        One writer, plain primitives, and an atomic rebind — so this cannot tear.
        Reading it lock-free is what lets a client see ``busy`` *while* the lock
        is held, which is the whole point.
        """
        return {**self._snapshot, "busy": self._busy}

    def cache_viewers(self, documents: dict[str, str]) -> None:
        """Keep the 3D documents a render just built. Call while holding the lock."""
        self._viewers = dict(documents)

    def viewer_document(self, slot: str) -> str | None:
        """The 3D document for *slot*, or None if no render has built one.

        Read without the lock, and that is the point. The ``<iframe>`` is
        fetched by the browser on its own schedule, not through the client's
        request queue, so it used to arrive alongside whatever the client asked
        for next and get refused as busy -- putting a JSON error where the
        structure should be.

        Serving what the render already computed removes the contention rather
        than waiting it out, and it is also the more correct answer: the
        document now matches the state that handed out its URL, instead of
        being recomputed against whatever the controller holds by the time the
        browser gets round to asking. It saves the duplicated analysis too --
        building it here meant a second full ``render_payload()`` per render.
        """
        return self._viewers.get(slot)

    # -------------------------------------------------------------- locking

    @contextmanager
    def use(self, *, wait: float = 0.0):
        """Take exclusive use of the controller, or raise :class:`Busy`.

        Args:
            wait: Seconds to wait for the lock before giving up. Zero -- the
                default -- fails immediately, which is right for anything that
                changes state: a double-clicked Search should be dropped, not
                run twice.

                Read-only routes pass a few seconds instead. The client
                serialises its own calls, but the browser does not: an
                ``<iframe src>`` is fetched on its own schedule, concurrently
                with whatever the client asks for next. Refusing that instantly
                put a raw JSON error where the 3D structure should be. Waiting
                is still bounded, so this never becomes the queue the class
                docstring rules out.
        """
        acquired = self._lock.acquire(timeout=wait) if wait > 0 else self._lock.acquire(False)
        if not acquired:
            raise Busy("The explorer is still working on the previous request.")
        self._busy = True
        try:
            yield self.require_controller()
        finally:
            self._busy = False
            self._lock.release()

    def require_controller(self) -> ExplorerController:
        if self.controller is None:
            raise NotReady("No data is open yet.")
        return self.controller

    def touch(self, build_snapshot: Callable[[ExplorerController], dict[str, Any]]) -> int:
        """Bump the revision and rebuild the snapshot. Call while holding the lock."""
        self._revision += 1
        controller = self.controller
        payload = build_snapshot(controller) if controller is not None else {"ready": False}
        self._snapshot = {**payload, "revision": self._revision}
        return self._revision

    # ------------------------------------------------------------------- open

    def open(self, resolution: Resolution | None = None) -> None:
        """Build the controller from a resolution. The only way one is created."""
        if resolution is not None:
            self.resolution = resolution
            self.problems = list(resolution.problems)
        resolved = self.resolution

        # One forgiving path: a single .h5, a repo data directory, or a drive
        # root all work, and the index is chosen by whether it matches the data
        # rather than by filename.
        target = resolved.drive_root or resolved.data_dir
        if target is None:
            raise NotReady("No data directory or file to open.")

        controller, notes = controller_for_path(
            target,
            dataset=resolved.dataset,
            index_path=resolved.index_path,
            data_dir=resolved.data_dir,
            max_stats_molecules=resolved.max_stats_molecules,
        )

        self.controller = controller
        self.warnings = list(notes)
        # Build the first snapshot now: /api/state reads it without taking the
        # lock, so a session that never touched it would report itself not ready.
        from rpfr_visualizer.gui.payloads import state_payload

        self.touch(state_payload)

        # controller_for_path already rejects a non-matching index, so anything
        # still here is worth saying out loud rather than checking again.

    def close(self) -> None:
        self.controller = None


class NotReady(RuntimeError):
    """Raised when a request needs a controller and there is none."""


__all__ = ["Busy", "ExplorerSession", "NotReady"]
