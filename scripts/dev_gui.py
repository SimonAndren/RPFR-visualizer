"""Start the explorer the way a developer — or an agent — wants it started.

Unlike ``rpfr-visualizer gui``: never opens a browser, logs every request, and turns
on the dev-only client-error endpoint so a browser-side crash lands in this
terminal instead of a devtools console nobody is reading.

``--fixture`` builds a five-molecule dataset and its search index once and reuses
it, so the app is up in about a second with no dependency on ``data/raw`` or on
the external drive.

Two header lines are printed before anything else and before the log is
redirected, so a caller can read the URL without parsing the banner::

    URL http://127.0.0.1:8765
    LOG /tmp/gui.log

Usage::

    uv run python scripts/dev_gui.py --fixture --port 8765 \\
        --log /tmp/gui.log --pid-file /tmp/gui.pid
    until curl -sf http://127.0.0.1:8765/healthz >/dev/null; do sleep 0.2; done
    ...
    kill "$(cat /tmp/gui.pid)"

A front-end edit needs no restart: the static assets are re-read per request and
served no-store, so reloading the browser is enough.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where --fixture keeps its dataset, so the second run is instant.
FIXTURE_ROOT = REPO_ROOT / "data" / "cache" / "dev-gui"


def ensure_fixture(root: Path) -> Path:
    """Build the throwaway dataset and index under *root*, once.

    Returns the directory to point ``--data`` at. The work is done by
    :func:`rpfr_visualizer.demo.ensure_demo_data`, the same builder behind
    ``rpfr-visualizer gui --demo``; this registers it as ``qm9s`` because that is the
    dataset name the e2e suite asserts against.
    """
    from rpfr_visualizer.demo import ensure_demo_data

    return ensure_demo_data(root, dataset="qm9s").root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dev_gui",
        description="Run the explorer for development: no browser, verbose, dev endpoints on.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="use a generated five-molecule dataset instead of real data",
    )
    parser.add_argument("--data", help="a .h5 file, a data directory, or a drive root")
    parser.add_argument("--index", help="a search index to use")
    parser.add_argument("--dataset", help="which dataset to open")
    parser.add_argument("--port", type=int, default=8765, help="0 picks any free port")
    parser.add_argument("--log", help="redirect stdout and stderr here")
    parser.add_argument("--pid-file", help="write the process id here, for teardown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.fixture and args.data:
        raise SystemExit("Pass either --fixture or --data, not both.")

    # Before importing gui.config, which reads this at import time. Without it a
    # dev session that clicks "Use this" would rewrite the real remembered
    # settings — the config module offers this override for exactly this case.
    scratch = FIXTURE_ROOT if args.fixture else REPO_ROOT / "data" / "cache" / "dev-gui"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["RPFR_VISUALIZER_GUI_CONFIG"] = str(scratch / "gui.json")

    data = args.data
    if args.fixture:
        data = str(ensure_fixture(FIXTURE_ROOT))

    if args.pid_file:
        Path(args.pid_file).write_text(f"{os.getpid()}\n", encoding="utf-8")

    from rpfr_visualizer.gui.server import DEFAULT_HOST, _bind

    # Bind here rather than inside launch() so the URL can be printed before the
    # log is redirected; launch() would otherwise only reveal it in the banner.
    probe = _bind(DEFAULT_HOST, args.port)
    port = probe.server_address[1]
    probe.server_close()

    print(f"URL http://{DEFAULT_HOST}:{port}", flush=True)
    if args.log:
        print(f"LOG {args.log}", flush=True)

    if args.log:
        stream = open(args.log, "w", buffering=1, encoding="utf-8")  # noqa: SIM115 — lives on
        sys.stdout = stream
        sys.stderr = stream

    from rpfr_visualizer.gui.server import launch

    return launch(
        data_dir=data,
        index_path=args.index,
        dataset=args.dataset,
        port=port,
        open_browser=False,
        dev=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
