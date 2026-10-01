#!/usr/bin/env python3
"""Fail when a tracked notebook carries execution outputs.

Notebook outputs bloat the repository (an embedded 3Dmol viewer is ~1.8 MB per
cell) and make diffs unreviewable. ``nbstripout`` handles this via pre-commit,
but hooks are opt-in, so this script is the enforcing check for CI.

Usage::

    python scripts/check_notebook_outputs.py [notebook ...]

With no arguments every git-tracked ``.ipynb`` is checked. Pass ``--fix`` to
strip the offending notebooks in place instead of failing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def tracked_notebooks() -> list[Path]:
    """Return every git-tracked notebook path."""
    result = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in result.stdout.split() if line]


def dirty_cells(notebook: dict) -> list[int]:
    """Return indices of cells carrying outputs or an execution count."""
    dirty = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("outputs") or cell.get("execution_count") is not None:
            dirty.append(i)
    return dirty


def strip(notebook: dict) -> dict:
    """Clear outputs, execution counts, and volatile metadata in place."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    notebook.get("metadata", {}).pop("widgets", None)
    return notebook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="notebooks to check")
    parser.add_argument("--fix", action="store_true", help="strip in place instead of failing")
    args = parser.parse_args(argv)

    paths = args.paths or tracked_notebooks()
    failures: list[tuple[Path, list[int]]] = []

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        dirty = dirty_cells(notebook)
        if not dirty:
            continue
        if args.fix:
            path.write_text(
                json.dumps(strip(notebook), indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"stripped {path.relative_to(REPO_ROOT)} ({len(dirty)} cell(s))")
        else:
            failures.append((path, dirty))

    if failures:
        print("Notebooks must be committed without outputs:", file=sys.stderr)
        for path, dirty in failures:
            print(f"  {path.relative_to(REPO_ROOT)}: cells {dirty}", file=sys.stderr)
        print(
            "\nStrip them with:\n"
            "  python scripts/check_notebook_outputs.py --fix\n"
            "or install the hooks once with:\n"
            "  uv run pre-commit install",
            file=sys.stderr,
        )
        return 1

    print(f"{len(paths)} notebook(s) clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
