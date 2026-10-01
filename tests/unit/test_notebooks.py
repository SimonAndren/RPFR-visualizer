"""The shipped notebooks, checked without running them.

A notebook is the first thing a new user opens, and a broken one is a bad
first impression that no other test catches: nothing imports it, so a renamed
function or a stale keyword argument sits there until someone hits it by hand.

These are static checks -- valid JSON, parseable Python, no committed outputs.
Executing them needs the data files and belongs in the integration suite.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "docs").glob("*.ipynb"))

#: Magics and shell escapes are valid in a cell and not valid Python, so a
#: cell using them is skipped rather than reported as a syntax error.
_NOT_PYTHON = ("%", "!", "?")


def _code_cells(path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(_NOT_PYTHON) for line in source.splitlines()):
            continue
        out.append((index, source))
    return out


def test_there_are_notebooks_to_check() -> None:
    """Guards the glob: an empty list would make every test below vacuous."""
    assert NOTEBOOKS, "no notebooks found under docs/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_the_notebook_is_valid(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    assert notebook["cells"], "a notebook with no cells"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_parses(path: Path) -> None:
    for index, source in _code_cells(path):
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"{path.name} cell {index}: {exc}") from exc


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_outputs_are_committed(path: Path) -> None:
    """nbstripout is configured, but only as a hook a fresh clone may not have.

    Committed outputs make every diff a re-render and can carry data from
    whatever file the author happened to have mounted.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    dirty = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    assert not dirty, f"{path.name} has committed output in cells {dirty}"
