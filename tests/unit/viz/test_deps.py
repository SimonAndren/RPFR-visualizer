"""The optional-dependency guards.

These exist so a missing extra produces one actionable sentence at call time
rather than an ImportError at import time. The message is the feature, so it is
worth asserting on.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import pytest

from rpfr_visualizer.viz import _deps


@pytest.mark.parametrize(
    ("require", "blocked", "named"),
    [
        (_deps.require_matplotlib, "matplotlib.pyplot", "matplotlib"),
        (_deps.require_py3dmol, "py3Dmol", "py3Dmol"),
        (_deps.require_rdkit, "rdkit", "RDKit"),
        (_deps.require_rdkit_draw, "rdkit", "RDKit"),
        (_deps.require_widgets, "ipywidgets", "ipywidgets"),
    ],
)
def test_a_missing_package_names_itself_the_feature_and_the_fix(
    monkeypatch: pytest.MonkeyPatch,
    require: Callable[..., Any],
    blocked: str,
    named: str,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked or name.startswith(f"{blocked}."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        require("drawing a molecule")

    message = str(excinfo.value)
    assert named in message
    assert "drawing a molecule" in message
    assert "uv sync --extra viz" in message


def test_the_guard_chains_the_original_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """`raise ... from exc` keeps the real cause reachable for debugging."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "py3Dmol":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        _deps.require_py3dmol()

    assert isinstance(excinfo.value.__cause__, ImportError)
    assert "boom" in str(excinfo.value.__cause__)


def test_each_guard_has_a_default_feature_name() -> None:
    """Called with no argument the message must still read as a sentence, so
    every guard needs a sensible default rather than an empty string."""
    for require in (
        _deps.require_matplotlib,
        _deps.require_py3dmol,
        _deps.require_rdkit,
        _deps.require_rdkit_draw,
        _deps.require_widgets,
    ):
        default = require.__defaults__
        assert default and isinstance(default[0], str) and default[0].strip()


def test_present_packages_are_returned() -> None:
    """With the viz extra installed the guards are transparent."""
    pytest.importorskip("rdkit")
    chem = _deps.require_rdkit()
    assert hasattr(chem, "MolFromSmiles")

    chem2, draw = _deps.require_rdkit_draw()
    assert chem2 is chem
    assert hasattr(draw, "rdMolDraw2D")
