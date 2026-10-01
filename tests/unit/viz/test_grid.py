"""Multi-molecule grids.

molecule_grid() only ever touches ``.title`` and ``.build_viewer()`` on what it
is given, so a stand-in exercises the layout logic without building real 3D
viewers -- which is the part that needs py3Dmol and a browser to mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rpfr_visualizer.viz.grid import GridCell, display_grid, molecule_grid


@dataclass
class FakeView:
    """Stands in for EnrichmentView."""

    title: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def build_viewer(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"viewer<{self.title}>"


def test_no_views_is_an_empty_grid_not_an_error() -> None:
    assert molecule_grid([]) == []


def test_each_view_becomes_one_cell() -> None:
    views = [FakeView("a"), FakeView("b"), FakeView("c")]

    cells = molecule_grid(views)  # type: ignore[arg-type]

    assert [c.label for c in cells] == ["a", "b", "c"]
    assert [c.viewer for c in cells] == ["viewer<a>", "viewer<b>", "viewer<c>"]


def test_labels_override_the_view_title() -> None:
    views = [FakeView("from-view"), FakeView("also-from-view")]

    cells = molecule_grid(views, labels=["given"])  # type: ignore[arg-type]

    assert [c.label for c in cells] == ["given", "also-from-view"]


def test_too_many_labels_are_ignored_rather_than_raising() -> None:
    cells = molecule_grid([FakeView("x")], labels=["a", "b", "c"])  # type: ignore[arg-type]
    assert [c.label for c in cells] == ["a"]


def test_size_and_label_options_reach_the_viewer() -> None:
    view = FakeView("m")

    molecule_grid([view], width=123, height=45, show_labels=True)  # type: ignore[arg-type]

    assert view.calls == [{"width": 123, "height": 45, "show_labels": True}]


def test_defaults_are_passed_explicitly() -> None:
    """The viewer should never have to guess: build_viewer is always called with
    all three, so a default change here cannot silently change a rendering."""
    view = FakeView("m")

    molecule_grid([view])  # type: ignore[arg-type]

    assert set(view.calls[0]) == {"width", "height", "show_labels"}


# --- display ---------------------------------------------------------------


def test_display_grid_says_so_when_there_is_nothing_to_show() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")

    display_grid([])  # must not raise


def test_display_grid_lays_out_cells_in_rows() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")
    import rpfr_visualizer.viz.grid as grid_module

    captured: list[Any] = []

    def fake_display_view(viewer: Any) -> None:
        captured.append(viewer)

    original = grid_module.display_view
    grid_module.display_view = fake_display_view  # type: ignore[assignment]
    try:
        cells = [GridCell(label=f"m{i}", viewer=f"v{i}") for i in range(5)]
        display_grid(cells, cols=2)
    finally:
        grid_module.display_view = original  # type: ignore[assignment]

    # Every cell is rendered exactly once, including the odd one that closes the
    # last, short row.
    assert captured == ["v0", "v1", "v2", "v3", "v4"]
