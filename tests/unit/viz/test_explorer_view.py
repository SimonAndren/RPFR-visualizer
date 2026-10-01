"""Tests for the explorer's widget layer.

The old panel's ``show()`` was a ~590-line method with no coverage at all,
because it built and displayed in one step. The view now builds a tree without
displaying, so the wiring can be exercised headlessly: widgets are constructed,
handlers are invoked with synthetic change events, and the resulting controller
state is checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.io.providers import DatasetCatalog, HDF5Provider
from rpfr_visualizer.viz.explorer import ExplorerController, ExplorerView, open_explorer

pytest.importorskip("ipywidgets")
pytest.importorskip("rdkit")

import ipywidgets as widgets  # noqa: E402


@pytest.fixture
def view(synthetic_h5_path: Path) -> ExplorerView:
    controller = ExplorerController(HDF5Provider(synthetic_h5_path), max_stats_molecules=2)
    return ExplorerView(controller)


def walk(widget):
    """Yield every widget in a tree."""
    for child in getattr(widget, "children", []):
        yield child
        yield from walk(child)


def find(tree, kind, description: str | None = None):
    """First widget of *kind* whose description matches, if given."""
    for widget in walk(tree):
        if isinstance(widget, kind) and (
            description is None or getattr(widget, "description", None) == description
        ):
            return widget
    raise AssertionError(f"no {kind.__name__} with description={description!r}")


def test_build_produces_a_tree_without_displaying(view: ExplorerView) -> None:
    tree = view.build()
    assert isinstance(tree, widgets.VBox)
    assert len(list(walk(tree))) > 20


def test_headings_cover_the_workflow(view: ExplorerView) -> None:
    tree = view.build()
    headings = " ".join(
        w.value for w in walk(tree) if isinstance(w, widgets.HTML) and "<h4" in str(w.value)
    )
    for expected in ("Find molecules", "Functional groups", "What to compute", "View and rank"):
        assert expected in headings


def test_functional_groups_are_all_offered(view: ExplorerView) -> None:
    from rpfr_visualizer.analysis.functional_groups import primary_fg_presets

    tree = view.build()
    group = find(tree, widgets.Dropdown, "Group")
    # every curated preset, plus the "(none)" entry
    assert len(group.options) == len(primary_fg_presets()) + 1
    labels = [label for label, _value in group.options]
    assert any(label.startswith("alcohol") for label in labels)
    assert any(label.startswith("carboxylic_acid") for label in labels)


def test_element_change_updates_state(view: ExplorerView) -> None:
    tree = view.build()
    element = find(tree, widgets.Dropdown, "Element")
    element.value = "N"
    assert view.controller.state.element == "N"
    element.value = None
    assert view.controller.state.element is None


def test_quantity_change_toggles_the_reference_control(view: ExplorerView) -> None:
    tree = view.build()
    kind = find(tree, widgets.Dropdown, "Quantity")
    reference = find(tree, widgets.Dropdown, "Reference")

    kind.value = AnalysisKind.BETA
    assert view.controller.state.kind == AnalysisKind.BETA
    # Beta is absolute, so a reference frame would be meaningless.
    assert reference.disabled

    kind.value = AnalysisKind.ALPHA
    assert not reference.disabled


def test_temperature_and_custom_temperature(view: ExplorerView) -> None:
    tree = view.build()
    grid = view.controller.temperatures
    temperature = find(tree, widgets.Dropdown, "T")
    custom = find(tree, widgets.FloatText, "or T =")

    temperature.value = float(grid[1])
    assert view.controller.state.temperature == float(grid[1])

    off_grid = float(grid[1]) + 3.0
    custom.value = off_grid
    assert view.controller.state.temperature == off_grid
    assert view.controller.warnings()


def test_display_toggles_update_state(view: ExplorerView) -> None:
    tree = view.build()
    for description, field in (
        ("Show H", "show_hydrogens"),
        ("Label atoms", "show_labels"),
        ("3D view", "use_3d"),
        ("Classic 2D", "use_rdkit_2d"),
    ):
        box = find(tree, widgets.Checkbox, description)
        box.value = False
        assert getattr(view.controller.state, field) is False


def test_ranking_controls_update_state(view: ExplorerView) -> None:
    tree = view.build()
    find(tree, widgets.BoundedIntText, "Top N").value = 7
    assert view.controller.state.top_n == 7
    find(tree, widgets.Checkbox, "Heavy atoms only").value = False
    assert view.controller.state.heavy_only is False


def test_status_bar_tracks_selection(view: ExplorerView) -> None:
    view.build()
    assert "(none)" in view._status.value

    view.controller.load(view.controller.provider.list_keys()[0])
    view.refresh_status()
    assert "QC=" in view._status.value


# Outside a live kernel widgets.Output does not capture, so handler output is
# checked on stdout rather than on the Output widget.


def test_render_handler_reports_missing_molecule(view: ExplorerView, capsys) -> None:
    view.build()
    view._on_render()  # must report, not raise
    assert "Cannot render" in capsys.readouterr().out


def test_render_handler_prints_site_table(view: ExplorerView, capsys) -> None:
    view.build()
    key = view.controller.provider.list_keys()[0]
    view.controller.load(key)
    view.controller.state.use_3d = False  # skip the 537 KB 3Dmol bundle in tests
    view._on_render()
    out = capsys.readouterr().out
    assert f"=== A {view.controller.dataset}:{key} ===" in out
    assert "site" in out and "elem" in out


def test_render_handler_surfaces_off_grid_warning(view: ExplorerView, capsys) -> None:
    view.build()
    view.controller.load(view.controller.provider.list_keys()[0])
    view.controller.state.use_3d = False
    view.controller.state.temperature = float(view.controller.temperatures[0]) + 4.0
    view._on_render()
    assert "fitted" in capsys.readouterr().out


def test_rank_handler_reports_when_empty_and_when_ranked(view: ExplorerView, capsys) -> None:
    view.build()
    view._on_rank()
    assert "Nothing to rank" in capsys.readouterr().out

    view.controller.load(view.controller.provider.list_keys()[0])
    view._on_rank()
    assert "Top" in capsys.readouterr().out
    assert view.controller.last_ranks


def test_qc_handler_runs(view: ExplorerView, capsys) -> None:
    view.build()
    view._on_qc()
    assert "Load molecule A first" in capsys.readouterr().out

    view.controller.load(view.controller.provider.list_keys()[0])
    view._on_qc()
    assert "verdict" in capsys.readouterr().out


def _dataset_row(tree):
    """The HBox holding the dataset dropdown."""
    for box in walk(tree):
        children = getattr(box, "children", ())
        if any(isinstance(c, widgets.Dropdown) and c.description == "Dataset" for c in children):
            return box
    raise AssertionError("dataset row not found")


def test_dataset_picker_hidden_for_a_single_dataset(view: ExplorerView) -> None:
    tree = view.build()
    dataset = find(tree, widgets.Dropdown, "Dataset")
    assert dataset.value == view.controller.dataset
    # Nothing to switch between, so the row is collapsed rather than shown empty.
    assert _dataset_row(tree).layout.display == "none"


def test_dataset_picker_switches_datasets(synthetic_h5_path: Path) -> None:
    catalog = DatasetCatalog()
    catalog.register("first", synthetic_h5_path)
    catalog.register("second", synthetic_h5_path)
    view = ExplorerView(ExplorerController(catalog, max_stats_molecules=2))
    tree = view.build()

    view.controller.load(view.controller.provider.list_keys()[0])
    dataset = find(tree, widgets.Dropdown, "Dataset")
    assert list(dataset.options) == ["first", "second"]
    assert _dataset_row(tree).layout.display != "none"  # visible with a choice

    dataset.value = "second"
    assert view.controller.dataset == "second"
    assert view.controller.state.mol_a is None  # selection does not cross datasets


def test_open_explorer_accepts_a_path(synthetic_h5_path: Path) -> None:
    view = open_explorer(synthetic_h5_path, max_stats_molecules=2)
    assert isinstance(view, ExplorerView)
    assert isinstance(view.build(), widgets.VBox)


def test_open_explorer_accepts_a_catalog(synthetic_h5_path: Path) -> None:
    catalog = DatasetCatalog()
    catalog.register("only", synthetic_h5_path)
    view = open_explorer(catalog, dataset="only", max_stats_molecules=2)
    assert view.controller.dataset == "only"
