"""Widget groups for the explorer panel.

Each builder returns ``(box, bind)``: the widgets to lay out, and a function
that wires them to a controller. Handlers only ever write to
``controller.state`` and then ask for a refresh — no section keeps its own copy
of a setting, which is what stops the old panel's closure-captured state from
coming back.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rpfr_visualizer.analysis.functional_groups import primary_fg_presets
from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.viz._deps import require_widgets
from rpfr_visualizer.viz.explorer.controller import ExplorerController
from rpfr_visualizer.viz.explorer.state import ELEMENTS

Refresh = Callable[[], None]
Section = tuple[Any, Callable[[ExplorerController, Refresh], None]]

_LABEL = {"description_width": "110px"}


def _widgets():
    widgets, _display, _clear, _html = require_widgets("the molecule explorer")
    return widgets


def dataset_section(controller: ExplorerController) -> Section:
    """Dataset picker. Hidden when only one dataset is registered."""
    widgets = _widgets()
    names = controller.available_datasets()
    dropdown = widgets.Dropdown(
        options=list(names),
        value=controller.dataset,
        description="Dataset",
        style=_LABEL,
        layout=widgets.Layout(width="320px"),
    )
    box = widgets.HBox([dropdown])
    box.layout.display = "" if len(names) > 1 else "none"

    def bind(ctrl: ExplorerController, refresh: Refresh) -> None:
        def on_change(change):
            ctrl.use_dataset(change["new"])
            refresh()

        dropdown.observe(on_change, names="value")

    return box, bind


def search_section(controller: ExplorerController) -> Section:
    """Free-text search plus the hit list and load buttons."""
    widgets = _widgets()
    query = widgets.Text(
        placeholder="name, formula, SMILES, key, smarts:… or fg:…",
        description="Search",
        style=_LABEL,
        layout=widgets.Layout(width="520px"),
    )
    search_btn = widgets.Button(description="Search", button_style="primary")
    hits = widgets.SelectMultiple(
        options=[],
        rows=8,
        description="Hits",
        style=_LABEL,
        layout=widgets.Layout(width="720px"),
    )
    load_a_btn = widgets.Button(description="Load as A")
    load_b_btn = widgets.Button(description="Load as B")
    pool_btn = widgets.Button(description="All hits → ranking pool")
    clear_pool_btn = widgets.Button(description="Clear pool")
    output = widgets.Output()

    box = widgets.VBox(
        [
            widgets.HBox([query, search_btn]),
            hits,
            widgets.HBox([load_a_btn, load_b_btn, pool_btn, clear_pool_btn]),
            output,
        ]
    )

    def bind(ctrl: ExplorerController, refresh: Refresh) -> None:
        def show_hits() -> None:
            hits.options = list(zip(ctrl.hit_labels(), ctrl.last_hits, strict=False))

        def on_search(_btn=None):
            with output:
                output.clear_output(wait=True)
                text = query.value.strip()
                if not text:
                    print("Enter a name, formula, SMILES, key, smarts: or fg: query.")
                    return
                try:
                    ctrl.search(text)
                except Exception as exc:  # noqa: BLE001 — surfaced to the user
                    print(f"Search failed: {exc}")
                    return
                show_hits()
                print(f"{len(ctrl.last_hits)} hit(s).")

        def selected_keys() -> list[str]:
            return [key for _ds, key in hits.value]

        def load_into(slot: str):
            def handler(_btn):
                with output:
                    output.clear_output(wait=True)
                    keys = selected_keys()
                    if not keys:
                        print("Select a molecule in the Hits list first.")
                        return
                    ctrl.load(keys[0], slot=slot)
                    print(f"Loaded {slot.upper()} = {ctrl.dataset}:{keys[0]}")
                refresh()

            return handler

        def on_pool(_btn):
            with output:
                output.clear_output(wait=True)
                n = ctrl.load_hits_into_pool()
                print(f"Ranking pool now holds {n} molecule(s).")
            refresh()

        def on_clear_pool(_btn):
            ctrl.clear_pool()
            with output:
                output.clear_output(wait=True)
                print("Ranking pool cleared.")
            refresh()

        search_btn.on_click(on_search)
        load_a_btn.on_click(load_into("a"))
        load_b_btn.on_click(load_into("b"))
        pool_btn.on_click(on_pool)
        clear_pool_btn.on_click(on_clear_pool)
        box._show_hits = show_hits  # let the FG section refresh the same list

    return box, bind


def functional_group_section(controller: ExplorerController) -> Section:
    """Curated functional groups plus free-text substructure search."""
    widgets = _widgets()
    preset = widgets.Dropdown(
        options=[("(none)", None)]
        + [(f"{p.name} — {p.description}", p.name) for p in primary_fg_presets()],
        value=None,
        description="Group",
        style=_LABEL,
        layout=widgets.Layout(width="420px"),
    )
    free_text = widgets.Text(
        placeholder="smarts:[CX3](=O)[OX2H1]  or a SMILES substring",
        description="or pattern",
        style=_LABEL,
        layout=widgets.Layout(width="420px"),
    )
    max_hits = widgets.BoundedIntText(
        value=controller.state.max_fg_hits,
        min=1,
        max=100_000,
        description="Max hits",
        style=_LABEL,
        layout=widgets.Layout(width="200px"),
    )
    search_btn = widgets.Button(description="Find molecules", button_style="primary")
    counts_btn = widgets.Button(description="What's in this dataset?")
    output = widgets.Output()

    box = widgets.VBox(
        [
            widgets.HBox([preset, max_hits]),
            widgets.HBox([free_text, search_btn, counts_btn]),
            output,
        ]
    )

    def bind(ctrl: ExplorerController, refresh: Refresh) -> None:
        def on_search(_btn=None):
            with output:
                output.clear_output(wait=True)
                ctrl.state.max_fg_hits = int(max_hits.value)
                try:
                    hits = ctrl.search_functional_group(
                        preset_name=preset.value, fg_text=free_text.value
                    )
                except Exception as exc:  # noqa: BLE001 — surfaced to the user
                    print(f"Search failed: {exc}")
                    return
                what = preset.value or free_text.value
                print(
                    f"{len(hits)} molecule(s) contain {what}. "
                    "They are in the Hits list above — load one, or send them all "
                    "to the ranking pool."
                )
            refresh()

        def on_counts(_btn):
            with output:
                output.clear_output(wait=True)
                print(f"Functional groups present in {ctrl.dataset}:")
                rows = [row for row in ctrl.functional_group_counts() if row[2]]
                if not rows:
                    print("  (none of the curated groups appear in this dataset)")
                for name, description, n in sorted(rows, key=lambda r: -r[2]):
                    print(f"  {name:18s} {n:7d}   {description}")

        search_btn.on_click(on_search)
        counts_btn.on_click(on_counts)
        preset.observe(lambda change: _sync_preset(ctrl, change), names="value")
        free_text.observe(
            lambda change: setattr(ctrl.state, "fg_text", change["new"]), names="value"
        )

    def _sync_preset(ctrl: ExplorerController, change) -> None:
        ctrl.state.fg_preset = change["new"]

    return box, bind


def quantity_section(controller: ExplorerController) -> Section:
    """What to compute: element, quantity, reference frame, temperature."""
    widgets = _widgets()
    state = controller.state

    element = widgets.Dropdown(
        options=[("all elements", None)] + [(e, e) for e in ELEMENTS],
        value=state.element,
        description="Element",
        style=_LABEL,
        layout=widgets.Layout(width="220px"),
    )
    kind = widgets.Dropdown(
        # Site-level quantities only: clumping and mass-law are pair/array
        # quantities that this per-site view cannot draw.
        options=[
            ("beta (stored)", AnalysisKind.BETA),
            ("alpha", AnalysisKind.ALPHA),
            ("epsilon", AnalysisKind.EPSILON),
            ("ln alpha", AnalysisKind.LN_ALPHA),
        ],
        value=state.kind,
        description="Quantity",
        style=_LABEL,
        layout=widgets.Layout(width="240px"),
    )
    reference = widgets.Dropdown(
        options=[
            ("in-molecule element mean", "element_mean"),
            ("dataset standard", "standard"),
            ("site in molecule B", "mol_b"),
            ("anchor site in A", "self"),
        ],
        value=state.reference_mode if state.reference_mode != "none" else "element_mean",
        description="Reference",
        style=_LABEL,
        layout=widgets.Layout(width="320px"),
    )
    grid = controller.temperatures_for_display
    temperature = widgets.Dropdown(
        options=[(f"{t:g} K", float(t)) for t in grid],
        value=float(state.temperature) if state.temperature in grid else float(grid[0]),
        description="T",
        style=_LABEL,
        layout=widgets.Layout(width="220px"),
    )
    custom_t = widgets.FloatText(
        value=0.0,
        description="or T =",
        style=_LABEL,
        layout=widgets.Layout(width="200px"),
    )
    note = widgets.HTML()

    box = widgets.VBox(
        [
            widgets.HBox([element, kind, reference]),
            widgets.HBox([temperature, custom_t]),
            note,
        ]
    )

    def bind(ctrl: ExplorerController, refresh: Refresh) -> None:
        def update_note() -> None:
            warnings = ctrl.warnings()
            note.value = f"<i style='color:#a15c00'>{' '.join(warnings)}</i>" if warnings else ""

        def on_element(change):
            ctrl.state.element = change["new"]
            refresh()

        def on_kind(change):
            ctrl.state.kind = change["new"]
            # Reference only means something for derived quantities.
            reference.disabled = change["new"] == AnalysisKind.BETA
            refresh()

        def on_reference(change):
            ctrl.state.reference_mode = change["new"]
            refresh()

        def on_temperature(change):
            ctrl.state.temperature = float(change["new"])
            custom_t.value = 0.0
            update_note()
            refresh()

        def on_custom(change):
            value = float(change["new"])
            if value > 0:
                ctrl.state.temperature = value
                update_note()
                refresh()

        element.observe(on_element, names="value")
        kind.observe(on_kind, names="value")
        reference.observe(on_reference, names="value")
        temperature.observe(on_temperature, names="value")
        custom_t.observe(on_custom, names="value")
        reference.disabled = ctrl.state.kind == AnalysisKind.BETA
        update_note()

    return box, bind


def display_section(controller: ExplorerController) -> Section:
    """Toggles for how the molecule is drawn."""
    widgets = _widgets()
    state = controller.state
    show_h = widgets.Checkbox(value=state.show_hydrogens, description="Show H", indent=False)
    # Governs both panes: "3D labels" was misleading once the skeletal
    # depiction started honouring the same flag.
    labels = widgets.Checkbox(value=state.show_labels, description="Label atoms", indent=False)
    three_d = widgets.Checkbox(value=state.use_3d, description="3D view", indent=False)
    two_d = widgets.Checkbox(value=state.use_rdkit_2d, description="Classic 2D", indent=False)
    box = widgets.HBox([show_h, labels, three_d, two_d])

    def bind(ctrl: ExplorerController, refresh: Refresh) -> None:
        pairs = (
            (show_h, "show_hydrogens"),
            (labels, "show_labels"),
            (three_d, "use_3d"),
            (two_d, "use_rdkit_2d"),
        )
        for widget, field in pairs:
            widget.observe(
                lambda change, field=field: setattr(ctrl.state, field, bool(change["new"])),
                names="value",
            )

    return box, bind


def ranking_section(controller: ExplorerController) -> Section:
    """Ranking controls."""
    widgets = _widgets()
    state = controller.state
    top_n = widgets.BoundedIntText(
        value=state.top_n,
        min=1,
        max=500,
        description="Top N",
        style=_LABEL,
        layout=widgets.Layout(width="180px"),
    )
    heavy_only = widgets.Checkbox(
        value=state.heavy_only, description="Heavy atoms only", indent=False
    )
    use_pool = widgets.Checkbox(
        value=state.use_match_pool, description="Rank the pool", indent=False
    )
    box = widgets.HBox([top_n, heavy_only, use_pool])

    def bind(ctrl: ExplorerController, refresh: Refresh) -> None:
        top_n.observe(lambda c: setattr(ctrl.state, "top_n", int(c["new"])), names="value")
        heavy_only.observe(
            lambda c: setattr(ctrl.state, "heavy_only", bool(c["new"])), names="value"
        )
        use_pool.observe(
            lambda c: setattr(ctrl.state, "use_match_pool", bool(c["new"])), names="value"
        )

    return box, bind


__all__ = [
    "Section",
    "dataset_section",
    "display_section",
    "functional_group_section",
    "quantity_section",
    "ranking_section",
    "search_section",
]
