"""The explorer's ipywidgets front end.

The only module in the explorer that imports ipywidgets. It assembles the
section builders, wires them to a controller, and renders what the controller
returns — it holds no analysis logic of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpfr_visualizer.io.providers import DatasetCatalog, HDF5Provider
from rpfr_visualizer.viz._deps import require_widgets
from rpfr_visualizer.viz.depict import depict_enrichment, format_site_value_table
from rpfr_visualizer.viz.explorer.controller import ExplorerController
from rpfr_visualizer.viz.explorer.sections import (
    dataset_section,
    display_section,
    functional_group_section,
    quantity_section,
    ranking_section,
    search_section,
)
from rpfr_visualizer.viz.py3dmol_local import display_view

_HEADINGS = {
    "find": "1 · Find molecules",
    "groups": "2 · Functional groups",
    "quantity": "3 · What to compute",
    "render": "4 · View and rank",
}


class ExplorerView:
    """Widget layer over an :class:`ExplorerController`."""

    def __init__(self, controller: ExplorerController) -> None:
        self.controller = controller
        self._widgets, self._display, _clear, self._html = require_widgets("the molecule explorer")
        self._status: Any = None
        self._main_out: Any = None
        self._rank_out: Any = None
        self._qc_out: Any = None

    # ------------------------------------------------------------------ build

    def build(self) -> Any:
        """Assemble the widget tree. Does not display it."""
        widgets = self._widgets
        ctrl = self.controller

        self._status = widgets.HTML()
        self._main_out = widgets.Output()
        self._rank_out = widgets.Output()
        self._qc_out = widgets.Output()

        sections = {
            "dataset": dataset_section(ctrl),
            "search": search_section(ctrl),
            "groups": functional_group_section(ctrl),
            "quantity": quantity_section(ctrl),
            "display": display_section(ctrl),
            "ranking": ranking_section(ctrl),
        }
        for _box, bind in sections.values():
            bind(ctrl, self.refresh_status)

        render_btn = widgets.Button(description="Render", button_style="primary")
        rank_btn = widgets.Button(description="Rank sites")
        qc_btn = widgets.Button(description="QC")
        render_btn.on_click(self._on_render)
        rank_btn.on_click(self._on_rank)
        qc_btn.on_click(self._on_qc)

        tree = widgets.VBox(
            [
                self._status,
                sections["dataset"][0],
                self._heading(_HEADINGS["find"]),
                sections["search"][0],
                self._heading(_HEADINGS["groups"]),
                sections["groups"][0],
                self._heading(_HEADINGS["quantity"]),
                sections["quantity"][0],
                sections["display"][0],
                self._heading(_HEADINGS["render"]),
                widgets.HBox([render_btn, rank_btn, qc_btn]),
                sections["ranking"][0],
                self._main_out,
                self._rank_out,
                self._qc_out,
            ]
        )
        self.refresh_status()
        return tree

    def show(self) -> None:
        """Display the panel."""
        self._display(self.build())

    def _heading(self, text: str) -> Any:
        return self._widgets.HTML(
            f"<h4 style='margin:12px 0 4px 0;border-bottom:1px solid #ddd'>{text}</h4>"
        )

    def refresh_status(self) -> None:
        """Redraw the one-line status bar."""
        if self._status is None:
            return
        ctrl = self.controller
        state = ctrl.state
        parts = [f"<b>Dataset</b>: {state.dataset}"]
        if state.mol_a is not None:
            flags = ctrl.qc_flags()
            verdict = "pass" if flags and flags.qc_pass else "check"
            parts.append(
                f"<b>A</b>: {state.mol_a.key} (n={state.mol_a.molecule.n_atoms}, QC={verdict})"
            )
        else:
            parts.append("<b>A</b>: (none)")
        parts.append(f"<b>B</b>: {state.mol_b.key}" if state.mol_b else "<b>B</b>: (none)")
        if ctrl.match_pool:
            parts.append(f"<b>pool</b>: {len(ctrl.match_pool)}")
        parts.append(f"<b>{state.describe_quantity()}</b>")
        self._status.value = " &nbsp;|&nbsp; ".join(parts)

    # --------------------------------------------------------------- handlers

    def _on_render(self, _btn=None) -> None:
        widgets = self._widgets
        ctrl = self.controller
        with self._main_out:
            self._main_out.clear_output(wait=True)
            try:
                payload = ctrl.render_payload()
            except Exception as exc:  # noqa: BLE001 — surfaced to the user
                print(f"Cannot render: {exc}")
                return

            for warning in payload.warnings:
                print(f"note: {warning}")

            rows = [self._molecule_row(payload.view_a, payload.label_a, slot="a")]
            if payload.view_b is not None:
                rows.append(self._molecule_row(payload.view_b, payload.label_b, slot="b"))
            self._display(widgets.VBox(rows))

            if payload.caption:
                print(payload.caption)
            if payload.caption_b:
                print(payload.caption_b)
            if payload.note_b:
                print(f"note: {payload.note_b}")
            self._print_site_table(payload)
        self.refresh_status()

    def _molecule_row(self, view, label: str, *, slot: str) -> Any:
        """One molecule: 3D viewer beside its classic 2D depiction."""
        widgets = self._widgets
        ctrl = self.controller
        state = ctrl.state
        loaded = state.mol_a if slot == "a" else state.mol_b
        highlight = None
        if slot == "b":
            highlight = int(state.ref_atom_index_b)
        elif state.reference_mode == "self":
            highlight = int(state.anchor_atom_index)

        columns = []
        if state.use_3d:
            out = widgets.Output()
            with out:
                try:
                    viewer = view.build_viewer(
                        element=state.element,
                        width=420,
                        height=340,
                        show_labels=state.show_labels,
                        show_hydrogens=state.show_hydrogens,
                        highlight_atom_index=highlight,
                    )
                    display_view(viewer)
                except Exception as exc:  # noqa: BLE001 — 2D pane still works
                    print(f"3D unavailable ({exc})")
            columns.append(widgets.VBox([widgets.HTML("<b>3D</b>"), out]))

        if state.use_rdkit_2d and loaded is not None:
            out = widgets.Output()
            with out:
                depiction = depict_enrichment(
                    loaded.molecule,
                    view.values,
                    element_filter=state.element,
                    size=(400, 320),
                    highlight_atom_index=highlight,
                    show_labels=state.show_labels,
                )
                if depiction.svg:
                    self._display(self._html(depiction.svg))
                if depiction.note:
                    print(depiction.note)
            columns.append(widgets.VBox([widgets.HTML("<b>classic 2D</b>"), out]))

        return widgets.VBox([widgets.HTML(f"<b>{label}</b>"), widgets.HBox(columns)])

    def _print_site_table(self, payload) -> None:
        state = self.controller.state
        if state.mol_a is not None and payload.view_a is not None:
            print(f"\n=== A {state.mol_a.label} ===")
            print(
                format_site_value_table(
                    state.mol_a.molecule,
                    payload.view_a.values,
                    element_filter=state.element,
                    show_hydrogens=state.show_hydrogens,
                    quantity_kind=payload.view_a.display_kind,
                )
            )

    def _on_rank(self, _btn=None) -> None:
        with self._rank_out:
            self._rank_out.clear_output(wait=True)
            ranks = self.controller.rank()
            if not ranks:
                print("Nothing to rank — load a molecule or fill the ranking pool.")
                return
            source = "ranking pool" if self.controller.match_pool else "loaded molecules"
            print(f"Top {len(ranks)} site(s) from the {source}:")
            self._display(self.controller.ranks_frame())

    def _on_qc(self, _btn=None) -> None:
        with self._qc_out:
            self._qc_out.clear_output(wait=True)
            print(self.controller.qc_text())


def open_explorer(
    source: HDF5Provider | DatasetCatalog | Path | str,
    *,
    dataset: str | None = None,
    index: Any = None,
    max_stats_molecules: int | None = 200,
) -> ExplorerView:
    """Open the molecule explorer.

    *source* may be a catalog, a provider, or a path to an HDF5 file::

        open_explorer(catalog, dataset="qm9s").show()
    """
    if isinstance(source, (str, Path)):
        provider: HDF5Provider | DatasetCatalog = HDF5Provider(source)
    else:
        provider = source
    controller = ExplorerController(
        provider,
        dataset=dataset,
        index=index,
        max_stats_molecules=max_stats_molecules,
    )
    return ExplorerView(controller)


def open_full_dataset_explorer(
    dataset: str = "qm9s",
    *,
    drive_root: Path | str | None = None,
    index_path: Path | str | None = None,
    max_stats_molecules: int | None = 200,
) -> ExplorerView:
    """Open the explorer on the **full** datasets from the external drive.

    The full files are tens of gigabytes, so nothing is read up front: molecules
    are loaded from HDF5 one at a time and all searching is answered by the
    SQLite index.

    Raises
    ------
    FileNotFoundError
        If the drive is not mounted, or no search index is found. Building an
        index over the full data takes tens of minutes, so it is never started
        implicitly — run ``rpfr-visualizer index build --drive ...`` first.
    """
    from rpfr_visualizer.viz.explorer.factory import controller_for_drive

    return ExplorerView(
        controller_for_drive(
            dataset,
            drive_root=drive_root,
            index_path=index_path,
            max_stats_molecules=max_stats_molecules,
        )
    )


__all__ = ["ExplorerView", "open_explorer", "open_full_dataset_explorer"]
