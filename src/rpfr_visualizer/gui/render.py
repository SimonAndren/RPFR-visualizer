"""Producing the figures, and drawing the trust boundary around them.

Everything here returns JSON-shaped dicts, but two fields carry **markup** and
that distinction is a safety rule rather than a style choice:

* ``svg`` and the viewer document are **trusted renderer output** — RDKit and
  py3Dmol build them from our own coordinate arrays and formatted numbers — so a
  client may insert them as markup.
* Every other string is **data-derived** (keys, SMILES, formulas, fixed-width
  table text) and must be inserted with ``textContent``. Molecule synonyms can
  come from PubChem, so this is a real boundary, not a hypothetical one. Nothing
  here concatenates data into markup, which is what makes that safe.

Anything a client must *decide* from goes back through
:mod:`rpfr_visualizer.gui.payloads` instead.
"""

from __future__ import annotations

from html import escape
from typing import Any

from rpfr_visualizer.viz.depict import depict_enrichment, format_site_value_table
from rpfr_visualizer.viz.explorer import ExplorerController

#: Figure sizes. The stage is laid out for these, so they are not per-request.
VIEW_WIDTH = 420
VIEW_HEIGHT = 340
DEPICT_SIZE = (400, 320)

#: Where the patched 3Dmol bundle is served from.
BUNDLE_URL = "/static/3Dmol-min.js"


def render_fragment(controller: ExplorerController) -> dict[str, Any]:
    """Everything the stage shows for the current state.

    Returns a dict rather than one blob so the client can place the panes
    independently and swap only what changed.
    """
    payload = controller.render_payload()
    state = controller.state

    panes = [_molecule_pane(controller, payload, slot="a")]
    if payload.view_b is not None:
        panes.append(_molecule_pane(controller, payload, slot="b"))

    return {
        "panes": panes,
        "caption": payload.caption,
        "caption_b": payload.caption_b,
        "note_b": payload.note_b,
        # The provenance line already says whether the temperature was read or
        # fitted, in the same words; listing it again as a notice put the one
        # fact on screen twice.
        "warnings": [w for w in payload.warnings if w != controller.temperature_note()],
        "provenance": provenance_line(controller),
        "quantity": state.kind.value,
        "temperature": float(state.temperature),
        "element": state.element,
    }


#: Said on the 2D pane when the stored SMILES no longer describes the geometry.
#:
#: The structure is still drawn -- refusing to draw it would hide a molecule
#: that is otherwise fine -- but the drawing comes from the SMILES while every
#: number comes from the coordinates, and for these molecules those are two
#: different things. The 3D view is built from the coordinates themselves, so
#: it stays authoritative and carries no such caveat.
GEOMETRY_MISMATCH_NOTE = (
    "The stored SMILES does not match this geometry, so the 2D structure may be "
    "wrong. The 3D view uses the real coordinates."
)


def geometry_mismatch_note(molecule: Any) -> str:
    """The caveat for this molecule's 2D depiction, or an empty string."""
    return "" if getattr(molecule, "geometry_match", True) else GEOMETRY_MISMATCH_NOTE


def _join_notes(*notes: str) -> str:
    """Join the notes that apply, keeping order and dropping the empty ones."""
    return " ".join(note for note in notes if note)


def _molecule_pane(controller: ExplorerController, payload: Any, *, slot: str) -> dict[str, Any]:
    """One molecule: a 3D viewer URL, a 2D SVG, its value table, and any caveat."""
    state = controller.state
    loaded = state.mol_a if slot == "a" else state.mol_b
    view = payload.view_a if slot == "a" else payload.view_b

    highlight: int | None = None
    if slot == "b" and state.reference_mode == "mol_b":
        highlight = int(state.ref_atom_index_b)
    elif slot == "a" and state.reference_mode == "self":
        highlight = int(state.anchor_atom_index)

    pane: dict[str, Any] = {
        "slot": slot,
        "label": payload.label_a if slot == "a" else payload.label_b,
        "viewer_url": None,
        "svg": None,
        "depiction_note": "",
        "table": "",
    }

    if state.use_3d:
        # A URL, not inline HTML: the iframe fetches it, so the 537 KB bundle is
        # cached once instead of re-parsed per render.
        pane["viewer_url"] = f"/viewer/3d?slot={slot}&r={controller.state.temperature:g}"

    if loaded is None or view is None:
        return pane

    if state.use_rdkit_2d:
        depiction = depict_enrichment(
            loaded.molecule,
            view.values,
            element_filter=state.element,
            size=DEPICT_SIZE,
            highlight_atom_index=highlight,
            show_labels=state.show_labels,
        )
        pane["svg"] = depiction.svg  # trusted: RDKit output
        pane["depiction_note"] = _join_notes(
            geometry_mismatch_note(loaded.molecule), depiction.note
        )

    pane["table"] = format_site_value_table(
        loaded.molecule,
        view.values,
        element_filter=state.element,
        show_hydrogens=state.show_hydrogens,
        quantity_kind=view.display_kind,
        highlight_atom_index=highlight,
    )
    return pane


def provenance_line(controller: ExplorerController) -> dict[str, str]:
    """Whether the temperature was read or interpolated — stated every time.

    The widget layer only spoke up when the value was off-grid, so the good case
    was silent and a user never learned the distinction existed.
    """
    state = controller.state
    grid = controller.temperatures
    if state.temperature_is_on_grid(grid):
        return {"level": "ok", "text": controller.temperature_note()}

    low, high = min(grid), max(grid)
    if not low <= state.temperature <= high:
        return {
            "level": "error",
            "text": (
                f"{state.temperature:g} K is outside the stored range ({low:g} to "
                f"{high:g} K), so β is extrapolated and may be unreliable."
            ),
        }
    return {"level": "warn", "text": controller.temperature_note()}


def viewer_document(controller: ExplorerController, slot: str) -> str:
    """A whole HTML document for the 3D iframe.

    Trusted output: py3Dmol builds it from coordinates and formatted numbers.
    """
    from rpfr_visualizer.viz.py3dmol_local import view_to_document

    payload = controller.render_payload()
    view = payload.view_a if slot == "a" else payload.view_b
    if view is None:
        return _placeholder_document("Nothing loaded in this slot.")

    state = controller.state
    highlight: int | None = None
    if slot == "b" and state.reference_mode == "mol_b":
        highlight = int(state.ref_atom_index_b)
    elif slot == "a" and state.reference_mode == "self":
        highlight = int(state.anchor_atom_index)

    viewer = view.build_viewer(
        element=state.element,
        width=VIEW_WIDTH,
        height=VIEW_HEIGHT,
        show_labels=state.show_labels,
        show_hydrogens=state.show_hydrogens,
        highlight_atom_index=highlight,
        bundle_src=BUNDLE_URL,
    )
    return view_to_document(viewer)


def _placeholder_document(message: str) -> str:
    """A minimal document for the iframe when there is no viewer to show."""
    return (
        "<!doctype html><meta charset='utf-8'>"
        '<div style="font:13px/1.5 system-ui,sans-serif;color:#8a8a84;padding:12px">'
        f"{escape(message)}</div>"
    )


def qc_fragment(controller: ExplorerController, *, structured: bool = False) -> dict[str, Any]:
    """The QC report as text plus a pass/fail verdict.

    The verdict never travels as colour alone: the client shows a dot *and* the
    words, so it reads the same for a colour-blind user.
    """
    flags = controller.qc_flags()
    return {
        "text": controller.qc_text(structured=structured),
        "verdict": None if flags is None else ("pass" if flags.qc_pass else "check"),
        "structured": structured,
    }


__all__ = [
    "BUNDLE_URL",
    "DEPICT_SIZE",
    "VIEW_HEIGHT",
    "VIEW_WIDTH",
    "provenance_line",
    "qc_fragment",
    "render_fragment",
    "viewer_document",
]
