"""Turning controller objects into JSON-safe dicts.

Knows nothing about HTTP and produces no HTML. Everything here must survive
``json.dumps`` — which is less obvious than it sounds, since the analysis path
deals in NumPy scalars that are not JSON-serialisable.

Molecule-derived text (keys, SMILES, formulas, and especially synonyms, which
can come from PubChem) travels as JSON precisely so the client can insert it
with ``textContent``. It is never concatenated into markup here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import fields
from typing import Any

from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.analysis.temperature import (
    DEFAULT_METHOD,
    INTERPOLATION_METHODS,
    parse_method,
)
from rpfr_visualizer.viz.colors import quantity_symbol
from rpfr_visualizer.viz.depict import site_anchor_options
from rpfr_visualizer.viz.explorer import ExplorerController, ExplorerState
from rpfr_visualizer.viz.explorer.state import ELEMENTS

#: Fields a client may write. Derived from the dataclass rather than listed, so
#: it cannot drift when a field is added.
#:
#: ``dataset`` routes through its own endpoint because switching datasets also
#: resets the selection; ``mol_a``/``mol_b`` hold whole molecules and are set by
#: loading, not by assignment.
_NOT_WRITABLE = {"dataset", "mol_a", "mol_b"}

WRITABLE_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(ExplorerState) if f.name not in _NOT_WRITABLE
)

#: Quantities the per-site view can draw. Clumping and mass-law are pair/array
#: quantities that a per-atom colour cannot represent.
QUANTITIES: tuple[tuple[str, str], ...] = (
    (AnalysisKind.BETA.value, "β — partition function ratio"),
    (AnalysisKind.ALPHA.value, "α — fractionation factor"),
    (AnalysisKind.LN_ALPHA.value, "1000 ln α (‰)"),
)

#: Reference modes offered, in plain language. ``mol_b`` is deliberately absent:
#: it needs a second molecule *and* an atom picker inside it, and cross-molecule
#: alpha is the easiest result to misread. Still reachable from Python.
REFERENCE_MODES: tuple[tuple[str, str], ...] = (
    ("element_mean", "the average {element} site in this molecule"),
    ("standard", "the dataset average for {element}"),
    ("self", "a site I choose"),
)


def jsonable(value: Any) -> Any:
    """Coerce NumPy scalars and non-finite floats into JSON-safe values.

    ``json.dumps`` accepts NaN and writes a bare ``NaN`` token, which is not
    valid JSON and which ``JSON.parse`` rejects. A site with no value must
    arrive as ``null`` so the client can render an em dash rather than a zero.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):  # NumPy scalar
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def slot_payload(controller: ExplorerController, slot: str) -> dict[str, Any] | None:
    """A loaded molecule projected for display. Never leaks the molecule itself."""
    loaded = controller.state.mol_a if slot == "a" else controller.state.mol_b
    if loaded is None:
        return None
    molecule = loaded.molecule
    flags = controller.qc_flags() if slot == "a" else None
    return {
        "dataset": loaded.dataset,
        "key": loaded.key,
        "label": loaded.label,
        "n_atoms": int(molecule.n_atoms),
        "smiles": getattr(molecule, "smiles", "") or "",
        "qc_pass": None if flags is None else bool(flags.qc_pass),
    }


def anchor_options(controller: ExplorerController, slot: str) -> list[dict[str, Any]]:
    """Choices for the reference-site picker.

    Wired here because the widget layer never had this control at all: choosing
    the anchor reference mode silently anchored to atom 0 with nothing on screen
    saying so.
    """
    loaded = controller.state.mol_a if slot == "a" else controller.state.mol_b
    if loaded is None:
        return []
    options = site_anchor_options(
        loaded.molecule, element=controller.state.element, heavy_only=True
    )
    return [{"label": label, "value": int(index)} for label, index in options]


def subsample_label(controller: ExplorerController) -> str:
    """ "1% subsample" when the open file is one, else an empty string.

    A search that finds nothing in a 1% file is not evidence the molecule is
    absent from QM9s, and nothing on screen used to say which one was open.
    """
    fraction = getattr(controller.provider, "subsample_fraction", None)
    if not fraction or not 0 < fraction < 1:
        return ""
    percent = fraction * 100
    shown = f"{percent:g}" if percent >= 1 else f"{percent:.1g}"
    return f"{shown}% subsample"


def state_payload(controller: ExplorerController) -> dict[str, Any]:
    """Everything a client needs to draw every control, plus why.

    The response to a settings change *is* this document, and the client
    re-applies all of it — so a clamp or coercion the server performed is
    immediately visible instead of leaving a control disagreeing with the
    analysis.
    """
    state = controller.state
    settings = {name: jsonable(getattr(state, name)) for name in sorted(WRITABLE_FIELDS)}
    settings["kind"] = state.kind.value

    grid = controller.temperatures_for_display
    return {
        "ready": True,
        "dataset": controller.dataset,
        "datasets": list(controller.available_datasets()),
        "subsample": subsample_label(controller),
        "temperatures": [float(t) for t in grid],
        "on_grid": state.temperature_is_on_grid(controller.temperatures),
        "elements": list(ELEMENTS),
        "quantities": [{"value": v, "label": label} for v, label in QUANTITIES],
        "reference_modes": [
            {"value": v, "label": label.format(element=state.element or "element")}
            for v, label in REFERENCE_MODES
        ],
        "needs_reference": state.needs_reference(),
        "beta_methods": [{"value": v, "label": label} for v, label in INTERPOLATION_METHODS],
        "settings": settings,
        "slots": {"a": slot_payload(controller, "a"), "b": slot_payload(controller, "b")},
        "anchor_options": {
            "a": anchor_options(controller, "a"),
            "b": anchor_options(controller, "b"),
        },
        "hits": {"count": len(controller.last_hits)},
        "warnings": list(controller.warnings()),
        "status": controller.status_summary(),
    }


def hit_payload(hits: list, *, limit: int | None, elapsed_ms: float) -> dict[str, Any]:
    """Search results, with an honest account of what was left out.

    There is no pagination: the index has a ``limit`` but no cursor, and results
    come back in internal id order with no relevance ranking. Paging deeper into
    an arbitrary order would feel like progress without being progress, so the
    contract is a cap plus a truthful label.
    """
    rows = []
    for hit in hits:
        entry = hit.entry
        rows.append(
            {
                "dataset": entry.dataset,
                "key": entry.key,
                "formula": entry.formula or "",
                "smiles": entry.smiles or "",
                "names": list(entry.names or ()),
                "match_type": hit.match_type,
                "why": match_reason(hit.match_type),
            }
        )
    return {
        "hits": rows,
        "shown": len(rows),
        "limit": limit,
        "capped": limit is not None and len(rows) >= limit,
        "elapsed_ms": round(elapsed_ms, 1),
    }


#: Plain-language rendering of a hit's ``match_type``, so the results list can
#: say *why* a molecule came back.
_MATCH_REASONS: dict[str, str] = {
    "key": "matched ID",
    "key_substring": "ID contains this",
    "formula": "matched formula",
    "canonical_smiles": "same structure",
    "smiles_substring": "contains this structure",
    "substructure": "contains this structure",
    "inchikey": "matched InChIKey",
    "smarts": "contains this pattern",
    "functional_group": "contains this group",
}


def match_reason(match_type: str) -> str:
    if match_type.startswith("name"):
        return "name starts with this" if match_type.endswith("prefix") else "matched name"
    return _MATCH_REASONS.get(match_type, match_type.replace("_", " "))


def rank_payload(controller: ExplorerController) -> dict[str, Any]:
    """The site ranking, as rows rather than a rendered table."""
    ranks = controller.last_ranks
    rows = [
        {
            "dataset": rank.dataset,
            "key": rank.key,
            "atom_index": int(rank.atom_index),
            "element": rank.element,
            "value": jsonable(rank.value),
            "r1": (rank.local_smiles[0] if rank.local_smiles else ""),
            "molecule_smiles": rank.molecule_smiles or "",
        }
        for rank in ranks
    ]
    state = controller.state
    return {
        "rows": rows,
        "quantity": state.kind.value,
        "quantity_symbol": quantity_symbol(state.kind.value),
        "temperature": float(state.temperature),
        "element": state.element,
        "scope": "results" if controller.match_pool else "molecule",
        "pool_size": len(controller.match_pool),
    }


def coerce_setting(name: str, value: Any, *, grid: Sequence[float] | None = None) -> Any:
    """Convert one incoming setting to the type its field declares.

    Args:
        name: The setting being written.
        value: What the client sent.
        grid: The open dataset's temperatures, where a setting needs to be
            judged against them rather than against a package constant. The
            stored range moved from 420 K to 500 K between schema versions.

    Raises ``ValueError`` with a message meant for a person, since it goes
    straight back to the browser.
    """
    if name not in WRITABLE_FIELDS:
        if name == "dataset":
            raise ValueError("Change the dataset through its own action, not as a setting.")
        raise ValueError(f"Unknown setting {name!r}.")

    if name == "kind":
        # Validated against what this view offers, not against the enum.
        # Clumping and mass_law are real AnalysisKind members kept out of the
        # colour picker because the per-atom view normalises against cached
        # dataset percentiles, which exist for beta, alpha and epsilon only.
        # They are shown as charts in the Derived panel instead, where an
        # absolute scale is fine. (Mass law is a plain per-atom scalar, and
        # clumping's per-atom value is a mean over that atom's partners -- a
        # summary, not a site measurement.)
        offered = {v for v, _ in QUANTITIES}
        if value not in offered:
            allowed = ", ".join(v for v, _ in QUANTITIES)
            raise ValueError(f"Cannot show {value!r} per site. Choose one of: {allowed}.")
        return AnalysisKind(value)

    if name == "reference_mode":
        allowed_modes = {"none", "element_mean", "standard", "mol_b", "self"}
        if value not in allowed_modes:
            raise ValueError(f"Unknown reference {value!r}.")
        return value

    if name == "element":
        if value in (None, "", "all"):
            return None
        if value not in ELEMENTS:
            raise ValueError(f"No data for element {value!r}.")
        return value

    if name == "derived_sites":
        if value in (None, ""):
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("derived_sites must be a list of atom indices.")
        return tuple(int(v) for v in value)

    if name == "highlight_pair":
        if value in (None, "", []):
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("highlight_pair must be two atom indices.")
        first, second = (int(v) for v in value)
        if first == second:
            raise ValueError("A clumping pair needs two different atoms.")
        return (first, second)

    if name == "beta_method":
        # parse_method owns the vocabulary; repeating it here would let the
        # two drift and let the UI offer something the fit cannot do.
        parse_method(value if value not in ("", None) else None)
        return (value or "").strip().lower() or DEFAULT_METHOD

    if name == "download_temperatures":
        # Imported here: download.py imports payloads for jsonable(), so a
        # module-level import would close the cycle.
        from rpfr_visualizer.gui.download import parse_temperature_list

        return parse_temperature_list(value, grid=grid)

    if name == "derived_aggregation":
        if value not in {"mean", "max"}:
            raise ValueError("Aggregate a clumping matrix by 'mean' or 'max'.")
        return value

    declared = str({f.name: f.type for f in fields(ExplorerState)}[name])
    # Order matters: dict[str, float] contains both "dict" and "float", so the
    # container has to be recognised before the scalar it holds.
    if "dict" in declared:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object.")
        return {str(k): float(v) for k, v in value.items()}
    if "bool" in declared:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if "float" in declared:
        return float(value)
    if "int" in declared:
        return int(value)
    return value if value not in ("", None) else None


__all__ = [
    "QUANTITIES",
    "REFERENCE_MODES",
    "WRITABLE_FIELDS",
    "anchor_options",
    "coerce_setting",
    "hit_payload",
    "jsonable",
    "match_reason",
    "rank_payload",
    "slot_payload",
    "state_payload",
]
