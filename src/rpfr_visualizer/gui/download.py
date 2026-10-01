"""CSV download of the current molecule or search results.

The explorer's job for this audience is "look at a subset, then take it to
another calculation." The download is that subset at the temperature and
quantity already on screen.

**Every match, not every row shown.** The result list stops at the display
limit; the export does not. A download that silently handed back the first 200
of 376 matches would be worse than one that refused, because nothing about the
file says it is partial. :meth:`ExplorerController.all_matching_keys` re-runs
the query without a limit to get the real set.

Rows are generated and written straight into a spooled temporary file rather
than accumulated: a large export then costs bounded memory instead of holding
both the row dicts and the finished text at once.
"""

from __future__ import annotations

import csv
import io
import math
import re
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

from rpfr_visualizer.analysis.analyze import run_molecule_analysis
from rpfr_visualizer.analysis.temperature import T_MAX_K, T_MIN_K, is_on_grid
from rpfr_visualizer.viz.colors import quantity_symbol
from rpfr_visualizer.viz.explorer import ExplorerController
from rpfr_visualizer.viz.explorer.state import LoadedMolecule

#: Spill the CSV to disk past this many bytes. Small exports never touch the
#: filesystem; large ones stop being a memory question.
SPOOL_MAX_BYTES = 8 << 20

CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "molecule_key",
    "smiles",
    "formula",
    "n_atoms",
    "qc_pass",
    "atom_index",
    "element",
    "temperature_K",
    "on_grid",
    #: Which interpolant produced ``beta`` and ``reference_beta``. Empty at a
    #: stored temperature, where nothing is fitted. Without this the beta
    #: column depends on a sidebar setting the file does not record, and two
    #: exports of "the same" query would differ with no way to tell why.
    "beta_method",
    "quantity",
    "value",
    "beta",
    "reference_beta",
    "reference_mode",
    "local_smiles",
)

#: How many temperatures one export may cover. The cost is one analysis per
#: molecule per temperature, so a thousand-molecule search at fifty
#: temperatures is fifty thousand analyses -- enough to want a deliberate
#: limit rather than a surprised user.
MAX_DOWNLOAD_TEMPERATURES = 50


def parse_temperature_list(
    value: Any,
    grid: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Read "250, 298.15, 300" -- or a list of numbers -- into a tuple.

    Empty means "just the temperature on screen", which is what the export did
    before it could do anything else.

    Args:
        value: The text or list to read.
        grid: The open dataset's stored temperatures, which define the range
            worth exporting. Without one the package defaults are used -- but
            those describe the 20-420 K schema, and a 2.2 file reaches 500 K,
            so a hardcoded ceiling would refuse temperatures the file holds.

    Raises:
        ValueError: With a message meant for a person; it goes straight back to
            the browser.
    """
    low = float(min(grid)) if grid else T_MIN_K
    high = float(max(grid)) if grid else T_MAX_K
    if value is None or value == "" or value == []:
        return ()
    items = value if isinstance(value, (list, tuple)) else str(value).replace(";", ",").split(",")

    out: list[float] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        try:
            kelvin = float(text)
        except ValueError:
            raise ValueError(
                f"{text!r} is not a temperature. Use commas, as in “250, 298.15, 300”."
            ) from None
        if not math.isfinite(kelvin) or kelvin <= 0:
            raise ValueError(f"{text} is not a temperature in kelvin.")
        if kelvin < low or kelvin > high:
            raise ValueError(f"{kelvin:g} K is outside the stored range ({low:g} to {high:g} K).")
        if kelvin not in out:  # a repeat would duplicate every row
            out.append(kelvin)

    if len(out) > MAX_DOWNLOAD_TEMPERATURES:
        raise ValueError(
            f"{len(out)} temperatures is more than the {MAX_DOWNLOAD_TEMPERATURES} "
            f"this export allows at once."
        )
    return tuple(out)


def download_temperatures(controller: ExplorerController) -> tuple[float, ...]:
    """The temperatures this export covers, always at least one."""
    chosen = controller.state.download_temperatures
    return chosen if chosen else (float(controller.state.temperature),)


_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_part(text: str, *, limit: int = 40) -> str:
    cleaned = _SAFE_PART.sub("_", text).strip("._")
    return (cleaned[:limit] or "export").rstrip("._")


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{float(value):.8g}"


def _bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _temperature_note(controller: ExplorerController, wanted: Sequence[float]) -> str:
    """Read or fitted, in the same words the Explore page uses."""
    if len(wanted) == 1:
        return controller.temperature_note(wanted[0])
    from rpfr_visualizer.analysis.temperature import describe_fit

    stored = sum(1 for t in wanted if is_on_grid(float(t), controller.temperatures))
    fitted = len(wanted) - stored
    if not fitted:
        return "All of these are stored temperatures."
    return (
        f"{stored} stored, {fitted} fitted with {describe_fit(controller.state.beta_method)} "
        "(see the beta_method column)."
    )


def download_filename(controller: ExplorerController, scope: str) -> str:
    """A stable, filesystem-safe name for the current download.

    A multi-temperature export is named for how many it covers rather than for
    one of them: picking the first would read as a claim about the contents.
    """
    dataset = _safe_part(controller.dataset)
    wanted = download_temperatures(controller)
    temperature = f"{wanted[0]:g}K" if len(wanted) == 1 else f"{len(wanted)}T"
    if scope == "molecule":
        loaded = controller.state.mol_a
        key = _safe_part(loaded.key if loaded is not None else "molecule")
        return f"{dataset}_{key}_{temperature}.csv"
    return f"{dataset}_search_{temperature}.csv"


def download_preview(controller: ExplorerController, scope: str) -> dict[str, Any]:
    """What a download would contain, without opening every HDF5 group."""
    if scope not in {"molecule", "results"}:
        raise ValueError("Download either this molecule or the current search results.")

    state = controller.state
    n_molecules = 0
    n_sites_estimate: int | None = None
    reason = ""

    if scope == "molecule":
        loaded = state.mol_a
        if loaded is None:
            reason = "Load a molecule or run a search first."
        else:
            n_molecules = 1
            n_sites_estimate = int(loaded.molecule.n_atoms)
            if state.element:
                n_sites_estimate = sum(
                    1 for site in loaded.molecule.sites if site.element == state.element
                )
    else:
        # Every match, not the page. The result list stops at the display limit;
        # the file will not, so the count shown here has to be the real one or
        # the preview understates what is about to be downloaded.
        keys = controller.all_matching_keys() or list(controller.last_hits)
        n_molecules = len(keys)
        if n_molecules == 0:
            reason = "Run a search first."
        elif controller.index is not None:
            entries = controller.index.get_many(keys)
            n_sites_estimate = 0
            for pair in keys:
                entry = entries.get(pair)
                if entry is None:
                    continue
                if state.element:
                    counts = getattr(entry, "element_counts", None) or {}
                    n_sites_estimate += int(counts.get(state.element, 0))
                else:
                    n_sites_estimate += int(entry.n_atoms or 0)

    if not reason and n_sites_estimate == 0:
        which = state.element or "the current filter"
        reason = f"No {which} sites in this selection. Choose “all elements” or another element."

    wanted = download_temperatures(controller)
    return {
        "scope": scope,
        "dataset": controller.dataset,
        "n_molecules": n_molecules,
        # One row per site *per temperature*: a four-temperature export of a
        # thousand sites is four thousand rows, and the count has to say so
        # before the click rather than after it. None means "not countable
        # from the index" and stays None -- multiplying it would invent a
        # number out of an unknown.
        "n_sites_estimate": (None if n_sites_estimate is None else n_sites_estimate * len(wanted)),
        "shown": len(controller.last_hits) if scope == "results" else n_molecules,
        "temperature": float(state.temperature),
        "temperatures": [float(t) for t in wanted],
        "on_grid": state.temperature_is_on_grid(controller.temperatures),
        "quantity": state.kind.value,
        "quantity_symbol": quantity_symbol(state.kind.value),
        "temperature_note": _temperature_note(controller, wanted),
        "element": state.element,
        "reference_mode": state.reference_mode if state.needs_reference() else None,
        "filename": download_filename(controller, scope),
        "can_download": not reason,
        "reason": reason,
        "columns": list(CSV_COLUMNS),
    }


def _molecules(controller: ExplorerController, scope: str) -> Iterator[LoadedMolecule]:
    if scope == "molecule":
        loaded = controller.state.mol_a
        if loaded is None:
            raise ValueError("Load a molecule first.")
        yield loaded
        return

    keys = controller.all_matching_keys() or controller.last_hits
    if not keys:
        raise ValueError("Search for molecules first.")

    provider = controller.provider
    for dataset, key in keys:
        if dataset != controller.dataset:
            continue
        try:
            yield LoadedMolecule(
                dataset=dataset,
                key=key,
                molecule=provider.get_molecule_by_key(key),
                arrays=provider.load_arrays_by_key(key),
            )
        except (KeyError, OSError):
            continue


def iter_download_rows(controller: ExplorerController, scope: str) -> Iterator[dict[str, str]]:
    """One row per site, using the same analysis as the figure on screen.

    A generator rather than a list: the caller writes each row out as it arrives,
    so an export of every match never holds the whole thing in memory.
    """
    if scope not in {"molecule", "results"}:
        raise ValueError("Download either this molecule or the current search results.")

    state = controller.state
    base_request = controller.build_request()
    grid = controller.temperatures
    wanted = download_temperatures(controller)
    formulas: dict[tuple[str, str], str] = {}
    if controller.index is not None:
        pairs = (
            [(controller.dataset, state.mol_a.key)]
            if scope == "molecule" and state.mol_a is not None
            else controller.all_matching_keys() or list(controller.last_hits)
        )
        formulas = {
            pair: (entry.formula or "") for pair, entry in controller.index.get_many(pairs).items()
        }

    # Each molecule is read from the HDF5 once and analysed at every
    # temperature asked for, then written out site by site so that a site's
    # temperature series lands contiguously -- which is the shape you want if
    # you are fitting a calibration to it. Holding one molecule's results
    # across the temperatures costs a few floats per site.
    for loaded in _molecules(controller, scope):
        formula = formulas.get((loaded.dataset, loaded.key), "")
        qc = getattr(loaded.molecule, "qc_pass", None)
        sites_by_index = {site.atom_index: site for site in loaded.molecule.sites}

        by_temperature = [
            (
                float(temperature),
                is_on_grid(float(temperature), grid),
                run_molecule_analysis(
                    loaded.molecule,
                    replace(base_request, temperature=float(temperature)),
                    arrays=loaded.arrays,
                ),
            )
            for temperature in wanted
        ]

        if not by_temperature:
            continue
        # Same molecule and same element lock at every temperature, so the
        # site lists line up position for position. A site with no value at
        # one temperature is still present in the list, just empty.
        for position in range(len(by_temperature[0][2].sites)):
            for temperature, on_grid, result in by_temperature:
                yield _row(
                    loaded,
                    result.sites[position],
                    sites_by_index,
                    state=state,
                    temperature=temperature,
                    on_grid=on_grid,
                    formula=formula,
                    qc=qc,
                )


def _row(
    loaded: LoadedMolecule,
    site: Any,
    sites_by_index: dict[int, Any],
    *,
    state: Any,
    temperature: float,
    on_grid: bool,
    formula: str,
    qc: Any,
) -> dict[str, str]:
    """One row: one site at one temperature."""
    stored = sites_by_index.get(site.atom_index)
    local = ""
    beta = ""
    if stored is not None:
        local = next((fp for fp in stored.local_smiles if fp), "")
        try:
            beta = _fmt(stored.beta_at(temperature, warn=False, method=state.beta_method))
        except (ValueError, TypeError):
            beta = ""
    return {
        "dataset": loaded.dataset,
        "molecule_key": loaded.key,
        "smiles": getattr(loaded.molecule, "smiles", "") or "",
        "formula": formula,
        "n_atoms": str(int(loaded.molecule.n_atoms)),
        "qc_pass": _bool(None if qc is None else bool(qc)),
        "atom_index": str(int(site.atom_index)),
        "element": site.element,
        "temperature_K": _fmt(temperature),
        "on_grid": _bool(on_grid),
        "beta_method": "" if on_grid else state.beta_method,
        "quantity": state.kind.value,
        "value": _fmt(site.value),
        "beta": beta,
        "reference_beta": _fmt(site.reference_value),
        "reference_mode": (state.reference_mode if state.needs_reference() else ""),
        "local_smiles": local,
    }


def download_rows(controller: ExplorerController, scope: str) -> list[dict[str, str]]:
    """Every row at once. Convenient for tests and notebooks; the HTTP path uses
    :func:`iter_download_rows` so a large export stays bounded."""
    return list(iter_download_rows(controller, scope))


def download_csv(controller: ExplorerController, scope: str) -> tuple[str, str]:
    """Return ``(filename, csv_text)``. Prefer :func:`open_download_csv` for HTTP."""
    name, handle, _size = open_download_csv(controller, scope)
    try:
        return name, handle.read().decode("utf-8")
    finally:
        handle.close()


def open_download_csv(controller: ExplorerController, scope: str) -> tuple[str, Any, int]:
    """Write the CSV to a spooled temporary file.

    Returns ``(filename, open_binary_file, size_bytes)`` positioned at the
    start. Spooled rather than in-memory so a large export costs bounded memory,
    and sized rather than chunked so the response still carries Content-Length
    and the browser shows real progress.

    The caller owns the handle and must close it.
    """
    spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_MAX_BYTES, mode="w+b")
    wrapper = io.TextIOWrapper(spool, encoding="utf-8", newline="", write_through=True)
    writer = csv.DictWriter(wrapper, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    wrote = False
    try:
        for row in iter_download_rows(controller, scope):
            writer.writerow(row)
            wrote = True
        if not wrote:
            raise ValueError("Nothing to download at these settings.")
        wrapper.flush()
        size = spool.tell()
        spool.seek(0)
    except BaseException:
        wrapper.detach()
        spool.close()
        raise

    # Detach so closing the wrapper later cannot close the file out from under
    # the transport; the caller closes the spool itself.
    wrapper.detach()
    return download_filename(controller, scope), spool, size


__all__ = [
    "CSV_COLUMNS",
    "SPOOL_MAX_BYTES",
    "download_csv",
    "download_filename",
    "download_preview",
    "download_rows",
    "iter_download_rows",
    "open_download_csv",
]
