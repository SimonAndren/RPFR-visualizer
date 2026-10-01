"""Everything the explorer does, with no user interface attached.

This module deliberately imports no ipywidgets and calls no ``display()``: it
loads molecules, runs analyses, searches, ranks, and returns objects. The view
layer turns those into widgets. That split is what makes the behaviour testable
and would let a different front end reuse the same logic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpfr_visualizer.analysis.analyze import run_molecule_analysis
from rpfr_visualizer.analysis.clumping import clumping_delta
from rpfr_visualizer.analysis.element_stats import ElementStatsCache
from rpfr_visualizer.analysis.fingerprint_explorer import rank_sites, ranks_to_frame
from rpfr_visualizer.analysis.functional_groups import (
    filter_ranks_by_fg,
    index_query_for_fg,
    observed_r1_shells,
    parse_free_text_filter,
    preset_by_name,
    primary_fg_presets,
    search_molecules_by_fg,
)
from rpfr_visualizer.analysis.models import AnalysisKind, AnalysisRequest
from rpfr_visualizer.analysis.reference import (
    ElementMeanReference,
    MoleculeReference,
    SelfReference,
    StandardReference,
)
from rpfr_visualizer.analysis.temperature import beta_curve, grid_index
from rpfr_visualizer.io.molecule_index import MoleculeIndex, build_or_load_index, index_cache_path
from rpfr_visualizer.io.providers import DatasetCatalog, HDF5Provider
from rpfr_visualizer.viz.colors import quantity_symbol
from rpfr_visualizer.viz.electron_cloud import EnrichmentView
from rpfr_visualizer.viz.explorer.state import ExplorerState, LoadedMolecule, ReferenceMode
from rpfr_visualizer.viz.qc_strip import qc_panel_text, run_qc_for_key, stored_qc_flags


@dataclass
class RenderPayload:
    """What a render produced, before anything is displayed.

    Holding this as data (rather than drawing inside the analysis code) is what
    lets tests check a render without a notebook, and lets the view decide how
    to lay it out.
    """

    view_a: EnrichmentView | None = None
    view_b: EnrichmentView | None = None
    label_a: str = ""
    label_b: str = ""
    caption: str = ""
    warnings: tuple[str, ...] = ()

    #: Colour-scale caption for the second molecule, when one is loaded.
    caption_b: str = ""
    #: What the second molecule is actually showing. Usually the selected
    #: quantity, but see :attr:`note_b`.
    kind_b: AnalysisKind | None = None
    #: Why the second molecule shows a different quantity from the selection,
    #: or why its values are not comparable with the first molecule's. Empty
    #: when there is nothing to warn about. A caller displaying two molecules
    #: under one legend must surface this.
    note_b: str = ""


class ExplorerController:
    """Drives molecule selection, analysis, search and ranking."""

    def __init__(
        self,
        provider_or_catalog: HDF5Provider | DatasetCatalog,
        *,
        dataset: str | None = None,
        index: MoleculeIndex | None = None,
        stats: ElementStatsCache | None = None,
        state: ExplorerState | None = None,
        index_cache: Path | str | None = None,
        data_dir: Path | str | None = None,
        max_stats_molecules: int | None = 200,
    ) -> None:
        if isinstance(provider_or_catalog, DatasetCatalog):
            self.catalog = provider_or_catalog
            names = self.catalog.datasets
            if not names:
                raise ValueError("Catalog has no registered datasets")
            chosen = dataset or names[0]
            if chosen not in names:
                raise KeyError(f"Dataset {chosen!r} not in catalog. Available: {', '.join(names)}")
        else:
            self.catalog = DatasetCatalog()
            self.catalog.register_provider(provider_or_catalog.dataset, provider_or_catalog)
            chosen = dataset or provider_or_catalog.dataset

        self.state = state or ExplorerState()
        self.state.dataset = chosen

        self._index = index
        self._index_cache = Path(index_cache) if index_cache else None
        # Where the index and statistics caches live. Relative by default for
        # backwards compatibility, which resolves against the process cwd — pass
        # an absolute path from anything that can be launched from an arbitrary
        # directory, or it will silently miss an existing index and rebuild
        # statistics that are already on disk.
        self.data_dir = Path(data_dir) if data_dir is not None else Path("data")
        self._max_stats_molecules = max_stats_molecules
        # Per-dataset statistics are expensive on full-size files, so they are
        # built on first use and kept per dataset rather than eagerly.
        self._stats_by_dataset: dict[str, ElementStatsCache] = {}
        if stats is not None:
            self._stats_by_dataset[chosen] = stats

        self.match_pool: list[LoadedMolecule] = []
        self.last_hits: list[tuple[str, str]] = []
        #: The query behind :attr:`last_hits`, so an export can ask what the
        #: *query* matched rather than what the page happened to show.
        self.last_query: str | None = None
        self.last_ranks: list = []
        self._structured_qc: Any = None

    # --------------------------------------------------------------- datasets

    @property
    def dataset(self) -> str:
        """Dataset currently being explored."""
        return self.state.dataset

    @property
    def provider(self) -> HDF5Provider:
        """Provider for the active dataset."""
        return self.catalog.get_provider(self.state.dataset)

    def available_datasets(self) -> tuple[str, ...]:
        """Every dataset this controller can switch to."""
        return self.catalog.datasets

    def use_dataset(self, name: str) -> None:
        """Switch datasets, clearing selections that belong to the old one."""
        if name not in self.catalog.datasets:
            raise KeyError(
                f"Dataset {name!r} not registered. Available: {', '.join(self.catalog.datasets)}"
            )
        if name == self.state.dataset:
            return
        self.state.dataset = name
        self.state.mol_a = None
        self.state.mol_b = None
        self.clear_pool()
        self.last_hits = []
        self.last_query = None
        self.last_ranks = []
        self._structured_qc = None

    @property
    def temperatures(self) -> tuple[float, ...]:
        """The stored grid, in file order.

        The order is positionally coupled to the beta arrays: ``grid_index()``
        returns an index into *this* tuple and callers use it to index
        ``rpfr_single``. Never sort this — use :attr:`temperatures_for_display`.
        """
        return tuple(self.provider.temperatures)

    @property
    def temperatures_for_display(self) -> tuple[float, ...]:
        """The stored grid in ascending order, for pickers and captions.

        These files append the chemistry temperatures after the sweep, so the
        stored order runs ``20 … 420, 273.15, 298.15``. Listing that verbatim
        puts ``273.15 K`` after ``420 K`` in a dropdown. Sorting is safe here
        because only the values are shown; nothing indexes by position.
        """
        return tuple(sorted(self.temperatures))

    @property
    def stats(self) -> ElementStatsCache:
        """Per-element dataset statistics, built on first use and persisted.

        Sampling molecules out of a full-size file is dominated by cold random
        reads (about a minute on external storage), so the small result is
        cached to disk and reused across sessions.
        """
        name = self.state.dataset
        if name not in self._stats_by_dataset:
            self._stats_by_dataset[name] = ElementStatsCache.from_hdf5_cached(
                self.provider.path,
                cache_path=self._stats_cache_path(name),
                max_molecules=self._max_stats_molecules,
                temperatures=self.temperatures,
            )
        return self._stats_by_dataset[name]

    def _stats_cache_path(self, dataset: str) -> Path:
        """Where this dataset's statistics are persisted.

        Keyed by the source filename and sample size, so a different file or a
        different cap does not silently reuse the wrong numbers.
        """
        stem = self.provider.path.stem
        n = self._max_stats_molecules or "all"
        return self.data_dir / "cache" / "stats" / f"{dataset}.{stem}.n{n}.json"

    @property
    def index(self) -> MoleculeIndex | None:
        """Search index, loaded or built on first use."""
        if self._index is None:
            if self._index_cache is None:
                self._index_cache = index_cache_path(self.data_dir)
            self._index = build_or_load_index(self.catalog, self._index_cache)
        return self._index

    # -------------------------------------------------------------- selection

    def load(self, key: str, *, slot: str = "a") -> LoadedMolecule:
        """Load a molecule from the active dataset into slot ``a`` or ``b``."""
        if slot not in ("a", "b"):
            raise ValueError(f"slot must be 'a' or 'b', got {slot!r}")
        provider = self.provider
        loaded = LoadedMolecule(
            dataset=self.state.dataset,
            key=key,
            molecule=provider.get_molecule_by_key(key),
            arrays=provider.load_arrays_by_key(key),
        )
        if slot == "a":
            self.state.mol_a = loaded
            self._structured_qc = None
        else:
            self.state.mol_b = loaded
            self.state.ref_atom_index_b = 0
        return loaded

    def clear(self, slot: str) -> None:
        """Empty slot ``a`` or ``b``."""
        if slot == "a":
            self.state.mol_a = None
            self._structured_qc = None
        elif slot == "b":
            self.state.mol_b = None
            self.state.ref_atom_index_b = 0
        else:
            raise ValueError(f"slot must be 'a' or 'b', got {slot!r}")

    # ------------------------------------------------------------ ranking pool

    def load_pool(self, keys: Sequence[str], *, clear: bool = True) -> int:
        """Load molecules into the ranking pool. Returns how many are in it."""
        if clear:
            self.match_pool.clear()
        existing = {m.key for m in self.match_pool}
        provider = self.provider
        for key in keys:
            if key in existing:
                continue
            try:
                self.match_pool.append(
                    LoadedMolecule(
                        dataset=self.state.dataset,
                        key=key,
                        molecule=provider.get_molecule_by_key(key),
                        arrays=provider.load_arrays_by_key(key),
                    )
                )
                existing.add(key)
            except (KeyError, OSError):
                # A hit that will not load should not abort the whole pool.
                continue
        return len(self.match_pool)

    def load_hits_into_pool(self, *, clear: bool = True) -> int:
        """Load every molecule from the last search into the ranking pool."""
        return self.load_pool([key for _ds, key in self.last_hits], clear=clear)

    def clear_pool(self) -> None:
        self.match_pool.clear()

    # ----------------------------------------------------------------- search

    def search(self, query: str, *, limit: int | None = 200) -> list:
        """Free-text search (name, formula, SMILES, key, ``smarts:``, ``fg:``)."""
        idx = self.index
        if idx is None:
            raise ValueError("Search requires a molecule index")
        hits = idx.search(query, datasets=(self.state.dataset,), limit=limit)
        self.last_hits = [(h.entry.dataset, h.entry.key) for h in hits]
        self.last_query = query
        return hits

    def all_matching_keys(self) -> list[tuple[str, str]]:
        """Every molecule the last query matched, not just the page shown.

        ``last_hits`` stops at the display limit, which is the right size for a
        result list and the wrong size for an export: a download that silently
        gave you the first 200 of 376 matches is worse than one that refuses.
        Re-running the query without a limit is what makes "all of them" mean
        what it says.
        """
        if self.last_query is None:
            return []
        idx = self.index
        if idx is None:
            raise ValueError("Search requires a molecule index")
        hits = idx.search(self.last_query, datasets=(self.state.dataset,), limit=None)
        return [(h.entry.dataset, h.entry.key) for h in hits]

    def search_functional_group(
        self,
        *,
        preset_name: str | None = None,
        fg_text: str = "",
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        """Find molecules containing a functional group.

        A curated preset is answered from the index's precomputed membership
        table when available — a lookup rather than a substructure scan.
        """
        idx = self.index
        if idx is None:
            raise ValueError("Functional-group search requires a molecule index")

        preset_name = preset_name or self.state.fg_preset
        fg_text = fg_text or self.state.fg_text
        limit = limit if limit is not None else self.state.max_fg_hits
        preset = preset_by_name(preset_name) if preset_name else None

        precomputed = (
            preset is not None
            and not fg_text.strip()
            and getattr(idx.backend, "has_functional_group", lambda _n: False)(preset.name)
        )
        if precomputed:
            hits = idx.search_functional_group(
                preset.name, datasets=(self.state.dataset,), limit=limit
            )
        else:
            if not index_query_for_fg(preset=preset, fg_text=fg_text):
                raise ValueError("Choose a functional group or enter smarts:/substring text")
            hits = search_molecules_by_fg(
                idx,
                preset=preset,
                fg_text=fg_text,
                datasets=(self.state.dataset,),
                max_hits=limit,
            )

        self.last_hits = [(h.entry.dataset, h.entry.key) for h in hits]
        # Record the equivalent query text, so all_matching_keys() has one way
        # to re-run a search regardless of which route produced it. A bare
        # preset goes back through the same precomputed membership table.
        self.last_query = f"fg:{preset_name}" if preset_name and not fg_text.strip() else None
        self.state.fg_preset = preset_name
        return list(self.last_hits)

    def functional_group_counts(self) -> list[tuple[str, str, int]]:
        """``(name, description, n_molecules)`` for every curated group.

        Shows what the dataset actually contains, so an empty search result
        reads as "none here" rather than "the search is broken". Every preset is
        listed, including those with no members — absence is information.
        """
        idx = self.index
        if idx is None:
            raise ValueError("Functional-group counts require a molecule index")
        counts = idx.functional_group_counts(datasets=(self.state.dataset,))
        return [
            (preset.name, preset.description, counts.get(preset.name, 0))
            for preset in primary_fg_presets()
        ]

    def dataset_inventory(self) -> dict[str, dict[str, int]]:
        """Molecule, structure-coverage and QC counts for every indexed dataset.

        Structure coverage is what makes a zero-hit structural search legible:
        some datasets store no SMILES at all, so "no matches" and "unanswerable
        here" look identical without it.
        """
        idx = self.index
        if idx is None:
            raise ValueError("Dataset inventory requires a molecule index")
        return idx.dataset_inventory()

    def hit_labels(self) -> list[str]:
        """Readable labels for the last search hits, enriched from the index."""
        idx = self.index
        if idx is None:
            return [f"{ds}:{key}" for ds, key in self.last_hits]
        entries = idx.get_many(self.last_hits)
        labels = []
        for ds, key in self.last_hits:
            entry = entries.get((ds, key))
            if entry is None:
                labels.append(f"{ds}:{key}")
            else:
                labels.append(f"{ds}:{key}  {entry.formula or ''}  {(entry.smiles or '')[:40]}")
        return labels

    # --------------------------------------------------------------- analysis

    def build_request(self) -> AnalysisRequest:
        """Turn the current state into an analysis request."""
        state = self.state
        if state.kind == AnalysisKind.BETA:
            return AnalysisRequest(
                kind=AnalysisKind.BETA,
                temperature=state.temperature,
                reference=None,
                element=state.element,
                beta_method=state.beta_method,
            )
        return AnalysisRequest(
            kind=state.kind,
            temperature=state.temperature,
            reference=self._reference_for(state.reference_mode),
            element=state.element,
            beta_method=state.beta_method,
        )

    def _request_for_slot_b(self) -> tuple[AnalysisRequest, str]:
        """The analysis molecule B can honestly show, and any caveat about it.

        B is comparable with A only under a reference that does not depend on A.
        The in-molecule element mean and the dataset standard both qualify. An
        anchor site in A does not, and when B *is* the reference there is
        nothing left to reference it against — so those two fall back to stored
        beta rather than putting a differently-defined number under the same
        colour scale.
        """
        state = self.state
        note = ""
        kind = state.kind

        if kind != AnalysisKind.BETA:
            if state.reference_mode == "mol_b":
                kind = AnalysisKind.BETA
                note = (
                    f"Showing β, not {quantity_symbol(state.kind.value)}: "
                    "this molecule is the reference."
                )
            elif state.reference_mode == "self":
                kind = AnalysisKind.BETA
                note = (
                    f"Showing β, not {quantity_symbol(state.kind.value)}: "
                    "the reference site is in the other molecule."
                )
            elif state.reference_mode in ("none", "element_mean"):
                note = (
                    f"Each molecule is referenced to its own mean "
                    f"{state.element or 'element'} site. Use the dataset average to "
                    f"compare the two."
                )

        reference = None if kind == AnalysisKind.BETA else self._reference_for(state.reference_mode)
        request = AnalysisRequest(
            kind=kind,
            temperature=state.temperature,
            reference=reference,
            element=state.element,
            beta_method=state.beta_method,
        )
        return request, note

    def _reference_for(self, mode: ReferenceMode):
        if mode == "standard":
            return StandardReference(
                beta_by_element=dict(self.state.standard_overrides), stats=self.stats
            )
        if mode == "mol_b":
            if self.state.mol_b is None:
                raise ValueError("Molecule B is required for the mol_b reference mode")
            return MoleculeReference(
                reference_molecule=self.state.mol_b.molecule,
                reference_atom_index=int(self.state.ref_atom_index_b),
            )
        if mode == "self":
            return SelfReference(anchor_atom_index=self.state.anchor_atom_index)
        # "none" and "element_mean" both reference the in-molecule element mean.
        return ElementMeanReference()

    def analyze(self, slot: str = "a"):
        """Run the current analysis on a loaded molecule."""
        loaded = self.state.mol_a if slot == "a" else self.state.mol_b
        if loaded is None:
            raise ValueError(f"Load molecule {slot.upper()} first")
        return run_molecule_analysis(loaded.molecule, self.build_request(), arrays=loaded.arrays)

    def temperature_note(self, temperature: float | None = None) -> str:
        """Whether β at *temperature* is read or fitted, and how.

        The single wording for this, used by the provenance line, the notebook
        warning and the download page. Each had its own, three of them
        differed, and one still described the fit that stopped being the
        default two changes ago.
        """
        from rpfr_visualizer.analysis.temperature import describe_fit

        t = float(self.state.temperature if temperature is None else temperature)
        if any(abs(float(g) - t) <= 1e-6 for g in self.temperatures):
            return f"{t:g} K is a stored temperature."
        return (
            f"{t:g} K is not a stored temperature, so β is fitted with "
            f"{describe_fit(self.state.beta_method)}."
        )

    def warnings(self) -> tuple[str, ...]:
        """Caveats that apply to the current settings.

        Currently the off-grid temperature case: beta at such a temperature is
        fitted, not read from the file, and the user should know.
        """
        messages: list[str] = []
        if not self.state.temperature_is_on_grid(self.temperatures):
            messages.append(self.temperature_note())
        return tuple(messages)

    def render_payload(self) -> RenderPayload:
        """Build the enrichment views for the current state, without displaying."""
        state = self.state
        if state.mol_a is None:
            raise ValueError("Load molecule A first")

        result_a = self.analyze("a")
        view_a = EnrichmentView.from_analysis(
            state.mol_a.molecule,
            result_a,
            stats=self.stats,
            show_hydrogens=state.show_hydrogens,
        )
        label_a = state.mol_a.label + (f" [{state.element}]" if state.element else "")

        view_b = None
        label_b = ""
        caption_b = ""
        kind_b: AnalysisKind | None = None
        note_b = ""
        if state.mol_b is not None:
            request_b, note_b = self._request_for_slot_b()
            kind_b = request_b.kind
            result_b = run_molecule_analysis(
                state.mol_b.molecule,
                request_b,
                arrays=state.mol_b.arrays,
            )
            view_b = EnrichmentView.from_analysis(
                state.mol_b.molecule,
                result_b,
                stats=self.stats,
                show_hydrogens=state.show_hydrogens,
            )
            caption_b = view_b.caption_for(state.element)
            label_b = f"{state.mol_b.label} [{kind_b.value}]"
            if state.reference_mode == "mol_b":
                ref_element = state.mol_b.molecule.sites[int(state.ref_atom_index_b)].element
                label_b += f" (reference site {state.ref_atom_index_b}: {ref_element})"

        return RenderPayload(
            view_a=view_a,
            view_b=view_b,
            label_a=label_a,
            label_b=label_b,
            # Ask for the caption rather than reading view_a.caption: that
            # attribute is only populated while a 3D viewer is being built, so
            # reading it here always yielded "".
            caption=view_a.caption_for(state.element),
            warnings=self.warnings(),
            caption_b=caption_b,
            kind_b=kind_b,
            note_b=note_b,
        )

    # ---------------------------------------------------------------- ranking

    def rank(self) -> list:
        """Rank sites by |value| using the current analysis settings.

        Ranks the ranking pool when it is non-empty, otherwise the loaded A (and
        B) molecules.
        """
        state = self.state
        molecules: list[tuple[str, str, Any]] = []
        arrays_by_key: dict[tuple[str, str], Any] = {}

        pool: Sequence[LoadedMolecule]
        if state.use_match_pool and self.match_pool:
            pool = self.match_pool
        else:
            pool = [m for m in (state.mol_a, state.mol_b) if m is not None]
        for loaded in pool:
            molecules.append((loaded.dataset, loaded.key, loaded.molecule))
            arrays_by_key[(loaded.dataset, loaded.key)] = loaded.arrays
        if not molecules:
            return []

        ranks = rank_sites(
            molecules,
            self.build_request(),
            # Over-fetch so functional-group filtering below still has
            # candidates left to return top_n from.
            top_n=max(state.top_n * 5, state.top_n),
            element=state.element,
            arrays_by_key=arrays_by_key,
        )
        if state.heavy_only:
            ranks = [r for r in ranks if r.element != "H"]

        # A functional-group preset selects *molecules*, and has already done so
        # by the time they reach the pool. Re-applying it here as a site filter
        # silently empties the result whenever the locked element differs from
        # the group's element — searching for alcohols and then ranking carbon
        # sites returned nothing. Site-level narrowing is the element lock plus
        # any explicit R1 / SMARTS filter the user typed.
        free = parse_free_text_filter(state.fg_text)
        ranks = filter_ranks_by_fg(
            ranks,
            preset=None,
            r1_exact=state.fg_observed_r1 or free.get("r1_exact"),
            smiles_substring=free.get("smiles_substring"),
            smarts=free.get("smarts"),
        )
        self.last_ranks = ranks[: state.top_n]
        return self.last_ranks

    def ranks_frame(self):
        """Last ranking as a pandas DataFrame."""
        return ranks_to_frame(self.last_ranks)

    def observed_r1(self) -> list[str]:
        """R1 shells present in the last ranking, for the FG filter dropdown."""
        return observed_r1_shells(self.last_ranks, element=self.state.element)

    # --------------------------------------------------------------------- QC

    # ------------------------------------------------- derived properties

    #: Sites of one element track each other closely against temperature, so a
    #: long series list spends colours without adding information. Four also
    #: keeps every line directly labelled rather than legend-only.
    MAX_DERIVED_SITES = 4

    def _derived_site_indices(self, loaded: LoadedMolecule) -> list[int]:
        """Which sites the curves show: the explicit choice, or the strongest few."""
        molecule = loaded.molecule
        chosen = [i for i in self.state.derived_sites if 0 <= i < molecule.n_atoms]
        if chosen:
            return chosen[: self.MAX_DERIVED_SITES]

        element = self.state.element
        candidates = [
            site
            for site in molecule.sites
            if (not element or site.element == element)
            and (not self.state.heavy_only or site.element != "H")
        ]
        if not candidates:
            candidates = list(molecule.sites)

        def strength(site: Any) -> float:
            try:
                value = site.beta_at(
                    self.state.temperature, warn=False, method=self.state.beta_method
                )
            except (ValueError, TypeError):
                return 0.0
            return abs(float(value)) if value is not None else 0.0

        candidates.sort(key=strength, reverse=True)
        return [s.atom_index for s in candidates[: self.MAX_DERIVED_SITES]]

    def _nearest_grid_temperatures(self) -> list[float]:
        """The two stored temperatures bracketing the current one."""
        grid = sorted(float(t) for t in self.temperatures)
        target = float(self.state.temperature)
        below = [t for t in grid if t <= target]
        above = [t for t in grid if t >= target]
        nearest = []
        if below:
            nearest.append(below[-1])
        if above and above[0] not in nearest:
            nearest.append(above[0])
        return nearest

    def derived_payload(self) -> dict[str, Any]:
        """Temperature curves, mass-law values and the clumping matrix.

        Everything here comes from arrays already resident: ``load()`` reads
        them for every molecule regardless of what is on screen.

        Clumping and mass law exist only on the stored grid -- neither has an
        interpolation model, unlike beta -- so an off-grid temperature makes
        them *unavailable* rather than approximate, and this says so instead of
        raising. An error would read as a bug; a typed-in temperature is normal.
        """
        loaded = self.state.mol_a
        if loaded is None:
            return {"available": False, "message": "Load molecule A first."}

        molecule = loaded.molecule
        grid = [float(t) for t in self.temperatures]
        indices = self._derived_site_indices(loaded)
        by_index = {site.atom_index: site for site in molecule.sites}
        on_grid = grid_index(float(self.state.temperature), tuple(grid)) is not None

        def label(idx: int) -> str:
            site = by_index.get(idx)
            return f"{site.element}{idx}" if site is not None else str(idx)

        # --- beta against temperature -----------------------------------
        curves = []
        for idx in indices:
            site = by_index.get(idx)
            if site is None:
                continue
            temps = [float(t) for t in site.temperatures]
            betas = [float(b) for b in site.beta]
            points = sorted(zip(temps, betas, strict=False))
            try:
                dense_t, dense_b = beta_curve(temps, betas, method=self.state.beta_method)
            except ValueError:
                dense_t, dense_b = [], []
            curves.append(
                {
                    "atom_index": idx,
                    "element": site.element,
                    "label": label(idx),
                    "temperatures": [p[0] for p in points],
                    "values": [p[1] for p in points],
                    "curve_temperatures": [float(t) for t in dense_t],
                    "curve_values": [float(b) for b in dense_b],
                }
            )

        payload: dict[str, Any] = {
            "available": True,
            "message": "",
            "molecule": f"{self.dataset}:{loaded.key}",
            "temperature": float(self.state.temperature),
            "on_grid": on_grid,
            "nearest_grid": self._nearest_grid_temperatures(),
            "beta": {
                "sites": curves,
                "scale": "log",
                "method": self.state.beta_method,
            },
        }

        off_grid_note = "Only available at stored temperatures."

        # --- mass law, per stored temperature ----------------------------
        payload["mass_law"] = self._mass_law_section(loaded, indices, grid, label, off_grid_note)

        # --- clumping, at the chosen temperature -------------------------
        payload["clumping"] = self._clumping_section(loaded, on_grid, label, off_grid_note)
        return payload

    #: Mass law is the triple-oxygen exponent, and the files store it for oxygen
    #: only -- every C, H and N row is exactly 0.0, meaning "not applicable"
    #: rather than "lambda is zero". Drawing those would assert a measurement
    #: that was never made.
    MASS_LAW_ELEMENT = "O"

    def _mass_law_section(
        self,
        loaded: LoadedMolecule,
        indices: Sequence[int],
        grid: Sequence[float],
        label: Any,
        off_grid_note: str,
    ) -> dict[str, Any]:
        del indices  # mass law picks its own sites; see MASS_LAW_ELEMENT
        arrays = loaded.arrays
        if arrays is None or getattr(arrays, "mass_law", None) is None:
            return {"available": False, "message": "This dataset stores no mass-law values."}

        oxygens = [
            site.atom_index
            for site in loaded.molecule.sites
            if site.element == self.MASS_LAW_ELEMENT
        ]
        if not oxygens:
            return {
                "available": False,
                "message": "No oxygen in this molecule.",
            }

        order = sorted(range(len(grid)), key=lambda i: grid[i])
        table = arrays.mass_law
        sites = []
        for idx in oxygens[: self.MAX_DERIVED_SITES]:
            if idx >= table.shape[0]:
                continue
            row = [float(table[idx, i]) for i in order]
            sites.append(
                {
                    "atom_index": idx,
                    "element": loaded.molecule.sites[idx].element
                    if idx < len(loaded.molecule.sites)
                    else "",
                    "label": label(idx),
                    "temperatures": [float(grid[i]) for i in order],
                    "values": row,
                }
            )
        return {
            "available": bool(sites),
            "message": "" if sites else "No mass-law values for these sites.",
            "note": "Oxygen only, at stored temperatures.",
            "element": self.MASS_LAW_ELEMENT,
            "sites": sites,
        }

    def _clumping_section(
        self,
        loaded: LoadedMolecule,
        on_grid: bool,
        label: Any,
        off_grid_note: str,
    ) -> dict[str, Any]:
        arrays = loaded.arrays
        if arrays is None or getattr(arrays, "clumping", None) is None:
            return {"available": False, "message": "This dataset stores no clumping values."}
        if not on_grid:
            return {"available": False, "message": off_grid_note}

        request = AnalysisRequest(
            kind=AnalysisKind.CLUMPING,
            temperature=float(self.state.temperature),
            aggregation=self.state.derived_aggregation,
        )
        try:
            result = run_molecule_analysis(loaded.molecule, request, arrays=arrays)
        except ValueError as exc:
            return {"available": False, "message": str(exc)}

        matrix = result.pair_values
        if matrix is None:
            return {"available": False, "message": "No pair values at this temperature."}

        n = int(matrix.shape[0])
        labels = [label(i) for i in range(n)]
        sites_by_index = {s.atom_index: s for s in loaded.molecule.sites}
        elements = [sites_by_index[i].element if i in sites_by_index else "" for i in range(n)]
        rows: list[list[float | None]] = []
        pairs: list[dict[str, Any]] = []
        for i in range(n):
            row: list[float | None] = []
            for j in range(n):
                # pair_values holds the ratio beta_ij / (beta_i * beta_j): the
                # doubly substituted isotopologue's beta against the product of
                # the two singly substituted ones. Reported as the clumped
                # isotope Delta in per mil, 1000 ln of that ratio -- signed,
                # zero for no excess, and a size a reader can hold in their
                # head. Plotting the ratio directly put every cell at ~1.0 and
                # painted the whole matrix one colour.
                factor = float(matrix[i, j])
                delta = clumping_delta(factor) if math.isfinite(factor) else None
                value = None if delta is None else delta * 1000.0
                # The diagonal is an atom paired with itself, not a pair: it is
                # NaN in schema 2.2 and a placeholder 1 before that, and either
                # way it must not enter the colour scale.
                if i == j or value is None:
                    row.append(None)
                    continue
                row.append(value)
                if j > i:
                    pairs.append(
                        {
                            "i": i,
                            "j": j,
                            "label_i": labels[i],
                            "label_j": labels[j],
                            "delta": value,
                            # H-H pairs dominate by an order of magnitude, so
                            # without a way to set them aside the heavy-atom
                            # clumping -- the part a clumped-isotopologue
                            # measurement is usually about -- is invisible.
                            "has_h": elements[i] == "H" or elements[j] == "H",
                        }
                    )
            rows.append(row)

        pairs.sort(key=lambda p: abs(p["delta"]), reverse=True)
        return {
            "available": True,
            "message": "",
            "temperature": float(self.state.temperature),
            "unit": "Δ (‰)",
            "labels": labels,
            "elements": elements,
            "is_hydrogen": [e == "H" for e in elements],
            "matrix": rows,
            # Enough of each class that filtering to heavy-only still has a
            # table to show.
            "pairs": pairs[:120],
            "aggregation": self.state.derived_aggregation,
            "per_atom": [
                {"atom_index": s.atom_index, "label": label(s.atom_index), "value": s.value}
                for s in result.sites
            ],
        }

    def qc_text(self, *, structured: bool = False) -> str:
        """QC summary for molecule A."""
        if self.state.mol_a is None:
            return "Load molecule A first."
        report = self._structured_qc
        if structured:
            report = run_qc_for_key(self.provider.path, self.state.mol_a.key)
            self._structured_qc = report
        return qc_panel_text(self.state.mol_a.molecule, structured=report)

    def qc_flags(self):
        """Stored QC flags for molecule A."""
        if self.state.mol_a is None:
            return None
        return stored_qc_flags(self.state.mol_a.molecule)

    def status_summary(self) -> str:
        """What is loaded, for the line beside the dataset picker.

        It used to read ``dataset=qm9s | A=100506 (n=18, QC=pass) | B=(none) |
        beta @ 298.15 K`` -- a debug dump, repeating the dataset the picker
        already shows and the quantity the controls already show. Now it names
        the molecule and nothing the rest of the header says for it.
        """
        state = self.state
        if state.mol_a is None:
            return ""
        flags = self.qc_flags()
        verdict = "passes QC" if flags and flags.qc_pass else "check QC"
        parts = [f"{state.mol_a.key}", f"{state.mol_a.molecule.n_atoms} atoms", verdict]
        if state.mol_b is not None:
            parts.append(f"compared with {state.mol_b.key}")
        if self.match_pool:
            parts.append(f"{len(self.match_pool):,} in the results")
        return " · ".join(parts)


__all__ = ["ExplorerController", "RenderPayload"]
