"""Tests for the explorer controller — the panel's behaviour without any UI."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.analysis.reference import (
    ElementMeanReference,
    MoleculeReference,
    SelfReference,
    StandardReference,
)
from rpfr_visualizer.io.providers import DatasetCatalog, HDF5Provider
from rpfr_visualizer.viz.explorer import ExplorerController, ExplorerState

pytest.importorskip("rdkit")


@pytest.fixture
def controller(synthetic_h5_path: Path) -> ExplorerController:
    provider = HDF5Provider(synthetic_h5_path)
    return ExplorerController(provider, max_stats_molecules=2)


@pytest.fixture
def two_dataset_controller(synthetic_h5_path: Path) -> ExplorerController:
    catalog = DatasetCatalog()
    catalog.register("first", synthetic_h5_path)
    catalog.register("second", synthetic_h5_path)
    return ExplorerController(catalog, dataset="first", max_stats_molecules=2)


def test_loads_into_slots(controller: ExplorerController) -> None:
    keys = controller.provider.list_keys()
    a = controller.load(keys[0])
    assert controller.state.mol_a is a
    assert a.label == f"{controller.dataset}:{keys[0]}"

    controller.load(keys[1], slot="b")
    assert controller.state.mol_b is not None
    assert controller.state.mol_b.key == keys[1]

    controller.clear("b")
    assert controller.state.mol_b is None
    with pytest.raises(ValueError, match="slot must be"):
        controller.load(keys[0], slot="c")


def test_beta_needs_no_reference(controller: ExplorerController) -> None:
    controller.load(controller.provider.list_keys()[0])
    controller.state.kind = AnalysisKind.BETA
    controller.state.reference_mode = "standard"
    request = controller.build_request()
    assert request.reference is None  # beta is absolute regardless of mode


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("none", ElementMeanReference),
        ("element_mean", ElementMeanReference),
        ("standard", StandardReference),
        ("self", SelfReference),
    ],
)
def test_reference_modes_build_the_right_reference(
    controller: ExplorerController, mode: str, expected: type
) -> None:
    controller.load(controller.provider.list_keys()[0])
    controller.state.kind = AnalysisKind.ALPHA
    controller.state.reference_mode = mode  # type: ignore[assignment]
    assert isinstance(controller.build_request().reference, expected)


def test_mol_b_reference_requires_b(controller: ExplorerController) -> None:
    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.state.kind = AnalysisKind.ALPHA
    controller.state.reference_mode = "mol_b"
    with pytest.raises(ValueError, match="Molecule B is required"):
        controller.build_request()

    controller.load(keys[1], slot="b")
    assert isinstance(controller.build_request().reference, MoleculeReference)


def test_analyze_respects_element_filter(controller: ExplorerController) -> None:
    controller.load(controller.provider.list_keys()[0])
    controller.state.element = "C"
    result = controller.analyze()
    assert result.sites
    assert all(site.element == "C" for site in result.sites)


def test_analyze_without_a_molecule_is_an_error(controller: ExplorerController) -> None:
    with pytest.raises(ValueError, match="Load molecule A first"):
        controller.analyze()


def test_off_grid_temperature_is_flagged(controller: ExplorerController) -> None:
    controller.load(controller.provider.list_keys()[0])
    grid = controller.temperatures

    controller.state.temperature = grid[0]
    assert controller.state.temperature_is_on_grid(grid)
    assert controller.warnings() == ()

    controller.state.temperature = float(grid[0]) + 7.5
    assert not controller.state.temperature_is_on_grid(grid)
    warnings = controller.warnings()
    assert len(warnings) == 1
    assert "fitted" in warnings[0]


def test_slot_b_shows_the_selected_quantity_under_a_shared_reference(
    controller: ExplorerController,
) -> None:
    """B is comparable with A when the reference does not depend on A."""
    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.load(keys[1], slot="b")
    controller.state.kind = AnalysisKind.EPSILON
    controller.state.reference_mode = "standard"

    payload = controller.render_payload()
    assert payload.kind_b == AnalysisKind.EPSILON
    assert payload.note_b == ""  # dataset standard puts both on one scale
    assert payload.caption_b


def test_element_mean_warns_that_two_molecules_are_not_comparable(
    controller: ExplorerController,
) -> None:
    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.load(keys[1], slot="b")
    controller.state.kind = AnalysisKind.EPSILON
    controller.state.reference_mode = "element_mean"

    payload = controller.render_payload()
    assert payload.kind_b == AnalysisKind.EPSILON  # computable, just not comparable
    assert "its own mean" in payload.note_b


@pytest.mark.parametrize("mode", ["self", "mol_b"])
def test_slot_b_falls_back_to_beta_and_says_why(controller: ExplorerController, mode: str) -> None:
    """A reference that depends on A cannot be applied to B.

    Silently showing a differently-defined number under the same colour scale
    is the one outcome that is not acceptable, so the payload carries the
    reason.
    """
    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.load(keys[1], slot="b")
    controller.state.kind = AnalysisKind.EPSILON
    controller.state.reference_mode = mode  # type: ignore[assignment]

    payload = controller.render_payload()
    assert payload.kind_b == AnalysisKind.BETA
    # ...and names what it replaced, so the substitution is never silent.
    assert "Showing β, not ε" in payload.note_b
    assert payload.label_b.startswith(f"{controller.dataset}:{keys[1]} [beta]")


def test_beta_needs_no_slot_b_caveat(controller: ExplorerController) -> None:
    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.load(keys[1], slot="b")
    controller.state.kind = AnalysisKind.BETA

    payload = controller.render_payload()
    assert payload.kind_b == AnalysisKind.BETA
    assert payload.note_b == ""


def test_render_payload_carries_the_colour_scale_caption(
    controller: ExplorerController,
) -> None:
    """The sentence explaining what the colours mean must reach the payload."""
    loaded = controller.load(controller.provider.list_keys()[0])
    element = next(s.element for s in loaded.molecule.sites if s.element != "H")
    controller.state.element = element

    payload = controller.render_payload()
    assert payload.caption, "caption is empty — the legend never reaches the user"
    assert element in payload.caption
    assert "5th–95th percentile" in payload.caption


def test_display_grid_is_sorted_but_stored_order_is_not(
    controller: ExplorerController,
) -> None:
    """Pickers show ascending temperatures; the stored order stays untouched.

    The stored grid appends the chemistry temperatures after the sweep, so it
    ends ``420, 273.15, 298.15``. That order is positionally coupled to the beta
    arrays via ``grid_index()``, so sorting it in place would silently return
    the wrong beta for every temperature.
    """
    stored = controller.temperatures
    assert stored != tuple(sorted(stored)), "fixture no longer exercises an unsorted grid"

    shown = controller.temperatures_for_display
    assert shown == tuple(sorted(stored))
    assert set(shown) == set(stored)
    assert controller.temperatures == stored


def test_ranking_uses_pool_when_present(controller: ExplorerController) -> None:
    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.state.element = "C"
    controller.state.top_n = 10

    ranked_from_a = controller.rank()
    assert ranked_from_a
    assert {r.key for r in ranked_from_a} == {keys[0]}

    controller.load_pool(keys)
    assert len(controller.match_pool) == len(keys)
    ranked_from_pool = controller.rank()
    # Only molecules that actually have a carbon site can be ranked on C.
    carbon_keys = {
        m.key
        for m in controller.match_pool
        if any(site.element == "C" for site in m.molecule.sites)
    }
    assert {r.key for r in ranked_from_pool} == carbon_keys
    assert len(carbon_keys) > 1

    controller.state.use_match_pool = False
    assert {r.key for r in controller.rank()} == {keys[0]}


def test_pool_load_is_idempotent_and_clearable(controller: ExplorerController) -> None:
    keys = controller.provider.list_keys()
    assert controller.load_pool(keys) == len(keys)
    assert controller.load_pool(keys, clear=False) == len(keys)  # no duplicates
    controller.clear_pool()
    assert controller.match_pool == []


def test_pool_skips_keys_that_do_not_load(controller: ExplorerController) -> None:
    keys = controller.provider.list_keys()
    assert controller.load_pool([keys[0], "not-a-real-key"]) == 1


def test_heavy_only_filters_hydrogens(controller: ExplorerController) -> None:
    controller.load(controller.provider.list_keys()[0])
    controller.state.element = None
    controller.state.top_n = 50

    controller.state.heavy_only = False
    assert any(r.element == "H" for r in controller.rank())

    controller.state.heavy_only = True
    assert all(r.element != "H" for r in controller.rank())


def test_render_payload_includes_b_when_loaded(controller: ExplorerController) -> None:
    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.state.element = "C"

    payload = controller.render_payload()
    assert payload.view_a is not None
    assert payload.view_b is None
    assert keys[0] in payload.label_a
    assert "[C]" in payload.label_a

    controller.load(keys[1], slot="b")
    payload = controller.render_payload()
    assert payload.view_b is not None
    assert keys[1] in payload.label_b

    # The reference-site annotation is only meaningful when B *is* the
    # reference; under any other mode it described a role B does not have.
    assert "reference site" not in payload.label_b
    controller.state.kind = AnalysisKind.EPSILON
    controller.state.reference_mode = "mol_b"
    assert "reference site 0" in controller.render_payload().label_b


def test_render_payload_carries_warnings(controller: ExplorerController) -> None:
    controller.load(controller.provider.list_keys()[0])
    controller.state.temperature = float(controller.temperatures[0]) + 3.0
    assert any("fitted" in w for w in controller.render_payload().warnings)


def test_qc_text_and_status(controller: ExplorerController) -> None:
    assert "Load molecule A first" in controller.qc_text()
    assert controller.qc_flags() is None

    keys = controller.provider.list_keys()
    controller.load(keys[0])
    assert "verdict" in controller.qc_text()
    assert controller.qc_flags() is not None

    status = controller.status_summary()
    assert keys[0] in status


def test_dataset_switching_resets_selection(two_dataset_controller: ExplorerController) -> None:
    controller = two_dataset_controller
    assert controller.available_datasets() == ("first", "second")
    assert controller.dataset == "first"

    keys = controller.provider.list_keys()
    controller.load(keys[0])
    controller.load_pool(keys)

    controller.use_dataset("second")
    assert controller.dataset == "second"
    # Keys belong to the dataset they came from, so selections cannot carry over.
    assert controller.state.mol_a is None
    assert controller.match_pool == []

    with pytest.raises(KeyError, match="not registered"):
        controller.use_dataset("nope")


def test_switching_to_the_same_dataset_keeps_selection(
    two_dataset_controller: ExplorerController,
) -> None:
    controller = two_dataset_controller
    controller.load(controller.provider.list_keys()[0])
    controller.use_dataset("first")
    assert controller.state.mol_a is not None


def test_stats_are_built_per_dataset_on_demand(
    two_dataset_controller: ExplorerController,
) -> None:
    controller = two_dataset_controller
    assert controller._stats_by_dataset == {}  # nothing computed at construction
    assert controller.stats is not None
    assert set(controller._stats_by_dataset) == {"first"}

    controller.use_dataset("second")
    assert controller.stats is not None
    assert set(controller._stats_by_dataset) == {"first", "second"}


def test_empty_catalog_is_rejected() -> None:
    with pytest.raises(ValueError, match="no registered datasets"):
        ExplorerController(DatasetCatalog())


def test_unknown_dataset_is_rejected(synthetic_h5_path: Path) -> None:
    catalog = DatasetCatalog()
    catalog.register("only", synthetic_h5_path)
    with pytest.raises(KeyError, match="not in catalog"):
        ExplorerController(catalog, dataset="missing")


def test_state_can_be_supplied(synthetic_h5_path: Path) -> None:
    state = ExplorerState(element="N", top_n=3, kind=AnalysisKind.ALPHA)
    controller = ExplorerController(
        HDF5Provider(synthetic_h5_path), state=state, max_stats_molecules=2
    )
    assert controller.state is state
    assert controller.state.element == "N"
    assert controller.state.dataset  # controller fills the dataset in


def test_group_search_does_not_block_ranking_a_different_element(
    controller: ExplorerController,
) -> None:
    """Selecting a functional group must not empty an unrelated element ranking.

    A group preset chooses *molecules*. Applying it again as a site filter meant
    that searching for alcohols (an oxygen group) and then ranking carbon sites
    returned nothing at all.
    """
    keys = controller.provider.list_keys()
    controller.load_pool(keys)
    controller.state.element = "C"
    controller.state.top_n = 10

    baseline = controller.rank()
    assert baseline

    controller.state.fg_preset = "alcohol"  # an oxygen group
    assert controller.rank() == baseline


def test_explicit_site_filters_still_narrow_ranking(controller: ExplorerController) -> None:
    """A typed R1/SMARTS filter is site-level and must still apply."""
    controller.load_pool(controller.provider.list_keys())
    controller.state.element = "C"
    controller.state.top_n = 20
    assert controller.rank()

    controller.state.fg_text = "smarts:[NX3]"  # no carbon site sits in an amine here
    controller.state.fg_preset = None
    narrowed = controller.rank()
    assert len(narrowed) < 20


def test_statistics_are_persisted_between_controllers(
    synthetic_h5_path: Path, tmp_path, monkeypatch
) -> None:
    """The stats sample is written once and reused, not recomputed per session."""
    monkeypatch.chdir(tmp_path)  # cache lands under ./data/cache/stats

    first = ExplorerController(HDF5Provider(synthetic_h5_path), max_stats_molecules=2)
    stats = first.stats
    cache_file = first._stats_cache_path(first.dataset)
    assert cache_file.exists()

    calls: list[int] = []
    from rpfr_visualizer.analysis.element_stats import ElementStatsCache

    original = ElementStatsCache.from_hdf5

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ElementStatsCache, "from_hdf5", staticmethod(counting))

    second = ExplorerController(HDF5Provider(synthetic_h5_path), max_stats_molecules=2)
    reloaded = second.stats
    assert calls == []  # served from the cache file
    assert reloaded.elements() == stats.elements()
    assert reloaded.n_sites == stats.n_sites


def test_stats_cache_path_distinguishes_sample_size(synthetic_h5_path: Path) -> None:
    small = ExplorerController(HDF5Provider(synthetic_h5_path), max_stats_molecules=2)
    large = ExplorerController(HDF5Provider(synthetic_h5_path), max_stats_molecules=50)
    assert small._stats_cache_path("d") != large._stats_cache_path("d")


def test_explicit_data_dir_needs_no_chdir(synthetic_h5_path: Path, tmp_path) -> None:
    """Caches must be findable without the process sitting in the repo root.

    The default is relative, so anything launched from an arbitrary directory
    (a GUI, a console script) would silently miss an existing index and rebuild
    statistics already on disk. Note there is deliberately no ``chdir`` here.
    """
    controller = ExplorerController(
        HDF5Provider(synthetic_h5_path),
        data_dir=tmp_path / "somewhere",
        max_stats_molecules=2,
    )

    cache_file = controller._stats_cache_path(controller.dataset)
    assert cache_file.is_absolute()
    assert cache_file.parent == tmp_path / "somewhere" / "cache" / "stats"

    assert controller.stats is not None  # builds and persists
    assert cache_file.exists()


def test_data_dir_defaults_to_the_relative_legacy_path(synthetic_h5_path: Path) -> None:
    controller = ExplorerController(HDF5Provider(synthetic_h5_path), max_stats_molecules=2)
    assert controller.data_dir == Path("data")


# ------------------------------------------------ one wording, everywhere


@pytest.mark.parametrize(
    ("method", "phrase"),
    [
        ("local2", "local quadratic through the four nearest stored points"),
        ("pchip", "monotone spline"),
        ("poly5", "degree-5 polynomial"),
    ],
)
def test_the_temperature_note_names_the_fit_actually_in_use(
    controller: ExplorerController, method: str, phrase: str
) -> None:
    """It said "a fit ... over the 15 stored points" whatever the fit was --
    still describing a global polynomial after the default became local."""
    controller.load(controller.provider.list_keys()[0])
    controller.state.beta_method = method
    controller.state.temperature = 210.0
    note = controller.temperature_note()
    assert "not a stored temperature" in note
    assert phrase in note


def test_a_stored_temperature_says_it_is_read(controller: ExplorerController) -> None:
    controller.state.temperature = 298.15
    assert controller.temperature_note() == "298.15 K is a stored temperature."


def test_the_header_line_is_not_a_debug_dump(controller: ExplorerController) -> None:
    """It read "dataset=qm9s | A=1 (n=5, QC=pass) | B=(none) | beta @ 298.15 K",
    repeating what the picker and the controls already show."""
    assert controller.status_summary() == ""
    key = controller.provider.list_keys()[0]
    controller.load(key)
    status = controller.status_summary()
    assert status.startswith(key)
    for leftover in ("dataset=", "A=", "B=", "(none)", "beta @", "QC="):
        assert leftover not in status
