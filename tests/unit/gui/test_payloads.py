"""Tests for the session and the JSON projection layer."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import fields
from pathlib import Path

import pytest

from rpfr_visualizer.analysis.models import AnalysisKind
from rpfr_visualizer.analysis.temperature import DEFAULT_METHOD, INTERPOLATION_METHODS
from rpfr_visualizer.gui.config import Resolution
from rpfr_visualizer.gui.payloads import (
    WRITABLE_FIELDS,
    coerce_setting,
    hit_payload,
    jsonable,
    match_reason,
    rank_payload,
    state_payload,
)
from rpfr_visualizer.gui.session import Busy, ExplorerSession, NotReady
from rpfr_visualizer.viz.explorer import ExplorerState

pytest.importorskip("rdkit")


@pytest.fixture
def session(synthetic_h5_path: Path, tmp_path: Path) -> ExplorerSession:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(synthetic_h5_path, raw / "qm9s_tr_project_1pct.h5")
    session = ExplorerSession(Resolution(data_dir=tmp_path, max_stats_molecules=2))
    session.open()
    return session


# ----------------------------------------------------------------- the contract


def test_writable_fields_track_the_dataclass() -> None:
    """The guard against drift.

    This fails the moment a field is added to ExplorerState, which is the point:
    a setting the server silently refuses to accept is worse than a loud one.
    """
    expected = {f.name for f in fields(ExplorerState)} - {"dataset", "mol_a", "mol_b"}
    assert set(WRITABLE_FIELDS) == expected


def test_state_payload_is_json_serialisable(session: ExplorerSession) -> None:
    """One assertion that catches every stray NumPy scalar."""
    with session.use() as controller:
        controller.load(controller.provider.list_keys()[0])
        payload = state_payload(controller)

    assert json.loads(json.dumps(payload)) == payload


def test_state_payload_never_leaks_a_molecule(session: ExplorerSession) -> None:
    with session.use() as controller:
        controller.load(controller.provider.list_keys()[0])
        payload = state_payload(controller)

    slot = payload["slots"]["a"]
    assert set(slot) == {"dataset", "key", "label", "n_atoms", "smiles", "qc_pass"}
    # A serialized Molecule would drag its per-site records along with it, and
    # each site carries a beta vector over the whole temperature grid. Matched
    # as a JSON key rather than a bare substring, so an unrelated setting whose
    # name happens to end in "sites" does not read as a leak.
    blob = json.dumps(payload)
    assert '"sites"' not in blob
    assert '"atom_symbol"' not in blob
    assert len(blob) < 8_000


def test_temperatures_are_offered_in_order(session: ExplorerSession) -> None:
    with session.use() as controller:
        payload = state_payload(controller)
    assert payload["temperatures"] == sorted(payload["temperatures"])


def test_anchor_options_are_populated(session: ExplorerSession) -> None:
    """The picker the widget layer never had.

    Choosing the anchor reference mode used to anchor to atom 0 silently.
    """
    with session.use() as controller:
        controller.load(controller.provider.list_keys()[0])
        payload = state_payload(controller)

    options = payload["anchor_options"]["a"]
    assert options
    assert all(set(o) == {"label", "value"} for o in options)
    assert payload["anchor_options"]["b"] == []


# ----------------------------------------------------------------------- jsonable


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), None),
        (float("inf"), None),
        (1.5, 1.5),
        (None, None),
        (True, True),
        ("x", "x"),
    ],
)
def test_jsonable_handles_non_finite_floats(value, expected) -> None:
    """NaN must become null, not the bare NaN token json.dumps would write.

    A site with no value has to arrive as null so the client shows an em dash
    rather than a zero.
    """
    assert jsonable(value) == expected


def test_jsonable_unwraps_numpy_scalars() -> None:
    import numpy as np

    assert jsonable(np.float64(2.5)) == 2.5
    assert jsonable(np.int64(3)) == 3
    assert jsonable(np.float64("nan")) is None


# ------------------------------------------------------------------- coercion


@pytest.mark.parametrize(
    ("name", "given", "expected"),
    [
        ("show_hydrogens", "false", False),
        ("show_hydrogens", 1, True),
        ("temperature", "298.15", 298.15),
        ("top_n", "25", 25),
        ("element", "C", "C"),
        ("element", "all", None),
        ("element", "", None),
        ("kind", "ln_alpha", AnalysisKind.LN_ALPHA),
        ("reference_mode", "standard", "standard"),
        ("standard_overrides", {"C": "1.1"}, {"C": 1.1}),
    ],
)
def test_coerce_setting_by_declared_type(name, given, expected) -> None:
    assert coerce_setting(name, given) == expected


def test_coercing_the_dataset_points_at_the_right_action() -> None:
    """Switching datasets also resets the selection, so it is not a setting."""
    with pytest.raises(ValueError, match="its own action"):
        coerce_setting("dataset", "qm9s")


def test_unknown_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown setting"):
        coerce_setting("colour_scheme", "dark")


def test_quantities_the_view_cannot_draw_are_rejected() -> None:
    """clumping and mass_law are real AnalysisKind members, not per-site ones.

    Validating against the enum would let them through and then fail deeper in,
    so the check is against what this view actually offers.
    """
    for kind in ("clumping", "mass_law"):
        with pytest.raises(ValueError, match="Cannot show"):
            coerce_setting("kind", kind)
    with pytest.raises(ValueError, match="beta"):
        coerce_setting("kind", "nonsense")


def test_unknown_element_is_rejected() -> None:
    with pytest.raises(ValueError, match="No data for element"):
        coerce_setting("element", "Fe")


@pytest.mark.parametrize(
    "method", ["poly5", "poly4", "poly3", "poly2", "pchip", "PCHIP", " poly3 "]
)
def test_an_offered_interpolation_method_is_accepted(method: str) -> None:
    assert coerce_setting("beta_method", method) == method.strip().lower()


@pytest.mark.parametrize("method", ["poly1", "poly6", "spline", "linear"])
def test_an_unoffered_interpolation_method_is_rejected(method: str) -> None:
    """The vocabulary lives in analysis.temperature; this only checks it is
    consulted, so the picker cannot offer something the fit cannot do."""
    with pytest.raises(ValueError, match="Unknown interpolation method"):
        coerce_setting("beta_method", method)


@pytest.mark.parametrize("blank", ["", None])
def test_a_blank_interpolation_method_falls_back_to_the_default(blank: str | None) -> None:
    assert coerce_setting("beta_method", blank) == DEFAULT_METHOD


def test_the_state_payload_offers_every_interpolation_method(session: ExplorerSession) -> None:
    """The picker is built from this list, so it has to carry the labels too."""
    with session.use() as controller:
        payload = state_payload(controller)

    offered = payload["beta_methods"]
    assert [m["value"] for m in offered] == [v for v, _l in INTERPOLATION_METHODS]
    assert all(m["label"] for m in offered)
    assert payload["settings"]["beta_method"] == DEFAULT_METHOD


# ------------------------------------------------------------------ hit payload


def test_hit_payload_reports_truncation_honestly() -> None:
    """There is no pagination, so the cap has to be visible."""
    hits = [_hit(f"k{i}") for i in range(5)]
    payload = hit_payload(hits, limit=5, elapsed_ms=12.34)

    assert payload["shown"] == 5
    assert payload["capped"] is True
    assert payload["limit"] == 5
    assert payload["elapsed_ms"] == 12.3

    assert hit_payload(hits[:2], limit=5, elapsed_ms=1.0)["capped"] is False


def test_hit_payload_carries_synonyms_and_a_reason() -> None:
    payload = hit_payload([_hit("k", names=("ethanol",))], limit=None, elapsed_ms=0.0)
    row = payload["hits"][0]
    assert row["names"] == ["ethanol"]
    assert row["why"] == "matched name"


@pytest.mark.parametrize(
    ("match_type", "expected"),
    [
        ("key", "matched ID"),
        ("formula", "matched formula"),
        ("canonical_smiles", "same structure"),
        ("substructure", "contains this structure"),
        ("smarts", "contains this pattern"),
        ("functional_group", "contains this group"),
        ("name:local", "matched name"),
        ("name:prefix", "name starts with this"),
    ],
)
def test_match_reasons_are_plain_language(match_type, expected) -> None:
    assert match_reason(match_type) == expected


# ----------------------------------------------------------------- rank payload


def test_rank_payload_reports_its_scope(session: ExplorerSession) -> None:
    with session.use() as controller:
        controller.load(controller.provider.list_keys()[0])
        controller.state.element = None
        controller.rank()
        payload = rank_payload(controller)

    assert payload["scope"] == "molecule"
    assert payload["pool_size"] == 0
    assert json.loads(json.dumps(payload)) == payload
    if payload["rows"]:
        assert set(payload["rows"][0]) == {
            "dataset",
            "key",
            "atom_index",
            "element",
            "value",
            "r1",
            "molecule_smiles",
        }


# --------------------------------------------------------------------- session


def test_busy_is_reported_not_queued(session: ExplorerSession) -> None:
    """Failing fast beats stacking six requests behind one slow query."""
    session._lock.acquire()
    try:
        with pytest.raises(Busy, match="still working"):
            with session.use():
                pass
    finally:
        session._lock.release()

    # And the lock is usable again afterwards.
    with session.use() as controller:
        assert controller is not None


def test_a_read_can_wait_for_a_lock_that_is_about_to_be_released(
    session: ExplorerSession,
) -> None:
    """Read-only routes take turns rather than refusing.

    The client serialises its own calls, but the browser does not: an
    <iframe src> is fetched on its own schedule, alongside whatever the client
    asks for next. Failing that instantly put a raw JSON error where the 3D
    structure should be.
    """
    import threading

    session._lock.acquire()
    threading.Timer(0.1, session._lock.release).start()

    started = time.monotonic()
    with session.use(wait=5.0) as controller:
        assert controller is not None
    assert time.monotonic() - started >= 0.05, "it did not actually wait"


def test_a_read_still_gives_up_rather_than_queueing(session: ExplorerSession) -> None:
    """Bounded, so a stuck session answers instead of hanging the browser."""
    session._lock.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(Busy, match="still working"):
            with session.use(wait=0.2):
                pass
        assert time.monotonic() - started < 2.0
    finally:
        session._lock.release()


def test_snapshot_is_readable_without_the_lock(session: ExplorerSession) -> None:
    """This is what lets a client see `busy` while the lock is held."""
    with session.use() as controller:
        session.touch(state_payload)
        assert session.snapshot()["busy"] is True
        assert controller is not None

    assert session.snapshot()["busy"] is False


def test_open_builds_the_first_snapshot(session: ExplorerSession) -> None:
    """A session that never touched its snapshot would report itself not ready.

    /api/state reads the snapshot without taking the lock, so opening has to
    populate it.
    """
    assert session.revision >= 1
    snapshot = session.snapshot()
    assert snapshot["ready"] is True
    assert snapshot["dataset"]


def test_revision_advances_on_every_change(session: ExplorerSession) -> None:
    before = session.revision
    with session.use():
        session.touch(state_payload)
        session.touch(state_payload)
    assert session.revision == before + 2
    assert session.snapshot()["revision"] == before + 2


def test_a_session_without_data_is_not_ready() -> None:
    session = ExplorerSession(Resolution())
    assert not session.ready
    with pytest.raises(NotReady):
        session.require_controller()
    with pytest.raises(NotReady, match="No data directory"):
        session.open()


def test_opening_reports_what_was_found(session: ExplorerSession) -> None:
    """No index means no search, but every molecule is still reachable by ID.

    The notes go on screen: "7 datasets and an 881,757-molecule index" and "one
    file, no search" are very different situations to be in.
    """
    assert session.ready
    assert any("Opened" in w for w in session.warnings)
    assert any("Search index" in w for w in session.warnings)
    assert any("open molecules by ID" in w for w in session.warnings)


def _hit(key: str, *, names: tuple[str, ...] = ()):
    from rpfr_visualizer.io.search.models import MoleculeIndexEntry, MoleculeSearchHit

    entry = MoleculeIndexEntry(
        dataset="qm9s",
        key=key,
        smiles="CCO",
        canonical_smiles="CCO",
        n_atoms=3,
        label=f"qm9s:{key}",
        formula="C2H6O",
        names=names,
    )
    return MoleculeSearchHit(entry, "ethanol", "name:local" if names else "key")


# ------------------------------------------------------------- subsample badge


class _Provider:
    def __init__(self, fraction):
        self.subsample_fraction = fraction


class _Controller:
    def __init__(self, fraction):
        self.provider = _Provider(fraction)


@pytest.mark.parametrize(
    ("fraction", "label"),
    [(0.01, "1% subsample"), (0.05, "5% subsample"), (0.001, "0.1% subsample")],
)
def test_a_subsample_is_named_in_the_header(fraction: float, label: str) -> None:
    """An empty search on a 1% file says nothing about QM9s. The header has to
    say which of the two is open, or that is easy to forget."""
    from rpfr_visualizer.gui.payloads import subsample_label

    assert subsample_label(_Controller(fraction)) == label


@pytest.mark.parametrize("fraction", [None, 0.0, 1.0])
def test_a_complete_file_carries_no_badge(fraction: float | None) -> None:
    from rpfr_visualizer.gui.payloads import subsample_label

    assert subsample_label(_Controller(fraction)) == ""


def test_the_state_payload_says_whether_this_is_a_subsample(session: ExplorerSession) -> None:
    with session.use() as controller:
        payload = state_payload(controller)
    assert "subsample" in payload
    assert isinstance(payload["subsample"], str)


def test_epsilon_is_not_offered_per_site() -> None:
    """The per-mil quantity is 1000 ln alpha, the isotope-geochemistry
    convention. Offering epsilon beside it put two near-identical per-mil
    numbers in one dropdown with nothing to say which to use. It stays in the
    Python API; the explorer just does not offer it."""
    from rpfr_visualizer.gui.payloads import QUANTITIES

    offered = [value for value, _label in QUANTITIES]
    assert offered == ["beta", "alpha", "ln_alpha"]
    with pytest.raises(ValueError, match="Cannot show"):
        coerce_setting("kind", "epsilon")
