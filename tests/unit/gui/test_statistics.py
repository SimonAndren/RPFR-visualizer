"""The Dataset page payload.

This page is headed "dataset statistics" while its percentiles come from a
*sample*, so the tests that matter here are about honest labelling as much as
about shape: a page that quietly reported sample percentiles as population ones
would be misleading in exactly the way a reader could not detect.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rpfr_visualizer.gui.config import Resolution
from rpfr_visualizer.gui.session import ExplorerSession
from rpfr_visualizer.gui.statistics import _coverage_note, statistics_payload

pytest.importorskip("rdkit")

SAMPLED = 2


@pytest.fixture
def session(synthetic_h5_path: Path, tmp_path: Path) -> ExplorerSession:
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(synthetic_h5_path, raw / "qm9s_tr_project_1pct.h5")
    (tmp_path / "cache").mkdir()

    catalog = DatasetCatalog()
    catalog.register("qm9s", raw / "qm9s_tr_project_1pct.h5")
    index_path = tmp_path / "cache" / "qm9s_rpfr_index.sqlite"
    build_index(catalog, index_path, progress=False)

    opened = ExplorerSession(
        Resolution(data_dir=tmp_path, index_path=index_path, max_stats_molecules=SAMPLED)
    )
    opened.open()
    return opened


# --- honesty ----------------------------------------------------------------


def test_the_payload_states_how_many_molecules_were_sampled(session: ExplorerSession) -> None:
    """Without this number the percentiles below it read as population values."""
    payload = statistics_payload(session.controller)
    assert payload["sampled_molecules"] == SAMPLED


def test_provenance_reports_the_index_and_its_rdkit(session: ExplorerSession) -> None:
    """Canonical SMILES, InChIKeys and the screening keys all depend on the
    RDKit the index was built with, so a drift has to be visible."""
    provenance = statistics_payload(session.controller)["provenance"]

    assert provenance["index"]
    assert provenance["molecules"] == 5
    assert provenance["rdkit_running"]
    assert provenance["rdkit_drift"] is False


# --- dataset rows -----------------------------------------------------------


def test_dataset_rows_carry_counts_and_qc_census(session: ExplorerSession) -> None:
    rows = statistics_payload(session.controller)["datasets"]

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "qm9s"
    assert row["active"] is True
    assert row["available"] is True
    assert row["n"] == 5
    assert row["n_qc_pass"] + row["n_qc_fail"] + row["n_qc_unknown"] == row["n"]


def test_the_qc_census_reflects_the_fixture(session: ExplorerSession) -> None:
    """The fixture is built with both passing and failing molecules; the census
    has to show both or the Data quality story is untestable."""
    row = statistics_payload(session.controller)["datasets"][0]
    assert row["n_qc_pass"] > 0
    assert row["n_qc_fail"] > 0


def test_structure_percentage_is_a_percentage(session: ExplorerSession) -> None:
    row = statistics_payload(session.controller)["datasets"][0]
    assert 0.0 <= row["structure_pct"] <= 100.0


# --- coverage notes ---------------------------------------------------------


def test_full_structure_coverage_gets_no_note() -> None:
    assert _coverage_note("qm9s", {"n": 10, "n_with_structure": 10}) == ""


def test_no_structures_at_all_says_the_search_is_unanswerable() -> None:
    """A zero-hit search there means "cannot be asked here", not "none here"."""
    note = _coverage_note("unstructured", {"n": 10, "n_with_structure": 0})
    assert "stores no structures" in note
    assert "only be searched by molecule ID" in note


def test_a_partial_gap_elsewhere_is_explained_generically() -> None:
    note = _coverage_note("qm9s", {"n": 10, "n_with_structure": 7})
    assert "no stored structure" in note
    assert "found by ID" in note


# --- elements and groups ----------------------------------------------------


def test_element_rows_are_ordered_percentiles(session: ExplorerSession) -> None:
    rows = statistics_payload(session.controller)["elements"]

    assert rows, "the colour scale normalises against these; an empty list is a bug"
    for row in rows:
        assert row["p5"] <= row["p50"] <= row["p95"]


def test_group_rows_keep_empty_groups(session: ExplorerSession) -> None:
    """A chemist looking for anhydrides should learn there are none, rather than
    wonder where the row went."""
    rows = statistics_payload(session.controller)["groups"]

    assert rows
    assert any(row["count"] == 0 for row in rows)
    assert all({"name", "description", "count"} <= set(row) for row in rows)


# --- header -----------------------------------------------------------------


def test_the_temperature_grid_is_reported_with_the_chosen_temperature(
    session: ExplorerSession,
) -> None:
    payload = statistics_payload(session.controller)

    assert payload["dataset"] == "qm9s"
    assert isinstance(payload["temperature"], float)
    assert payload["temperatures"]
    assert all(isinstance(t, float) for t in payload["temperatures"])
