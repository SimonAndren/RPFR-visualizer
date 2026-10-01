"""Filtering a search by the stored QC verdict.

The verdict is three-valued -- passed, failed, never recorded -- and the whole
point of this filter is that the third case is not quietly folded into one of
the other two.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from rpfr_visualizer.io.molecule_index import MoleculeIndex
from rpfr_visualizer.io.schema import TEMPERATURES_K
from rpfr_visualizer.io.search.chem import parse_qc_token, stored_qc_pass
from rpfr_visualizer.io.search.models import RowFilter, SizeFilter

pytest.importorskip("rdkit")

#: key, SMILES, QC state -- "pass", "fail", or "unknown" (no flags recorded).
MOLECULES = (
    ("1", "CCO", "pass"),
    ("2", "CC(=O)O", "pass"),
    ("3", "c1ccccc1", "fail"),
    ("4", "CC#N", "unknown"),
)


def _write(path: Path) -> Path:
    n_temps = len(TEMPERATURES_K)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.0"
        h5.attrs["dataset"] = "qc"
        h5.attrs["temperatures_K"] = json.dumps(list(TEMPERATURES_K))
        for key, smiles, state in MOLECULES:
            group = h5.create_group(key)
            group.attrs["mol_id"] = key
            group.attrs["smiles"] = smiles
            group.attrs["n_atoms"] = len(smiles)
            if state != "unknown":
                ok = state == "pass"
                for flag in ("freq_qc_ok", "rpfr_qc_ok", "tr_converged", "rdkit_ok"):
                    group.attrs[flag] = ok
                group.attrs["smiles_order_match"] = ok
                group.attrs["n_imag_vib"] = 0 if ok else 1
            group.create_dataset("rpfr_single", data=np.ones((len(smiles), n_temps)))
            group.create_dataset("atom_symbol", data=np.array(["C"] * len(smiles), dtype="S2"))
    return path


@pytest.fixture
def indexes(tmp_path: Path):
    """The same mixed-verdict data, in memory and as a built index."""
    from rpfr_visualizer.io.providers import DatasetCatalog
    from rpfr_visualizer.io.search import build_index

    catalog = DatasetCatalog()
    catalog.register("qc", _write(tmp_path / "qc.h5"))
    out = tmp_path / "qc.sqlite"
    build_index(catalog, out, progress=False)

    memory = MoleculeIndex.build_from_catalog(catalog)
    memory.allow_pubchem = False
    return memory, MoleculeIndex.open(out, allow_pubchem=False)


def keys(hits) -> set[str]:
    return {h.entry.key for h in hits}


# --- grammar ----------------------------------------------------------------


@pytest.mark.parametrize("token", ["qc:pass", "QC:PASS", " qc:passed ", "qc:ok"])
def test_pass_spellings(token: str) -> None:
    parsed = parse_qc_token(token)
    assert parsed is not None and parsed.qc_pass is True


@pytest.mark.parametrize("token", ["qc:fail", "qc:failed", "QC:Fail"])
def test_fail_spellings(token: str) -> None:
    parsed = parse_qc_token(token)
    assert parsed is not None and parsed.qc_pass is False


@pytest.mark.parametrize("token", ["qc:maybe", "qc:", "qc", "ethanol", "C:5-9", ""])
def test_non_qc_tokens_fall_through(token: str) -> None:
    """Anything that is not a verdict must reach the normal classifier rather
    than being guessed at."""
    assert parse_qc_token(token) is None


# --- filter semantics -------------------------------------------------------


def test_an_empty_filter_is_falsy_but_a_qc_one_is_not() -> None:
    assert not RowFilter()
    assert RowFilter(qc_pass=False)


def test_merge_lets_a_later_verdict_win() -> None:
    merged = RowFilter(qc_pass=True).merge(RowFilter(qc_pass=False))
    assert merged.qc_pass is False


def test_merge_keeps_an_earlier_verdict_when_the_later_has_none() -> None:
    merged = RowFilter(qc_pass=True).merge(RowFilter(element_counts=(("C", 1, 2),)))
    assert merged.qc_pass is True


def test_describe_reads_as_a_reason() -> None:
    assert RowFilter(qc_pass=True).describe() == "QC pass"
    assert RowFilter(qc_pass=False).describe() == "QC fail"
    assert "QC pass" in RowFilter(element_counts=(("C", 5, 9),), qc_pass=True).describe()


def test_size_filter_is_still_importable_under_its_old_name() -> None:
    assert SizeFilter is RowFilter


# --- derivation -------------------------------------------------------------


def test_stored_qc_pass_is_none_when_nothing_is_recorded(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "x.h5", "w") as h5:
        group = h5.create_group("1")
        assert stored_qc_pass(group) is None


def test_an_imaginary_frequency_fails_even_when_the_flags_are_set(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "x.h5", "w") as h5:
        group = h5.create_group("1")
        group.attrs["freq_qc_ok"] = True
        group.attrs["rpfr_qc_ok"] = True
        group.attrs["n_imag_vib"] = 1
        assert stored_qc_pass(group) is False


# --- searching --------------------------------------------------------------


@pytest.mark.parametrize("backend", [0, 1], ids=["memory", "sqlite"])
def test_qc_pass_selects_only_recorded_passes(indexes, backend: int) -> None:
    assert keys(indexes[backend].search("qc:pass", limit=None)) == {"1", "2"}


@pytest.mark.parametrize("backend", [0, 1], ids=["memory", "sqlite"])
def test_qc_fail_selects_only_recorded_failures(indexes, backend: int) -> None:
    assert keys(indexes[backend].search("qc:fail", limit=None)) == {"3"}


@pytest.mark.parametrize("backend", [0, 1], ids=["memory", "sqlite"])
def test_an_unrecorded_verdict_matches_neither(indexes, backend: int) -> None:
    """Molecule 4 records no QC flags. "Not known to have failed" is a weaker
    claim than "passed", so it must not be returned by either branch."""
    index = indexes[backend]
    assert "4" not in keys(index.search("qc:pass", limit=None))
    assert "4" not in keys(index.search("qc:fail", limit=None))
    assert "4" in keys(index.search("CC#N", limit=None))


@pytest.mark.parametrize("backend", [0, 1], ids=["memory", "sqlite"])
def test_qc_narrows_rather_than_widens(indexes, backend: int) -> None:
    """A comma normally means OR. This one means AND, so adding it can only
    ever remove hits."""
    index = indexes[backend]
    unfiltered = keys(index.search("fg:alcohol", limit=None))
    filtered = keys(index.search("fg:alcohol, qc:pass", limit=None))
    assert filtered <= unfiltered


@pytest.mark.parametrize("backend", [0, 1], ids=["memory", "sqlite"])
def test_qc_combines_with_a_size_token(indexes, backend: int) -> None:
    index = indexes[backend]
    assert keys(index.search("qc:pass, C:2-2", limit=None)) == {"1", "2"}


# --- OR'd tokens name molecules, not (molecule, reason) pairs ---------------


@pytest.mark.parametrize("backend", [0, 1], ids=["memory", "sqlite"])
def test_a_molecule_matching_two_ord_tokens_is_listed_once(indexes, backend: int) -> None:
    """Dedup is per molecule, not per (molecule, query).

    Molecule 1 is ethanol: it matches both `CCO` and `C2H6O`. Listing it once
    per reason double-counted it in the result list, spent the hit budget twice
    on one molecule, and wrote it twice into an uncapped CSV export.
    """
    hits = indexes[backend].search("CCO, C2H6O", limit=None)

    assert [h.entry.key for h in hits] == ["1"]


@pytest.mark.parametrize("backend", [0, 1], ids=["memory", "sqlite"])
def test_or_of_two_groups_obeys_inclusion_exclusion(indexes, backend: int) -> None:
    """|A| + |B| - |A and B| == |A or B|, which only holds once each molecule is
    counted once."""
    index = indexes[backend]
    a = len(index.search("fg:alcohol", limit=None))
    b = len(index.search("fg:carboxylic_acid", limit=None))
    both = len(index.search("fg:alcohol&carboxylic_acid", limit=None))
    either = len(index.search("fg:alcohol, fg:carboxylic_acid", limit=None))

    assert either == a + b - both
