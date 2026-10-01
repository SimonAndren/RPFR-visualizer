"""Unit tests for the size-token search grammar.

The grammar sits in front of every other classifier, so the cases that matter
most are the negative ones: a name, a formula or a SMILES must never be read
as a size constraint, and a SMARTS body's commas must survive intact.
"""

from __future__ import annotations

import pytest

from rpfr_visualizer.io.search.chem import parse_size_token, split_search_query, split_size_tokens
from rpfr_visualizer.io.search.models import SizeFilter

pytest.importorskip("rdkit")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("C:5-9", SizeFilter(element_counts=(("C", 5, 9),))),
        ("C:7", SizeFilter(element_counts=(("C", 7, 7),))),
        ("nC:7", SizeFilter(element_counts=(("C", 7, 7),))),
        ("O:1-2", SizeFilter(element_counts=(("O", 1, 2),))),
        ("Br:1", SizeFilter(element_counts=(("Br", 1, 1),))),
        ("nC>3", SizeFilter(element_counts=(("C", 4, 1_000_000),))),
        ("nC>=3", SizeFilter(element_counts=(("C", 3, 1_000_000),))),
        ("C<12", SizeFilter(element_counts=(("C", 0, 11),))),
        ("C<=12", SizeFilter(element_counts=(("C", 0, 12),))),
        ("C=7", SizeFilter(element_counts=(("C", 7, 7),))),
        ("heavy<=12", SizeFilter(column_counts=(("n_heavy", 0, 12),))),
        ("heavy:5-9", SizeFilter(column_counts=(("n_heavy", 5, 9),))),
        ("atoms=20", SizeFilter(column_counts=(("n_atoms", 20, 20),))),
    ],
)
def test_parses_size_tokens(token: str, expected: SizeFilter) -> None:
    assert parse_size_token(token) == expected


@pytest.mark.parametrize(
    "token",
    [
        "methanol",
        "benzoic acid",
        "CCO",
        "CC(=O)O",
        "C6H6",
        "C3H7NO2",
        "[NX3;H2,H1,H0]",
        "fg:alcohol",
        "smarts:[CX3](=O)[OX2H1]",
        "inchikey:LFQSCWFLJHTTHZ",
        "dsgdb9nsd_000002",
        "version:2",
        "C:9-5",  # reversed range is not a constraint
        "",
    ],
)
def test_rejects_non_size_tokens(token: str) -> None:
    assert parse_size_token(token) is None


def test_split_size_tokens_separates_and_merges() -> None:
    rest, size = split_size_tokens(["fg:alcohol", "C:5-9", "O:1"])
    assert rest == ["fg:alcohol"]
    assert size == SizeFilter(element_counts=(("C", 5, 9), ("O", 1, 1)))


def test_smarts_body_keeps_its_commas() -> None:
    assert split_search_query("smarts:[NX3;H2,H1,H0]") == ["smarts:[NX3;H2,H1,H0]"]


def test_smarts_query_still_peels_trailing_size_tokens() -> None:
    assert split_search_query("smarts:[NX3;H2,H1,H0], C:5-9, O:1") == [
        "smarts:[NX3;H2,H1,H0]",
        "C:5-9",
        "O:1",
    ]


def test_size_filter_describe_reads_as_prose() -> None:
    size = SizeFilter(element_counts=(("C", 5, 9), ("O", 1, 1)))
    assert size.describe() == "C 5-9, O 1"


def test_empty_size_filter_is_falsy() -> None:
    assert not SizeFilter()
    assert SizeFilter(element_counts=(("C", 1, 2),))
