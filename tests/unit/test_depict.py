"""Tests for classic RDKit enrichment depiction."""

from __future__ import annotations

import re

import numpy as np
import pytest

from rpfr_visualizer.io.schema import MoleculeRecord, SiteRecord
from rpfr_visualizer.viz.depict import (
    depict_enrichment,
    depict_enrichment_svg,
    format_site_value_table,
)

pytest.importorskip("rdkit")


def _mol() -> MoleculeRecord:
    # Heavy atoms first (QM9s convention), then H — matches CC=O + 4H
    sites = [
        SiteRecord(molecule_id=1, atom_index=0, element="C", beta=(1.1,), temperatures=(300.0,)),
        SiteRecord(molecule_id=1, atom_index=1, element="C", beta=(1.2,), temperatures=(300.0,)),
        SiteRecord(molecule_id=1, atom_index=2, element="O", beta=(1.05,), temperatures=(300.0,)),
        SiteRecord(molecule_id=1, atom_index=3, element="H", beta=(1.0,), temperatures=(300.0,)),
        SiteRecord(molecule_id=1, atom_index=4, element="H", beta=(1.0,), temperatures=(300.0,)),
        SiteRecord(molecule_id=1, atom_index=5, element="H", beta=(1.0,), temperatures=(300.0,)),
        SiteRecord(molecule_id=1, atom_index=6, element="H", beta=(1.0,), temperatures=(300.0,)),
    ]
    return MoleculeRecord(molecule_id=1, smiles="CC=O", sites=sites, n_atoms=7)


def test_depict_enrichment_svg_returns_svg() -> None:
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]
    svg = depict_enrichment_svg(mol, values, element_filter="C")
    assert svg is not None
    assert "<svg" in svg.lower()
    # Kekulé double bond should appear as a path pair / bond in SVG output
    assert "C=O" not in svg  # SMILES not embedded; bond geometry is
    assert len(svg) > 500


def test_the_numbers_can_be_turned_off_without_touching_the_structure() -> None:
    """The locants are optional; the skeleton underneath them is not."""
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]

    labelled = depict_enrichment(mol, values, element_filter="C", show_labels=True)
    plain = depict_enrichment(mol, values, element_filter="C", show_labels=False)
    assert labelled.svg is not None and plain.svg is not None

    # Locants are drawn as annotations, which RDKit marks class='note'. The
    # atom glyphs are untouched either way -- that is the point of using a note
    # rather than replacing the atom's symbol.
    notes = re.compile(r"class='note'")
    atoms = re.compile(r"class='atom-\d+'")
    assert len(notes.findall(labelled.svg)) > 0
    assert len(notes.findall(plain.svg)) == 0
    assert len(atoms.findall(labelled.svg)) == len(atoms.findall(plain.svg))

    # No discs either way: this pane draws structure, not values.
    assert plain.svg.lower().count("<ellipse") == labelled.svg.lower().count("<ellipse") == 0


def test_atom_labels_carry_identity_not_value() -> None:
    """A label names the atom; the colour is what carries the number.

    Printing both crowded the skeleton badly enough that the structure was hard
    to read. Two renders whose values differ only in how many digits they would
    print must therefore produce the same label glyphs -- if a value were still
    being drawn, the longer numbers would emit more of them.
    """
    mol = _mol()
    short = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]
    long = [1.123456789, 1.234567891, 1.056789123, 1.0, 1.0, 1.0, 1.0]

    a = depict_enrichment(mol, short, element_filter="C", show_labels=True)
    b = depict_enrichment(mol, long, element_filter="C", show_labels=True)
    assert a.svg is not None and b.svg is not None

    glyphs = re.compile(r"class='note'")
    assert len(glyphs.findall(a.svg)) == len(glyphs.findall(b.svg))


def _disc_alphas(svg: str) -> list[int]:
    """Alpha byte of every highlight disc, in document order.

    RDKit writes a four-component highlight colour as ``#rrggbbaa``; a
    three-component one has no alpha at all. Reading the byte back is the only
    way to assert from outside that the fingerprint reached the drawing.
    """
    return [
        int(match, 16)
        for match in re.findall(r"<ellipse[^>]*fill:#[0-9a-fA-F]{6}([0-9a-fA-F]{2})", svg)
    ]


def test_no_value_tint_is_drawn_on_the_skeleton() -> None:
    """The 2D pane is a structural diagram, not a heat map.

    A shaded disc behind every vertex is not something a chemist draws, and on
    a one-carbon molecule -- where RDKit writes the condensed formula rather
    than a skeleton -- it came out as a grey circle around the letters CH4.
    The value is stated in the 3D labels, the site table and the CSV instead.
    """
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]

    result = depict_enrichment(mol, values, element_filter="C", show_labels=True)
    assert result.svg is not None
    assert "<ellipse" not in result.svg.lower()


def test_the_picked_atom_still_gets_a_disc() -> None:
    """Selection is not a value. Clicking a reference site, or one half of a
    clumping pair, still has to show which atom it landed on."""
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]

    result = depict_enrichment(mol, values, element_filter="C", highlight_atom_index=1)
    assert result.svg is not None
    assert result.svg.lower().count("<ellipse") == 1


def test_bonds_keep_their_own_colour_and_weight() -> None:
    """Highlighting every atom used to repaint every bond between them.

    RDKit tints a bond whose two ends are both highlighted, so a molecule with
    all its atoms coloured came back as a hairline skeleton in the default
    highlight pink -- no element colours, and double bonds barely visible.
    """
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]

    result = depict_enrichment(mol, values, element_filter=None, show_labels=False)
    assert result.svg is not None

    strokes = re.findall(
        r"class='bond-" + r"\d" + r"+[^']*'[^>]*stroke:(#[0-9A-Fa-f]{6})", result.svg
    )
    assert strokes, "no bonds were drawn at all"
    assert "#FF7F7F" not in strokes, "bonds are still wearing the highlight colour"

    # The tint also came at the highlight's own hairline width, which is what
    # made the double bond hard to see. Bonds must be drawn at the width the
    # drawer was configured with.
    widths = {
        float(w)
        for w in re.findall(
            r"class='bond-" + r"\d" + r"+[^']*'[^>]*stroke-width:([0-9.]+)px", result.svg
        )
    }
    assert widths == {2.2}, widths


#: A fused polycyclic -- an androstane acetate. Ring systems are where the two
#: layout engines visibly disagree, and where a chemist notices immediately.
FUSED_RING_SMILES = "CC(=O)OC1CCC2(C)C(CCC3C2CCC2(C)C3CCC2=O)C1"


def test_the_layout_uses_the_coordgen_engine() -> None:
    """CoordGen is what makes a fused ring system come out textbook-shaped.

    Asserted by agreement with a direct CoordGen call rather than by any
    quality metric: bond lengths and atom spacing come out the same under both
    engines, and what actually differs -- how the rings sit against each other
    -- has no honest scalar to measure it by.
    """
    from rdkit import Chem
    from rdkit.Chem import rdCoordGen

    from rpfr_visualizer.viz.depict import _compute_2d_coords

    def xy(mol) -> np.ndarray:
        conf = mol.GetConformer()
        return np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                for i in range(mol.GetNumAtoms())
            ]
        )

    ours = Chem.MolFromSmiles(FUSED_RING_SMILES)
    _compute_2d_coords(ours)

    coordgen = Chem.MolFromSmiles(FUSED_RING_SMILES)
    rdCoordGen.AddCoords(coordgen)

    assert np.allclose(xy(ours), xy(coordgen))


def test_a_layout_engine_failure_still_produces_coordinates(monkeypatch) -> None:
    """An awkward drawing beats no drawing."""
    from rdkit import Chem
    from rdkit.Chem import rdCoordGen

    from rpfr_visualizer.viz.depict import _compute_2d_coords

    def boom(*args, **kwargs):
        raise RuntimeError("coordgen said no")

    monkeypatch.setattr(rdCoordGen, "AddCoords", boom)

    mol = Chem.MolFromSmiles(FUSED_RING_SMILES)
    _compute_2d_coords(mol)
    assert mol.GetNumConformers() == 1


def test_a_locant_is_drawn_beside_the_atom_not_over_it() -> None:
    """The skeleton keeps its shape; carbons stay bare vertices.

    Writing "C4" into the atom's place put a piece of text at every vertex and
    broke the bond lines around it. A note sits alongside instead, which is how
    a numbered structure is drawn in an organic chemistry text.
    """
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]

    result = depict_enrichment(mol, values, element_filter="C", show_labels=True)
    assert result.svg is not None

    # One note glyph per digit of each heavy atom's index: 0, 1, 2 here.
    assert len(re.findall(r"class='note'", result.svg)) == 3


def _single_heavy_atom(smiles: str, element: str, n_h: int) -> MoleculeRecord:
    sites = [
        SiteRecord(molecule_id=1, atom_index=0, element=element, beta=(1.2,), temperatures=(300.0,))
    ]
    sites += [
        SiteRecord(molecule_id=1, atom_index=i, element="H", beta=(1.0,), temperatures=(300.0,))
        for i in range(1, n_h + 1)
    ]
    return MoleculeRecord(molecule_id=1, smiles=smiles, sites=sites, n_atoms=1 + n_h)


def test_methane_is_drawn_as_a_structure_not_a_formula() -> None:
    """One heavy atom leaves skeletal notation with nothing to draw.

    RDKit falls back to the condensed formula, so methane arrived as the four
    characters CH4 -- a formula sitting in a pane full of structures. The
    hydrogens are made explicit instead, which is how every textbook draws it.
    """
    mol = _single_heavy_atom("C", "C", 4)
    result = depict_enrichment(mol, [1.2, 1.0, 1.0, 1.0, 1.0], element_filter="C")
    assert result.svg is not None

    bonds = set(re.findall(r"class='bond-(\d+)", result.svg))
    assert len(bonds) == 4, "expected four C-H bonds"
    # Five atom glyphs: the carbon and its four hydrogens, each spelled out.
    assert len(set(re.findall(r"class='atom-(\d+)'", result.svg))) == 5


def test_a_lone_heavy_atom_is_not_numbered() -> None:
    """A locant that distinguishes nothing is noise, and RDKit strands it.

    With no neighbours to place the annotation against it landed half a canvas
    away from the atom it belonged to.
    """
    mol = _single_heavy_atom("C", "C", 4)
    result = depict_enrichment(mol, [1.2, 1.0, 1.0, 1.0, 1.0], element_filter="C")
    assert result.svg is not None
    assert "class='note'" not in result.svg


def test_water_is_drawn_bent() -> None:
    """Every other hydrogen count divides the circle evenly, which would put
    H-O-H in a straight line and state a geometry water does not have."""
    import math

    mol = _single_heavy_atom("O", "O", 2)
    result = depict_enrichment(mol, [1.2, 1.0, 1.0], element_filter="O")
    assert result.svg is not None

    # RDKit splits each bond into two coloured halves, so group the segments
    # by bond and keep the one leaving the oxygen.
    segments = re.findall(
        r"class='bond-(\d+)[^']*'[^>]*d='M ([0-9.]+),([0-9.]+) L ([0-9.]+),([0-9.]+)", result.svg
    )
    first: dict[str, tuple[float, float]] = {}
    for bond, x0, y0, x1, y1 in segments:
        first.setdefault(bond, (float(x1) - float(x0), float(y1) - float(y0)))
    assert len(first) == 2, segments
    vectors = list(first.values())
    cos = sum(u * v for u, v in zip(*vectors, strict=True)) / (
        math.hypot(*vectors[0]) * math.hypot(*vectors[1])
    )
    assert cos > -0.99, "the two O-H bonds came out collinear"


def test_hydrogen_filter_explains_the_bare_skeleton() -> None:
    """Skeletal depictions leave hydrogens implicit, so H can never be labelled.

    Returning a valueless skeleton with no explanation made this look like a
    rendering choice instead of a structural limitation.
    """
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]

    depiction = depict_enrichment(mol, values, element_filter="H")
    assert depiction.svg is not None  # the skeleton is still drawn
    assert "hydrogen" in depiction.note.lower()
    assert "3D view" in depiction.note


def test_missing_smiles_reports_why_nothing_was_drawn() -> None:
    sites = [
        SiteRecord(molecule_id=2, atom_index=0, element="C", beta=(1.1,), temperatures=(300.0,))
    ]
    mol = MoleculeRecord(molecule_id=2, smiles="", sites=sites, n_atoms=1)

    depiction = depict_enrichment(mol, [1.1])
    assert depiction.svg is None
    assert "No SMILES" in depiction.note


def test_unparseable_smiles_reports_why() -> None:
    sites = [
        SiteRecord(molecule_id=3, atom_index=0, element="C", beta=(1.1,), temperatures=(300.0,))
    ]
    mol = MoleculeRecord(molecule_id=3, smiles="not a molecule", sites=sites, n_atoms=1)

    depiction = depict_enrichment(mol, [1.1])
    assert depiction.svg is None
    assert "could not read the SMILES" in depiction.note


def test_clean_render_carries_no_note() -> None:
    mol = _mol()
    depiction = depict_enrichment(mol, [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0], element_filter="C")
    assert depiction.svg is not None
    assert depiction.note == ""


def test_svg_wrapper_matches_the_full_result() -> None:
    mol = _mol()
    values = [1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0]
    assert (
        depict_enrichment_svg(mol, values, element_filter="C")
        == depict_enrichment(mol, values, element_filter="C").svg
    )


def test_site_anchor_options() -> None:
    from rpfr_visualizer.viz.depict import site_anchor_options

    mol = _mol()
    opts = site_anchor_options(mol, element="C", heavy_only=True)
    assert all(v in {0, 1} for _, v in opts)
    assert any(label.startswith("0:") for label, _ in opts)


def test_format_site_value_table() -> None:
    mol = _mol()
    values = np.array([1.1, 1.2, 1.05, 1.0, 1.0, 1.0, 1.0])
    text = format_site_value_table(mol, values, element_filter="C", show_hydrogens=False)
    assert "site" in text
    assert "C" in text
    assert "1.1" in text or "1.10000" in text


def test_depict_fg_pattern_prefers_highlighted_example() -> None:
    from rpfr_visualizer.analysis.functional_groups import preset_by_name
    from rpfr_visualizer.viz.depict import depict_fg_pattern

    preset = preset_by_name("carboxylic_acid")
    svg, mode = depict_fg_pattern(
        smarts=preset.smarts,
        smiles=preset.smiles_substring,
        example_smiles=preset.example_smiles,
    )
    assert svg is not None
    assert "<svg" in svg.lower()
    assert mode == "example_highlight"


def test_depict_fg_pattern_falls_back_to_smarts() -> None:
    from rpfr_visualizer.viz.depict import depict_fg_pattern

    svg, mode = depict_fg_pattern(smarts="[CX3](=O)[OX2H1]")
    assert svg is not None
    assert mode == "smarts"


def test_depict_fg_pattern_reports_none_when_undrawable() -> None:
    from rpfr_visualizer.viz.depict import depict_fg_pattern

    assert depict_fg_pattern() == (None, "none")


def test_depict_smiles_and_highlight_render() -> None:
    from rpfr_visualizer.viz.depict import depict_smiles_svg, depict_with_highlight

    plain = depict_smiles_svg("CC(=O)O")
    assert plain is not None and "<svg" in plain.lower()

    highlighted = depict_with_highlight("CC(=O)O", "[CX3](=O)[OX2H1]")
    assert highlighted is not None and "<svg" in highlighted.lower()
    # A matched substructure adds highlight geometry the plain drawing lacks.
    assert len(highlighted) > len(plain)

    assert depict_smiles_svg("not-a-smiles") is None


def test_active_fg_patterns_summarizes_filters() -> None:
    from rpfr_visualizer.analysis.functional_groups import preset_by_name
    from rpfr_visualizer.viz.depict import active_fg_patterns

    patterns = active_fg_patterns(
        preset=preset_by_name("alcohol"),
        smarts="[OX2H]",
        observed_r1=("CO",),
    )
    sources = [p["source"] for p in patterns]
    assert sources == ["preset", "smarts", "observed_r1"]
    assert all(set(p) >= {"source", "name", "description", "depict"} for p in patterns)
    assert patterns[0]["depict"] == preset_by_name("alcohol").example_smiles
