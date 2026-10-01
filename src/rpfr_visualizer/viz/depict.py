"""Classic RDKit 2D depiction: skeletal formulas, enrichment labels, FG patterns.

Textbook skeletal style throughout — carbons as vertices, heteroatoms labelled,
aromatic rings drawn Kekulé so double and triple bonds stay readable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from rpfr_visualizer.analysis.functional_groups import FunctionalGroupPreset
    from rpfr_visualizer.io.loader import Molecule


# Soft pastel fill for pattern depictions; enrichment views use outline-only
# highlights instead so bond multiplicity is never obscured.
PATTERN_HIGHLIGHT_RGB = (0.85, 0.92, 1.0)


class _DrawerStyle(TypedDict):
    """Keyword arguments for :func:`_classic_drawer`.

    A plain dict here infers as ``dict[str, float]`` -- bool is a subclass of
    int, so the mixed literal joins to float -- and splatting that into a bool
    parameter is a type error on three live drawing paths. Naming the shape
    fixes those and makes a mistyped key a failure rather than a silently
    ignored argument.
    """

    bond_line_width: float
    scaling_factor: float
    per_atom_highlights: bool


# Pattern depictions are small and filled; enrichment views use the defaults.
_PATTERN_STYLE: _DrawerStyle = {
    "bond_line_width": 2.0,
    "scaling_factor": 22.0,
    "per_atom_highlights": False,
}


def _heavy_site_indices(molecule: Molecule) -> list[int]:
    return [i for i, site in enumerate(molecule.sites) if site.element != "H"]


def _compute_2d_coords(draw_mol) -> None:
    """Lay the molecule out the way a chemist would draw it.

    CoordGen -- the layout engine from Schrodinger that RDKit ships -- is what
    makes a fused polycyclic come out in its textbook shape. Both engines give
    even bond lengths and neither lets atoms collide; what differs is how the
    ring systems are oriented against each other, and on a steroid that is the
    whole difference between the drawing in a textbook and a sheared version of
    it that is correct but nobody would draw.

    Falls back to ``Compute2DCoords`` if CoordGen is unavailable or refuses a
    structure, because a slightly awkward drawing beats no drawing.
    """
    from rdkit.Chem import AllChem

    try:
        from rdkit.Chem import rdCoordGen

        rdCoordGen.AddCoords(draw_mol)
        return
    except Exception:
        pass
    try:
        AllChem.Compute2DCoords(draw_mol)
    except Exception:
        pass


#: Angle between the two bonds when a lone heavy atom carries exactly two
#: hydrogens. Water is bent, and drawing H-O-H in a straight line states a
#: geometry the molecule does not have; every other count divides the circle
#: evenly, which is already the conventional picture.
_BENT_BOND_ANGLE_DEG = 104.5


def _lewis_mol(mol):
    """A one-heavy-atom molecule drawn with its hydrogens explicit.

    Skeletal notation has nothing to draw here -- no bonds between heavy atoms
    means no skeleton -- so RDKit falls back to the condensed formula and
    methane comes out as the four characters ``CH4``. That is a formula, not a
    structure, and it looks broken sitting in a pane full of structures.

    So hydrogens are made explicit and placed by hand: evenly around the
    central atom, starting at twelve o'clock. Four of them give the cross that
    every textbook draws methane as. CoordGen would splay them tetrahedrally
    instead, which is a fair picture of the 3D shape but not how the molecule
    is written.
    """
    import math

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    with_h = Chem.AddHs(Chem.Mol(mol))
    heavy = next(
        (a.GetIdx() for a in with_h.GetAtoms() if a.GetSymbol() != "H"),
        0,
    )
    hydrogens = [a.GetIdx() for a in with_h.GetAtoms() if a.GetSymbol() == "H"]
    if not hydrogens:
        return with_h

    if len(hydrogens) == 2:
        half = math.radians(_BENT_BOND_ANGLE_DEG) / 2.0
        angles = [math.pi / 2 - half, math.pi / 2 + half]
    else:
        step = 2 * math.pi / len(hydrogens)
        angles = [math.pi / 2 - n * step for n in range(len(hydrogens))]

    conf = Chem.Conformer(with_h.GetNumAtoms())
    conf.SetAtomPosition(heavy, Point3D(0.0, 0.0, 0.0))
    for idx, angle in zip(hydrogens, angles, strict=True):
        conf.SetAtomPosition(idx, Point3D(math.cos(angle), math.sin(angle), 0.0))
    with_h.RemoveAllConformers()
    with_h.AddConformer(conf, assignId=True)
    return with_h


def _prepare_classic_mol(mol):
    """Kekulé 2D mol with explicit double/triple bonds (textbook RDKit style)."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    draw_mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(draw_mol, clearAromaticFlags=True)
    except Exception:
        pass
    _compute_2d_coords(draw_mol)
    try:
        # wedgeBonds is the default and is why stereocentres come out as the
        # bold and hashed bonds an organic chemist expects rather than as bare
        # lines with a parity flag nobody can see.
        return rdMolDraw2D.PrepareMolForDrawing(draw_mol, kekulize=True)
    except Exception:
        return draw_mol


def _classic_drawer(
    width: int,
    height: int,
    *,
    bond_line_width: float = 2.2,
    scaling_factor: float = 28.0,
    per_atom_highlights: bool = True,
):
    """Return a ``MolDraw2DSVG`` configured for classic skeletal output.

    *per_atom_highlights* gives each atom its own disc (enrichment views, which
    colour atoms rather than substructures, and where one continuous blob along
    the bonds would hide a C=O). Otherwise RDKit's continuous highlight is used
    with a soft pastel fill, which reads better for a whole-substructure match.

    RDKit draws a radius-based highlight filled whatever ``fillHighlights``
    says, so a disc here is always solid; the only lever is its alpha. The
    enrichment view uses one disc and one only, to mark the atom the user
    picked.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False
    opts.bondLineWidth = bond_line_width
    opts.scalingFactor = scaling_factor
    opts.clearBackground = True
    opts.additionalAtomLabelPadding = 0.12 if per_atom_highlights else 0.05
    if hasattr(opts, "prepareMolsBeforeDrawing"):
        opts.prepareMolsBeforeDrawing = True
    if per_atom_highlights:
        if hasattr(opts, "continuousHighlight"):
            opts.continuousHighlight = False
        if hasattr(opts, "highlightBondWidthMultiplier"):
            opts.highlightBondWidthMultiplier = 1
    else:
        try:
            opts.setHighlightColour(PATTERN_HIGHLIGHT_RGB)
        except Exception:
            pass
    return drawer


@dataclass(frozen=True)
class EnrichmentDepiction:
    """A skeletal depiction and, when something is missing from it, why.

    ``svg`` is ``None`` when nothing could be drawn at all. A non-empty ``note``
    with an ``svg`` present means the picture is real but incomplete — most
    often that values could not be placed on the atoms. Callers must surface it:
    a skeleton with no numbers on it looks like a rendering choice rather than a
    missing mapping.
    """

    svg: str | None
    note: str = ""


def depict_enrichment(
    molecule: Molecule,
    values: Sequence[float | None] | np.ndarray,
    *,
    element_filter: str | None = None,
    size: tuple[int, int] = (420, 360),
    highlight_atom_index: int | None = None,
    show_labels: bool = True,
) -> EnrichmentDepiction:
    """Draw a numbered skeletal structure, and report what could not be drawn.

    This is a structural diagram: CoordGen layout, carbons as bare vertices,
    heteroatoms labelled, Kekulé bonds, stereocentres as wedges, and each
    atom's database index as an annotation beside it.

    *values* does not reach the drawing. It once set a per-atom tint, but a
    shaded disc behind every vertex is not something a chemist draws, and on a
    one-carbon molecule -- where RDKit writes ``CH4`` rather than a skeleton --
    it was a grey circle around the letters. The numbers live in the 3D labels,
    the site table and the CSV, which can state a value rather than merely rank
    it. The parameter stays because it is this pane's contract with the caller
    and because a future annotation may want it.

    Skeletal structures leave hydrogens implicit, so hydrogen sites can never
    be labelled here — with ``element_filter="H"`` the returned ``note`` says
    so instead of handing back a bare skeleton with no explanation.
    """
    if not molecule.smiles:
        return EnrichmentDepiction(
            None,
            "No SMILES stored, so there is no 2D structure.",
        )
    try:
        from rdkit import Chem
    except ImportError:
        return EnrichmentDepiction(None, "RDKit is not installed, so there is no 2D structure.")

    del values  # structural drawing; see the docstring
    mol = Chem.MolFromSmiles(molecule.smiles)
    if mol is None:
        return EnrichmentDepiction(
            None,
            f"RDKit could not read the SMILES {molecule.smiles!r}.",
        )

    heavy_sites = _heavy_site_indices(molecule)
    # One heavy atom means no skeleton to draw, so the hydrogens come out
    # explicit instead of RDKit writing a condensed formula. See _lewis_mol.
    lone_heavy = len(heavy_sites) == 1
    draw_mol = _lewis_mol(mol) if lone_heavy else _prepare_classic_mol(mol)

    # Compare heavy atoms only: the lone-heavy drawing carries explicit
    # hydrogens, which the database's heavy-site list does not.
    draw_heavy = [
        idx
        for idx in range(draw_mol.GetNumAtoms())
        if draw_mol.GetAtomWithIdx(idx).GetSymbol() != "H"
    ]
    mapped = len(draw_heavy) == len(heavy_sites) and all(
        molecule.sites[site_idx].element == draw_mol.GetAtomWithIdx(rd_idx).GetSymbol()
        for rd_idx, site_idx in zip(draw_heavy, heavy_sites, strict=True)
    )

    note = ""
    if not mapped:
        note = (
            "Atom numbers hidden: the stored atom order does not match the SMILES. "
            "The 3D view and table are correct."
        )
    elif element_filter == "H":
        note = "Hydrogens are implicit here. See the 3D view and table."
    # No note for an element the molecule lacks. It used to say "none are
    # labelled", but the 2D numbers every heavy atom whatever the element lock,
    # so the note contradicted the picture above it.

    atom_labels: dict[int, str] = {}
    highlight: list[int] = []
    colors: dict[int, tuple[float, float, float]] = {}
    radii: dict[int, float] = {}

    if mapped:
        for rd_idx, site_idx in zip(draw_heavy, heavy_sites, strict=True):
            # The locant beside the atom, not a symbol replacing it. This is how
            # a numbered structure is drawn in an organic chemistry text: the
            # skeleton keeps its shape, carbons stay bare vertices, heteroatoms
            # keep their own symbols, and the index sits alongside. Writing
            # "C4" into the atom's place instead put a letter at every vertex
            # and broke the bond lines around it.
            #
            # These are database atom indices, not IUPAC locants -- they are
            # what ties an atom here to a row in the site table.
            atom_labels[rd_idx] = str(site_idx)

            # The only disc drawn here marks the atom the user *picked* -- a
            # reference site, or one half of a clumping pair. The per-value
            # tint that used to sit under every focus atom is gone: this is a
            # structural diagram, and a shaded blob behind each vertex is not
            # something a chemist draws. On a one-carbon molecule, where RDKit
            # writes the condensed formula rather than a skeleton, it was a
            # grey circle around the letters CH4.
            #
            # The value now lives where there is room to state it rather than
            # merely rank it: the 3D labels, the site table and the CSV.
            if highlight_atom_index is not None and site_idx == int(highlight_atom_index):
                highlight.append(rd_idx)
                colors[rd_idx] = (1.0, 0.85, 0.2, 0.75)
                radii[rd_idx] = 0.55

    drawer = _classic_drawer(size[0], size[1])
    opts = drawer.drawOptions()
    opts.annotationFontScale = 0.75
    if lone_heavy:
        # Skeletal notation hides a carbon at a vertex, which would leave the
        # bonds meeting at nothing. With the hydrogens spelled out, the central
        # atom has to be spelled out too.
        for rd_idx in draw_heavy:
            opts.atomLabels[rd_idx] = draw_mol.GetAtomWithIdx(rd_idx).GetSymbol()
    # A lone heavy atom gets no number. There is only one site to name, so the
    # locant distinguishes nothing -- and with no neighbours to place it
    # against, RDKit strands it half a canvas below the formula. Methane came
    # out as "CH4" with an orphaned "0" floating underneath it.
    number_atoms = show_labels and len(heavy_sites) > 1
    if number_atoms:
        # atomNote, not atomLabels: a note is drawn *beside* the atom and
        # leaves the skeleton alone, which is what a numbered structure in a
        # textbook looks like. atomLabels replaces the atom's symbol, so every
        # carbon vertex became a piece of text with the bonds broken around it.
        for idx, text in atom_labels.items():
            draw_mol.GetAtomWithIdx(idx).SetProp("atomNote", text)

    # Always draw kekulized molecule. The colour highlights stay even with
    # labels off — they are the fingerprint.
    if highlight:
        # Empty highlightBonds on purpose: without it RDKit tints every bond
        # between two highlighted atoms, and a molecule whose atoms are all
        # highlighted comes back with a pink skeleton and no bond orders
        # readable through it.
        drawer.DrawMolecule(
            draw_mol,
            highlightAtoms=highlight,
            highlightBonds=[],
            highlightAtomColors=colors,
            highlightAtomRadii=radii,
        )
    else:
        drawer.DrawMolecule(draw_mol)
    drawer.FinishDrawing()
    return EnrichmentDepiction(drawer.GetDrawingText(), note)


def depict_enrichment_svg(
    molecule: Molecule,
    values: Sequence[float | None] | np.ndarray,
    *,
    element_filter: str | None = None,
    size: tuple[int, int] = (420, 360),
    highlight_atom_index: int | None = None,
    show_labels: bool = True,
) -> str | None:
    """Return just the SVG from :func:`depict_enrichment`.

    Convenient when the caller has nowhere to put a note. Prefer
    :func:`depict_enrichment` in a UI, which explains an unlabelled skeleton
    rather than leaving the user to guess.
    """
    return depict_enrichment(
        molecule,
        values,
        element_filter=element_filter,
        size=size,
        highlight_atom_index=highlight_atom_index,
        show_labels=show_labels,
    ).svg


def format_site_value_table(
    molecule: Molecule,
    values: Sequence[float | None] | np.ndarray,
    *,
    element_filter: str | None = None,
    show_hydrogens: bool = True,
    quantity_kind: str = "beta",
    digits: int = 6,
    highlight_atom_index: int | None = None,
) -> str:
    """Return a plain-text table of per-site enrichment values."""
    vals = np.asarray(
        [np.nan if v is None else float(v) for v in values],
        dtype=np.float64,
    )
    # The column is headed with the quantity's symbol, and the environment
    # column is "neighbourhood", as the ranked list calls it. Headed "beta" and
    # "R1", the same column had two names on one page, and a 1000 ln α table
    # was headed "epsilon" after the colour scale it borrows.
    from rpfr_visualizer.viz.colors import quantity_symbol

    kind = quantity_symbol(quantity_kind)
    lines = [f"site  element  {kind:>12s}   neighbourhood"]
    lines.append("-" * max(40, len(lines[0])))
    for idx, site in enumerate(molecule.sites):
        el = site.element
        if not show_hydrogens and el == "H":
            continue
        if element_filter is not None and el != element_filter:
            continue
        v = vals[idx]
        v_txt = f"{float(v):.{digits}g}" if np.isfinite(v) else "nan"
        r1 = ""
        if site.local_smiles:
            r1 = (site.local_smiles[0] or "")[:28]
        mark = (
            " <<" if highlight_atom_index is not None and idx == int(highlight_atom_index) else ""
        )
        lines.append(f"{idx:<5d} {el:<8s} {v_txt:>12s}   {r1}{mark}")
    return "\n".join(lines)


def site_anchor_options(
    molecule: Molecule,
    *,
    element: str | None = None,
    heavy_only: bool = True,
) -> list[tuple[str, int]]:
    """Dropdown labels for choosing a reference atom index on *molecule*."""
    options: list[tuple[str, int]] = []
    for idx, site in enumerate(molecule.sites):
        el = site.element
        if heavy_only and el == "H":
            continue
        if element is not None and el != element:
            continue
        r1 = (site.local_smiles[0] if site.local_smiles else "")[:20]
        label = f"{idx}: {el}" + (f"  {r1}" if r1 else "")
        options.append((label, idx))
    return options


def depict_smiles_svg(smiles: str, *, size: tuple[int, int] = (280, 200)) -> str | None:
    """Return a classic skeletal SVG of *smiles*, or None if it cannot be drawn."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = _classic_drawer(size[0], size[1], **_PATTERN_STYLE)
    drawer.DrawMolecule(_prepare_classic_mol(mol))
    drawer.FinishDrawing()
    return str(drawer.GetDrawingText())


def depict_smarts_svg(smarts: str, *, size: tuple[int, int] = (280, 200)) -> str | None:
    """Return an SVG depiction of a SMARTS query (best effort)."""
    if not smarts:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        return None
    try:
        drawer = _classic_drawer(size[0], size[1], **_PATTERN_STYLE)
        drawer.DrawMolecule(query)
        drawer.FinishDrawing()
    except Exception:
        return None
    return str(drawer.GetDrawingText())


def depict_with_highlight(
    smiles: str,
    smarts: str | None,
    *,
    size: tuple[int, int] = (280, 200),
) -> str | None:
    """Depict *smiles* in classic style, highlighting the *smarts* match."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    highlight: list[int] = []
    highlight_bonds: list[int] = []
    if smarts:
        query = Chem.MolFromSmarts(smarts)
        if query is not None:
            highlight = list(mol.GetSubstructMatch(query))
            matched = set(highlight)
            highlight_bonds = [
                bond.GetIdx()
                for bond in mol.GetBonds()
                if bond.GetBeginAtomIdx() in matched and bond.GetEndAtomIdx() in matched
            ]

    drawer = _classic_drawer(size[0], size[1], **_PATTERN_STYLE)
    draw_mol = _prepare_classic_mol(mol)
    if highlight:
        drawer.DrawMolecule(draw_mol, highlightAtoms=highlight, highlightBonds=highlight_bonds)
    else:
        drawer.DrawMolecule(draw_mol)
    drawer.FinishDrawing()
    return str(drawer.GetDrawingText())


def depict_fg_pattern(
    *,
    smarts: str | None = None,
    smiles: str | None = None,
    example_smiles: str | None = None,
    size: tuple[int, int] = (280, 200),
) -> tuple[str | None, str]:
    """Best-effort functional-group depiction in classic skeletal style.

    Returns ``(svg_or_none, mode)`` where *mode* is ``example_highlight``,
    ``smiles``, ``smarts``, or ``none`` — depicting a real example molecule with
    the group highlighted reads far better than depicting a bare query.
    """
    if example_smiles:
        svg = depict_with_highlight(example_smiles, smarts, size=size)
        if svg:
            return svg, "example_highlight"
    if smiles:
        svg = depict_smiles_svg(smiles, size=size)
        if svg:
            return svg, "smiles"
        svg = depict_smarts_svg(smiles, size=size)
        if svg:
            return svg, "smarts"
    if smarts:
        svg = depict_smarts_svg(smarts, size=size)
        if svg:
            return svg, "smarts"
    return None, "none"


def active_fg_patterns(
    *,
    preset: FunctionalGroupPreset | None = None,
    r1_exact: str | None = None,
    smiles_substring: str | None = None,
    smarts: str | None = None,
    observed_r1: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Summarize the currently active functional-group filters for the viewer pane."""
    patterns: list[dict[str, str]] = []

    def _entry(source: str, name: str, description: str, **fields: str) -> dict[str, str]:
        entry = {
            "source": source,
            "name": name,
            "description": description,
            "r1": "",
            "smarts": "",
            "smiles_substring": "",
            "example_smiles": "",
            "depict": "",
        }
        entry.update(fields)
        return entry

    if preset is not None:
        patterns.append(
            _entry(
                "preset",
                preset.name,
                preset.description,
                r1=preset.r1 or "",
                smarts=preset.smarts or "",
                smiles_substring=preset.smiles_substring or "",
                example_smiles=preset.example_smiles or "",
                depict=(
                    preset.example_smiles
                    or preset.smiles_substring
                    or preset.smarts
                    or preset.r1
                    or ""
                ),
            )
        )
    if r1_exact:
        patterns.append(_entry("r1", "R1 exact", r1_exact, r1=r1_exact, depict=r1_exact))
    if smiles_substring:
        patterns.append(
            _entry(
                "substring",
                "SMILES substring",
                smiles_substring,
                smiles_substring=smiles_substring,
                depict=smiles_substring,
            )
        )
    if smarts:
        patterns.append(_entry("smarts", "SMARTS", smarts, smarts=smarts, depict=smarts))
    for r1 in observed_r1:
        patterns.append(_entry("observed_r1", "Observed R1", r1, r1=r1, depict=r1))
    return patterns


__all__ = [
    "EnrichmentDepiction",
    "active_fg_patterns",
    "depict_enrichment",
    "depict_enrichment_svg",
    "depict_fg_pattern",
    "depict_smarts_svg",
    "depict_smiles_svg",
    "depict_with_highlight",
    "format_site_value_table",
    "site_anchor_options",
]
