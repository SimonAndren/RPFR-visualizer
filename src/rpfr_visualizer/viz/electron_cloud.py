"""Electron-cloud style enrichment visualization with py3Dmol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from rpfr_visualizer.viz._deps import require_py3dmol
from rpfr_visualizer.viz.colors import cpk_fingerprint_style, opacity_caption, range_caption
from rpfr_visualizer.viz.py3dmol_local import create_view as create_py3dmol_view
from rpfr_visualizer.viz.py3dmol_local import display_view

if TYPE_CHECKING:
    from rpfr_visualizer.analysis.element_stats import ElementStatsCache
    from rpfr_visualizer.analysis.models import MoleculeAnalysisResult
    from rpfr_visualizer.io.loader import Molecule

# Van der Waals-like sphere radii (Å) for ball-and-stick style rendering.
ATOM_BALL_RADIUS: dict[str, float] = {
    "H": 0.18,
    "C": 0.35,
    "N": 0.33,
    "O": 0.32,
    "F": 0.28,
}
DEFAULT_BALL_RADIUS = 0.30
STICK_COLOR = "#888888"
STICK_RADIUS = 0.12
# Heavy–heavy covalent distance cutoff when drawing sticks without H.
HEAVY_BOND_CUTOFF_A = 1.85


def _value_bounds(values: np.ndarray, indices: list[int]) -> tuple[float, float]:
    finite = [float(values[i]) for i in indices if np.isfinite(values[i])]
    if not finite:
        return 0.0, 1.0
    return float(min(finite)), float(max(finite))


class EnrichmentView:
    """Render per-atom enrichment as CPK-colored balls with fingerprint fill.

    Fill intensity is normalized within an element using dataset p5–p95 when
    :class:`~rpfr_visualizer.analysis.element_stats.ElementStatsCache` is provided;
    otherwise falls back to the visible same-element value range.
    """

    def __init__(
        self,
        molecule: Molecule,
        values: Sequence[float | None] | np.ndarray,
        *,
        temperature: float = 300.0,
        title: str | None = None,
        stats: ElementStatsCache | None = None,
        quantity_kind: str = "beta",
        show_hydrogens: bool = True,
        display_kind: str | None = None,
    ) -> None:
        self.molecule = molecule
        self.values = np.asarray(
            [np.nan if v is None else float(v) for v in values],
            dtype=np.float64,
        )
        self.temperature = temperature
        self.title = title or f"mol {molecule.molecule_id} @ {temperature:.0f} K"
        self.symbols = [site.element for site in molecule.sites]
        self.coords = np.array(
            [site.coordinates or (0.0, 0.0, 0.0) for site in molecule.sites],
            dtype=np.float64,
        )
        self.stats = stats
        self.quantity_kind = quantity_kind
        #: The quantity on screen, which is not always the one whose percentile
        #: scale colours it: 1000 ln α borrows epsilon's, and naming the view
        #: after its scale put "ε" on a 1000 ln α figure.
        self.display_kind = display_kind or quantity_kind
        self.show_hydrogens = show_hydrogens
        self.caption = ""

    @classmethod
    def from_analysis(
        cls,
        molecule: Molecule,
        result: MoleculeAnalysisResult,
        *,
        stats: ElementStatsCache | None = None,
        show_hydrogens: bool = True,
    ) -> EnrichmentView:
        """Build a view from a :class:`MoleculeAnalysisResult`."""
        values = result.per_atom_array(molecule.n_atoms)
        kind = result.kind.value
        # No silent default. A clumping or mass-law result routed through here
        # would have been coloured and captioned as beta -- wrong numbers under
        # a confident label, which is the worst failure this view can have.
        scales = {
            "beta": "beta",
            "alpha": "alpha",
            "epsilon": "epsilon",
            "ln_alpha": "epsilon",
        }
        if kind not in scales:
            raise ValueError(
                f"{kind!r} has no colour scale. The per-atom view normalises "
                f"against dataset percentiles, which exist for "
                f"{', '.join(sorted(set(scales)))} only."
            )
        quantity_kind = scales[kind]
        return cls(
            molecule,
            values,
            temperature=result.temperature,
            title=f"mol {molecule.molecule_id} ({kind})",
            stats=stats,
            quantity_kind=quantity_kind,
            show_hydrogens=show_hydrogens,
            display_kind=kind,
        )

    def _visible_indices(self, element_filter: str | None = None) -> list[int]:
        """Indices drawn as structure (element lock does not hide other atoms)."""
        indices = list(range(self.molecule.n_atoms))
        if not self.show_hydrogens:
            indices = [i for i in indices if self.symbols[i] != "H"]
        # Keep *element_filter* for coloring only — full connectivity stays visible
        # so heteroatoms (e.g. N in nitriles/amines) are never dropped from 3D.
        _ = element_filter
        return indices

    def _focus_indices(self, element_filter: str | None) -> list[int]:
        """Indices that receive fingerprint fill / value labels."""
        indices = self._visible_indices()
        if element_filter is None:
            return indices
        return [i for i in indices if self.symbols[i] == element_filter]

    def _fill_bounds(
        self,
        element: str,
        indices: list[int],
    ) -> tuple[float, float]:
        """Return (vmin, vmax) for fill normalization of *element*."""
        if self.stats is not None:
            rng = self.stats.quantity_range(
                element,
                self.temperature,
                kind=self.quantity_kind,
                warn=False,
            )
            if rng is not None:
                if self.display_kind == "ln_alpha" and self.quantity_kind == "epsilon":
                    # The cache holds epsilon's percentiles. 1000 ln α is
                    # 1000 ln(1 + ε/1000), which is monotonic, so each
                    # percentile converts exactly rather than approximately.
                    low, high = rng
                    return (
                        1000.0 * float(np.log1p(low / 1000.0)),
                        1000.0 * float(np.log1p(high / 1000.0)),
                    )
                return rng
        elem_indices = [i for i in indices if self.symbols[i] == element]
        return _value_bounds(self.values, elem_indices)

    def range_caption_for(self, element: str) -> str:
        """Return the dataset range caption for *element* at the view temperature."""
        vmin, vmax = self._fill_bounds(element, self._focus_indices(element))
        return range_caption(
            element,
            kind=self.display_kind,
            temperature=self.temperature,
            vmin=vmin,
            vmax=vmax,
        )

    def caption_for(self, element_filter: str | None = None) -> str:
        """Colour-scale caption for the element lock, or for every focus element.

        This is the sentence that says what the colours mean, so it has to be
        obtainable without building a 3D viewer — a caller may be rendering only
        the 2D depiction, or no figure at all.
        """
        elements = {self.symbols[i] for i in self._focus_indices(element_filter)}
        if element_filter is not None:
            return self.range_caption_for(element_filter) if element_filter in elements else ""
        ranges = [(el, *self._fill_bounds(el, self._focus_indices(el))) for el in sorted(elements)]
        return opacity_caption(kind=self.display_kind, temperature=self.temperature, ranges=ranges)

    def _empty_viewer(self, width: int, height: int, bundle_src: str | None = None):
        require_py3dmol()
        return create_py3dmol_view(width=width, height=height, bundle_src=bundle_src)

    def _add_sticks(self, view, indices: Sequence[int]) -> None:
        """Add stick bonds for the visible atom set."""
        if self.show_hydrogens and len(indices) == self.molecule.n_atoms:
            from rpfr_visualizer.viz.structure import molecule_to_xyz

            view.addModel(molecule_to_xyz(self.molecule), "xyz")
            view.setStyle({}, {"stick": {"radius": STICK_RADIUS, "color": STICK_COLOR}})
            return

        # Heavy-only (or subset): draw distance-based cylinders.
        coords = self.coords
        for i_pos, i in enumerate(indices):
            for j in indices[i_pos + 1 :]:
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                if dist > HEAVY_BOND_CUTOFF_A or dist < 0.4:
                    continue
                view.addCylinder(
                    {
                        "start": {
                            "x": float(coords[i][0]),
                            "y": float(coords[i][1]),
                            "z": float(coords[i][2]),
                        },
                        "end": {
                            "x": float(coords[j][0]),
                            "y": float(coords[j][1]),
                            "z": float(coords[j][2]),
                        },
                        "radius": STICK_RADIUS,
                        "color": STICK_COLOR,
                        "fromCap": True,
                        "toCap": True,
                    }
                )

    def _add_spheres(
        self,
        view,
        *,
        element_filter: str | None = None,
        show_labels: bool = False,
        highlight_atom_index: int | None = None,
    ):
        indices = self._visible_indices()
        if not indices:
            view.zoomTo()
            return view

        self._add_sticks(view, indices)

        focus = set(self._focus_indices(element_filter))
        # Per-element fill bounds among focus atoms only.
        bounds_by_element: dict[str, tuple[float, float]] = {}
        for el in {self.symbols[i] for i in focus}:
            bounds_by_element[el] = self._fill_bounds(el, list(focus))

        self.caption = self.caption_for(element_filter)

        for idx in indices:
            element = self.symbols[idx]
            value = self.values[idx]
            val = float(value) if np.isfinite(value) else None
            is_focus = idx in focus
            if is_focus and element in bounds_by_element:
                vmin, vmax = bounds_by_element[element]
                symmetric = self.quantity_kind != "beta"
                color, opacity = cpk_fingerprint_style(
                    element,
                    val,
                    vmin,
                    vmax,
                    symmetric=symmetric,
                )
            else:
                # Non-focus: solid CPK so N stays blue when the lock is on carbon.
                color, opacity = cpk_fingerprint_style(element, None, 0.0, 1.0)

            radius = ATOM_BALL_RADIUS.get(element, DEFAULT_BALL_RADIUS)
            if highlight_atom_index is not None and idx == int(highlight_atom_index):
                radius *= 1.35
                opacity = 1.0
            pos = self.coords[idx]
            view.addSphere(
                {
                    "center": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                    "radius": radius,
                    "color": color,
                    "opacity": opacity,
                }
            )
            # Identity *and* value. The sphere's opacity encodes the value but
            # only comparatively -- it answers "which of these is strongest?",
            # never "what is it?". In 3D there is room to answer both, and a
            # reader who wants the number should not have to cross-reference
            # the site table to get it. The 2D skeleton is the one that cannot
            # afford this: there the number goes beside a bare vertex and the
            # structure is what gets crowded out.
            #
            # A site with no value still gets its identity, since "which atom
            # is this?" is a question it still has.
            if show_labels and is_focus:
                view.addLabel(
                    f"{element}{idx}" if val is None else f"{element}{idx}\n{val:.4g}",
                    {
                        "position": {
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "z": float(pos[2]),
                        },
                        "fontSize": 9,
                        "fontColor": "black",
                        "backgroundColor": "white",
                        "backgroundOpacity": 0.75,
                    },
                )
            elif highlight_atom_index is not None and idx == int(highlight_atom_index):
                view.addLabel(
                    f"ref {element}{idx}",
                    {
                        "position": {
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "z": float(pos[2]),
                        },
                        "fontSize": 10,
                        "fontColor": "black",
                        "backgroundColor": "#ffe082",
                        "backgroundOpacity": 0.9,
                    },
                )
        view.zoomTo()
        return view

    def build_viewer(
        self,
        *,
        element: str | None = None,
        width: int = 700,
        height: int = 500,
        show_labels: bool = False,
        show_hydrogens: bool | None = None,
        highlight_atom_index: int | None = None,
        bundle_src: str | None = None,
    ):
        """Build a py3Dmol ball-and-stick viewer (does not display yet).

        *bundle_src* references 3Dmol.js at a URL instead of inlining it — see
        :func:`rpfr_visualizer.viz.py3dmol_local.create_view`.
        """
        if show_hydrogens is not None:
            self.show_hydrogens = show_hydrogens
        view = self._empty_viewer(width=width, height=height, bundle_src=bundle_src)
        return self._add_spheres(
            view,
            element_filter=element,
            show_labels=show_labels,
            highlight_atom_index=highlight_atom_index,
        )

    def show(
        self,
        *,
        element: str | None = None,
        width: int = 700,
        height: int = 500,
        show_labels: bool = False,
        show_hydrogens: bool | None = None,
    ):
        """Render ball-and-stick enrichment view in the notebook."""
        viewer = self.build_viewer(
            element=element,
            width=width,
            height=height,
            show_labels=show_labels,
            show_hydrogens=show_hydrogens,
        )
        display_view(viewer)
        return viewer


__all__ = ["EnrichmentView", "HEAVY_BOND_CUTOFF_A"]
