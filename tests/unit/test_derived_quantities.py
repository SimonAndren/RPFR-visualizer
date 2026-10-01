"""Clumping, mass law, and beta as a function of temperature.

``compute_mass_law`` had no test at all, and ``compute_clumping`` was covered
only on its happy path -- both are now reachable from the UI, so their error
branches are reachable from the UI too.
"""

from __future__ import annotations

import numpy as np
import pytest

from rpfr_visualizer.analysis.analyze import run_molecule_analysis
from rpfr_visualizer.analysis.clumping import clumping_delta, compute_clumping
from rpfr_visualizer.analysis.mass_law import compute_mass_law
from rpfr_visualizer.analysis.models import AnalysisKind, AnalysisRequest
from rpfr_visualizer.analysis.temperature import (
    DEFAULT_METHOD,
    INTERPOLATION_METHODS,
    T_MAX_K,
    T_MIN_K,
    beta_curve,
    interpolate_beta,
    parse_method,
)
from rpfr_visualizer.io.providers import MoleculeArrays
from rpfr_visualizer.io.schema import TEMPERATURES_K, MoleculeRecord, SiteRecord

N_TEMPS = len(TEMPERATURES_K)
ON_GRID = 298.15


def _molecule(elements: tuple[str, ...] = ("C", "O", "H")) -> MoleculeRecord:
    sites = [
        SiteRecord(
            molecule_id=1,
            atom_index=i,
            element=el,
            beta=tuple(float(np.exp(90.0 / t)) for t in TEMPERATURES_K),
            temperatures=tuple(float(t) for t in TEMPERATURES_K),
        )
        for i, el in enumerate(elements)
    ]
    return MoleculeRecord(molecule_id=1, smiles="CO", sites=sites, n_atoms=len(elements))


def _arrays(n: int, *, clumping: bool = True, mass_law: bool = True) -> MoleculeArrays:
    c = None
    if clumping:
        c = np.full((n, n, N_TEMPS), 1.01)
        for k in range(N_TEMPS):
            np.fill_diagonal(c[:, :, k], 1.0)
    m = np.full((n, N_TEMPS), 0.52) if mass_law else None
    return MoleculeArrays(clumping=c, mass_law=m)


# --- beta_curve -------------------------------------------------------------


def test_beta_curve_reproduces_the_stored_points() -> None:
    """The drawn line and the value a caller reads must be the same model."""
    temps = np.array(TEMPERATURES_K, dtype=float)
    betas = np.exp(120.0 / temps)

    grid, curve = beta_curve(temps, betas, num=400)

    for t, stored in zip(temps, betas, strict=True):
        drawn = float(np.interp(t, grid, curve))
        assert drawn == pytest.approx(stored, rel=1e-3)


def test_beta_curve_is_ascending_in_temperature() -> None:
    """The stored grid is not sorted -- it ends 273.15, 298.15 -- so a curve
    built from it positionally would zig-zag."""
    temps = np.array(TEMPERATURES_K, dtype=float)
    assert not np.all(np.diff(temps) > 0), "fixture no longer exercises the unsorted case"

    grid, _curve = beta_curve(temps, np.exp(100.0 / temps))

    assert np.all(np.diff(grid) > 0)


def test_beta_curve_spans_the_stored_range_by_default() -> None:
    temps = np.array(TEMPERATURES_K, dtype=float)
    grid, _ = beta_curve(temps, np.exp(100.0 / temps))
    assert grid[0] == pytest.approx(T_MIN_K)
    assert grid[-1] == pytest.approx(T_MAX_K)


def test_beta_curve_agrees_with_interpolate_beta_off_grid() -> None:
    temps = np.array(TEMPERATURES_K, dtype=float)
    betas = np.exp(120.0 / temps)
    grid, curve = beta_curve(temps, betas, num=600)

    drawn = float(np.interp(305.0, grid, curve))
    read = interpolate_beta(temps, betas, 305.0, warn=False).value

    assert drawn == pytest.approx(read, rel=1e-3)


def test_beta_curve_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        beta_curve([100.0, 200.0], [1.0])


# --- mass law ---------------------------------------------------------------


def test_mass_law_reads_the_stored_value() -> None:
    mol = _molecule()
    arrays = _arrays(mol.n_atoms)

    result = compute_mass_law(mol, arrays, temperature=ON_GRID)

    assert result.kind is AnalysisKind.MASS_LAW
    assert len(result.sites) == mol.n_atoms
    assert all(site.value == pytest.approx(0.52) for site in result.sites)


def test_mass_law_without_arrays_is_an_error() -> None:
    mol = _molecule()
    with pytest.raises(ValueError, match="[Mm]ass"):
        compute_mass_law(mol, _arrays(mol.n_atoms, mass_law=False), temperature=ON_GRID)


def test_mass_law_with_the_wrong_shape_is_an_error() -> None:
    mol = _molecule()
    arrays = MoleculeArrays(mass_law=np.full((mol.n_atoms + 3, N_TEMPS), 0.5))
    with pytest.raises(ValueError):
        compute_mass_law(mol, arrays, temperature=ON_GRID)


def test_mass_law_off_grid_is_an_error_naming_the_grid() -> None:
    """There is no interpolation model for lambda, unlike beta. The UI has to
    be able to tell the user which temperatures do work."""
    mol = _molecule()
    with pytest.raises(ValueError, match="not on the grid"):
        compute_mass_law(mol, _arrays(mol.n_atoms), temperature=305.0)


# --- clumping ---------------------------------------------------------------


def test_clumping_delta_is_the_log_of_the_factor() -> None:
    assert clumping_delta(1.0) == 0.0
    assert clumping_delta(np.e) == pytest.approx(1.0)
    assert clumping_delta(0.0) is None
    assert clumping_delta(-1.0) is None


def test_clumping_keeps_the_raw_factor_matrix() -> None:
    """pair_values is C, not Delta. Anything drawing the matrix has to take the
    logarithm itself -- plotting C put every cell at ~1.0."""
    mol = _molecule()
    result = compute_clumping(mol, _arrays(mol.n_atoms), temperature=ON_GRID)

    assert result.pair_values is not None
    off_diagonal = result.pair_values[0, 1]
    assert off_diagonal == pytest.approx(1.01)
    assert result.pair_values[0, 0] == pytest.approx(1.0)


def test_clumping_max_differs_from_mean() -> None:
    mol = _molecule(("C", "O", "H", "H"))
    arrays = _arrays(mol.n_atoms)
    arrays.clumping[0, 1, :] = 2.0  # one strong partner for atom 0

    mean = compute_clumping(mol, arrays, temperature=ON_GRID, aggregation="mean")
    biggest = compute_clumping(mol, arrays, temperature=ON_GRID, aggregation="max")

    assert biggest.sites[0].value > mean.sites[0].value


def test_clumping_rejects_an_unknown_aggregation() -> None:
    mol = _molecule()
    with pytest.raises(ValueError):
        compute_clumping(mol, _arrays(mol.n_atoms), temperature=ON_GRID, aggregation="median")


def test_clumping_without_arrays_is_an_error() -> None:
    mol = _molecule()
    with pytest.raises(ValueError, match="[Cc]lumping"):
        compute_clumping(mol, _arrays(mol.n_atoms, clumping=False), temperature=ON_GRID)


def test_clumping_off_grid_is_an_error() -> None:
    mol = _molecule()
    with pytest.raises(ValueError, match="not on the grid"):
        compute_clumping(mol, _arrays(mol.n_atoms), temperature=305.0)


# --- dispatch ---------------------------------------------------------------


@pytest.mark.parametrize("kind", [AnalysisKind.CLUMPING, AnalysisKind.MASS_LAW])
def test_dispatch_requires_arrays(kind: AnalysisKind) -> None:
    mol = _molecule()
    with pytest.raises(ValueError, match="requires MoleculeArrays"):
        run_molecule_analysis(mol, AnalysisRequest(kind=kind, temperature=ON_GRID))


def test_dispatch_carries_the_aggregation_through() -> None:
    """AnalysisRequest gained the field so a caller going through the
    dispatcher can reach compute_clumping's existing "max" mode."""
    mol = _molecule(("C", "O", "H", "H"))
    arrays = _arrays(mol.n_atoms)
    arrays.clumping[0, 1, :] = 2.0

    mean = run_molecule_analysis(
        mol, AnalysisRequest(kind=AnalysisKind.CLUMPING, temperature=ON_GRID), arrays=arrays
    )
    biggest = run_molecule_analysis(
        mol,
        AnalysisRequest(kind=AnalysisKind.CLUMPING, temperature=ON_GRID, aggregation="max"),
        arrays=arrays,
    )

    assert biggest.sites[0].value > mean.sites[0].value


# --- the shape of the interpolation error near 20 K -------------------------


#: A real stored series: carbon 0 of molecule 100506, ascending in temperature.
#: Synthetic betas of the form exp(a/T) are exactly linear in 1/T and the fit
#: reproduces them perfectly, which hides the behaviour this pins. Real ln β
#: saturates towards low temperature, and that curvature is what the polynomial
#: cannot follow across the grid's largest gap.
REAL_BETA = (
    17.74909690791943,
    2.4702943316110195,
    1.671512077081535,
    1.4166782398184243,
    1.2937005200824854,
    1.2219122236264086,
    1.1751802318226219,
    1.1632328615149843,
    1.1438442471473065,
    1.1425573964291074,
    1.1186557330317157,
    1.1005106196158982,
    1.0863533486805947,
)
REAL_TEMPS = tuple(sorted(float(t) for t in TEMPERATURES_K))


def test_the_beta_fit_overshoots_just_above_the_lowest_grid_point() -> None:
    """A known, bounded defect, pinned so it cannot quietly get worse.

    The grid is even in temperature but the fit runs in 1/T, where the 20 K
    point sits ~60x further from its neighbour than the rest of the grid is
    spaced. The degree-5 polynomial overshoots into that gap and returns a β
    that rises with temperature for a few kelvin -- unphysical, and visible as
    a kink at the left edge of the curve.

    This asserts the damage stays confined to the low edge. If it ever reaches
    further up the range, the model needs replacing rather than documenting.
    """
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    grid, curve = beta_curve(temps, betas, num=800, method="poly5")
    rising = np.where(np.diff(curve) > 0)[0]

    assert rising.size, "the documented overshoot is gone -- update the docs with it"
    assert grid[rising[-1] + 1] < 30.0, "the overshoot has spread beyond the low edge"


def test_stored_grid_points_are_never_fitted() -> None:
    """Whatever the fit does between them, a grid temperature is read."""
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    for t, stored in zip(temps, betas, strict=True):
        got = interpolate_beta(temps, betas, float(t), warn=False)
        assert got.interpolated is False
        assert got.value == pytest.approx(stored)


# --- choosing an interpolation method ---------------------------------------
#
# The two families fail in different places, which is the whole reason both are
# offered: a polynomial misses the stored points but is a closed form, a
# monotone spline hits them all but cannot overshoot. These tests pin each
# property to the one method that has it, so neither can quietly acquire the
# other's defect.

METHOD_NAMES = [name for name, _label in INTERPOLATION_METHODS]
POLY_NAMES = [name for name in METHOD_NAMES if name.startswith("poly")]


def _fitted_at(temperature: float, method: str | None = None) -> float:
    """The *fitted* beta at a stored temperature.

    ``interpolate_beta`` short-circuits to the stored value on the grid, which
    is exactly what hides the fit's behaviour there. ``beta_curve`` never does
    that lookup, so asking it for a hair-thin range starting at a grid point
    reads what the fit actually says.
    """
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)
    _grid, curve = beta_curve(
        temps, betas, num=2, t_min=temperature, t_max=temperature + 1e-9, method=method
    )
    return float(curve[0])


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_every_method_returns_the_stored_value_on_the_grid(method: str) -> None:
    """The choice governs what happens between the points, never at them."""
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    for t, stored in zip(temps, betas, strict=True):
        got = interpolate_beta(temps, betas, float(t), warn=False, method=method)
        assert got.interpolated is False
        assert got.value == pytest.approx(stored)
        assert got.method == method


def test_the_monotone_spline_passes_through_every_stored_point() -> None:
    for t, stored in zip(REAL_TEMPS, REAL_BETA, strict=True):
        assert _fitted_at(t, "pchip") == pytest.approx(stored, rel=1e-9)


def test_a_polynomial_does_not_pass_through_the_stored_points() -> None:
    """The contrast that makes the previous test mean something.

    A least-squares fit is not an interpolant. Because the package returns the
    stored value *at* a grid point and the fitted value beside it, this gap is
    a discontinuity a user can land on -- it is the cost of the closed form.
    """
    misses = [
        abs(_fitted_at(t, "poly5") - stored) / stored
        for t, stored in zip(REAL_TEMPS, REAL_BETA, strict=True)
    ]
    assert max(misses) > 1e-4


@pytest.mark.parametrize("method", ["local2", "pchip", "poly2", "poly3", "poly4"])
def test_the_alternatives_do_not_overshoot_below_60_k(method: str) -> None:
    """Beta falls monotonically with temperature. Only degree 5 forgets.

    The companion test above pins degree 5's overshoot as a known defect; this
    pins that every other offered method is free of it, which is what makes
    offering them worthwhile.
    """
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    _grid, curve = beta_curve(temps, betas, num=800, method=method)
    assert np.all(np.diff(curve) <= 1e-12), f"{method} turns back on itself"


def test_the_default_is_the_method_that_stays_physical() -> None:
    """Degree 5 was the default until 0.5.0 and overshoots below 60 K.

    It returns a beta that rises with temperature there -- unphysical, and in
    every site measured. The default is now a local quadratic: physical
    everywhere, accurate to about a third of a per mil, and the
    second-degree polynomial the accompanying paper describes.
    """
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    assert DEFAULT_METHOD == "local2"
    _grid, curve = beta_curve(temps, betas, num=800)
    assert np.all(np.diff(curve) <= 1e-12), "the default fit rises with temperature"
    assert interpolate_beta(temps, betas, 305.0, warn=False).value == (
        interpolate_beta(temps, betas, 305.0, warn=False, method="local2").value
    )


def test_changing_the_default_left_stored_temperatures_alone() -> None:
    """Only off-grid values moved. A stored temperature is read, never fitted,
    so every published value at one is exactly what it was under degree 5."""
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    for t, stored in zip(temps, betas, strict=True):
        now = interpolate_beta(temps, betas, float(t), warn=False).value
        before = interpolate_beta(temps, betas, float(t), warn=False, method="poly5").value
        assert now == before == pytest.approx(stored)


@pytest.mark.parametrize("method", POLY_NAMES)
def test_polynomial_degree_is_read_from_the_name(method: str) -> None:
    family, degree = parse_method(method)
    assert family == "poly"
    assert degree == int(method[4:])


def test_the_spline_carries_no_degree() -> None:
    assert parse_method("pchip") == ("pchip", 0)


@pytest.mark.parametrize("empty", [None, "", "  "])
def test_no_method_means_the_default(empty: str | None) -> None:
    """A form that submits a blank select gets the default, not an error."""
    assert parse_method(empty) == parse_method(DEFAULT_METHOD)


@pytest.mark.parametrize(
    "bad", ["poly1", "poly6", "poly", "spline", "linear", "PCHIP!", "local", "local3"]
)
def test_an_unoffered_method_is_refused_by_name(bad: str) -> None:
    """Degree 1 and degree 6 are as wrong as a typo, and fail the same way."""
    with pytest.raises(ValueError, match="Unknown interpolation method"):
        parse_method(bad)


def test_a_bad_method_is_refused_even_at_a_stored_temperature() -> None:
    """Otherwise a typo would pass unnoticed until someone typed 305 K."""
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    with pytest.raises(ValueError, match="Unknown interpolation method"):
        interpolate_beta(temps, betas, float(REAL_TEMPS[0]), warn=False, method="poly9")


# --- the local quadratic -------------------------------------------------------


def _local_windows(temps: np.ndarray, targets: np.ndarray) -> list[tuple[int, ...]]:
    """Which stored points the local quadratic uses for each target."""
    grid = np.sort(temps)
    out = []
    for t in targets:
        below = int(np.searchsorted(grid, t, side="right")) - 1
        lo = min(max(below - 1, 0), len(grid) - 4)
        out.append(tuple(range(lo, lo + 4)))
    return out


def test_the_local_window_brackets_the_temperature() -> None:
    """Two stored points below, two above -- the four "around" it."""
    temps = np.array(REAL_TEMPS, dtype=float)
    grid = np.sort(temps)
    for t in (80.0, 150.0, 250.0, 305.0, 350.0):
        window = grid[list(_local_windows(temps, np.array([t]))[0])]
        assert (window < t).sum() == 2 and (window > t).sum() == 2, (t, window)


def test_the_local_window_changes_only_at_stored_temperatures() -> None:
    """Choosing the four nearest by distance moved the window partway through
    an interval, and the curve took a visible step there. Bracketing changes
    it only at stored points, which are read rather than fitted, so the drawn
    curve never steps."""
    temps = np.array(REAL_TEMPS, dtype=float)
    scan = np.linspace(temps.min(), temps.max(), 200_001)
    windows = _local_windows(temps, scan)
    switches = [
        0.5 * (scan[i] + scan[i + 1]) for i in range(len(scan) - 1) if windows[i] != windows[i + 1]
    ]
    assert switches
    for t in switches:
        assert np.min(np.abs(temps - t)) < 0.01, f"window changes mid-interval at {t:.2f} K"


def test_the_local_quadratic_beats_a_global_one() -> None:
    """The reason a local window is worth having: one quadratic across 20 to
    500 K cannot follow the curvature, and misses by several per mil."""
    temps = np.array(REAL_TEMPS, dtype=float)
    betas = np.array(REAL_BETA, dtype=float)

    def worst_leave_one_out(method: str) -> float:
        worst = 0.0
        for j in range(2, len(temps) - 2):
            keep = np.ones(len(temps), bool)
            keep[j] = False
            got = interpolate_beta(
                temps[keep], betas[keep], float(temps[j]), warn=False, method=method
            ).value
            worst = max(worst, abs(1000 * (np.log(got) - np.log(betas[j]))))
        return worst

    assert worst_leave_one_out("local2") < worst_leave_one_out("poly2")


def test_the_local_quadratic_copes_with_a_short_grid() -> None:
    """Fewer than four stored points still has to give an answer rather than
    fail inside numpy -- a test fixture or a partial file can have two."""
    got = interpolate_beta([200.0, 300.0], [1.2, 1.1], 250.0, warn=False, method="local2")
    assert 1.1 < got.value < 1.2
