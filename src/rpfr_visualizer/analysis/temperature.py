"""Temperature-grid lookup and off-grid beta interpolation.

A stored temperature is read. Anything else interpolates ``ln β`` against
``1/T``. Measured on 9,011 schema 2.2 sites, in per mil of 1000 ln β:

``local2`` (the default)
    A quadratic fitted to the four stored points around the temperature --
    two below, two above, shifted inward at the ends of the grid. Typical
    error 0.3 per mil, worst 5% 9 per mil, and it never overshoots. A quadratic
    in 1/T carries the 1/T² term that dominates ln β as temperature rises, and
    a local window follows curvature a single global quadratic cannot: the
    global one is about 30 times worse.

``pchip``
    Monotone piecewise-cubic spline, Fritsch-Carlson slopes. The most accurate
    (0.1 per mil typical, 1.7 worst 5%), exact at every stored point, never
    overshoots.

``poly2`` ... ``poly5``
    Least-squares polynomial over the whole grid. Degree 5 was the default
    until 0.5.0; it overshoots below 60 K in every site measured, returning a
    β that rises with temperature. Lower degrees do not overshoot but miss the
    stored points by several per mil.

No method can be *accurate* between 20 K and 60 K, where nothing is stored.
Prefer the stored grid below 60 K.

Exact grid temperatures return the stored β under every method.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Fallback validity window, for a caller with no grid to hand.
#:
#: The real window is whatever the open file stores, and that is what the
#: functions below use: schema 2.2 extended the grid to 500 K, so a constant
#: would have called every 460 K and 500 K reading an extrapolation and warned
#: about values the file holds outright.
T_MIN_K = 20.0
T_MAX_K = 420.0
POLY_DEGREE = 5
_GRID_TOL = 1e-6

#: Lowest and highest polynomial degree offered. Degree 1 is straight-line
#: ln β vs 1/T, which no real site follows; above 5 the fit starts chasing
#: noise between the closely spaced 273.15 / 298.15 / 300 points.
MIN_POLY_DEGREE = 2
MAX_POLY_DEGREE = 5

#: The method used when a caller does not name one.
#:
#: Degree 5 was the default until 0.5.0, and it is the only offered method
#: that overshoots below 60 K -- in every site measured it returns a β that
#: rises with temperature there, which is unphysical. The local quadratic is
#: physical everywhere, accurate to about a third of a per mil, and a
#: second-degree polynomial -- the method the accompanying paper describes.
#: Values at stored temperatures are unaffected by the change.
DEFAULT_METHOD = "local2"

#: Offered methods as ``(name, label)``, in picker order. Measured on v2.2
#: (leave-one-out, per mil of 1000 ln β, median/worst): local2 0.3/9,
#: pchip 0.1/1.7, poly5 1.0/171 (overshoots below 60 K), poly4 3.6/75,
#: poly3 5.7/49, poly2 7.7/52.
INTERPOLATION_METHODS: tuple[tuple[str, str], ...] = (
    ("local2", "Local quadratic"),
    ("pchip", "Monotone spline"),
    ("poly5", "Polynomial, degree 5"),
    ("poly4", "Polynomial, degree 4"),
    ("poly3", "Polynomial, degree 3"),
    ("poly2", "Polynomial, degree 2"),
)


def describe_fit(method: str | None = None) -> str:
    """How an off-grid β is obtained, as the end of a sentence.

    One phrase, used by every message that says a value was fitted -- the
    provenance line, the notebook warning, the download page -- so they cannot
    describe the same fit three different ways, which is what they had done.
    """
    family, degree = parse_method(method)
    if family == "local":
        return "a local quadratic through the four nearest stored points"
    if family == "pchip":
        return "a monotone spline through the stored points"
    return f"a degree-{degree} polynomial over all the stored points"


@dataclass(frozen=True)
class BetaAtResult:
    """Beta evaluated at a temperature, with interpolation metadata."""

    value: float
    interpolated: bool
    out_of_range: bool
    #: Which method produced *value*. Meaningful only when ``interpolated``;
    #: a grid point returns the stored number whatever the method.
    method: str = DEFAULT_METHOD


def _canonical(method: str | None) -> str:
    """Normalise a method name, treating blank as "not given"."""
    return (method or "").strip().lower() or DEFAULT_METHOD


def parse_method(method: str | None) -> tuple[str, int]:
    """Split a method name into ``(family, degree)``.

    Args:
        method: One of :data:`INTERPOLATION_METHODS`, or None for the default.

    Returns:
        ``("local", 2)``, ``("poly", degree)`` or ``("pchip", 0)``.

    Raises:
        ValueError: If the name is not one this module implements. The message
            is written for a person; it reaches the browser unchanged.
    """
    name = _canonical(method)
    if name == "pchip":
        return ("pchip", 0)
    if name == "local2":
        return ("local", 2)
    if name.startswith("poly") and name[4:].isdigit():
        degree = int(name[4:])
        if MIN_POLY_DEGREE <= degree <= MAX_POLY_DEGREE:
            return ("poly", degree)
    offered = ", ".join(n for n, _label in INTERPOLATION_METHODS)
    raise ValueError(f"Unknown interpolation method {method!r}. Choose one of: {offered}.")


def is_on_grid(temperature: float, temperatures: tuple[float, ...] | list[float]) -> bool:
    """Return True if *temperature* matches a grid point within tolerance."""
    return any(abs(float(t) - temperature) < _GRID_TOL for t in temperatures)


def grid_index(temperature: float, temperatures: tuple[float, ...] | list[float]) -> int | None:
    """Return the index of an exact grid temperature, or None."""
    for idx, temp in enumerate(temperatures):
        if abs(float(temp) - temperature) < _GRID_TOL:
            return idx
    return None


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson slopes: the monotonicity-preserving choice.

    A plain cubic spline picks slopes for smoothness and may overshoot between
    points. These slopes are capped so each segment stays monotone, which is
    what keeps β from turning back on itself in the 20–60 K gap.
    """
    h = np.diff(x)
    d = np.diff(y) / h
    if y.size == 2:
        return np.array([d[0], d[0]])

    m = np.zeros_like(y)
    for i in range(1, y.size - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0  # a local extremum: flat, so neither side overshoots
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    def end(dk: float, dk1: float, hk: float, hk1: float) -> float:
        slope = ((2 * hk + hk1) * dk - hk * dk1) / (hk + hk1)
        if slope * dk <= 0:
            return 0.0
        if dk * dk1 <= 0 and abs(slope) > abs(3 * dk):
            return 3 * dk
        return slope

    m[0] = end(d[0], d[1], h[0], h[1])
    m[-1] = end(d[-1], d[-2], h[-1], h[-2])
    return m


def _pchip_evaluator(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Build a monotone cubic Hermite interpolant over ascending *x*."""
    m = _pchip_slopes(x, y)

    def evaluate(xq: np.ndarray) -> np.ndarray:
        k = np.clip(np.searchsorted(x, xq) - 1, 0, x.size - 2)
        h = x[k + 1] - x[k]
        t = (xq - x[k]) / h
        t2 = t * t
        t3 = t2 * t
        out: np.ndarray = (
            (2 * t3 - 3 * t2 + 1) * y[k]
            + (t3 - 2 * t2 + t) * h * m[k]
            + (-2 * t3 + 3 * t2) * y[k + 1]
            + (t3 - t2) * h * m[k + 1]
        )
        return out

    return evaluate


#: How many stored points each local fit uses, and the degree fitted to them.
#: Four points under a quadratic leave one degree of freedom, so the fit
#: smooths rather than interpolating -- which is what keeps it from ringing
#: the way a quartic through the same four points would.
LOCAL_WINDOW = 4
LOCAL_DEGREE = 2


def _local_quadratic_evaluator(
    temperatures: np.ndarray, log_betas: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """A quadratic in 1/T through the four stored points around each target.

    The window is the two stored points below the target and the two above,
    shifted inward where the grid runs out -- so between 20 K and 60 K it is
    {20, 60, 100, 140}. That choice matters. Taking the four *nearest by
    distance* instead changes the window partway through four of the
    intervals (at 206.6, 239.1, 326.6 and 359.1 K on the 2.2 grid), and the
    curve takes a visible step of up to about a per mil there. Bracketing
    changes it only at stored temperatures, which are read rather than
    fitted, so the drawn curve never steps -- at no cost in accuracy on the
    9,011 sites measured.

    One quadratic per window, fitted up front: there are only as many windows
    as stored temperatures, so evaluating a dense curve costs a lookup rather
    than a least-squares solve per point.
    """
    order = np.argsort(temperatures)
    grid = np.asarray(temperatures, dtype=float)[order]
    y = np.asarray(log_betas, dtype=float)[order]
    n = int(grid.size)
    width = min(LOCAL_WINDOW, n)
    degree = min(LOCAL_DEGREE, width - 1)
    fits = {
        lo: np.polyfit(1.0 / grid[lo : lo + width], y[lo : lo + width], degree)
        for lo in range(n - width + 1)
    }

    def evaluate(xq: np.ndarray) -> np.ndarray:
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        below = np.searchsorted(grid, 1.0 / xq, side="right") - 1
        start = np.clip(below - 1, 0, n - width)
        out = np.empty_like(xq)
        for lo in np.unique(start):
            mask = start == lo
            out[mask] = np.polyval(fits[int(lo)], xq[mask])
        return out

    return evaluate


def _fit_ln_beta(
    temperatures: np.ndarray,
    betas: np.ndarray,
    method: str | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit ln(β) against 1/T and return a callable evaluating it at 1/T.

    The callable takes *reciprocal* temperature, matching the space the fit
    lives in, so a caller drawing a curve converts once rather than per point.
    It is fitted once, so a plot costs one fit rather than one per pixel.
    """
    if temperatures.size < 2:
        raise ValueError("Need at least two temperature points to interpolate beta")
    if np.any(temperatures <= 0):
        raise ValueError("Temperatures must be positive Kelvin")
    if np.any(betas <= 0):
        raise ValueError("Beta values must be positive for ln(beta) interpolation")

    family, degree = parse_method(method)
    # Sort by 1/T, not by T. The grid is stored out of order (273.15 and
    # 298.15 are appended), and the spline needs ascending knots.
    x = 1.0 / temperatures
    y = np.log(betas)
    order = np.argsort(x)
    x, y = x[order], y[order]

    if family == "poly":
        return np.poly1d(np.polyfit(x, y, min(degree, int(x.size) - 1)))
    if family == "local":
        return _local_quadratic_evaluator(1.0 / x, y)
    return _pchip_evaluator(x, y)


def _fit_ln_beta_poly(temperatures: np.ndarray, betas: np.ndarray) -> np.poly1d:
    """Fit ln(β) vs 1/T with degree min(POLY_DEGREE, n-1).

    Retained for callers that want the coefficients themselves rather than an
    opaque evaluator; :func:`_fit_ln_beta` is what the interpolators use.
    """
    if temperatures.size < 2:
        raise ValueError("Need at least two temperature points to interpolate beta")
    if np.any(temperatures <= 0):
        raise ValueError("Temperatures must be positive Kelvin")
    if np.any(betas <= 0):
        raise ValueError("Beta values must be positive for ln(beta) interpolation")

    x = 1.0 / temperatures
    y = np.log(betas)
    return np.poly1d(np.polyfit(x, y, min(POLY_DEGREE, int(temperatures.size) - 1)))


def interpolate_beta(
    temperatures: tuple[float, ...] | list[float] | np.ndarray,
    betas: tuple[float, ...] | list[float] | np.ndarray,
    temperature: float,
    *,
    warn: bool = True,
    method: str | None = None,
) -> BetaAtResult:
    """Return β at *temperature*, interpolating off-grid via ln(β) vs 1/T.

    Parameters
    ----------
    temperatures:
        Grid temperatures in Kelvin.
    betas:
        Beta values aligned with *temperatures*.
    temperature:
        Requested temperature in Kelvin.
    warn:
        If True, emit a :class:`UserWarning` when *temperature* is outside
        ``[T_MIN_K, T_MAX_K]``.
    method:
        One of :data:`INTERPOLATION_METHODS`; None uses
        :data:`DEFAULT_METHOD`. Ignored at exact grid temperatures, which
        return the stored value.
    """
    temps = np.asarray(temperatures, dtype=np.float64)
    vals = np.asarray(betas, dtype=np.float64)
    if temps.shape != vals.shape:
        raise ValueError(f"temperatures and betas length mismatch: {temps.shape} vs {vals.shape}")
    if temps.size == 0:
        raise ValueError("Empty temperature grid")

    name = _canonical(method)
    parse_method(name)  # reject a bad name even when the answer is a stored value
    idx = grid_index(temperature, tuple(float(t) for t in temps))
    # The validated range is the stored one, not a constant: asking a 20-500 K
    # file to justify 460 K against a hardcoded 420 K ceiling would warn about
    # a temperature it actually holds.
    low, high = float(temps.min()), float(temps.max())
    out_of_range = temperature < low - _GRID_TOL or temperature > high + _GRID_TOL
    if out_of_range and warn:
        warnings.warn(
            f"Temperature {temperature} K is outside this dataset's range "
            f"[{low:g}, {high:g}] K; extrapolated beta may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    if idx is not None:
        return BetaAtResult(
            value=float(vals[idx]),
            interpolated=False,
            out_of_range=out_of_range,
            method=name,
        )

    evaluate = _fit_ln_beta(temps, vals, name)
    ln_beta = float(np.atleast_1d(evaluate(np.atleast_1d(1.0 / temperature)))[0])
    return BetaAtResult(
        value=float(math.exp(ln_beta)),
        interpolated=True,
        out_of_range=out_of_range,
        method=name,
    )


def beta_curve(
    temperatures: tuple[float, ...] | list[float] | np.ndarray,
    betas: tuple[float, ...] | list[float] | np.ndarray,
    *,
    num: int = 160,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the ln β vs 1/T fit across a dense temperature range.

    :func:`interpolate_beta` answers one temperature and refits every time,
    which is the wrong shape for drawing a curve. This fits once and evaluates
    across *num* points, so a plot costs one fit rather than one per pixel.

    Args:
        temperatures: The stored grid, in any order.
        betas: Stored β, aligned with *temperatures*.
        num: How many points to evaluate.
        t_min: Low end, defaulting to the lowest stored temperature.
        t_max: High end, defaulting to the highest.
        method: One of :data:`INTERPOLATION_METHODS`; None uses
            :data:`DEFAULT_METHOD`.

    Returns:
        ``(temperatures, betas)``, both ascending in temperature.

    Raises:
        ValueError: If the inputs disagree in length, hold values the
            logarithm cannot take, or name a method that does not exist.
    """
    temps = np.asarray(temperatures, dtype=float)
    vals = np.asarray(betas, dtype=float)
    if temps.shape != vals.shape:
        raise ValueError("temperatures and betas must have the same length")

    evaluate = _fit_ln_beta(temps, vals, method)

    low = float(t_min if t_min is not None else temps.min())
    high = float(t_max if t_max is not None else temps.max())
    if high <= low:
        raise ValueError("t_max must be greater than t_min")

    grid = np.linspace(low, high, max(2, int(num)))
    return grid, np.exp(evaluate(1.0 / grid))


def beta_at_temperature(
    temperatures: tuple[float, ...] | list[float] | np.ndarray,
    betas: tuple[float, ...] | list[float] | np.ndarray,
    temperature: float,
    *,
    warn: bool = True,
    method: str | None = None,
) -> float:
    """Convenience wrapper returning only the beta value."""
    return interpolate_beta(temperatures, betas, temperature, warn=warn, method=method).value


__all__ = [
    "DEFAULT_METHOD",
    "INTERPOLATION_METHODS",
    "MAX_POLY_DEGREE",
    "MIN_POLY_DEGREE",
    "POLY_DEGREE",
    "T_MAX_K",
    "T_MIN_K",
    "BetaAtResult",
    "beta_at_temperature",
    "beta_curve",
    "grid_index",
    "interpolate_beta",
    "is_on_grid",
    "describe_fit",
    "parse_method",
]
