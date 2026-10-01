"""Per-element dataset statistics for standards and color grounding.

Computes p5 / p50 (median) / p95 of site β on the temperature grid, plus
α and ε derived vs the median β (for grounding relative color scales).
Off-grid temperatures interpolate each percentile curve in ``ln(y)`` vs
``1/T`` with the same default fit β uses (for positive β/α curves; ε is
interpolated linearly in 1/T because it can be negative).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from rpfr_visualizer.analysis.temperature import T_MAX_K, T_MIN_K, beta_at_temperature
from rpfr_visualizer.io.loader import Molecule, _molecule_keys, _temperatures_from_file
from rpfr_visualizer.io.schema import TEMPERATURES_K


@dataclass(frozen=True)
class ElementPercentiles:
    """Percentile summary for one element at one temperature."""

    p5: float
    p50: float
    p95: float


@dataclass(frozen=True)
class ElementQuantityStats:
    """β plus α/ε-vs-median percentiles for one element across the grid."""

    element: str
    temperatures: tuple[float, ...]
    beta: tuple[ElementPercentiles, ...]
    alpha_vs_median: tuple[ElementPercentiles, ...]
    epsilon_vs_median: tuple[ElementPercentiles, ...]

    def _series(self, kind: str) -> tuple[ElementPercentiles, ...]:
        if kind == "beta":
            return self.beta
        if kind == "alpha":
            return self.alpha_vs_median
        if kind == "epsilon":
            return self.epsilon_vs_median
        raise KeyError(f"Unknown quantity kind: {kind!r}")

    def at(
        self,
        kind: str,
        temperature: float,
        *,
        warn: bool = False,
        method: str | None = None,
    ) -> ElementPercentiles:
        """Return percentiles at *temperature* (grid lookup or interpolated).

        *method* is passed through to the β interpolator so the colour scale
        is built the same way as the values it normalises. ε keeps its own
        linear interpolation: it can be negative, so ln β vs 1/T does not
        apply to it at all.
        """
        series = self._series(kind)
        temps = self.temperatures
        # Exact grid
        for idx, temp in enumerate(temps):
            if abs(temp - temperature) < 1e-6:
                return series[idx]

        p5s = tuple(s.p5 for s in series)
        p50s = tuple(s.p50 for s in series)
        p95s = tuple(s.p95 for s in series)

        if kind == "epsilon":
            # ε can be negative — linear interp in 1/T on the sorted grid.
            return ElementPercentiles(
                p5=_interp_linear_inv_t(temps, p5s, temperature),
                p50=_interp_linear_inv_t(temps, p50s, temperature),
                p95=_interp_linear_inv_t(temps, p95s, temperature),
            )

        return ElementPercentiles(
            p5=beta_at_temperature(temps, p5s, temperature, warn=warn, method=method),
            p50=beta_at_temperature(temps, p50s, temperature, warn=warn, method=method),
            p95=beta_at_temperature(temps, p95s, temperature, warn=warn, method=method),
        )


def _interp_linear_inv_t(
    temperatures: Sequence[float],
    values: Sequence[float],
    temperature: float,
) -> float:
    """Piecewise-linear interpolation of *values* vs 1/T."""
    t = np.asarray(temperatures, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    x = 1.0 / t
    order = np.argsort(x)
    return float(np.interp(1.0 / temperature, x[order], v[order]))


def _percentiles(values: np.ndarray) -> ElementPercentiles:
    p5, p50, p95 = np.percentile(values, [5.0, 50.0, 95.0])
    return ElementPercentiles(p5=float(p5), p50=float(p50), p95=float(p95))


def _build_from_beta_lists(
    temperatures: tuple[float, ...],
    betas_by_element: dict[str, list[list[float]]],
) -> dict[str, ElementQuantityStats]:
    """Build stats from per-element lists of beta vectors (one list per site)."""
    n_t = len(temperatures)
    out: dict[str, ElementQuantityStats] = {}
    for element, site_betas in betas_by_element.items():
        if not site_betas:
            continue
        arr = np.asarray(site_betas, dtype=np.float64)  # (n_sites, n_t)
        if arr.ndim != 2 or arr.shape[1] != n_t:
            continue
        beta_pct: list[ElementPercentiles] = []
        alpha_pct: list[ElementPercentiles] = []
        eps_pct: list[ElementPercentiles] = []
        for t_idx in range(n_t):
            col = arr[:, t_idx]
            col = col[np.isfinite(col) & (col > 0)]
            if col.size == 0:
                continue
            b = _percentiles(col)
            beta_pct.append(b)
            alphas = col / b.p50
            a = _percentiles(alphas)
            alpha_pct.append(a)
            eps = 1000.0 * (alphas - 1.0)
            eps_pct.append(_percentiles(eps))
        if len(beta_pct) != n_t:
            # Skip elements that failed at some T
            continue
        out[element] = ElementQuantityStats(
            element=element,
            temperatures=temperatures,
            beta=tuple(beta_pct),
            alpha_vs_median=tuple(alpha_pct),
            epsilon_vs_median=tuple(eps_pct),
        )
    return out


@dataclass(frozen=True)
class ElementStatsCache:
    """Dataset-level per-element β / α / ε percentile cache."""

    temperatures: tuple[float, ...]
    by_element: dict[str, ElementQuantityStats]
    n_sites: int
    source: str = ""

    def elements(self) -> tuple[str, ...]:
        """Return sorted element symbols present in the cache."""
        return tuple(sorted(self.by_element))

    def median_beta(
        self,
        element: str,
        temperature: float,
        *,
        warn: bool = False,
        method: str | None = None,
    ) -> float | None:
        """Return dataset median β(T) for *element*, or None if missing."""
        stats = self.by_element.get(element)
        if stats is None:
            return None
        return stats.at("beta", temperature, warn=warn, method=method).p50

    def quantity_range(
        self,
        element: str,
        temperature: float,
        *,
        kind: str = "beta",
        warn: bool = False,
        method: str | None = None,
    ) -> tuple[float, float] | None:
        """Return (p5, p95) for the requested quantity, or None."""
        stats = self.by_element.get(element)
        if stats is None:
            return None
        pct = stats.at(kind, temperature, warn=warn, method=method)
        return pct.p5, pct.p95

    def percentiles(
        self,
        element: str,
        temperature: float,
        *,
        kind: str = "beta",
        warn: bool = False,
        method: str | None = None,
    ) -> ElementPercentiles | None:
        """Return full percentile triple for *element* / *kind* / *temperature*."""
        stats = self.by_element.get(element)
        if stats is None:
            return None
        return stats.at(kind, temperature, warn=warn, method=method)

    @classmethod
    def from_molecules(
        cls,
        molecules: Iterable[Molecule],
        *,
        temperatures: tuple[float, ...] | None = None,
        source: str = "molecules",
    ) -> ElementStatsCache:
        """Build cache by scanning in-memory :class:`Molecule` records."""
        temps = temperatures or TEMPERATURES_K
        n_t = len(temps)
        buckets: dict[str, list[list[float]]] = defaultdict(list)
        n_sites = 0
        for mol in molecules:
            for site in mol.sites:
                if len(site.beta) != n_t:
                    # Align via beta_at on the requested grid
                    try:
                        vec = [site.beta_at(t, warn=False) for t in temps]
                    except (ValueError, IndexError):
                        continue
                else:
                    # Reorder if site temperatures differ
                    if tuple(site.temperatures) == temps:
                        vec = list(site.beta)
                    else:
                        vec = [site.beta_at(t, warn=False) for t in temps]
                buckets[site.element].append(vec)
                n_sites += 1
        by_element = _build_from_beta_lists(temps, buckets)
        return cls(temperatures=temps, by_element=by_element, n_sites=n_sites, source=source)

    def to_dict(self) -> dict:
        """Serialize to plain JSON-compatible data."""
        return {
            "temperatures": list(self.temperatures),
            "n_sites": int(self.n_sites),
            "source": self.source,
            "by_element": {
                element: {
                    "beta": [[p.p5, p.p50, p.p95] for p in stats.beta],
                    "alpha_vs_median": [[p.p5, p.p50, p.p95] for p in stats.alpha_vs_median],
                    "epsilon_vs_median": [[p.p5, p.p50, p.p95] for p in stats.epsilon_vs_median],
                }
                for element, stats in self.by_element.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ElementStatsCache:
        """Rebuild from :meth:`to_dict` output."""
        temperatures = tuple(float(t) for t in payload["temperatures"])

        def series(rows) -> tuple[ElementPercentiles, ...]:
            return tuple(ElementPercentiles(float(a), float(b), float(c)) for a, b, c in rows)

        by_element = {
            element: ElementQuantityStats(
                element=element,
                temperatures=temperatures,
                beta=series(data["beta"]),
                alpha_vs_median=series(data["alpha_vs_median"]),
                epsilon_vs_median=series(data["epsilon_vs_median"]),
            )
            for element, data in payload["by_element"].items()
        }
        return cls(
            temperatures=temperatures,
            by_element=by_element,
            n_sites=int(payload.get("n_sites", 0)),
            source=str(payload.get("source", "")),
        )

    def save(self, path: Path | str) -> None:
        """Write the cache to *path* as JSON."""
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> ElementStatsCache:
        """Read a cache written by :meth:`save`."""
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_hdf5_cached(
        cls,
        path: Path | str,
        *,
        cache_path: Path | str,
        max_molecules: int | None = None,
        temperatures: tuple[float, ...] | None = None,
    ) -> ElementStatsCache:
        """Load persisted statistics, computing and saving them on a miss.

        Sampling molecules out of a multi-gigabyte file is dominated by cold
        random reads — a minute on external storage, instant once the OS cache
        is warm. Persisting the small result means that is paid once ever
        rather than once per session.
        """
        cache_file = Path(cache_path)
        if cache_file.exists():
            try:
                return cls.load(cache_file)
            except (OSError, ValueError, KeyError):
                pass  # unreadable or stale format: recompute
        stats = cls.from_hdf5(path, max_molecules=max_molecules, temperatures=temperatures)
        try:
            stats.save(cache_file)
        except OSError:
            pass  # a read-only data directory must not break the panel
        return stats

    @classmethod
    def from_hdf5(
        cls,
        path: Path | str,
        *,
        max_molecules: int | None = None,
        temperatures: tuple[float, ...] | None = None,
    ) -> ElementStatsCache:
        """Build cache by streaming ``rpfr_single`` / ``atom_symbol`` from HDF5."""
        path = Path(path)
        buckets: dict[str, list[list[float]]] = defaultdict(list)
        n_sites = 0
        with h5py.File(path, "r") as h5:
            temps = temperatures or _temperatures_from_file(h5)
            keys = _molecule_keys(h5)
            if max_molecules is not None:
                keys = keys[: max(0, int(max_molecules))]
            for key in keys:
                group = h5[key]
                if "rpfr_single" not in group or "atom_symbol" not in group:
                    continue
                rpfr = np.asarray(group["rpfr_single"][()], dtype=np.float64)
                symbols = group["atom_symbol"][()]
                if rpfr.ndim != 2 or rpfr.shape[0] != len(symbols):
                    continue
                for i, raw_sym in enumerate(symbols):
                    sym = raw_sym.decode("utf-8") if isinstance(raw_sym, bytes) else str(raw_sym)
                    row = rpfr[i]
                    if row.shape[0] != len(temps):
                        continue
                    buckets[sym].append(row.tolist())
                    n_sites += 1
        by_element = _build_from_beta_lists(temps, buckets)
        return cls(
            temperatures=temps,
            by_element=by_element,
            n_sites=n_sites,
            source=str(path),
        )


__all__ = [
    "ElementPercentiles",
    "ElementQuantityStats",
    "ElementStatsCache",
    "T_MAX_K",
    "T_MIN_K",
]
