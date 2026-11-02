"""Per-site training data, flattened into arrays.

Every model here predicts one thing: ``1000 ln β`` for a site, at each stored
temperature. :class:`SiteTable` holds that target for many sites alongside the
graph descriptors the descriptor models read -- the bond paths radiating from
the site and the R1-R5 local-environment SMILES -- with every string interned
to an integer, so a hundred thousand molecules fit in memory and the fits are
array operations rather than Python loops.

The same table is built from loaded :class:`~rpfr_visualizer.io.schema.MoleculeRecord`
objects for prediction, so training and prediction cannot read the descriptors
differently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np

from rpfr_visualizer.io.loader import _decode_str_array, _temperatures_from_file
from rpfr_visualizer.io.schema import MoleculeRecord

#: Radii of the local-environment fingerprints (``local_smiles`` columns).
N_RADII = 5

#: Characters that separate atoms in a bond path such as ``C-C=O``.
_BOND_CHARS = re.compile(r"[-=#:]")


def path_depth(path: str) -> int:
    """Number of bonds in a bond path: ``C-H`` is 1, ``C-C=O`` is 2."""
    return len(_BOND_CHARS.findall(path))


def to_target(beta: np.ndarray) -> np.ndarray:
    """``1000 ln β``, the per-mil quantity every model is fitted to.

    Non-positive or non-finite β map to NaN, which the fits skip.
    """
    beta = np.asarray(beta, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: np.ndarray = 1000.0 * np.log(beta)
    out[~np.isfinite(out)] = np.nan
    return out


def from_target(y: np.ndarray) -> np.ndarray:
    """Invert :func:`to_target`."""
    return np.exp(np.asarray(y, dtype=np.float64) / 1000.0)


class _Interner:
    """Assigns each distinct string the next integer."""

    def __init__(self) -> None:
        self.index: dict[str, int] = {}
        self.items: list[str] = []

    def __call__(self, text: str) -> int:
        found = self.index.get(text)
        if found is None:
            found = len(self.items)
            self.index[text] = found
            self.items.append(text)
        return found


@dataclass(frozen=True)
class SiteTable:
    """Many sites, their target and their descriptors, as flat arrays.

    Attributes:
        molecule_keys: One entry per molecule, in the order read.
        site_molecule: Index into *molecule_keys* for each site.
        site_atom: Atom index of each site within its molecule.
        element: Element symbol of each site.
        y: ``1000 ln β`` per site and temperature; NaN where undefined.
        has_graph: False for sites whose molecule lacks the descriptors.
        path_ptr: Site *i*'s bond paths are ``path_ids[path_ptr[i]:path_ptr[i+1]]``.
        path_ids: Interned bond paths, repeated when a path occurs twice.
        path_vocab: The string behind each path id.
        env_ids: Interned local-environment SMILES, ``(n_sites, 5)``; -1 if empty.
        env_vocab: The string behind each environment id.
        temperatures: The grid the columns of *y* refer to.
    """

    molecule_keys: tuple[str, ...]
    site_molecule: np.ndarray
    site_atom: np.ndarray
    element: np.ndarray
    y: np.ndarray
    has_graph: np.ndarray
    path_ptr: np.ndarray
    path_ids: np.ndarray
    path_vocab: tuple[str, ...]
    env_ids: np.ndarray
    env_vocab: tuple[str, ...]
    temperatures: tuple[float, ...]

    def __len__(self) -> int:
        return int(self.element.shape[0])

    @cached_property
    def path_depths(self) -> np.ndarray:
        """Bond count of each entry of :attr:`path_vocab`."""
        return np.array([path_depth(p) for p in self.path_vocab], dtype=np.int16)

    def usable(self) -> np.ndarray:
        """Sites with descriptors and a finite target at every temperature."""
        usable: np.ndarray = self.has_graph & np.isfinite(self.y).all(axis=1)
        return usable

    def rows_for(self, keys: Iterable[str], *, element: str | None = None) -> np.ndarray:
        """Indices of usable sites in molecules *keys*, optionally of one element."""
        wanted = set(keys)
        mol_mask = np.array([k in wanted for k in self.molecule_keys], dtype=bool)
        mask = self.usable()
        if mol_mask.size:
            mask &= mol_mask[self.site_molecule]
        else:
            mask &= False
        if element is not None:
            mask &= self.element == element
        return np.flatnonzero(mask)

    def paths_coo(
        self, rows: np.ndarray, *, max_depth: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bond-path counts of *rows*, as sparse ``(row, path_id, count)`` triples.

        *row* indexes into *rows*, not into the table. Paths with more than
        *max_depth* bonds are dropped. Triples are sorted by row.
        """
        rows = np.asarray(rows, dtype=np.int64)
        starts = self.path_ptr[rows]
        lens = self.path_ptr[rows + 1] - starts
        total = int(lens.sum())
        if total == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0, dtype=np.float64)
        local = np.repeat(np.arange(rows.size, dtype=np.int64), lens)
        offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
        ids = self.path_ids[np.repeat(starts, lens) + offsets].astype(np.int64)
        keep = self.path_depths[ids] <= max_depth
        local, ids = local[keep], ids[keep]
        width = max(len(self.path_vocab), 1)
        combined, counts = np.unique(local * width + ids, return_counts=True)
        return combined // width, combined % width, counts.astype(np.float64)

    # ----------------------------------------------------------- construction

    @classmethod
    def from_hdf5(
        cls,
        path: Path | str,
        keys: Sequence[str],
        *,
        progress: Callable[[Iterable[str]], Iterable[str]] | None = None,
    ) -> SiteTable:
        """Read the sites of molecules *keys* from a QM9s-RPFR file."""
        builder = _Builder()
        with h5py.File(path, "r") as h5:
            temperatures = _temperatures_from_file(h5)
            iterator: Iterable[str] = progress(keys) if progress else keys
            for key in iterator:
                group = h5[key]
                symbols = _decode_str_array(group["atom_symbol"][()])
                beta = np.asarray(group["rpfr_single"][()], dtype=np.float64)
                if "bond_order" in group and "local_smiles" in group:
                    bonds = [_json_paths(raw) for raw in group["bond_order"][()]]
                    envs = [_decode_str_array(row) for row in group["local_smiles"][()]]
                else:
                    bonds, envs = None, None
                builder.add(key, symbols, beta, bonds, envs)
        return builder.build(temperatures)

    @classmethod
    def from_molecules(
        cls, molecules: Sequence[MoleculeRecord], *, keys: Sequence[str] | None = None
    ) -> SiteTable:
        """Build a table from loaded records, for prediction."""
        builder = _Builder()
        temperatures: tuple[float, ...] = ()
        for i, molecule in enumerate(molecules):
            key = keys[i] if keys is not None else str(molecule.molecule_id)
            sites = molecule.sites
            if sites:
                temperatures = tuple(sites[0].temperatures)
            symbols = tuple(site.element for site in sites)
            beta = np.array([site.beta for site in sites], dtype=np.float64).reshape(len(sites), -1)
            bonds = [list(site.bond_order or []) for site in sites]
            envs = [tuple(site.local_smiles) for site in sites]
            graph = any(bonds) and any(env and env[0] for env in envs)
            builder.add(key, symbols, beta, bonds if graph else None, envs if graph else None)
        return builder.build(temperatures)


def _json_paths(raw: object) -> list[str]:
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    if not text:
        return []
    parsed = json.loads(text)
    return [str(p) for p in parsed] if isinstance(parsed, list) else [str(parsed)]


class _Builder:
    """Accumulates molecules and emits a :class:`SiteTable`."""

    def __init__(self) -> None:
        self.keys: list[str] = []
        self.site_molecule: list[int] = []
        self.site_atom: list[int] = []
        self.element: list[str] = []
        self.y: list[np.ndarray] = []
        self.has_graph: list[bool] = []
        self.path_lens: list[int] = []
        self.path_ids: list[int] = []
        self.env_ids: list[list[int]] = []
        self.paths = _Interner()
        self.envs = _Interner()

    def add(
        self,
        key: str,
        symbols: Sequence[str],
        beta: np.ndarray,
        bonds: Sequence[Sequence[str]] | None,
        envs: Sequence[Sequence[str]] | None,
    ) -> None:
        mol = len(self.keys)
        self.keys.append(key)
        n_atoms = len(symbols)
        if n_atoms == 0:
            return
        self.y.append(to_target(beta.reshape(n_atoms, -1)))
        for atom, symbol in enumerate(symbols):
            self.site_molecule.append(mol)
            self.site_atom.append(atom)
            self.element.append(symbol)
            if bonds is None or envs is None:
                self.has_graph.append(False)
                self.path_lens.append(0)
                self.env_ids.append([-1] * N_RADII)
                continue
            self.has_graph.append(True)
            site_paths = bonds[atom]
            self.path_lens.append(len(site_paths))
            self.path_ids.extend(self.paths(p) for p in site_paths)
            row = list(envs[atom])[:N_RADII]
            row += [""] * (N_RADII - len(row))
            self.env_ids.append([self.envs(s) if s else -1 for s in row])

    def build(self, temperatures: Sequence[float]) -> SiteTable:
        n_temps = len(temperatures)
        path_ptr = np.zeros(len(self.path_lens) + 1, dtype=np.int64)
        np.cumsum(self.path_lens, out=path_ptr[1:])
        return SiteTable(
            molecule_keys=tuple(self.keys),
            site_molecule=np.asarray(self.site_molecule, dtype=np.int64),
            site_atom=np.asarray(self.site_atom, dtype=np.int64),
            element=np.asarray(self.element, dtype="<U2"),
            y=np.vstack(self.y) if self.y else np.zeros((0, n_temps)),
            has_graph=np.asarray(self.has_graph, dtype=bool),
            path_ptr=path_ptr,
            path_ids=np.asarray(self.path_ids, dtype=np.int32),
            path_vocab=tuple(self.paths.items),
            env_ids=np.asarray(self.env_ids, dtype=np.int64).reshape(-1, N_RADII),
            env_vocab=tuple(self.envs.items),
            temperatures=tuple(float(t) for t in temperatures),
        )


__all__ = ["N_RADII", "SiteTable", "from_target", "path_depth", "to_target"]
