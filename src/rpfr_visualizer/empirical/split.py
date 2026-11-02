"""The train / validation / test split every empirical model is fitted against.

One split, written once and shared, is what makes the models comparable: a
bond-additivity scheme and an MPNN that were scored on different test sets
cannot be ranked against each other. The split is by *molecule*, never by
site, so no molecule contributes sites to both sides of the line.

Only molecules a model can learn from are split: those that pass the pipeline
QC and carry the graph fields (``bond_order``, ``local_smiles``) the models
read. The rest are recorded as excluded rather than silently dropped, so the
explorer can say why a molecule has no split.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from rpfr_visualizer.io.loader import _decode_scalar, _is_molecule_group
from rpfr_visualizer.io.validate import derive_qc_pass

SplitName = Literal["train", "val", "test"]

#: The protocol asked for: 100k molecules to fit on, 24k held out to score,
#: and whatever is left (about 4k of QM9s) to choose hyperparameters on.
DEFAULT_N_TRAIN = 100_000
DEFAULT_N_TEST = 24_000
DEFAULT_SEED = 0

#: Filename the trainer writes next to the models, and the explorer looks for.
SPLIT_FILENAME = "split.json"

#: Datasets a molecule needs before any model here can read it.
GRAPH_FIELDS = ("bond_order", "local_smiles", "rpfr_single", "atom_symbol")


def _bool_attr(attrs: h5py.AttributeManager, name: str, default: bool) -> bool:
    return bool(_decode_scalar(attrs.get(name, default)))


def molecule_eligibility(group: h5py.Group) -> str | None:
    """Return why a molecule cannot be used for training, or None if it can."""
    missing = [name for name in GRAPH_FIELDS if name not in group]
    if missing:
        return f"missing {', '.join(missing)}"
    attrs = group.attrs
    passed = derive_qc_pass(
        freq_qc_ok=_bool_attr(attrs, "freq_qc_ok", True),
        rpfr_qc_ok=_bool_attr(attrs, "rpfr_qc_ok", True),
        tr_converged=_bool_attr(attrs, "tr_converged", True),
        rdkit_ok=_bool_attr(attrs, "rdkit_ok", True),
        smiles_order_match=_bool_attr(attrs, "smiles_order_match", True),
        n_imag_vib=int(_decode_scalar(attrs.get("n_imag_vib", 0))),
    )
    if not passed:
        return "fails QC"
    return None


def scan_eligible(path: Path | str) -> tuple[list[str], dict[str, str]]:
    """Read every molecule's eligibility from an HDF5 file.

    Returns:
        The eligible molecule keys, sorted, and a map from each excluded key to
        the reason it was excluded.
    """
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    with h5py.File(path, "r") as h5:
        for key in sorted(h5.keys(), key=str):
            if not _is_molecule_group(h5, key):
                continue
            reason = molecule_eligibility(h5[key])
            if reason is None:
                eligible.append(key)
            else:
                excluded[key] = reason
    return eligible, excluded


@dataclass(frozen=True)
class DatasetSplit:
    """Molecule keys assigned to train, validation and test.

    Attributes:
        train: Keys the models are fitted on.
        val: Keys used only to choose hyperparameters.
        test: Keys used only to report the final score.
        seed: Seed of the permutation that produced the split.
        source: Where the keys came from (file name, dataset, molecule count).
        excluded: Keys left out, with the reason for each.
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    seed: int = DEFAULT_SEED
    source: Mapping[str, object] = field(default_factory=dict)
    excluded: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for name in ("train", "val", "test"):
            keys = set(getattr(self, name))
            overlap = seen & keys
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(f"Molecule {example!r} is in more than one split")
            seen |= keys

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def membership(self) -> dict[str, SplitName]:
        """Map every split molecule key to the split it is in."""
        out: dict[str, SplitName] = {}
        for key in self.train:
            out[key] = "train"
        for key in self.val:
            out[key] = "val"
        for key in self.test:
            out[key] = "test"
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "rpfr-visualizer/split",
            "version": 1,
            "seed": self.seed,
            "sizes": self.sizes,
            "source": dict(self.source),
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "excluded": dict(self.excluded),
        }

    @property
    def fingerprint(self) -> str:
        """A short digest of the assignment, so a model can name its split.

        Covers the three key lists only: the same assignment written from a
        different machine, with a different source path, is the same split.
        """
        digest = hashlib.sha256()
        for name in ("train", "val", "test"):
            digest.update(name.encode())
            digest.update("\n".join(getattr(self, name)).encode())
        return digest.hexdigest()[:16]

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict()))
        return target

    @classmethod
    def load(cls, path: Path | str) -> DatasetSplit:
        raw = json.loads(Path(path).read_text())
        if raw.get("format") != "rpfr-visualizer/split":
            raise ValueError(f"{path} is not a split file")
        return cls(
            train=tuple(raw["train"]),
            val=tuple(raw["val"]),
            test=tuple(raw["test"]),
            seed=int(raw.get("seed", DEFAULT_SEED)),
            source=raw.get("source", {}),
            excluded=raw.get("excluded", {}),
        )


def make_split(
    keys: Iterable[str],
    *,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    seed: int = DEFAULT_SEED,
    source: Mapping[str, object] | None = None,
    excluded: Mapping[str, str] | None = None,
) -> DatasetSplit:
    """Shuffle *keys* with *seed* and cut them into train, test and validation.

    The test set is cut first and the validation set is whatever remains after
    training, so changing *n_train* never moves a molecule into or out of the
    test set.

    Raises:
        ValueError: If there are not enough keys to leave a non-empty
            validation set.
    """
    ordered = sorted(set(keys), key=str)
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must both be positive")
    if len(ordered) < n_train + n_test + 1:
        raise ValueError(
            f"{len(ordered):,} eligible molecules cannot hold {n_train:,} train + "
            f"{n_test:,} test and still leave a validation set. Pass smaller "
            "--n-train/--n-test (the 1% subsample holds about 1,300)."
        )
    order = np.random.default_rng(seed).permutation(len(ordered))
    shuffled = [ordered[i] for i in order]
    test = shuffled[:n_test]
    train = shuffled[n_test : n_test + n_train]
    val = shuffled[n_test + n_train :]
    return DatasetSplit(
        train=tuple(sorted(train, key=str)),
        val=tuple(sorted(val, key=str)),
        test=tuple(sorted(test, key=str)),
        seed=seed,
        source=dict(source or {}),
        excluded=dict(excluded or {}),
    )


def split_hdf5(
    path: Path | str,
    *,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    seed: int = DEFAULT_SEED,
) -> DatasetSplit:
    """Scan an HDF5 file and split its eligible molecules."""
    source = Path(path)
    eligible, excluded = scan_eligible(source)
    with h5py.File(source, "r") as h5:
        dataset = str(_decode_scalar(h5.attrs.get("dataset", source.stem)))
    return make_split(
        eligible,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        source={
            "file": source.name,
            "dataset": dataset,
            "n_molecules": len(eligible) + len(excluded),
            "n_eligible": len(eligible),
        },
        excluded=excluded,
    )


__all__ = [
    "DEFAULT_N_TEST",
    "DEFAULT_N_TRAIN",
    "DEFAULT_SEED",
    "SPLIT_FILENAME",
    "DatasetSplit",
    "SplitName",
    "make_split",
    "molecule_eligibility",
    "scan_eligible",
    "split_hdf5",
]
