"""Fit, tune and score empirical models against a shared split.

The protocol is fixed so models stay comparable: fit on the training
molecules, pick hyperparameters by validation error, then score once on the
test molecules. The test set is never seen before that last step, and the
model is *not* refitted on train + validation afterwards -- the saved model is
exactly the one the test score describes.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rpfr_visualizer.empirical.base import MODEL_KINDS, SiteTableModel
from rpfr_visualizer.empirical.metrics import site_metrics
from rpfr_visualizer.empirical.sites import SiteTable
from rpfr_visualizer.empirical.split import SPLIT_FILENAME, DatasetSplit


@dataclass(frozen=True)
class SplitRows:
    """Table rows of each split's usable sites."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_rows(table: SiteTable, split: DatasetSplit) -> SplitRows:
    """Which rows of *table* belong to each part of *split*."""
    return SplitRows(
        train=table.rows_for(split.train),
        val=table.rows_for(split.val),
        test=table.rows_for(split.test),
    )


def read_split_table(
    path: Path | str,
    split: DatasetSplit,
    *,
    progress: Callable[[Iterable[str]], Iterable[str]] | None = None,
) -> SiteTable:
    """Read every split molecule's sites from *path* in one pass."""
    keys = sorted({*split.train, *split.val, *split.test}, key=str)
    return SiteTable.from_hdf5(path, keys, progress=progress)


def train_model(
    kind: str,
    table: SiteTable,
    split: DatasetSplit,
    *,
    name: str | None = None,
    rows: SplitRows | None = None,
    data_file: str | None = None,
) -> SiteTableModel:
    """Fit one model kind with validation tuning, and score it on test.

    The scores land in ``model.meta["validation"]`` and ``model.meta["test"]``.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind {kind!r}. Known: {', '.join(sorted(MODEL_KINDS))}")
    cls = MODEL_KINDS[kind]
    if not issubclass(cls, SiteTableModel):
        raise ValueError(f"{kind!r} is not trained from a site table")
    rows = rows or split_rows(table, split)
    if rows.train.size == 0:
        raise ValueError("No usable training sites: is the split from this file?")

    model = cls(name=name)
    model.fit(table, rows.train, rows.val)

    def score(part: np.ndarray) -> dict[str, Any]:
        if part.size == 0:
            return {}
        pred = model.predict_table(table, part)
        return site_metrics(table.y[part], pred, table.element[part], table.temperatures)

    model.meta.update(
        {
            "label": cls.label,
            "temperatures": list(table.temperatures),
            "target": "1000 ln beta (per mil)",
            "split": {
                "fingerprint": split.fingerprint,
                "seed": split.seed,
                "sizes": split.sizes,
                "source": dict(split.source),
            },
            "data_file": data_file,
            "n_sites": {
                "train": int(rows.train.size),
                "val": int(rows.val.size),
                "test": int(rows.test.size),
            },
            "validation": score(rows.val),
            "test": score(rows.test),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
    )
    return model


def train_all(
    kinds: Sequence[str],
    data: Path | str,
    split: DatasetSplit,
    out_dir: Path | str,
    *,
    progress: Callable[[Iterable[str]], Iterable[str]] | None = None,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Read *data* once, train every kind in *kinds*, save each into *out_dir*."""
    out = Path(out_dir)
    table = read_split_table(data, split, progress=progress)
    rows = split_rows(table, split)
    log(
        f"{len(table):,} sites from {len(table.molecule_keys):,} molecules; "
        f"usable: {rows.train.size:,} train, {rows.val.size:,} val, {rows.test.size:,} test"
    )
    split_path = out / SPLIT_FILENAME
    if not split_path.exists() or DatasetSplit.load(split_path).fingerprint != split.fingerprint:
        split.save(split_path)
    saved = []
    for kind in kinds:
        log(f"training {kind} ...")
        model = train_model(kind, table, split, rows=rows, data_file=Path(data).name)
        path = model.save(out / f"{model.name}.npz")
        log(format_scores(model.meta))
        log(f"saved {path}")
        saved.append(path)
    return saved


def format_scores(meta: dict[str, Any]) -> str:
    """A short table of test MAE / RMSE / R² at the reference temperature."""
    lines = [f"{meta.get('label', '')} — test set, 1000 ln β (‰) at ~298 K"]
    lines.append(f"  {'el':<3}{'sites':>9}{'MAE':>9}{'RMSE':>9}{'R²':>8}  hyperparameters")
    hyper = meta.get("hyperparameters", {})
    for element, entry in meta.get("test", {}).items():
        if "mae_ref" not in entry:
            continue
        params = ", ".join(
            f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in hyper.get(element, {}).items()
        )
        lines.append(
            f"  {element:<3}{entry['n_sites']:>9,}{entry['mae_ref']:>9.3f}"
            f"{entry['rmse_ref']:>9.3f}{entry['r2_ref']:>8.4f}  {params}"
        )
    return "\n".join(lines)


__all__ = [
    "SplitRows",
    "format_scores",
    "read_split_table",
    "split_rows",
    "train_all",
    "train_model",
]
