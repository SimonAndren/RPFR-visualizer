"""Time search + molecule load at two scales. Does not build an index."""

from __future__ import annotations

import time
from pathlib import Path

from rpfr_visualizer.io.molecule_index import MoleculeIndex, default_drive_root
from rpfr_visualizer.viz.explorer.factory import controller_for_path

QUERIES = (
    ("id", "100547"),
    ("formula", "C6H6"),
    ("smiles", "CCO"),
    ("fg", "fg:alcohol"),
    ("smarts", "smarts:C=C"),
)


def _ms(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000, result


def profile(label: str, data: Path, index: Path, dataset: str) -> dict:
    print(f"\n=== {label} ===")
    open_ms, idx = _ms(lambda: MoleculeIndex.open(index, allow_pubchem=False))
    print(f"  open index          {open_ms:8.1f} ms   ({len(idx):,} molecules)")

    rows = {"label": label, "index_open_ms": round(open_ms, 1), "molecules": len(idx)}
    for name, query in QUERIES:
        ms, hits = _ms(lambda q=query: idx.search(q, datasets=(dataset,), limit=200))
        n = len(hits)
        print(f"  search {name:8}  {ms:8.1f} ms   {n} hits  {query!r}")
        rows[f"search_{name}_ms"] = round(ms, 1)
        rows[f"search_{name}_n"] = n

    ctrl_ms, (ctrl, notes) = _ms(
        lambda: controller_for_path(data, dataset=dataset, index_path=index)
    )
    print(f"  open controller     {ctrl_ms:8.1f} ms")
    rows["controller_open_ms"] = round(ctrl_ms, 1)
    for note in notes:
        print(f"    ! {note}")

    key = None
    if rows["search_id_n"]:
        key = "100547"
    else:
        hits = idx.search("fg:alcohol", datasets=(dataset,), limit=1)
        key = hits[0].entry.key if hits else None
    if key:
        load_ms, _ = _ms(lambda: ctrl.load(key))
        render_ms, _ = _ms(ctrl.render_payload)
        print(f"  load {key:12} {load_ms:8.1f} ms")
        print(f"  render payload      {render_ms:8.1f} ms")
        rows["load_ms"] = round(load_ms, 1)
        rows["render_ms"] = round(render_ms, 1)
        rows["load_key"] = key
    return rows


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    profile(
        "1% subsample (SSD)",
        repo / "data",
        repo / "data/cache/qm9s_rpfr_index.sqlite",
        "qm9s",
    )
    profile(
        "100% full (T7 + existing index)",
        default_drive_root(),
        repo / "data/cache/qm9s_rpfr_full_index.sqlite",
        "qm9s",
    )


if __name__ == "__main__":
    main()
