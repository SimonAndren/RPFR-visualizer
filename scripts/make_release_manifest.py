"""Generate src/rpfr_visualizer/data/releases.json from the artifacts on this machine.

The manifest is what ``rpfr-visualizer data fetch`` verifies downloads against, so it
must describe the exact bytes that get uploaded as release assets. Generating it
from the local files — rather than writing checksums by hand — is the only way
those two stay in step.

It reads ``data/release/``, a staging tree holding exactly what gets uploaded
and nothing else. Assemble it first::

    mkdir -p data/release/raw data/release/cache
    cp data/raw/qm9s_v2.2_1pt.h5 data/release/raw/
    uv run rpfr-visualizer index build --raw data/release/raw \
        --out data/release/cache/qm9s_rpfr_index.sqlite

Then generate, and commit the result::

    uv run python scripts/make_release_manifest.py --tag v0.4.0

Staging, rather than pointing at ``data/raw`` directly, is what keeps the
release honest: the index is built over precisely the files that ship, so it
cannot describe molecules the download does not contain. Re-run both steps if
the subsample or the index is ever rebuilt — the checksums change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpfr_visualizer.fetch import sha256_of

REPO = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/SimonAndren/RPFR-visualizer"

#: The staging tree: exactly what gets uploaded, nothing else.
STAGING = REPO / "data" / "release"

#: Whatever is staged is the tier, as (source path, destination subdirectory).
#: Discovered rather than listed, so this file cannot come to disagree with the
#: index that was built over the tree.
TIER_QM9S = (
    *((p, "raw") for p in sorted((STAGING / "raw").glob("*.h5"))),
    *((p, "cache") for p in sorted((STAGING / "cache").glob("*.sqlite"))),
)


def describe_tier() -> str:
    """The tier description, with the molecule count read from what is staged.

    The count used to be written into this file by hand, which meant that
    rebuilding the subsample produced a manifest whose checksums were right and
    whose description was wrong -- and that description is what
    ``rpfr-visualizer data list`` prints to somebody deciding whether to download
    88 MB. Counting the staged index is the only way the two stay in step.
    """
    molecules = ""
    for index in sorted((STAGING / "cache").glob("*.sqlite")):
        import sqlite3

        with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as conn:
            n = conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
        molecules = f"{n:,} molecules, "
        break

    return (
        f"QM9s at 1% ({molecules}Eckart-projected TR removal) plus the search "
        "index built over exactly that file. Needs no external drive."
    )


def build_tier(sources: tuple[tuple[Path, str], ...], tag: str, description: str) -> dict:
    artifacts = []
    for path, subdir in sources:
        if not path.exists():
            raise SystemExit(f"Missing artifact: {path}")
        print(f"  hashing {path.name} ...", flush=True)
        artifacts.append(
            {
                "name": path.name,
                "subdir": subdir,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )
    return {
        "description": description,
        "base_url": f"{REPO_URL}/releases/download/{tag}",
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0.4.0", help="release tag the assets hang off")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "src" / "rpfr_visualizer" / "data" / "releases.json",
        help="where to write the manifest",
    )
    args = parser.parse_args(argv)

    if not TIER_QM9S:
        raise SystemExit(
            f"Nothing staged under {STAGING}. See this script's docstring for how to assemble it."
        )

    print(f"Building manifest for {args.tag}")
    manifest = {
        "manifest_version": 1,
        "tiers": {"qm9s-1pct": build_tier(TIER_QM9S, args.tag, describe_tier())},
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    artifacts = manifest["tiers"]["qm9s-1pct"]["artifacts"]
    total = sum(a["size_bytes"] for a in artifacts)
    print(f"Wrote {args.out} ({total / 1024 / 1024:.0f} MB across {len(artifacts)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
