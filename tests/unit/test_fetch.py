"""Downloading a published data tier.

Every test here runs against a real HTTP server on loopback rather than a mocked
urlopen: the thing most likely to break is the streaming/verify path itself, and
a mock would assert that the mock works.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from rpfr_visualizer.fetch import (
    Artifact,
    FetchError,
    fetch_tier,
    human_bytes,
    load_manifest,
    sha256_of,
    verify,
)

PAYLOAD = b"molecules" * 512


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Serve a directory over loopback; yield (base_url, directory)."""
    root = tmp_path / "server"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", root
    finally:
        httpd.shutdown()
        httpd.server_close()


def manifest_for(base_url: str, *, name: str = "data.h5", payload: bytes = PAYLOAD) -> str:
    return json.dumps(
        {
            "manifest_version": 1,
            "tiers": {
                "tiny": {
                    "description": "test tier",
                    "base_url": base_url,
                    "artifacts": [
                        {
                            "name": name,
                            "subdir": "raw",
                            "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    ],
                }
            },
        }
    )


# --- manifest ---------------------------------------------------------------


def test_load_manifest_parses_a_tier() -> None:
    tier = load_manifest(manifest_for("http://example.invalid/x"))["tiny"]

    assert tier.key == "tiny"
    assert tier.total_bytes == len(PAYLOAD)
    assert tier.url_for(tier.artifacts[0]) == "http://example.invalid/x/data.h5"


def test_url_join_does_not_double_the_slash() -> None:
    tier = load_manifest(manifest_for("http://example.invalid/x/"))["tiny"]
    assert tier.url_for(tier.artifacts[0]) == "http://example.invalid/x/data.h5"


def test_a_tier_with_no_artifacts_is_rejected() -> None:
    bad = json.dumps({"tiers": {"empty": {"base_url": "u", "artifacts": []}}})
    with pytest.raises(FetchError, match="no artifacts"):
        load_manifest(bad)


def test_a_manifest_with_no_tiers_is_rejected() -> None:
    with pytest.raises(FetchError, match="no tiers"):
        load_manifest(json.dumps({"tiers": {}}))


def test_the_shipped_manifest_loads() -> None:
    """The real releases.json must parse, or `data fetch` is dead on arrival."""
    tiers = load_manifest()
    assert "qm9s-1pct" in tiers
    tier = tiers["qm9s-1pct"]
    assert tier.artifacts
    assert {a.subdir for a in tier.artifacts} <= {"raw", "cache"}
    assert all(len(a.sha256) == 64 for a in tier.artifacts)


def test_the_shipped_tier_pairs_data_with_an_index() -> None:
    """A tier must never offer an index alone: it would answer searches for
    molecules that nothing on disk can open."""
    tier = load_manifest()["qm9s-1pct"]
    subdirs = {a.subdir for a in tier.artifacts}
    assert "raw" in subdirs and "cache" in subdirs


# --- verification -----------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(PAYLOAD)
    assert sha256_of(path) == hashlib.sha256(PAYLOAD).hexdigest()


def test_verify_is_false_when_absent(tmp_path: Path) -> None:
    artifact = Artifact("f", "raw", len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
    assert verify(artifact, tmp_path) is False


def test_verify_is_false_on_a_size_mismatch(tmp_path: Path) -> None:
    artifact = Artifact("f", "raw", len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
    target = artifact.destination(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD[:-1])
    assert verify(artifact, tmp_path) is False


def test_verify_is_false_on_a_content_mismatch(tmp_path: Path) -> None:
    """Same length, different bytes — the case a size check cannot catch."""
    other = bytes(len(PAYLOAD))
    artifact = Artifact("f", "raw", len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
    target = artifact.destination(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(other)
    assert verify(artifact, tmp_path) is False


def test_verify_is_true_for_a_good_file(tmp_path: Path) -> None:
    artifact = Artifact("f", "raw", len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
    target = artifact.destination(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    assert verify(artifact, tmp_path) is True


# --- fetching ---------------------------------------------------------------


def test_fetch_downloads_and_verifies(served: tuple[str, Path], tmp_path: Path) -> None:
    base_url, serve_root = served
    (serve_root / "data.h5").write_bytes(PAYLOAD)
    tier = load_manifest(manifest_for(base_url))["tiny"]
    dest = tmp_path / "dest"

    results = list(fetch_tier(tier, dest))

    assert [status for _, status in results] == ["fetched"]
    assert (dest / "raw" / "data.h5").read_bytes() == PAYLOAD


def test_fetch_skips_what_already_verifies(served: tuple[str, Path], tmp_path: Path) -> None:
    base_url, serve_root = served
    (serve_root / "data.h5").write_bytes(PAYLOAD)
    tier = load_manifest(manifest_for(base_url))["tiny"]
    dest = tmp_path / "dest"
    list(fetch_tier(tier, dest))

    again = list(fetch_tier(tier, dest))

    assert [status for _, status in again] == ["present"]


def test_force_redownloads_a_good_file(served: tuple[str, Path], tmp_path: Path) -> None:
    base_url, serve_root = served
    (serve_root / "data.h5").write_bytes(PAYLOAD)
    tier = load_manifest(manifest_for(base_url))["tiny"]
    dest = tmp_path / "dest"
    list(fetch_tier(tier, dest))

    again = list(fetch_tier(tier, dest, force=True))

    assert [status for _, status in again] == ["fetched"]


def test_a_corrupt_download_raises_and_leaves_nothing_behind(
    served: tuple[str, Path], tmp_path: Path
) -> None:
    """Same length so the size check passes and only the checksum catches it.
    The file must not survive: a bad file left in place is one that gets used."""
    base_url, serve_root = served
    (serve_root / "data.h5").write_bytes(bytes(len(PAYLOAD)))
    tier = load_manifest(manifest_for(base_url))["tiny"]
    dest = tmp_path / "dest"

    with pytest.raises(FetchError, match="checksum mismatch"):
        list(fetch_tier(tier, dest))

    assert not (dest / "raw" / "data.h5").exists()


def test_a_truncated_download_raises_and_leaves_no_part_file(
    served: tuple[str, Path], tmp_path: Path
) -> None:
    base_url, serve_root = served
    (serve_root / "data.h5").write_bytes(PAYLOAD[:10])
    tier = load_manifest(manifest_for(base_url))["tiny"]
    dest = tmp_path / "dest"

    with pytest.raises(FetchError, match="truncated"):
        list(fetch_tier(tier, dest))

    assert not (dest / "raw" / "data.h5").exists()
    assert list((dest / "raw").glob("*.part")) == []


def test_a_missing_asset_raises_a_readable_error(served: tuple[str, Path], tmp_path: Path) -> None:
    base_url, _ = served  # nothing written to the server
    tier = load_manifest(manifest_for(base_url))["tiny"]

    with pytest.raises(FetchError, match="Could not download"):
        list(fetch_tier(tier, tmp_path / "dest"))


# --- formatting -------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (2048, "2.0 KB"),
        (5 * 1024**2, "5.0 MB"),
        (3 * 1024**3, "3.0 GB"),
    ],
)
def test_human_bytes(count: int, expected: str) -> None:
    assert human_bytes(count) == expected
