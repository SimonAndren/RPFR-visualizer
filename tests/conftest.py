"""Shared pytest fixtures for rpfr_visualizer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.demo import build_demo_dataset


def pytest_addoption(parser: pytest.Parser) -> None:
    """Options the suite reads.

    ``--run-integration`` was read by tests/integration but never registered, so
    passing it was an "unrecognized arguments" error and those tests could only
    ever skip.
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that need the local subsample files in data/raw",
    )
    parser.addoption(
        "--gallery",
        action="store_true",
        default=False,
        help="write UI screenshots (see tests/e2e)",
    )
    parser.addoption(
        "--screens-dir",
        action="store",
        default=None,
        help="where --gallery writes screenshots; defaults to tests/e2e/_screens",
    )


def build_synthetic_h5(path: Path) -> Path:
    """Write a small synthetic v2.0 HDF5 file of five molecules at *path*.

    Thin wrapper over :func:`rpfr_visualizer.demo.build_demo_dataset`, which is where
    the schema knowledge now lives so that the shipped ``gui --demo`` command
    and this fixture cannot drift apart. Registers the file under ``qm9s``,
    which is the name the suite has always asserted against.
    """
    return build_demo_dataset(path, dataset="qm9s")


@pytest.fixture
def synthetic_h5_path(tmp_path: Path) -> Path:
    """A small synthetic v2.0 HDF5 file with five molecules."""
    return build_synthetic_h5(tmp_path / "test_qm9s.h5")


@pytest.fixture
def subsample_h5_path() -> Path | None:
    """Return path to local 1% subsample if it exists."""
    path = Path("data/raw/qm9s_v2.2_1pt.h5")
    return path if path.exists() else None
