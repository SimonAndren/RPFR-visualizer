"""Tests for analysis/stats.py."""

from rpfr_visualizer.analysis.stats import dataset_stats
from rpfr_visualizer.io.loader import QM9sDataset


def test_dataset_stats(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    stats = dataset_stats(ds, temperature=300.0)
    assert stats["n_molecules"] == 5
    assert stats["n_sites"] == 14  # 4 + 4 + 1 + 3 + 2
    assert stats["qc_pass_count"] == 3
    assert "C" in stats["element_counts"]
    assert "H" in stats["element_counts"]
    assert stats["beta_at_temperature"] == 300.0


def test_stats_via_dataset_method(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    stats = ds.stats()
    assert stats["source"] == "dft"
