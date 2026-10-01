"""Integration tests against the local 1% subsample."""

import pytest

from rpfr_visualizer.io.loader import QM9sDataset


@pytest.mark.skipif(
    "not config.getoption('--run-integration', default=False)",
    reason="Integration tests require local subsample; pass --run-integration to enable",
)
def test_subsample_loads(subsample_h5_path):
    if subsample_h5_path is None:
        pytest.skip("Local 1% subsample not found at data/raw/qm9s_v2.2_1pt.h5")
    ds = QM9sDataset(subsample_h5_path)
    assert len(ds) > 1000
    mol = ds.get_molecule(ds.molecule_ids[0])
    assert mol.smiles
    assert len(mol.sites) == mol.n_atoms
    assert mol.sites[0].beta_at(300.0) > 0


def test_subsample_integration_auto(subsample_h5_path):
    """Run against local subsample when present (no extra flag needed)."""
    if subsample_h5_path is None:
        pytest.skip("Local 1% subsample not found")
    ds = QM9sDataset(subsample_h5_path)
    assert len(ds) == 1298
    report = ds.validate()
    assert report.n_molecules == 1298
    stats = ds.stats(temperature=300.0)
    assert stats["n_sites"] > 0
