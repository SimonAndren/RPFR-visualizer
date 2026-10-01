"""Tests for io/loader.py."""

from rpfr_visualizer.io.loader import QM9sDataset


def test_dataset_length(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    assert len(ds) == 5
    assert ds.molecule_ids == [1, 2, 3, 4, 5]


def test_get_molecule(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    mol = ds.get_molecule(1)
    assert mol.smiles == "C"
    assert mol.n_atoms == 4
    assert mol.qc_pass
    assert len(mol.sites) == 4
    assert mol.sites[0].element == "C"
    assert mol.sites[0].source == "dft"


def test_beta_at_temperature(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    mol = ds.get_molecule(1)
    assert mol.sites[0].beta_at(300.0) == 1.12


def test_iteration(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    ids = [mol.molecule_id for mol in ds]
    assert ids == [1, 2, 3, 4, 5]


def test_missing_molecule_raises(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path)
    try:
        ds.get_molecule(999)
        raised = False
    except KeyError:
        raised = True
    assert raised


def test_custom_source_tag(synthetic_h5_path):
    ds = QM9sDataset(synthetic_h5_path, source="model_v1")
    mol = ds.get_molecule(1)
    assert mol.source == "model_v1"
    assert mol.sites[0].source == "model_v1"
