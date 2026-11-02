"""Empirical models: split, fitting, persistence, and swapping β on a molecule."""

from __future__ import annotations

import numpy as np
import pytest

from rpfr_visualizer.empirical import (
    BondAdditivityModel,
    ClusterMeanModel,
    DatasetSplit,
    ModelStore,
    SiteTable,
    apply_model,
    load_model,
    make_split,
    model_source_tag,
    split_hdf5,
    to_target,
    train_all,
    train_model,
)
from rpfr_visualizer.empirical.base import pack_strings, unpack_strings
from rpfr_visualizer.empirical.sites import path_depth
from rpfr_visualizer.io.schema import MoleculeRecord, SiteRecord

TEMPS = (200.0, 298.15, 400.0)

#: Exact per-path increments (‰ at each temperature) for the additive fixture.
INCREMENTS = {
    "C-H": (40.0, 25.0, 15.0),
    "C-C": (30.0, 20.0, 12.0),
    "C=O": (55.0, 35.0, 22.0),
    "C-C-H": (3.0, 2.0, 1.0),
    "H-C": (900.0, 600.0, 400.0),
    "H-C-C": (20.0, 12.0, 8.0),
    "H-C-H": (-5.0, -3.0, -2.0),
}
BASE = {"C": (10.0, 6.0, 4.0), "H": (50.0, 30.0, 20.0)}


def _additive_molecule(mol_id: int, rng: np.random.Generator) -> MoleculeRecord:
    """A fake molecule whose ln β is exactly bond-additive."""
    sites = []
    for atom in range(6):
        element = "C" if atom < 2 else "H"
        pool = [p for p in INCREMENTS if p.startswith(element)]
        paths = list(rng.choice(pool, size=int(rng.integers(1, 5))))
        y = np.array(BASE[element]) + sum(np.array(INCREMENTS[p]) for p in paths)
        r1 = f"{element}{len(paths)}"
        sites.append(
            SiteRecord(
                molecule_id=mol_id,
                atom_index=atom,
                element=element,
                beta=tuple(np.exp(y / 1000.0)),
                temperatures=TEMPS,
                local_smiles=(r1, r1 + "|" + "".join(sorted(paths)), "", "", ""),
                bond_order=paths,
            )
        )
    return MoleculeRecord(molecule_id=mol_id, smiles="C", n_atoms=len(sites), sites=sites)


@pytest.fixture
def additive():
    rng = np.random.default_rng(1)
    molecules = [_additive_molecule(i, rng) for i in range(60)]
    keys = [str(m.molecule_id) for m in molecules]
    split = make_split(keys, n_train=40, n_test=10, seed=3)
    return molecules, SiteTable.from_molecules(molecules), split


# ------------------------------------------------------------------ split


def test_split_is_disjoint_deterministic_and_sized():
    keys = [str(i) for i in range(100)]
    a = make_split(keys, n_train=60, n_test=25, seed=7)
    b = make_split(reversed(keys), n_train=60, n_test=25, seed=7)
    assert a.sizes == {"train": 60, "val": 15, "test": 25}
    assert a == b
    assert not set(a.train) & set(a.test)
    assert make_split(keys, n_train=60, n_test=25, seed=8).fingerprint != a.fingerprint


def test_changing_train_size_never_moves_the_test_set():
    keys = [str(i) for i in range(100)]
    assert (
        make_split(keys, n_train=60, n_test=25).test == make_split(keys, n_train=40, n_test=25).test
    )


def test_split_refuses_to_leave_no_validation_set():
    with pytest.raises(ValueError, match="validation"):
        make_split([str(i) for i in range(10)], n_train=8, n_test=2)


def test_split_round_trips(tmp_path):
    split = make_split([str(i) for i in range(20)], n_train=10, n_test=5, excluded={"99": "x"})
    loaded = DatasetSplit.load(split.save(tmp_path / "split.json"))
    assert loaded == split
    assert loaded.fingerprint == split.fingerprint
    assert loaded.membership()[split.test[0]] == "test"


def test_overlapping_split_is_rejected():
    with pytest.raises(ValueError, match="more than one"):
        DatasetSplit(train=("1",), val=("1",), test=("2",))


def test_split_hdf5_excludes_failed_molecules(synthetic_h5_path):
    split = split_hdf5(synthetic_h5_path, n_train=1, n_test=1)
    assert split.excluded, "the demo dataset has a QC-failing molecule"
    assert all(reason == "fails QC" for reason in split.excluded.values())
    assert not set(split.excluded) & set(split.train + split.val + split.test)


# ------------------------------------------------------------ site table


def test_path_depth_counts_bonds():
    assert [path_depth(p) for p in ("C-H", "C-C=O", "C-N-C-H", "c:c")] == [1, 2, 3, 1]


def test_site_table_reads_targets_and_descriptors(additive):
    molecules, table, _split = additive
    assert len(table) == 6 * len(molecules)
    first = molecules[0].sites[0]
    np.testing.assert_allclose(table.y[0], to_target(np.array(first.beta)))
    rows, ids, counts = table.paths_coo(np.array([0]), max_depth=1)
    got = {table.path_vocab[i]: c for i, c in zip(ids, counts, strict=True)}
    expected = {p: first.bond_order.count(p) for p in first.bond_order if path_depth(p) == 1}
    assert got == expected


def test_strings_pack_without_pickle():
    items = ["C([H])N", "", "c1ccccc1"]
    assert unpack_strings(pack_strings(items), len(items)) == items
    assert unpack_strings(pack_strings([]), 0) == []


# ---------------------------------------------------------------- models


def test_bond_additivity_recovers_an_exactly_additive_target(additive):
    _molecules, table, split = additive
    model = train_model("bond_additivity", table, split)
    for element in ("C", "H"):
        assert model.meta["test"][element]["mae_ref"] < 1e-3
    assert model.meta["hyperparameters"]["H"]["depth"] == 2


def test_cluster_mean_falls_back_to_narrower_radius_then_element_mean(additive):
    _molecules, table, split = additive
    model = train_model("cluster_mean", table, split)
    assert isinstance(model, ClusterMeanModel)
    assert set(model.meta["test"]) == {"C", "H"}

    unseen = SiteTable.from_molecules(
        [
            MoleculeRecord(
                molecule_id=999,
                smiles="C",
                n_atoms=1,
                sites=[
                    SiteRecord(
                        molecule_id=999,
                        atom_index=0,
                        element="C",
                        beta=(1.1, 1.1, 1.1),
                        temperatures=TEMPS,
                        local_smiles=("nowhere", "never", "", "", ""),
                        bond_order=["C-H"],
                    )
                ],
            )
        ]
    )
    pred = model.predict_table(unseen, np.array([0]))
    np.testing.assert_allclose(pred[0], model.fits["C"].mean)


@pytest.mark.parametrize("kind", ["bond_additivity", "cluster_mean"])
def test_saved_model_predicts_identically(additive, tmp_path, kind):
    molecules, table, split = additive
    model = train_model(kind, table, split)
    loaded = load_model(model.save(tmp_path / kind))
    assert loaded.meta["test"] == model.meta["test"]
    for molecule in molecules[:5]:
        np.testing.assert_allclose(
            loaded.predict_molecule(molecule), model.predict_molecule(molecule)
        )


def test_model_file_is_not_a_pickle(additive, tmp_path):
    _molecules, table, split = additive
    path = train_model("cluster_mean", table, split).save(tmp_path / "m")
    with np.load(path, allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)


def test_unknown_kind_is_refused(additive):
    _molecules, table, split = additive
    with pytest.raises(ValueError, match="Unknown model kind"):
        train_model("orbnet", table, split)


# ------------------------------------------------------- applying a model


def test_apply_model_replaces_beta_and_tags_the_source(additive):
    molecules, table, split = additive
    model = train_model("bond_additivity", table, split)
    swapped = apply_model(molecules[0], model)
    assert swapped.source == model_source_tag("bond_additivity")
    assert all(site.source == swapped.source for site in swapped.sites)
    assert swapped.smiles == molecules[0].smiles
    np.testing.assert_allclose(
        [s.beta for s in swapped.sites], [s.beta for s in molecules[0].sites], rtol=1e-6
    )


def test_apply_model_refuses_a_different_grid(additive):
    molecules, table, split = additive
    model = train_model("cluster_mean", table, split)
    model.meta["temperatures"] = [300.0]
    with pytest.raises(ValueError, match="grid"):
        apply_model(molecules[0], model)


def test_molecule_without_descriptors_gets_nan_not_a_guess(additive):
    molecules, table, split = additive
    model = train_model("cluster_mean", table, split)
    bare = MoleculeRecord(
        molecule_id=1,
        smiles="C",
        n_atoms=1,
        sites=[
            SiteRecord(
                molecule_id=1, atom_index=0, element="C", beta=(1.0,) * 3, temperatures=TEMPS
            )
        ],
    )
    assert np.isnan(model.predict_molecule(bare)).all()


# ------------------------------------------------------------ store / e2e


def test_train_all_writes_models_and_split_the_store_can_read(synthetic_h5_path, tmp_path):
    split = split_hdf5(synthetic_h5_path, n_train=1, n_test=1)
    paths = train_all(
        ["bond_additivity", "cluster_mean"], synthetic_h5_path, split, tmp_path, log=lambda _m: None
    )
    assert [p.name for p in paths] == ["bond_additivity.npz", "cluster_mean.npz"]

    store = ModelStore(tmp_path)
    assert store.names() == ("bond_additivity", "cluster_mean")
    assert isinstance(store.get("bond_additivity"), BondAdditivityModel)
    assert store.split_of(split.test[0], "cluster_mean") == "test"
    assert store.split_of(next(iter(split.excluded)), "cluster_mean") is None
    with pytest.raises(KeyError, match="Available"):
        store.get("mpnn")


def test_store_will_not_vouch_for_a_model_from_another_split(synthetic_h5_path, tmp_path):
    split = split_hdf5(synthetic_h5_path, n_train=1, n_test=1)
    train_all(["cluster_mean"], synthetic_h5_path, split, tmp_path, log=lambda _m: None)
    other = make_split([*split.train, *split.val, *split.test], n_train=1, n_test=1, seed=99)
    other.save(tmp_path / "split.json")
    assert ModelStore(tmp_path).split_of(split.test[0], "cluster_mean") is None


def test_empty_or_missing_directory_has_no_models(tmp_path):
    assert ModelStore(tmp_path / "absent").names() == ()
    (tmp_path / "junk.npz").write_bytes(b"not a model")
    assert ModelStore(tmp_path).names() == ()


def test_models_train_on_the_real_subsample(subsample_h5_path, tmp_path):
    if subsample_h5_path is None:
        pytest.skip("Local 1% subsample not found")
    split = split_hdf5(subsample_h5_path, n_train=1000, n_test=240)
    paths = train_all(
        ["bond_additivity", "cluster_mean"], subsample_h5_path, split, tmp_path, log=lambda _m: None
    )
    for path in paths:
        test = load_model(path).meta["test"]
        assert test["C"]["r2_ref"] > 0.8
        assert test["H"]["coverage"] == 1.0
