"""Reading a schema 2.2 file.

2.2 changed two things this package depends on. The temperature grid moved
from a JSON root *attribute* (``temperatures_K``) to a root *dataset*
(``/temperatures``), and it grew from thirteen values in file order to fifteen
in ascending order, reaching 500 K instead of 420 K.

Neither change raises anything on its own, which is what makes them worth
pinning: a 2.2 file read with the 2.0 rules came back with the old thirteen
temperatures, so every column from the eighth on was labelled with the wrong
temperature and the two new ones were invisible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from rpfr_visualizer.analysis.temperature import interpolate_beta  # noqa: E402
from rpfr_visualizer.io.loader import QM9sDataset  # noqa: E402

#: The 2.2 grid: ascending, fifteen values, up to 500 K.
GRID_2_2 = (
    20.0, 60.0, 100.0, 140.0, 180.0, 220.0, 260.0, 273.15,
    298.15, 300.0, 340.0, 380.0, 420.0, 460.0, 500.0,
)  # fmt: skip


def _write_v2_2(path: Path) -> None:
    """A minimal but structurally faithful 2.2 file."""
    grid = np.asarray(GRID_2_2, dtype=float)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.2"
        # A dataset, not an attribute. This is the whole point.
        h5.create_dataset("temperatures", data=grid)

        group = h5.create_group("8")  # ids are unpadded in 2.2
        group.attrs["mol_id"] = "8"
        group.attrs["n_atoms"] = 3
        group.attrs["smiles"] = "CO"
        group.create_dataset("atom_symbol", data=np.array([b"C", b"O", b"F"], dtype="S2"))
        group.create_dataset("coordinates", data=np.zeros((3, 3)))
        # F is monoisotopic, so its row is NaN by definition rather than missing.
        group.create_dataset(
            "rpfr_single",
            data=np.exp(np.outer([30.0, 25.0, np.nan], 1.0 / grid)),
        )


@pytest.fixture
def v2_2_file(tmp_path: Path) -> Path:
    path = tmp_path / "qm9s_rpfr_v2.2.h5"
    _write_v2_2(path)
    return path


def test_the_grid_comes_from_the_dataset_not_the_package_default(v2_2_file: Path) -> None:
    """The failure this replaces was silent: no error, just wrong labels."""
    dataset = QM9sDataset(v2_2_file)
    assert tuple(dataset.temperatures) == GRID_2_2


def test_the_two_new_temperatures_are_visible(v2_2_file: Path) -> None:
    dataset = QM9sDataset(v2_2_file)
    assert 460.0 in dataset.temperatures
    assert 500.0 in dataset.temperatures


def test_the_grid_is_not_silently_reordered(v2_2_file: Path) -> None:
    """2.0 stored the chemistry temperatures last; 2.2 stores them in place.

    Column order is how beta is matched to a temperature, so a reorder here
    would mislabel every value rather than merely look untidy.
    """
    dataset = QM9sDataset(v2_2_file)
    assert list(dataset.temperatures) == sorted(dataset.temperatures)
    assert dataset.temperatures[7] == pytest.approx(273.15)


def test_a_temperature_this_file_stores_is_not_called_an_extrapolation() -> None:
    """The validated range is the stored one.

    Against a hardcoded 420 K ceiling, every 460 K and 500 K reading in a 2.2
    file would have warned about a value the file holds outright.
    """
    import warnings

    betas = tuple(float(np.exp(30.0 / t)) for t in GRID_2_2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = interpolate_beta(GRID_2_2, betas, 460.0, warn=True)

    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert result.out_of_range is False
    assert result.interpolated is False  # it is a stored point


def test_beyond_the_stored_range_still_warns() -> None:
    betas = tuple(float(np.exp(30.0 / t)) for t in GRID_2_2)
    with pytest.warns(UserWarning, match=r"\[20, 500\] K"):
        result = interpolate_beta(GRID_2_2, betas, 600.0, warn=True)
    assert result.out_of_range is True


def test_a_molecule_with_no_rdkit_annotations_still_loads(tmp_path: Path) -> None:
    """About 0.4% of 2.2 molecules have none.

    Where the DFT geometry no longer matches the SMILES the generator writes
    no local_smiles, no bond orders and no SMILES at all -- the datasets are
    absent rather than empty. They are flagged, never dropped, so the reader
    has to cope with them rather than the file excluding them for it.
    """
    path = tmp_path / "no_rdkit.h5"
    grid = np.asarray(GRID_2_2, dtype=float)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.2"
        h5.create_dataset("temperatures", data=grid)
        group = h5.create_group("8")
        group.attrs["mol_id"] = "8"
        group.attrs["n_atoms"] = 2
        group.attrs["geometry_match"] = False
        group.create_dataset("atom_symbol", data=np.array([b"C", b"O"], dtype="S2"))
        group.create_dataset("coordinates", data=np.zeros((2, 3)))
        group.create_dataset("rpfr_single", data=np.exp(np.outer([30.0, 25.0], 1.0 / grid)))

    molecule = QM9sDataset(path).get_molecule(8)
    assert len(molecule.sites) == 2
    assert molecule.smiles == ""
    assert molecule.sites[0].local_smiles == ("", "", "", "", "")
    # The beta values are real even when the structure annotations are not.
    assert np.isfinite(molecule.sites[0].beta_at(298.15, warn=False))


def test_fluorine_reads_as_undefined_rather_than_zero(v2_2_file: Path) -> None:
    """NaN in 2.2 means "not defined", not "missing".

    Fluorine is monoisotopic, so it has no beta at all. Coercing that to 0.0
    would put an F site at the bottom of every ranking as though it were the
    weakest rather than absent, and 0 is a value the colour scale would map.
    """
    molecule = QM9sDataset(v2_2_file).get_molecule(8)
    fluorine = next(s for s in molecule.sites if s.element == "F")
    assert np.isnan(fluorine.beta_at(298.15, warn=False))


def test_an_undefined_value_reaches_the_client_as_null(v2_2_file: Path) -> None:
    """So the browser can render an em dash rather than a confident zero."""
    import json

    from rpfr_visualizer.gui.payloads import jsonable

    molecule = QM9sDataset(v2_2_file).get_molecule(8)
    fluorine = next(s for s in molecule.sites if s.element == "F")
    encoded = json.loads(json.dumps(jsonable({"beta": fluorine.beta_at(298.15, warn=False)})))
    assert encoded["beta"] is None


def _write_geometry_mismatch(path: Path) -> None:
    """A 2.2 molecule whose SMILES no longer describes its geometry.

    It keeps the SMILES -- that is what makes this dangerous rather than
    merely missing -- but loses every RDKit-derived dataset.
    """
    grid = np.asarray(GRID_2_2, dtype=float)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "2.2"
        h5.create_dataset("temperatures", data=grid)
        group = h5.create_group("113967")
        group.attrs["mol_id"] = "113967"
        group.attrs["n_atoms"] = 2
        group.attrs["smiles"] = "C1C2C3OC22CC1CC32"
        group.attrs["geometry_match"] = False
        group.create_dataset("atom_symbol", data=np.array([b"C", b"O"], dtype="S2"))
        group.create_dataset("coordinates", data=np.zeros((2, 3)))
        group.create_dataset("rpfr_single", data=np.exp(np.outer([30.0, 25.0], 1.0 / grid)))


def test_a_geometry_mismatch_is_carried_onto_the_molecule(tmp_path: Path) -> None:
    """The flag exists in the file; it has to reach the code that draws."""
    path = tmp_path / "mismatch.h5"
    _write_geometry_mismatch(path)

    molecule = QM9sDataset(path).get_molecule(113967)
    assert molecule.geometry_match is False
    # The SMILES survives, which is exactly why a warning is needed: something
    # will draw it.
    assert molecule.smiles


def test_older_schemas_are_not_retroactively_flagged(v2_2_file: Path) -> None:
    """2.0 and 2.1 never computed this, and absent must not read as False."""
    molecule = QM9sDataset(v2_2_file).get_molecule(8)
    assert molecule.geometry_match is True


def test_the_2d_pane_says_the_structure_does_not_match(tmp_path: Path) -> None:
    """The 2D is still drawn -- refusing would hide a usable molecule -- but
    it cannot be drawn silently, because the picture and the numbers come from
    two different arrangements of the same atoms."""
    from rpfr_visualizer.gui.render import GEOMETRY_MISMATCH_NOTE, geometry_mismatch_note

    path = tmp_path / "mismatch.h5"
    _write_geometry_mismatch(path)
    dataset = QM9sDataset(path)

    flagged = dataset.get_molecule(113967)
    assert geometry_mismatch_note(flagged) == GEOMETRY_MISMATCH_NOTE
    # ...and the 3D view is named as the one to trust, since it is built from
    # the coordinates the values were computed on.
    assert "3D" in GEOMETRY_MISMATCH_NOTE


def test_an_ordinary_molecule_gets_no_such_warning(v2_2_file: Path) -> None:
    """A caveat on every molecule is a caveat nobody reads."""
    from rpfr_visualizer.gui.render import geometry_mismatch_note

    assert geometry_mismatch_note(QM9sDataset(v2_2_file).get_molecule(8)) == ""
