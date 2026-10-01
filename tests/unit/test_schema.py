"""Tests for io/schema.py."""

from rpfr_visualizer.io.schema import DEFAULT_SOURCE, TEMPERATURES_K, SiteRecord
from rpfr_visualizer.io.validate import derive_qc_pass


def test_temperatures_k_has_13_values():
    assert len(TEMPERATURES_K) == 13
    assert 300.0 in TEMPERATURES_K
    assert 298.15 in TEMPERATURES_K


def test_site_beta_at_exact_temperature():
    site = SiteRecord(
        molecule_id=1,
        atom_index=0,
        element="C",
        beta=(1.1, 1.12, 1.15),
        temperatures=(20.0, 300.0, 420.0),
        source=DEFAULT_SOURCE,
    )
    assert site.beta_at(300.0) == 1.12


def test_site_beta_at_off_grid_interpolates():
    # Need enough points for a stable ln(β) vs 1/T fit.
    temperatures = (20.0, 100.0, 180.0, 260.0, 340.0, 420.0)
    beta = tuple(1.05 + 20.0 / t for t in temperatures)
    site = SiteRecord(
        molecule_id=1,
        atom_index=0,
        element="C",
        beta=beta,
        temperatures=temperatures,
    )
    value = site.beta_at(250.0, warn=False)
    assert value > 1.0
    assert abs(value - (1.05 + 20.0 / 250.0)) < 0.05


def test_derive_qc_pass():
    assert derive_qc_pass(
        freq_qc_ok=True,
        rpfr_qc_ok=True,
        tr_converged=True,
        rdkit_ok=True,
        smiles_order_match=True,
        n_imag_vib=0,
    )
    assert not derive_qc_pass(
        freq_qc_ok=True,
        rpfr_qc_ok=True,
        tr_converged=True,
        rdkit_ok=True,
        smiles_order_match=True,
        n_imag_vib=1,
    )
