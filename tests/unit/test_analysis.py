"""Unit tests for visualizer analysis modules."""

from __future__ import annotations

import math

import pytest

from rpfr_visualizer.analysis.analyze import run_molecule_analysis
from rpfr_visualizer.analysis.clumping import clumping_delta, compute_clumping
from rpfr_visualizer.analysis.fingerprint import (
    exact_shell_match,
    find_exact_matches,
    jaccard_similarity,
)
from rpfr_visualizer.analysis.fractionation import (
    alpha,
    compute_fractionation,
    epsilon_per_mil,
    ln_alpha,
)
from rpfr_visualizer.analysis.models import AnalysisKind, AnalysisRequest
from rpfr_visualizer.analysis.reference import MoleculeReference, SelfReference, StandardReference
from rpfr_visualizer.io.providers import HDF5Provider


def test_fractionation_alpha_and_epsilon(synthetic_h5_path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    ref = SelfReference(anchor_atom_index=1)  # H anchor for H comparisons
    result = compute_fractionation(
        mol,
        kind=AnalysisKind.ALPHA,
        temperature=300.0,
        reference=ref,
        element="H",
    )
    h_sites = [s for s in result.sites if s.element == "H"]
    assert len(h_sites) == 3
    assert all(s.value is not None for s in h_sites)
    assert h_sites[0].value == pytest.approx(
        mol.sites[1].beta_at(300.0) / mol.sites[1].beta_at(300.0)
    )


def test_cross_element_reference_returns_none(synthetic_h5_path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    ref = SelfReference(anchor_atom_index=0)  # C anchor
    beta = ref.resolve_beta(
        target_molecule=mol,
        target_atom_index=1,
        element="H",
        temperature=300.0,
    )
    assert beta is None


def test_standard_reference(synthetic_h5_path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    ref = StandardReference(beta_by_element={"C": 1.1, "H": 10.0})
    result = compute_fractionation(
        mol,
        kind=AnalysisKind.EPSILON,
        temperature=300.0,
        reference=ref,
        element="C",
    )
    c_site = next(s for s in result.sites if s.element == "C")
    expected_alpha = alpha(mol.sites[0].beta_at(300.0), 1.1)
    assert c_site.value == pytest.approx(epsilon_per_mil(expected_alpha))


def test_molecule_reference(synthetic_h5_path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol_a = provider.get_molecule(1)
    mol_b = provider.get_molecule(2)
    ref = MoleculeReference(reference_molecule=mol_b, reference_atom_index=0)
    beta = ref.resolve_beta(
        target_molecule=mol_a,
        target_atom_index=0,
        element="C",
        temperature=300.0,
    )
    assert beta == pytest.approx(mol_b.sites[0].beta_at(300.0))


def test_clumping_delta() -> None:
    assert clumping_delta(1.0) == 0.0
    assert clumping_delta(math.e) == pytest.approx(1.0)
    assert clumping_delta(0.0) is None


def test_compute_clumping(synthetic_h5_path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    arrays = provider.load_arrays(1)
    result = compute_clumping(mol, arrays, temperature=300.0)
    assert result.kind == AnalysisKind.CLUMPING
    assert len(result.sites) == mol.n_atoms
    assert result.pair_values is not None


def test_fingerprint_matching(synthetic_h5_path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mols = [provider.get_molecule(1), provider.get_molecule(2)]
    matches = find_exact_matches(mols, shell=0)
    assert isinstance(matches, list)
    assert exact_shell_match(("C", "CC"), ("C", "CN")) == [0]
    assert jaccard_similarity(("C", "CC"), ("C", "CN")) == pytest.approx(1 / 3)


def test_run_molecule_analysis_dispatch(synthetic_h5_path) -> None:
    provider = HDF5Provider(synthetic_h5_path)
    mol = provider.get_molecule(1)
    request = AnalysisRequest(kind=AnalysisKind.BETA, temperature=300.0)
    result = run_molecule_analysis(mol, request)
    assert result.sites[0].value == pytest.approx(mol.sites[0].beta_at(300.0))

    arrays = provider.load_arrays(1)
    clump_request = AnalysisRequest(kind=AnalysisKind.CLUMPING, temperature=300.0)
    clump_result = run_molecule_analysis(mol, clump_request, arrays=arrays)
    assert clump_result.kind == AnalysisKind.CLUMPING


def test_ln_alpha() -> None:
    assert ln_alpha(1.0) == pytest.approx(0.0)
    assert ln_alpha(math.e) == pytest.approx(1000.0)
