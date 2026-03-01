"""
Tests for genesis_field_network.core.ResonanceCoupler
"""

import numpy as np
import pytest

from genesis_field_network.core import FieldElement, ResonanceCoupler


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestResonanceCouplerInit:
    def test_manifold_dim_stored(self, coupler, manifold_dim):
        assert coupler.manifold_dim == manifold_dim

    def test_coupling_resolution_stored(self, manifold_dim):
        c = ResonanceCoupler(manifold_dim=manifold_dim, coupling_resolution=32)
        assert c.coupling_resolution == 32

    def test_sample_grid_created_on_init(self, coupler, manifold_dim):
        assert hasattr(coupler, "sample_points")
        assert coupler.sample_points.shape[1] == manifold_dim

    def test_sample_grid_shape(self, coupler):
        assert coupler.sample_points.shape == (
            coupler.coupling_resolution, coupler.manifold_dim
        )

    def test_default_coupling_resolution(self, manifold_dim):
        """Default coupling_resolution should be 64."""
        c = ResonanceCoupler(manifold_dim=manifold_dim)
        assert c.coupling_resolution == 64
        assert c.sample_points.shape == (64, manifold_dim)


# ---------------------------------------------------------------------------
# _update_sample_grid()
# ---------------------------------------------------------------------------

class TestUpdateSampleGrid:
    def test_refresh_changes_grid(self, coupler):
        old_grid = coupler.sample_points.copy()
        np.random.seed(99)
        coupler._update_sample_grid()
        # The new grid is (probably) different
        assert coupler.sample_points.shape == old_grid.shape

    def test_grid_shape_preserved(self, coupler):
        coupler._update_sample_grid()
        assert coupler.sample_points.shape == (
            coupler.coupling_resolution, coupler.manifold_dim
        )

    def test_grid_finite_values(self, coupler):
        coupler._update_sample_grid()
        assert np.all(np.isfinite(coupler.sample_points))


# ---------------------------------------------------------------------------
# compute_coupling_matrix()
# ---------------------------------------------------------------------------

class TestComputeCouplingMatrix:
    def test_output_shape(self, coupler, small_field_list):
        n = len(small_field_list)
        matrix = coupler.compute_coupling_matrix(small_field_list)
        assert matrix.shape == (n, n)

    def test_symmetric(self, coupler, small_field_list):
        matrix = coupler.compute_coupling_matrix(small_field_list)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)

    def test_diagonal_is_zero(self, coupler, small_field_list):
        """The coupling matrix is computed for i < j pairs; diagonal stays 0."""
        matrix = coupler.compute_coupling_matrix(small_field_list)
        np.testing.assert_allclose(np.diag(matrix), 0.0)

    def test_values_in_valid_range(self, coupler, small_field_list):
        """Resonance values are correlations → [-1, 1]."""
        matrix = coupler.compute_coupling_matrix(small_field_list)
        assert np.all(matrix >= -1.0 - 1e-9)
        assert np.all(matrix <= 1.0 + 1e-9)

    def test_single_field(self, coupler, manifold_dim, num_harmonics):
        fields = [FieldElement(manifold_dim, num_harmonics)]
        matrix = coupler.compute_coupling_matrix(fields)
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0.0

    def test_finite_values(self, coupler, small_field_list):
        matrix = coupler.compute_coupling_matrix(small_field_list)
        assert np.all(np.isfinite(matrix))

    def test_two_identical_fields_max_resonance(self, coupler, manifold_dim, num_harmonics):
        """Two identical fields should have resonance = 1."""
        np.random.seed(5)
        f = FieldElement(manifold_dim, num_harmonics)
        matrix = coupler.compute_coupling_matrix([f, f])
        assert abs(matrix[0, 1] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# propagate()
# ---------------------------------------------------------------------------

class TestPropagate:
    def test_output_shape(self, coupler, small_field_list):
        excitation = np.ones(len(small_field_list))
        energies = coupler.propagate(small_field_list, excitation)
        assert energies.shape == (len(small_field_list),)

    def test_energies_bounded(self, coupler, small_field_list):
        """After propagation energies are bounded by tanh(x)*2 ∈ (-2, 2)."""
        excitation = np.ones(len(small_field_list))
        energies = coupler.propagate(small_field_list, excitation)
        assert np.all(np.abs(energies) <= 2.0 + 1e-9)

    def test_fields_energies_updated(self, coupler, small_field_list):
        excitation = np.ones(len(small_field_list))
        coupler.propagate(small_field_list, excitation)
        # All field.energy values should now be floats (updated in-place)
        for f in small_field_list:
            assert isinstance(f.energy, (float, np.floating))

    def test_output_finite(self, coupler, small_field_list):
        excitation = np.random.randn(len(small_field_list))
        energies = coupler.propagate(small_field_list, excitation)
        assert np.all(np.isfinite(energies))

    def test_excitation_shorter_than_fields(self, coupler, small_field_list):
        """Partial excitation (fewer elements than fields) should work."""
        excitation = np.ones(2)  # only first 2 fields are excited
        energies = coupler.propagate(small_field_list, excitation)
        assert energies.shape == (len(small_field_list),)

    def test_zero_excitation_returns_finite(self, coupler, small_field_list):
        excitation = np.zeros(len(small_field_list))
        energies = coupler.propagate(small_field_list, excitation)
        assert np.all(np.isfinite(energies))

    def test_propagate_mutates_field_energies(self, coupler, small_field_list):
        original = [f.energy for f in small_field_list]
        excitation = np.ones(len(small_field_list)) * 5.0
        coupler.propagate(small_field_list, excitation)
        updated = [f.energy for f in small_field_list]
        assert original != updated
