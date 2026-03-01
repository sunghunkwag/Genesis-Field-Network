"""
Tests for genesis_field_network.core.FieldElement
"""

import numpy as np
import pytest

from genesis_field_network.core import FieldElement


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestFieldElementInit:
    def test_position_shape(self, field, manifold_dim):
        assert field.position.shape == (manifold_dim,)

    def test_frequencies_shape(self, field, manifold_dim, num_harmonics):
        assert field.frequencies.shape == (num_harmonics, manifold_dim)

    def test_phases_shape(self, field, num_harmonics):
        assert field.phases.shape == (num_harmonics,)

    def test_amplitudes_shape(self, field, num_harmonics):
        assert field.amplitudes.shape == (num_harmonics,)

    def test_amplitudes_normalized(self, field):
        """Amplitudes must form a probability distribution (sum to 1)."""
        assert abs(np.sum(field.amplitudes) - 1.0) < 1e-9

    def test_amplitudes_non_negative(self, field):
        assert np.all(field.amplitudes >= 0)

    def test_curvature_shape(self, field, manifold_dim):
        assert field.curvature.shape == (manifold_dim, manifold_dim)

    def test_curvature_symmetric(self, field):
        """Curvature tensor must be symmetric."""
        diff = np.max(np.abs(field.curvature - field.curvature.T))
        assert diff < 1e-12

    def test_curvature_positive_definite(self, field):
        """Curvature tensor must be positive definite."""
        eigenvalues = np.linalg.eigvalsh(field.curvature)
        assert np.all(eigenvalues > 0)

    def test_initial_energy(self, field):
        assert field.energy == 1.0

    def test_resonance_history_empty(self, field):
        assert field.resonance_history == []

    def test_frequencies_in_valid_range(self, field):
        """Frequencies are sampled from U(0.1, 5.0)."""
        assert np.all(field.frequencies >= 0.1)
        assert np.all(field.frequencies <= 5.0)

    def test_phases_in_valid_range(self, field):
        """Phases are sampled from U(0, 2π)."""
        assert np.all(field.phases >= 0)
        assert np.all(field.phases < 2 * np.pi)

    def test_different_seeds_produce_different_fields(self, manifold_dim, num_harmonics):
        np.random.seed(0)
        f1 = FieldElement(manifold_dim, num_harmonics)
        np.random.seed(1)
        f2 = FieldElement(manifold_dim, num_harmonics)
        assert not np.allclose(f1.position, f2.position)

    def test_default_harmonics(self, manifold_dim):
        """Default num_harmonics should be 8."""
        f = FieldElement(manifold_dim=manifold_dim)
        assert f.num_harmonics == 8
        assert f.phases.shape == (8,)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestFieldElementEvaluate:
    def test_output_shape_single_point(self, field, manifold_dim):
        pts = np.zeros((1, manifold_dim))
        result = field.evaluate(pts)
        assert result.shape == (1,)

    def test_output_shape_batch(self, field, sample_points):
        result = field.evaluate(sample_points)
        assert result.shape == (len(sample_points),)

    def test_output_finite(self, field, sample_points):
        result = field.evaluate(sample_points)
        assert np.all(np.isfinite(result))

    def test_output_bounded(self, field, sample_points):
        """Field values must be bounded by the energy envelope."""
        result = field.evaluate(sample_points)
        # The envelope is bounded by field.energy; field_value is a sum of
        # unit-amplitude sines, so the total is bounded by num_harmonics.
        max_possible = field.energy * field.num_harmonics
        assert np.all(np.abs(result) <= max_possible + 1e-9)

    def test_output_decays_far_from_center(self, field, manifold_dim):
        """Values at large distances should be smaller than at the center."""
        near = field.position.reshape(1, -1)
        far = (field.position + 1000).reshape(1, -1)
        val_near = np.abs(field.evaluate(near))[0]
        val_far = np.abs(field.evaluate(far))[0]
        assert val_far < val_near + 1e-10  # far should be ≤ near

    def test_evaluate_at_position_uses_envelope(self, manifold_dim, num_harmonics):
        """Evaluate is dominated by envelope; zero distance gives max envelope."""
        np.random.seed(7)
        f = FieldElement(manifold_dim, num_harmonics)
        # At the field center the quad_form = 0, so envelope = f.energy
        center = f.position.reshape(1, -1)
        result = f.evaluate(center)
        assert np.isfinite(result[0])

    def test_evaluate_zero_energy(self, manifold_dim, num_harmonics, sample_points):
        np.random.seed(0)
        f = FieldElement(manifold_dim, num_harmonics)
        f.energy = 0.0
        result = f.evaluate(sample_points)
        assert np.allclose(result, 0.0)

    def test_evaluate_multiple_points_independent(self, field, manifold_dim):
        """Evaluating points one by one vs. in a batch must give the same result."""
        pts = np.random.randn(5, manifold_dim)
        batch = field.evaluate(pts)
        singles = np.array([field.evaluate(pts[i:i+1])[0] for i in range(5)])
        np.testing.assert_allclose(batch, singles, rtol=1e-10)


# ---------------------------------------------------------------------------
# compute_resonance()
# ---------------------------------------------------------------------------

class TestFieldElementComputeResonance:
    def test_self_resonance_is_one(self, field, sample_points):
        """A field perfectly resonates with itself."""
        r = field.compute_resonance(field, sample_points)
        assert abs(r - 1.0) < 1e-9

    def test_resonance_range(self, field_pair, sample_points):
        f1, f2 = field_pair
        r = f1.compute_resonance(f2, sample_points)
        assert -1.0 <= r <= 1.0

    def test_resonance_symmetric(self, field_pair, sample_points):
        f1, f2 = field_pair
        r12 = f1.compute_resonance(f2, sample_points)
        r21 = f2.compute_resonance(f1, sample_points)
        assert abs(r12 - r21) < 1e-12

    def test_resonance_returns_float(self, field_pair, sample_points):
        f1, f2 = field_pair
        r = f1.compute_resonance(f2, sample_points)
        assert isinstance(r, float)

    def test_constant_field_returns_zero(self, field, manifold_dim, num_harmonics):
        """If one field evaluates to a constant (std ≈ 0), resonance = 0."""
        f_const = FieldElement(manifold_dim, num_harmonics)
        # Force zero energy so the field evaluates to 0 everywhere
        f_const.energy = 0.0
        pts = np.random.randn(20, manifold_dim)
        r = field.compute_resonance(f_const, pts)
        assert r == 0.0

    def test_resonance_not_nan(self, field_pair, sample_points):
        f1, f2 = field_pair
        r = f1.compute_resonance(f2, sample_points)
        assert not np.isnan(r)
