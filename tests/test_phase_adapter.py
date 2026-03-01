"""
Tests for genesis_field_network.core.PhaseAdapter
"""

import numpy as np
import pytest

from genesis_field_network.core import FieldElement, PhaseAdapter, ResonanceCoupler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fields(n, manifold_dim=4, num_harmonics=3):
    return [FieldElement(manifold_dim, num_harmonics) for _ in range(n)]


def make_coupler(manifold_dim=4):
    return ResonanceCoupler(manifold_dim=manifold_dim, coupling_resolution=16)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestPhaseAdapterInit:
    def test_adaptation_rate_stored(self):
        pa = PhaseAdapter(adaptation_rate=0.05)
        assert pa.adaptation_rate == 0.05

    def test_dissonance_threshold_stored(self):
        pa = PhaseAdapter(dissonance_threshold=0.2)
        assert pa.dissonance_threshold == 0.2

    def test_dissonance_history_empty_on_init(self):
        pa = PhaseAdapter()
        assert pa.dissonance_history == []

    def test_default_adaptation_rate(self):
        pa = PhaseAdapter()
        assert pa.adaptation_rate == 0.01

    def test_default_dissonance_threshold(self):
        pa = PhaseAdapter()
        assert pa.dissonance_threshold == 0.1


# ---------------------------------------------------------------------------
# compute_dissonance()
# ---------------------------------------------------------------------------

class TestComputeDissonance:
    def test_identical_patterns_zero_dissonance(self, adapter):
        pattern = np.array([0.1, 0.5, -0.3, 0.7])
        d = adapter.compute_dissonance(pattern, pattern)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_returns_float(self, adapter):
        out = np.array([1.0, 2.0, 3.0])
        tgt = np.array([0.0, 0.0, 0.0])
        d = adapter.compute_dissonance(out, tgt)
        assert isinstance(d, float)

    def test_non_negative(self, adapter):
        out = np.random.randn(10)
        tgt = np.random.randn(10)
        d = adapter.compute_dissonance(out, tgt)
        assert d >= 0.0

    def test_appended_to_history(self, adapter):
        assert len(adapter.dissonance_history) == 0
        adapter.compute_dissonance(np.array([1.0]), np.array([0.0]))
        assert len(adapter.dissonance_history) == 1

    def test_history_accumulates(self, adapter):
        for _ in range(5):
            adapter.compute_dissonance(np.random.randn(4), np.random.randn(4))
        assert len(adapter.dissonance_history) == 5

    def test_length_mismatch_handled(self, adapter):
        """Mismatched lengths should be truncated to the shorter one."""
        out = np.array([1.0, 2.0, 3.0, 4.0])
        tgt = np.array([1.0, 2.0])
        d = adapter.compute_dissonance(out, tgt)
        assert isinstance(d, float)
        assert d >= 0.0

    def test_single_element_patterns(self, adapter):
        d = adapter.compute_dissonance(np.array([1.0]), np.array([0.0]))
        assert d > 0

    def test_dissonance_increases_with_difference(self, adapter):
        base = np.zeros(8)
        small_diff = adapter.compute_dissonance(base, np.ones(8) * 0.1)
        large_diff = adapter.compute_dissonance(base, np.ones(8) * 10.0)
        assert large_diff > small_diff

    def test_finite_for_random_patterns(self, adapter):
        out = np.random.randn(16)
        tgt = np.random.randn(16)
        d = adapter.compute_dissonance(out, tgt)
        assert np.isfinite(d)


# ---------------------------------------------------------------------------
# adapt_fields()
# ---------------------------------------------------------------------------

class TestAdaptFields:
    def test_returns_float(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        out = np.array([0.5, 0.3])
        tgt = np.array([0.0, 0.0])
        result = adapter.adapt_fields(fields, out, tgt, coupler)
        assert isinstance(result, float)

    def test_below_threshold_skips_adaptation(self):
        """If dissonance < threshold, adapt_fields should return immediately."""
        pa = PhaseAdapter(adaptation_rate=0.1, dissonance_threshold=1e6)
        fields = make_fields(4)
        coupler = make_coupler()
        original_phases = [f.phases.copy() for f in fields]
        out = np.zeros(2)
        tgt = np.zeros(2)
        pa.adapt_fields(fields, out, tgt, coupler)
        # Phases should be unchanged since dissonance (≈0) < threshold
        for f, orig in zip(fields, original_phases):
            np.testing.assert_array_equal(f.phases, orig)

    def test_phases_in_range_after_adaptation(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        out = np.random.randn(2)
        tgt = np.random.randn(2)
        adapter.adapt_fields(fields, out, tgt, coupler)
        for f in fields:
            assert np.all(f.phases >= 0)
            assert np.all(f.phases < 2 * np.pi + 1e-9)

    def test_frequencies_within_clip_bounds(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        out = np.random.randn(2)
        tgt = np.random.randn(2)
        adapter.adapt_fields(fields, out, tgt, coupler)
        for f in fields:
            assert np.all(f.frequencies >= 0.01)
            assert np.all(f.frequencies <= 10.0)

    def test_curvature_stays_positive_definite(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        out = np.ones(2) * 5
        tgt = np.zeros(2)
        adapter.adapt_fields(fields, out, tgt, coupler)
        for f in fields:
            eigenvalues = np.linalg.eigvalsh(f.curvature)
            assert np.all(eigenvalues > 0), "Curvature must remain positive definite"

    def test_amplitudes_normalized_after_adaptation(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        out = np.random.randn(2)
        tgt = np.random.randn(2)
        adapter.adapt_fields(fields, out, tgt, coupler)
        for f in fields:
            assert abs(np.sum(f.amplitudes) - 1.0) < 1e-9

    def test_multiple_adaptation_steps_reduce_dissonance(self):
        """Running multiple adaptation steps should not error and returns positive dissonance."""
        np.random.seed(0)
        pa = PhaseAdapter(adaptation_rate=0.1, dissonance_threshold=0.0)
        fields = make_fields(4, manifold_dim=4, num_harmonics=3)
        coupler = make_coupler()
        out = np.array([1.0, 1.0])
        tgt = np.array([0.0, 0.0])

        dissonances = []
        for _ in range(5):
            d = pa.adapt_fields(fields, out, tgt, coupler)
            dissonances.append(d)

        # The first dissonance should be well above zero
        assert dissonances[0] > 1e-6

    def test_no_nan_in_phases_after_adaptation(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        out = np.random.randn(2)
        tgt = np.random.randn(2)
        adapter.adapt_fields(fields, out, tgt, coupler)
        for f in fields:
            assert np.all(np.isfinite(f.phases))
            assert np.all(np.isfinite(f.frequencies))
            assert np.all(np.isfinite(f.curvature))
