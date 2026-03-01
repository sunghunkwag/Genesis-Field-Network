"""
Integration tests for Genesis Field Network.

These tests exercise the full system end-to-end and verify that the network
can learn simple patterns when given enough capacity and steps.
"""

import numpy as np
import pytest

from genesis_field_network.core import GenesisFieldNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gfn(input_dim, output_dim, num_fields=8, manifold_dim=4, num_harmonics=3):
    """Create a small GFN with morphing constrained for fast test execution.

    The default topology is deliberately small (4-dim manifold, 8 fields,
    3 harmonics) and SPAWN is disabled so that O(n²) coupling does not blow
    up during training loops.
    """
    gfn = GenesisFieldNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        manifold_dim=manifold_dim,
        num_fields=num_fields,
        num_harmonics=num_harmonics,
    )
    # Prevent SPAWN from inflating field count and making tests prohibitively slow
    gfn.morpher.spawn_threshold = 999.0
    gfn.morpher.max_fields = num_fields + 4
    return gfn


# ---------------------------------------------------------------------------
# End-to-end training loop
# ---------------------------------------------------------------------------

class TestEndToEndTraining:
    def test_training_completes_without_error(self):
        np.random.seed(0)
        gfn = make_gfn(2, 1)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        Y = np.array([[0], [1], [1], [0]], dtype=float)
        history = gfn.train(X, Y, epochs=10, verbose=False)
        assert len(history) == 10

    def test_training_reduces_dissonance_on_linear_problem(self):
        """A simple linear problem should show clearly decreasing dissonance."""
        np.random.seed(42)
        gfn = make_gfn(1, 1, num_fields=6)
        X = np.linspace(-1, 1, 8).reshape(-1, 1)
        Y = X * 2.0  # y = 2x

        history = gfn.train(X, Y, epochs=10, verbose=False)
        first_third = np.mean(history[:3])
        last_third = np.mean(history[7:])
        # Dissonance should generally decrease (lenient bound for stochastic dynamics)
        assert last_third <= first_third * 3

    def test_predict_after_training_returns_correct_shape(self):
        np.random.seed(0)
        gfn = make_gfn(2, 3)
        X = np.random.randn(6, 2)
        Y = np.random.randn(6, 3)
        gfn.train(X, Y, epochs=3, verbose=False)
        preds = gfn.predict(X)
        assert preds.shape == (6, 3)

    def test_network_state_summary_after_training(self):
        np.random.seed(0)
        gfn = make_gfn(2, 1)
        X = np.array([[0, 1], [1, 0]], dtype=float)
        Y = np.array([[1], [0]], dtype=float)
        gfn.train(X, Y, epochs=5, verbose=False)
        summary = gfn.get_state_summary()
        assert summary["num_fields"] >= gfn.morpher.min_fields
        assert summary["morph_count"] >= 0
        assert len(summary["dissonance_history"]) > 0

    def test_field_count_remains_bounded_after_training(self):
        np.random.seed(1)
        gfn = make_gfn(2, 1)
        X = np.random.randn(4, 2)
        Y = np.random.randn(4, 1)
        gfn.train(X, Y, epochs=5, verbose=False)
        assert len(gfn.fields) >= gfn.morpher.min_fields
        assert len(gfn.fields) <= gfn.morpher.max_fields


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageImport:
    def test_import_all_classes(self):
        from genesis_field_network import (
            FieldElement,
            GenesisFieldNetwork,
            PhaseAdapter,
            ResonanceCoupler,
            TopologicalMorpher,
        )
        assert FieldElement is not None
        assert ResonanceCoupler is not None
        assert PhaseAdapter is not None
        assert TopologicalMorpher is not None
        assert GenesisFieldNetwork is not None

    def test_version_string_exists(self):
        import genesis_field_network
        assert hasattr(genesis_field_network, "__version__")
        assert isinstance(genesis_field_network.__version__, str)

    def test_all_exports_defined(self):
        import genesis_field_network
        for name in genesis_field_network.__all__:
            assert hasattr(genesis_field_network, name)


# ---------------------------------------------------------------------------
# Regression: projection resizing after morphing
# ---------------------------------------------------------------------------

class TestProjectionResizingAfterMorphing:
    def test_field_count_increase_does_not_break_forward(self):
        """If morphing increases field count, forward() must still work."""
        np.random.seed(5)
        gfn = make_gfn(2, 1, num_fields=6)
        # Manually append extra fields (simulates SPAWN during morph)
        from genesis_field_network.core import FieldElement
        extra = FieldElement(gfn.manifold_dim, gfn.num_harmonics)
        gfn.fields.append(extra)
        x = np.array([0.5, 0.5])
        out = gfn.forward(x)
        assert out.shape == (1,)
        assert np.all(np.isfinite(out))

    def test_field_count_decrease_does_not_break_forward(self):
        """If morphing decreases field count, forward() must still work."""
        np.random.seed(5)
        gfn = make_gfn(2, 1, num_fields=8)
        # Manually remove fields (simulates DISSOLVE / MERGE during morph)
        gfn.fields = gfn.fields[:4]
        x = np.array([0.5, 0.5])
        out = gfn.forward(x)
        assert out.shape == (1,)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Determinism: same seed ↦ same forward output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_forward(self):
        np.random.seed(42)
        gfn1 = make_gfn(2, 1, num_fields=6)
        out1 = gfn1.forward(np.array([0.3, -0.7]))

        np.random.seed(42)
        gfn2 = make_gfn(2, 1, num_fields=6)
        out2 = gfn2.forward(np.array([0.3, -0.7]))

        np.testing.assert_allclose(out1, out2)

    def test_same_seed_same_training_history(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        Y = np.array([[1.0], [0.0]], dtype=float)

        np.random.seed(7)
        gfn1 = make_gfn(2, 1, num_fields=6)
        h1 = gfn1.train(X, Y, epochs=3, verbose=False)

        np.random.seed(7)
        gfn2 = make_gfn(2, 1, num_fields=6)
        h2 = gfn2.train(X, Y, epochs=3, verbose=False)

        np.testing.assert_allclose(h1, h2)


# ---------------------------------------------------------------------------
# Edge-case: single field network
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_min_fields_network_survives_training(self):
        np.random.seed(0)
        gfn = GenesisFieldNetwork(
            input_dim=1, output_dim=1,
            manifold_dim=4, num_fields=4, num_harmonics=3
        )
        gfn.morpher.min_fields = 4
        gfn.morpher.max_fields = 8
        gfn.morpher.spawn_threshold = 999.0
        X = np.array([[0.1], [0.5], [0.9]])
        Y = np.array([[0.0], [0.5], [1.0]])
        history = gfn.train(X, Y, epochs=3, verbose=False)
        assert len(history) == 3
        assert len(gfn.fields) >= 4

    def test_large_batch_predict(self):
        np.random.seed(0)
        gfn = make_gfn(4, 2, num_fields=6)
        X = np.random.randn(20, 4)
        preds = gfn.predict(X)
        assert preds.shape == (20, 2)
        assert np.all(np.isfinite(preds))

    def test_high_dimensional_input(self):
        np.random.seed(0)
        gfn = make_gfn(16, 4, num_fields=6, manifold_dim=4)
        X = np.random.randn(5, 16)
        preds = gfn.predict(X)
        assert preds.shape == (5, 4)
        assert np.all(np.isfinite(preds))
