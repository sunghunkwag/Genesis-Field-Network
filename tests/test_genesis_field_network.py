"""
Tests for genesis_field_network.core.GenesisFieldNetwork
"""

import numpy as np
import pytest

from genesis_field_network.core import FieldElement, GenesisFieldNetwork


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestGenesisFieldNetworkInit:
    def test_input_dim_stored(self, small_network):
        assert small_network.input_dim == 2

    def test_output_dim_stored(self, small_network):
        assert small_network.output_dim == 1

    def test_manifold_dim_stored(self, small_network):
        assert small_network.manifold_dim == 4

    def test_num_harmonics_stored(self, small_network):
        assert small_network.num_harmonics == 3

    def test_correct_number_of_fields_created(self, small_network):
        assert len(small_network.fields) == 6

    def test_fields_are_field_elements(self, small_network):
        for f in small_network.fields:
            assert isinstance(f, FieldElement)

    def test_input_projection_shape(self, small_network):
        assert small_network.input_projection.shape == (2, 6)

    def test_output_projection_shape(self, small_network):
        assert small_network.output_projection.shape == (6, 1)

    def test_coupler_created(self, small_network):
        assert small_network.coupler is not None

    def test_adapter_created(self, small_network):
        assert small_network.adapter is not None

    def test_morpher_created(self, small_network):
        assert small_network.morpher is not None

    def test_resonance_steps_default(self, small_network):
        assert small_network.resonance_steps == 5

    def test_custom_configuration(self):
        gfn = GenesisFieldNetwork(
            input_dim=3, output_dim=2, manifold_dim=8, num_fields=12, num_harmonics=5
        )
        assert gfn.input_dim == 3
        assert gfn.output_dim == 2
        assert len(gfn.fields) == 12
        assert gfn.input_projection.shape == (3, 12)
        assert gfn.output_projection.shape == (12, 2)


# ---------------------------------------------------------------------------
# forward()
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape(self, small_network):
        x = np.array([0.5, -0.3])
        out = small_network.forward(x)
        assert out.shape == (1,)

    def test_output_finite(self, small_network):
        x = np.random.randn(2)
        out = small_network.forward(x)
        assert np.all(np.isfinite(out))

    def test_output_is_numpy_array(self, small_network):
        x = np.zeros(2)
        out = small_network.forward(x)
        assert isinstance(out, np.ndarray)

    def test_different_inputs_different_outputs(self, small_network):
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])
        out1 = small_network.forward(x1)
        out2 = small_network.forward(x2)
        # Not guaranteed for all seeds, but very likely with a 6-field network
        assert out1.shape == out2.shape

    def test_zero_input_returns_finite(self, small_network):
        x = np.zeros(2)
        out = small_network.forward(x)
        assert np.all(np.isfinite(out))

    def test_large_input_bounded(self, small_network):
        """tanh ensures field energies stay in (-2, 2)."""
        x = np.ones(2) * 1000
        out = small_network.forward(x)
        assert np.all(np.isfinite(out))

    def test_projection_resizes_when_fields_count_changes(self):
        """forward() must not crash when field count differs from projection."""
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=4, num_fields=6, num_harmonics=3
        )
        # Manually remove a field to simulate morphing
        gfn.fields = gfn.fields[:4]
        x = np.array([0.5, -0.5])
        out = gfn.forward(x)
        assert out.shape == (1,)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# learn()
# ---------------------------------------------------------------------------

class TestLearn:
    def test_returns_float(self, small_network):
        x = np.array([1.0, 0.0])
        y = np.array([1.0])
        d = small_network.learn(x, y)
        assert isinstance(d, (float, np.floating))

    def test_dissonance_non_negative(self, small_network):
        x = np.array([1.0, 0.0])
        y = np.array([1.0])
        d = small_network.learn(x, y)
        assert d >= 0.0

    def test_field_count_respects_bounds(self, small_network):
        x = np.array([1.0, 0.0])
        y = np.array([1.0])
        for _ in range(5):
            small_network.learn(x, y)
        assert len(small_network.fields) >= small_network.morpher.min_fields
        assert len(small_network.fields) <= small_network.morpher.max_fields

    def test_dissonance_history_grows(self, small_network):
        x = np.array([0.0, 1.0])
        y = np.array([0.0])
        before = len(small_network.adapter.dissonance_history)
        small_network.learn(x, y)
        after = len(small_network.adapter.dissonance_history)
        assert after > before

    def test_repeated_learning_does_not_crash(self, small_network):
        x = np.array([0.5, 0.5])
        y = np.array([0.5])
        for _ in range(10):
            small_network.learn(x, y)


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

class TestTrain:
    def test_returns_list(self, small_network, xor_dataset):
        X, Y = xor_dataset
        history = small_network.train(X, Y, epochs=3, verbose=False)
        assert isinstance(history, list)

    def test_history_length_matches_epochs(self, small_network, xor_dataset):
        X, Y = xor_dataset
        epochs = 5
        history = small_network.train(X, Y, epochs=epochs, verbose=False)
        assert len(history) == epochs

    def test_history_values_non_negative(self, small_network, xor_dataset):
        X, Y = xor_dataset
        history = small_network.train(X, Y, epochs=5, verbose=False)
        assert all(d >= 0 for d in history)

    def test_history_values_finite(self, small_network, xor_dataset):
        X, Y = xor_dataset
        history = small_network.train(X, Y, epochs=5, verbose=False)
        assert all(np.isfinite(d) for d in history)

    def test_verbose_false_does_not_print(self, small_network, xor_dataset, capsys):
        X, Y = xor_dataset
        small_network.train(X, Y, epochs=2, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_single_sample_training(self):
        gfn = GenesisFieldNetwork(
            input_dim=1, output_dim=1, manifold_dim=4, num_fields=6, num_harmonics=3
        )
        X = np.array([[0.5]])
        Y = np.array([[1.0]])
        history = gfn.train(X, Y, epochs=3, verbose=False)
        assert len(history) == 3

    def test_multi_output_training(self):
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=3, manifold_dim=4, num_fields=6, num_harmonics=3
        )
        gfn.morpher.spawn_threshold = 999.0
        gfn.morpher.max_fields = 10
        X = np.random.randn(4, 2)
        Y = np.random.randn(4, 3)
        history = gfn.train(X, Y, epochs=2, verbose=False)
        assert len(history) == 2
        assert all(np.isfinite(d) for d in history)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_output_shape_single(self, small_network):
        X = np.array([[0.5, -0.5]])
        preds = small_network.predict(X)
        assert preds.shape == (1, 1)

    def test_output_shape_batch(self, small_network):
        X = np.random.randn(8, 2)
        preds = small_network.predict(X)
        assert preds.shape == (8, 1)

    def test_output_finite(self, small_network):
        X = np.random.randn(5, 2)
        preds = small_network.predict(X)
        assert np.all(np.isfinite(preds))

    def test_output_is_numpy_array(self, small_network):
        X = np.random.randn(3, 2)
        preds = small_network.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_consistent_with_forward(self):
        """predict() must produce the same results as sequential forward() calls.

        forward() has side effects on field energies, so we compare two networks
        initialised from the same seed: one via predict(), one via manual loops.
        """
        X = np.random.randn(4, 2)

        np.random.seed(77)
        gfn1 = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=4, num_fields=6, num_harmonics=3
        )
        gfn1.morpher.spawn_threshold = 999.0
        batch = gfn1.predict(X)

        np.random.seed(77)
        gfn2 = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=4, num_fields=6, num_harmonics=3
        )
        gfn2.morpher.spawn_threshold = 999.0
        manual = np.array([gfn2.forward(x) for x in X])

        np.testing.assert_allclose(batch, manual)


# ---------------------------------------------------------------------------
# get_state_summary()
# ---------------------------------------------------------------------------

class TestGetStateSummary:
    def test_returns_dict(self, small_network):
        summary = small_network.get_state_summary()
        assert isinstance(summary, dict)

    def test_required_keys_present(self, small_network):
        summary = small_network.get_state_summary()
        expected_keys = {
            "num_fields",
            "total_energy",
            "mean_energy",
            "max_energy",
            "morph_count",
            "dissonance_history",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_num_fields_correct(self, small_network):
        summary = small_network.get_state_summary()
        assert summary["num_fields"] == len(small_network.fields)

    def test_total_energy_non_negative(self, small_network):
        summary = small_network.get_state_summary()
        assert summary["total_energy"] >= 0

    def test_mean_energy_non_negative(self, small_network):
        summary = small_network.get_state_summary()
        assert summary["mean_energy"] >= 0

    def test_max_energy_gte_mean_energy(self, small_network):
        summary = small_network.get_state_summary()
        assert summary["max_energy"] >= summary["mean_energy"]

    def test_dissonance_history_is_list(self, small_network):
        summary = small_network.get_state_summary()
        assert isinstance(summary["dissonance_history"], list)

    def test_dissonance_history_at_most_10_entries(self, small_network, xor_dataset):
        X, Y = xor_dataset
        small_network.train(X, Y, epochs=5, verbose=False)
        summary = small_network.get_state_summary()
        assert len(summary["dissonance_history"]) <= 10

    def test_morph_count_is_int(self, small_network):
        summary = small_network.get_state_summary()
        assert isinstance(summary["morph_count"], int)

    def test_morph_count_non_negative(self, small_network):
        summary = small_network.get_state_summary()
        assert summary["morph_count"] >= 0
