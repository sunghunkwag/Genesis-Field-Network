"""
Shared pytest fixtures for Genesis Field Network tests.
"""

import numpy as np
import pytest

from genesis_field_network.core import (
    FieldElement,
    GenesisFieldNetwork,
    PhaseAdapter,
    ResonanceCoupler,
    TopologicalMorpher,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_random_seed():
    """Reset the NumPy random seed before every test for reproducibility."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Primitive building-block fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manifold_dim():
    return 4


@pytest.fixture
def num_harmonics():
    return 3


@pytest.fixture
def field(manifold_dim, num_harmonics):
    """A single FieldElement with deterministic state."""
    return FieldElement(manifold_dim=manifold_dim, num_harmonics=num_harmonics)


@pytest.fixture
def field_pair(manifold_dim, num_harmonics):
    """Two FieldElements sharing the same manifold."""
    return (
        FieldElement(manifold_dim=manifold_dim, num_harmonics=num_harmonics),
        FieldElement(manifold_dim=manifold_dim, num_harmonics=num_harmonics),
    )


@pytest.fixture
def sample_points(manifold_dim):
    """A small batch of random manifold points for field evaluation."""
    return np.random.randn(16, manifold_dim)


@pytest.fixture
def small_field_list(manifold_dim, num_harmonics):
    """A list of 6 FieldElements – larger than min_fields (4)."""
    return [FieldElement(manifold_dim, num_harmonics) for _ in range(6)]


# ---------------------------------------------------------------------------
# Component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coupler(manifold_dim):
    return ResonanceCoupler(manifold_dim=manifold_dim, coupling_resolution=16)


@pytest.fixture
def adapter():
    return PhaseAdapter(adaptation_rate=0.05, dissonance_threshold=0.1)


@pytest.fixture
def morpher():
    return TopologicalMorpher(
        merge_threshold=0.95,
        split_threshold=3.0,
        spawn_threshold=2.0,
        dissolve_threshold=0.01,
        max_fields=16,
        min_fields=4,
    )


# ---------------------------------------------------------------------------
# Full-network fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_network():
    """A tiny GFN suitable for fast unit-level tests.

    Morphing is constrained (high spawn_threshold, low max_fields) so that
    field count stays bounded during training tests and runtimes stay short.
    """
    gfn = GenesisFieldNetwork(
        input_dim=2,
        output_dim=1,
        manifold_dim=4,
        num_fields=6,
        num_harmonics=3,
    )
    # Prevent uncontrolled SPAWN growth which makes O(n²) coupling explode
    gfn.morpher.spawn_threshold = 999.0
    gfn.morpher.max_fields = 8
    return gfn


@pytest.fixture
def xor_dataset():
    """Classic XOR problem (inputs and targets)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, Y
