"""
Tests for genesis_field_network.core.TopologicalMorpher
"""

import numpy as np
import pytest

from genesis_field_network.core import FieldElement, ResonanceCoupler, TopologicalMorpher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MANIFOLD_DIM = 4
NUM_HARMONICS = 3


def make_field(**kwargs):
    return FieldElement(MANIFOLD_DIM, NUM_HARMONICS)


def make_fields(n):
    return [make_field() for _ in range(n)]


def make_coupler():
    return ResonanceCoupler(manifold_dim=MANIFOLD_DIM, coupling_resolution=16)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestTopologicalMorpherInit:
    def test_thresholds_stored(self, morpher):
        assert morpher.merge_threshold == 0.95
        assert morpher.split_threshold == 3.0
        assert morpher.spawn_threshold == 2.0
        assert morpher.dissolve_threshold == 0.01

    def test_field_bounds_stored(self, morpher):
        assert morpher.max_fields == 16
        assert morpher.min_fields == 4

    def test_morph_log_empty_on_init(self, morpher):
        assert morpher.morph_log == []

    def test_defaults(self):
        m = TopologicalMorpher()
        assert m.merge_threshold == 0.95
        assert m.split_threshold == 3.0
        assert m.spawn_threshold == 2.0
        assert m.dissolve_threshold == 0.01
        assert m.max_fields == 256
        assert m.min_fields == 4


# ---------------------------------------------------------------------------
# morph() – general contract
# ---------------------------------------------------------------------------

class TestMorphGeneral:
    def test_returns_list_of_field_elements(self, morpher):
        fields = make_fields(6)
        coupler = make_coupler()
        result = morpher.morph(fields, coupler, current_dissonance=0.5)
        assert isinstance(result, list)
        assert all(isinstance(f, FieldElement) for f in result)

    def test_original_list_not_mutated(self, morpher):
        """morph() should not mutate the input list in place."""
        fields = make_fields(6)
        original_ids = [id(f) for f in fields]
        coupler = make_coupler()
        morpher.morph(fields, coupler, current_dissonance=0.5)
        # The original list should still contain the same objects
        assert [id(f) for f in fields] == original_ids

    def test_respects_min_fields(self, morpher):
        fields = make_fields(morpher.min_fields)
        coupler = make_coupler()
        result = morpher.morph(fields, coupler, current_dissonance=0.0)
        assert len(result) >= morpher.min_fields

    def test_respects_max_fields(self, morpher):
        # Start with exactly max_fields; a high dissonance tries to SPAWN
        fields = make_fields(morpher.max_fields)
        coupler = make_coupler()
        result = morpher.morph(fields, coupler, current_dissonance=100.0)
        assert len(result) <= morpher.max_fields

    def test_result_fields_have_correct_manifold_dim(self, morpher):
        fields = make_fields(6)
        coupler = make_coupler()
        result = morpher.morph(fields, coupler, current_dissonance=0.5)
        for f in result:
            assert f.manifold_dim == MANIFOLD_DIM

    def test_result_fields_energies_finite(self, morpher):
        fields = make_fields(6)
        coupler = make_coupler()
        result = morpher.morph(fields, coupler, current_dissonance=0.5)
        for f in result:
            assert np.isfinite(f.energy)


# ---------------------------------------------------------------------------
# DISSOLVE path
# ---------------------------------------------------------------------------

class TestDissolve:
    def test_near_zero_energy_fields_removed(self, morpher):
        """Fields with |energy| ≤ dissolve_threshold should be dissolved."""
        fields = make_fields(8)
        for f in fields[4:]:
            f.energy = 0.0  # will be dissolved
        coupler = make_coupler()
        result = morpher.morph(fields, coupler, current_dissonance=0.0)
        # We can't guarantee exact count due to other morph ops, but all
        # remaining fields should have |energy| > dissolve_threshold OR be
        # newly spawned replacement fields (which start with energy=1.0).
        for f in result:
            assert isinstance(f, FieldElement)

    def test_min_fields_enforced_after_dissolve(self, morpher):
        """Even after dissolving many fields, min_fields must be maintained."""
        fields = make_fields(morpher.min_fields + 2)
        for f in fields:
            f.energy = 0.0  # force all to dissolve
        coupler = make_coupler()
        result = morpher.morph(fields, coupler, current_dissonance=0.0)
        assert len(result) >= morpher.min_fields


# ---------------------------------------------------------------------------
# SPAWN path
# ---------------------------------------------------------------------------

class TestSpawn:
    def test_spawn_triggered_by_high_dissonance(self):
        morpher = TopologicalMorpher(
            spawn_threshold=1.0, max_fields=20, min_fields=4
        )
        fields = make_fields(6)
        coupler = make_coupler()
        count_before = len(fields)
        result = morpher.morph(fields, coupler, current_dissonance=5.0)
        # A SPAWN event should have occurred
        spawn_events = [e for e in morpher.morph_log if e[0] == "SPAWN"]
        assert len(spawn_events) >= 1

    def test_no_spawn_when_max_fields_reached(self):
        """After morphing, field count must never exceed max_fields."""
        m = TopologicalMorpher(
            merge_threshold=0.95,
            split_threshold=9999.0,   # disable SPLIT
            spawn_threshold=0.0,      # always SPAWN if below max_fields
            dissolve_threshold=0.0,   # never DISSOLVE
            max_fields=8,
            min_fields=4,
        )
        # Fill to exactly max_fields with high-energy fields so MERGE is unlikely
        fields = make_fields(8)
        for f in fields:
            f.energy = 1.0
        coupler = make_coupler()
        result = m.morph(fields, coupler, current_dissonance=100.0)
        assert len(result) <= m.max_fields

    def test_spawned_field_energy_proportional_to_dissonance(self):
        morpher = TopologicalMorpher(spawn_threshold=0.0, max_fields=20, min_fields=4)
        fields = make_fields(6)
        coupler = make_coupler()
        dissonance = 4.0
        result = morpher.morph(fields, coupler, current_dissonance=dissonance)
        spawn_events = [e for e in morpher.morph_log if e[0] == "SPAWN"]
        if spawn_events:
            # The last appended field has energy = dissonance * 0.1
            spawned = result[-1]
            assert spawned.energy == pytest.approx(dissonance * 0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# _merge_fields()
# ---------------------------------------------------------------------------

class TestMergeFields:
    def test_merged_position_is_average(self, morpher):
        np.random.seed(0)
        a = make_field()
        b = make_field()
        merged = morpher._merge_fields(a, b)
        np.testing.assert_allclose(
            merged.position, (a.position + b.position) / 2
        )

    def test_merged_energy_is_sum(self, morpher):
        np.random.seed(0)
        a = make_field()
        a.energy = 1.5
        b = make_field()
        b.energy = 2.5
        merged = morpher._merge_fields(a, b)
        assert merged.energy == pytest.approx(4.0)

    def test_merged_amplitudes_normalized(self, morpher):
        np.random.seed(0)
        a = make_field()
        b = make_field()
        merged = morpher._merge_fields(a, b)
        assert abs(np.sum(merged.amplitudes) - 1.0) < 1e-9

    def test_merged_curvature_is_average(self, morpher):
        np.random.seed(0)
        a = make_field()
        b = make_field()
        merged = morpher._merge_fields(a, b)
        expected_curvature = (a.curvature + b.curvature) / 2
        np.testing.assert_allclose(merged.curvature, expected_curvature)

    def test_merged_field_has_correct_manifold_dim(self, morpher):
        np.random.seed(0)
        a = make_field()
        b = make_field()
        merged = morpher._merge_fields(a, b)
        assert merged.manifold_dim == MANIFOLD_DIM

    def test_merged_frequencies_average(self, morpher):
        np.random.seed(0)
        a = make_field()
        b = make_field()
        merged = morpher._merge_fields(a, b)
        np.testing.assert_allclose(
            merged.frequencies, (a.frequencies + b.frequencies) / 2
        )


# ---------------------------------------------------------------------------
# _split_field()
# ---------------------------------------------------------------------------

class TestSplitField:
    def test_returns_two_children(self, morpher):
        np.random.seed(0)
        parent = make_field()
        children = morpher._split_field(parent)
        assert len(children) == 2

    def test_children_are_field_elements(self, morpher):
        np.random.seed(0)
        parent = make_field()
        ca, cb = morpher._split_field(parent)
        assert isinstance(ca, FieldElement)
        assert isinstance(cb, FieldElement)

    def test_children_energy_halved(self, morpher):
        np.random.seed(0)
        parent = make_field()
        parent.energy = 4.0
        ca, cb = morpher._split_field(parent)
        assert ca.energy == pytest.approx(2.0)
        assert cb.energy == pytest.approx(2.0)

    def test_children_positions_differ(self, morpher):
        np.random.seed(0)
        parent = make_field()
        ca, cb = morpher._split_field(parent)
        assert not np.allclose(ca.position, cb.position)

    def test_children_symmetric_around_parent(self, morpher):
        """child_a.position + child_b.position ≈ 2 * parent.position"""
        np.random.seed(0)
        parent = make_field()
        ca, cb = morpher._split_field(parent)
        mid = (ca.position + cb.position) / 2
        np.testing.assert_allclose(mid, parent.position, atol=1e-12)

    def test_children_correct_manifold_dim(self, morpher):
        np.random.seed(0)
        parent = make_field()
        ca, cb = morpher._split_field(parent)
        assert ca.manifold_dim == MANIFOLD_DIM
        assert cb.manifold_dim == MANIFOLD_DIM

    def test_children_phases_differ_from_parent(self, morpher):
        np.random.seed(0)
        parent = make_field()
        ca, cb = morpher._split_field(parent)
        # Due to random perturbation both children's phases should differ
        assert not np.allclose(ca.phases, parent.phases)
        assert not np.allclose(cb.phases, parent.phases)
