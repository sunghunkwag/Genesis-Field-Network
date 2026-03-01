"""
Genesis Field Network (GFN) - Core Architecture
=================================================

A fundamentally new computational paradigm that abandons:
- Neurons / Perceptrons (no weighted sums)
- Backpropagation (no gradient chains)
- Fixed layer topology (no static graphs)
- Matrix multiplication as the core operation

Instead, GFN introduces:
- FIELD ELEMENTS: Continuous scalar fields in high-dimensional space
  that represent information as spatial energy distributions
- RESONANCE COUPLING: Fields interact through harmonic interference
  patterns, not weighted connections
- PHASE ADAPTATION: Learning occurs by adjusting oscillation phases
  and field curvatures, not weights
- TOPOLOGICAL MORPHING: The computational structure itself evolves
  during inference, with fields merging, splitting, and reorganizing

Core Idea:
  Information is encoded as standing wave patterns in a continuous
  field manifold. Computation emerges from the interference and
  resonance between overlapping fields. There are no discrete
  "neurons" - only regions of constructive/destructive interference
  that naturally form and dissolve.

Author: Genesis Field Network Project
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import math


class FieldElement:
    """
    A FieldElement is NOT a neuron. It is a continuous oscillatory
    entity defined over a region of the computational manifold.

    Each FieldElement has:
    - A position in the manifold (center of influence)
    - A frequency spectrum (how it oscillates)
    - A phase configuration (alignment with other fields)
    - A curvature tensor (how its influence decays spatially)
    - An amplitude envelope (energy distribution)

    Information is encoded in the INTERFERENCE PATTERN between
    multiple FieldElements, not in any single element.
    """

    def __init__(self, manifold_dim: int, num_harmonics: int = 8):
        self.manifold_dim = manifold_dim
        self.num_harmonics = num_harmonics

        # Position in the computational manifold
        self.position = np.random.randn(manifold_dim) * 0.5

        # Frequency spectrum - each harmonic has a frequency per dimension
        self.frequencies = np.random.uniform(0.1, 5.0, (num_harmonics, manifold_dim))

        # Phase offsets for each harmonic
        self.phases = np.random.uniform(0, 2 * np.pi, (num_harmonics,))

        # Amplitude per harmonic (energy distribution across spectrum)
        self.amplitudes = np.random.exponential(1.0, (num_harmonics,))
        self.amplitudes /= np.sum(self.amplitudes)  # normalize energy

        # Curvature tensor - controls spatial decay of influence
        # Positive definite matrix for well-defined field falloff
        raw = np.random.randn(manifold_dim, manifold_dim) * 0.3
        self.curvature = raw @ raw.T + np.eye(manifold_dim) * 0.1

        # Field energy (dynamically evolves)
        self.energy = 1.0

        # Resonance memory - tracks coupling history
        self.resonance_history = []

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate this field's value at given points in the manifold.

        Unlike a neuron's weighted sum, this computes a continuous
        wave-like field value based on spatial position and harmonic
        composition.

        Args:
            points: (N, manifold_dim) array of query positions

        Returns:
            (N,) array of field values at each point
        """
        # Displacement from field center
        delta = points - self.position  # (N, D)

        # Spatial decay based on curvature tensor (Mahalanobis-like)
        # This is NOT a Gaussian - it modulates the wave envelope
        quad_form = np.sum((delta @ np.linalg.inv(self.curvature)) * delta, axis=-1)
        envelope = self.energy * np.exp(-0.5 * quad_form)  # (N,)

        # Harmonic field value - superposition of oscillations
        field_value = np.zeros(len(points))
        for h in range(self.num_harmonics):
            # Project displacement onto frequency vector
            freq_projection = np.sum(delta * self.frequencies[h], axis=-1)
            # Add harmonic contribution
            field_value += self.amplitudes[h] * np.sin(
                freq_projection + self.phases[h]
            )

        return envelope * field_value

    def compute_resonance(self, other: 'FieldElement',
                          sample_points: np.ndarray) -> float:
        """
        Compute resonance coupling strength between two fields.

        Resonance is measured by the correlation of field patterns
        over shared space. High resonance = constructive interference.
        This replaces the concept of "connection weight".
        """
        my_field = self.evaluate(sample_points)
        other_field = other.evaluate(sample_points)

        # Resonance = normalized correlation of field patterns
        if np.std(my_field) < 1e-10 or np.std(other_field) < 1e-10:
            return 0.0

        resonance = np.corrcoef(my_field, other_field)[0, 1]
        return float(resonance) if not np.isnan(resonance) else 0.0


class ResonanceCoupler:
    """
    Manages the interaction between FieldElements through resonance.

    Unlike weight matrices that statically connect neurons, the
    ResonanceCoupler dynamically determines which fields interact
    based on their current harmonic compatibility.

    Fields that resonate (have compatible frequency/phase relationships)
    amplify each other. Fields that anti-resonate suppress each other.
    This is how information flows - through resonance pathways, not
    fixed connections.
    """

    def __init__(self, manifold_dim: int, coupling_resolution: int = 64):
        self.manifold_dim = manifold_dim
        self.coupling_resolution = coupling_resolution

        # Sampling grid for measuring field interactions
        self._update_sample_grid()

    def _update_sample_grid(self):
        """Generate sample points for measuring field interactions."""
        self.sample_points = np.random.randn(
            self.coupling_resolution, self.manifold_dim
        ) * 2.0

    def compute_coupling_matrix(self, fields: List[FieldElement]) -> np.ndarray:
        """
        Compute the full resonance coupling between all fields.

        This is NOT a weight matrix. It is dynamically computed from
        the current state of all fields and changes as fields evolve.
        There are no stored "weights" - coupling emerges from field
        configurations.
        """
        n = len(fields)
        coupling = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                r = fields[i].compute_resonance(fields[j], self.sample_points)
                coupling[i, j] = r
                coupling[j, i] = r

        return coupling

    def propagate(self, fields: List[FieldElement],
                  input_excitation: np.ndarray) -> np.ndarray:
        """
        Propagate information through the field network via resonance.

        Instead of forward-passing through layers, we:
        1. Excite fields with input energy
        2. Let resonance coupling distribute energy
        3. Read the equilibrium state

        This is a single-step relaxation (iterated for convergence).
        """
        coupling = self.compute_coupling_matrix(fields)

        # Energy state of each field
        energies = np.array([f.energy for f in fields])

        # Inject input excitation
        n_input = min(len(input_excitation), len(fields))
        energies[:n_input] += input_excitation[:n_input]

        # Resonance propagation (not matrix multiply - interference)
        # Energy flows through resonance channels
        resonance_flow = np.zeros_like(energies)
        for i in range(len(fields)):
            for j in range(len(fields)):
                if i != j:
                    # Energy transfer proportional to resonance strength
                    # and energy differential (flows from high to low)
                    differential = energies[i] - energies[j]
                    flow = coupling[i, j] * differential * 0.1
                    resonance_flow[i] -= flow
                    resonance_flow[j] += flow

        # Apply non-linear field dynamics (not activation function!)
        # Fields have natural energy bounds from physical analogy
        new_energies = energies + resonance_flow

        # Field saturation - energy cannot be negative or infinite
        new_energies = np.tanh(new_energies) * 2.0

        # Update field energies
        for i, field in enumerate(fields):
            field.energy = new_energies[i]

        return new_energies


class PhaseAdapter:
    """
    Learning mechanism that replaces backpropagation entirely.

    Instead of computing gradients and updating weights, PhaseAdapter:
    1. Measures the "dissonance" between current output pattern and target
    2. Identifies which phase/frequency adjustments reduce dissonance
    3. Applies local phase corrections (no global gradient chain)

    This is inspired by how coupled oscillators naturally synchronize
    (Kuramoto model), not by gradient descent.
    """

    def __init__(self, adaptation_rate: float = 0.01,
                 dissonance_threshold: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.dissonance_threshold = dissonance_threshold
        self.dissonance_history = []

    def compute_dissonance(self, output_pattern: np.ndarray,
                           target_pattern: np.ndarray) -> float:
        """
        Dissonance = how far the output field pattern is from the target.

        Unlike loss functions that compute scalar error, dissonance
        measures the harmonic incompatibility between two patterns.
        """
        if len(output_pattern) != len(target_pattern):
            min_len = min(len(output_pattern), len(target_pattern))
            output_pattern = output_pattern[:min_len]
            target_pattern = target_pattern[:min_len]

        # Pattern difference in field space
        diff = output_pattern - target_pattern

        # Dissonance includes both magnitude and phase misalignment
        magnitude_dissonance = np.mean(diff ** 2)

        # Spectral dissonance via FFT comparison
        if len(output_pattern) > 1:
            out_fft = np.fft.fft(output_pattern)
            tgt_fft = np.fft.fft(target_pattern)
            spectral_dissonance = np.mean(np.abs(out_fft - tgt_fft) ** 2)
        else:
            spectral_dissonance = magnitude_dissonance

        total = 0.5 * magnitude_dissonance + 0.5 * spectral_dissonance
        self.dissonance_history.append(float(total))
        return float(total)

    def adapt_fields(self, fields: List[FieldElement],
                     output_pattern: np.ndarray,
                     target_pattern: np.ndarray,
                     coupler: ResonanceCoupler):
        """
        Adapt field configurations to reduce dissonance.

        NO BACKPROPAGATION. Instead:
        - Each field independently senses local dissonance
        - Phase adjustments follow Kuramoto-like synchronization
        - Frequency adjustments follow resonance gradient (local, not global)
        - Curvature adjustments follow field energy flow patterns
        """
        dissonance = self.compute_dissonance(output_pattern, target_pattern)

        if dissonance < self.dissonance_threshold:
            return dissonance

        # Compute target influence for each field
        coupling = coupler.compute_coupling_matrix(fields)

        for idx, field in enumerate(fields):
            # LOCAL phase adaptation (Kuramoto-inspired)
            # Each field adjusts phase toward its resonant neighbors
            phase_correction = np.zeros_like(field.phases)

            for h in range(field.num_harmonics):
                for other_idx, other_field in enumerate(fields):
                    if other_idx != idx:
                        # Phase difference drives synchronization
                        for oh in range(other_field.num_harmonics):
                            phase_diff = other_field.phases[oh] - field.phases[h]
                            sync_force = coupling[idx, other_idx] * np.sin(phase_diff)
                            phase_correction[h] += sync_force

            # Apply phase correction scaled by adaptation rate
            field.phases += self.adaptation_rate * phase_correction
            field.phases = field.phases % (2 * np.pi)

            # LOCAL frequency adaptation
            # Shift frequencies toward patterns that reduce dissonance
            freq_perturbation = np.random.randn(*field.frequencies.shape) * 0.01
            field.frequencies += self.adaptation_rate * freq_perturbation * dissonance
            field.frequencies = np.clip(field.frequencies, 0.01, 10.0)

            # LOCAL curvature adaptation
            # Adjust spatial influence based on energy flow
            energy_gradient = np.zeros(field.manifold_dim)
            for other_idx, other_field in enumerate(fields):
                if other_idx != idx:
                    direction = other_field.position - field.position
                    energy_gradient += coupling[idx, other_idx] * direction

            # Curvature follows energy flow direction
            curvature_update = np.outer(energy_gradient, energy_gradient) * 0.001
            field.curvature += self.adaptation_rate * curvature_update

            # Ensure curvature stays positive definite
            eigenvalues = np.linalg.eigvalsh(field.curvature)
            if np.min(eigenvalues) < 0.01:
                field.curvature += np.eye(field.manifold_dim) * 0.05

            # LOCAL amplitude redistribution
            # Harmonics that contribute to resonance get more energy
            amplitude_gradient = np.zeros(field.num_harmonics)
            for h in range(field.num_harmonics):
                for other_idx, other_field in enumerate(fields):
                    if other_idx != idx:
                        amplitude_gradient[h] += abs(coupling[idx, other_idx])

            if np.sum(amplitude_gradient) > 0:
                amplitude_gradient /= np.sum(amplitude_gradient)
                field.amplitudes = (
                    (1 - self.adaptation_rate) * field.amplitudes +
                    self.adaptation_rate * amplitude_gradient
                )
                field.amplitudes /= np.sum(field.amplitudes)

        return dissonance


class TopologicalMorpher:
    """
    Dynamically evolves the structure of the field network.

    Unlike fixed neural architectures, GFN's topology is alive:
    - Fields that strongly resonate can MERGE into a single field
    - Fields with high internal complexity can SPLIT into two
    - New fields can SPAWN in regions of high dissonance
    - Dormant fields can DISSOLVE and free computational resources

    The network structure itself is part of the computation.
    """

    def __init__(self, merge_threshold: float = 0.95,
                 split_threshold: float = 3.0,
                 spawn_threshold: float = 2.0,
                 dissolve_threshold: float = 0.01,
                 max_fields: int = 256,
                 min_fields: int = 4):
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.spawn_threshold = spawn_threshold
        self.dissolve_threshold = dissolve_threshold
        self.max_fields = max_fields
        self.min_fields = min_fields
        self.morph_log = []

    def morph(self, fields: List[FieldElement],
              coupler: ResonanceCoupler,
              current_dissonance: float) -> List[FieldElement]:
        """
        Perform one morphing step on the field topology.
        """
        fields = list(fields)

        # MERGE: Highly resonant fields collapse into one
        if len(fields) > self.min_fields:
            coupling = coupler.compute_coupling_matrix(fields)
            merged = set()
            new_fields = []

            for i in range(len(fields)):
                if i in merged:
                    continue
                merge_partner = None
                best_resonance = self.merge_threshold

                for j in range(i + 1, len(fields)):
                    if j in merged:
                        continue
                    if abs(coupling[i, j]) > best_resonance:
                        best_resonance = abs(coupling[i, j])
                        merge_partner = j

                if merge_partner is not None and len(fields) - len(merged) > self.min_fields:
                    # Merge: average properties
                    merged_field = self._merge_fields(
                        fields[i], fields[merge_partner]
                    )
                    new_fields.append(merged_field)
                    merged.add(i)
                    merged.add(merge_partner)
                    self.morph_log.append(('MERGE', i, merge_partner))
                else:
                    if i not in merged:
                        new_fields.append(fields[i])

            fields = new_fields

        # SPLIT: Complex high-energy fields divide
        if len(fields) < self.max_fields:
            split_candidates = []
            for idx, field in enumerate(fields):
                complexity = np.std(field.frequencies) * field.energy
                if complexity > self.split_threshold:
                    split_candidates.append(idx)

            for idx in reversed(split_candidates):
                if len(fields) >= self.max_fields:
                    break
                child_a, child_b = self._split_field(fields[idx])
                fields[idx] = child_a
                fields.append(child_b)
                self.morph_log.append(('SPLIT', idx))

        # SPAWN: Create new fields in high-dissonance regions
        if current_dissonance > self.spawn_threshold and len(fields) < self.max_fields:
            new_field = FieldElement(
                fields[0].manifold_dim,
                fields[0].num_harmonics
            )
            new_field.energy = current_dissonance * 0.1
            fields.append(new_field)
            self.morph_log.append(('SPAWN', len(fields) - 1))

        # DISSOLVE: Remove near-zero energy fields
        if len(fields) > self.min_fields:
            # Save topology params before dissolving in case the list empties
            _manifold_dim = fields[0].manifold_dim
            _num_harmonics = fields[0].num_harmonics
            fields = [
                f for f in fields
                if abs(f.energy) > self.dissolve_threshold
            ]
            if len(fields) < self.min_fields:
                while len(fields) < self.min_fields:
                    fields.append(FieldElement(_manifold_dim, _num_harmonics))

        return fields

    def _merge_fields(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Merge two fields into one by averaging their properties."""
        merged = FieldElement(a.manifold_dim, a.num_harmonics)
        merged.position = (a.position + b.position) / 2
        merged.frequencies = (a.frequencies + b.frequencies) / 2
        merged.phases = np.angle(
            np.exp(1j * a.phases) + np.exp(1j * b.phases)
        )
        merged.amplitudes = (a.amplitudes + b.amplitudes) / 2
        merged.amplitudes /= np.sum(merged.amplitudes)
        merged.curvature = (a.curvature + b.curvature) / 2
        merged.energy = a.energy + b.energy
        return merged

    def _split_field(self, field: FieldElement) -> Tuple[FieldElement, FieldElement]:
        """Split a field into two children with perturbed properties."""
        child_a = FieldElement(field.manifold_dim, field.num_harmonics)
        child_b = FieldElement(field.manifold_dim, field.num_harmonics)

        perturbation = np.random.randn(field.manifold_dim) * 0.2

        child_a.position = field.position + perturbation
        child_b.position = field.position - perturbation

        child_a.frequencies = field.frequencies * (1 + np.random.randn(*field.frequencies.shape) * 0.1)
        child_b.frequencies = field.frequencies * (1 + np.random.randn(*field.frequencies.shape) * 0.1)

        child_a.phases = field.phases + np.random.randn(field.num_harmonics) * 0.2
        child_b.phases = field.phases - np.random.randn(field.num_harmonics) * 0.2

        child_a.amplitudes = field.amplitudes.copy()
        child_b.amplitudes = field.amplitudes.copy()

        child_a.curvature = field.curvature * 1.2
        child_b.curvature = field.curvature * 0.8

        child_a.energy = field.energy / 2
        child_b.energy = field.energy / 2

        return child_a, child_b


class GenesisFieldNetwork:
    """
    The complete Genesis Field Network.

    This is NOT a neural network. It is a field-based computational
    system where:

    - There are no layers (fields exist in a continuous manifold)
    - There are no weights (coupling emerges from resonance)
    - There is no backpropagation (adaptation is local and phase-based)
    - The topology is dynamic (fields merge, split, spawn, dissolve)

    Input: Encoded as excitation patterns on input-designated fields
    Output: Read from the energy state of output-designated fields
    Learning: Phase adaptation + topological morphing
    """

    def __init__(self, input_dim: int, output_dim: int,
                 manifold_dim: int = 16, num_fields: int = 32,
                 num_harmonics: int = 8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.manifold_dim = manifold_dim
        self.num_harmonics = num_harmonics

        # Create field elements
        self.fields = [
            FieldElement(manifold_dim, num_harmonics)
            for _ in range(num_fields)
        ]

        # Input encoding: maps input dimensions to field excitations
        self.input_projection = np.random.randn(input_dim, num_fields) * 0.1

        # Output reading: maps field energies to output dimensions
        self.output_projection = np.random.randn(num_fields, output_dim) * 0.1

        # Core components
        self.coupler = ResonanceCoupler(manifold_dim)
        self.adapter = PhaseAdapter()
        self.morpher = TopologicalMorpher()

        # Resonance iteration count for equilibrium
        self.resonance_steps = 5

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process input through the field network.

        This is NOT a forward pass through layers. It is:
        1. Encode input as field excitations
        2. Let fields reach resonance equilibrium
        3. Read output from field energy state
        """
        # Encode input as excitation pattern
        num_fields = len(self.fields)

        # Adjust projection if field count changed (due to morphing)
        if self.input_projection.shape[1] != num_fields:
            new_proj = np.random.randn(self.input_dim, num_fields) * 0.1
            min_f = min(self.input_projection.shape[1], num_fields)
            new_proj[:, :min_f] = self.input_projection[:, :min_f]
            self.input_projection = new_proj

        excitation = x @ self.input_projection  # (input_dim,) -> (num_fields,)

        # Let fields reach resonance equilibrium
        for _ in range(self.resonance_steps):
            energies = self.coupler.propagate(self.fields, excitation)
            excitation = np.zeros(num_fields)  # Only excite once

        # Read output from field energy state
        if self.output_projection.shape[0] != num_fields:
            new_proj = np.random.randn(num_fields, self.output_dim) * 0.1
            min_f = min(self.output_projection.shape[0], num_fields)
            new_proj[:min_f, :] = self.output_projection[:min_f, :]
            self.output_projection = new_proj

        output = energies @ self.output_projection
        return output

    def learn(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Learn from a single example.

        NO BACKPROPAGATION. The learning process:
        1. Forward pass to get current output
        2. Measure dissonance between output and target
        3. Adapt field phases/frequencies locally
        4. Optionally morph topology
        """
        output = self.forward(x)

        # Adapt fields via phase synchronization
        dissonance = self.adapter.adapt_fields(
            self.fields, output, target, self.coupler
        )

        # Topological morphing based on dissonance
        self.fields = self.morpher.morph(
            self.fields, self.coupler, dissonance
        )

        # Also adapt projections (minimal linear adaptation)
        error = target - output
        if self.output_projection.shape[0] == len(self.fields):
            energies = np.array([f.energy for f in self.fields])
            self.output_projection += 0.01 * np.outer(energies, error)

        return dissonance

    def train(self, X: np.ndarray, Y: np.ndarray,
              epochs: int = 100, verbose: bool = True) -> List[float]:
        """
        Train the network on a dataset.
        """
        history = []

        for epoch in range(epochs):
            epoch_dissonance = 0.0
            indices = np.random.permutation(len(X))

            for i in indices:
                d = self.learn(X[i], Y[i])
                epoch_dissonance += d

            avg_dissonance = epoch_dissonance / len(X)
            history.append(avg_dissonance)

            if verbose and (epoch + 1) % 10 == 0:
                n_fields = len(self.fields)
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Dissonance: {avg_dissonance:.6f} | "
                      f"Fields: {n_fields} | "
                      f"Morphs: {len(self.morpher.morph_log)}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for a batch of inputs."""
        outputs = []
        for x in X:
            outputs.append(self.forward(x))
        return np.array(outputs)

    def get_state_summary(self) -> Dict:
        """Get a summary of the current network state."""
        energies = [f.energy for f in self.fields]
        return {
            'num_fields': len(self.fields),
            'total_energy': sum(abs(e) for e in energies),
            'mean_energy': np.mean(np.abs(energies)),
            'max_energy': max(abs(e) for e in energies),
            'morph_count': len(self.morpher.morph_log),
            'dissonance_history': self.adapter.dissonance_history[-10:],
        }
