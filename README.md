# Genesis Field Network (GFN)

**A radically new computational paradigm beyond deep learning.**

Genesis Field Network abandons neurons, weights, backpropagation, and fixed layer topologies entirely. Instead, it introduces a field-based resonance computation system where information is encoded as standing wave patterns in a continuous manifold, and learning emerges from harmonic interference and phase synchronization.

---

## What Makes This Different

| Concept | Traditional Neural Networks | Genesis Field Network |
|---------|---------------------------|----------------------|
| **Basic Unit** | Neuron (weighted sum + activation) | FieldElement (oscillatory field in manifold) |
| **Connections** | Static weight matrices | Dynamic resonance coupling (no stored weights) |
| **Learning** | Backpropagation (global gradient chain) | Phase adaptation (local Kuramoto synchronization) |
| **Architecture** | Fixed layers/topology | Topological morphing (merge, split, spawn, dissolve) |
| **Core Operation** | Matrix multiplication | Harmonic interference + resonance propagation |
| **Information** | Stored in weight values | Encoded in field interference patterns |

---

## Core Concepts

### 1. Field Elements
Not neurons. Each FieldElement is a continuous oscillatory entity with:
- A **position** in the computational manifold
- - A **frequency spectrum** (multiple harmonics)
  - - **Phase configurations** that determine alignment
    - - A **curvature tensor** controlling spatial influence
      - - An **amplitude envelope** for energy distribution
       
        - ### 2. Resonance Coupling
        - No weight matrices. Fields interact through harmonic interference:
        - - Compatible frequencies create **constructive interference** (amplification)
          - - Incompatible frequencies create **destructive interference** (suppression)
            - - Coupling strength is **dynamically computed**, not stored
             
              - ### 3. Phase Adaptation (No Backpropagation)
              - Learning without gradients:
              - - Each field independently senses local **dissonance**
                - - Phases adjust via **Kuramoto-inspired synchronization**
                  - - Frequencies shift based on **local resonance gradients**
                    - - No global error signal propagates backward
                     
                      - ### 4. Topological Morphing
                      - The network structure itself evolves:
                      - - **MERGE**: Highly resonant fields collapse into one
                        - - **SPLIT**: Complex high-energy fields divide into two
                          - - **SPAWN**: New fields appear in high-dissonance regions
                            - - **DISSOLVE**: Dormant low-energy fields disappear
                             
                              - ---

                              ## Project Structure

                              ```
                              Genesis-Field-Network/
                              ├── genesis_field_network/
                              │   ├── __init__.py          # Package exports
                              │   └── core.py              # Core architecture
                              │       ├── FieldElement          - Oscillatory field unit
                              │       ├── ResonanceCoupler      - Field interaction manager
                              │       ├── PhaseAdapter          - Non-gradient learning
                              │       ├── TopologicalMorpher    - Dynamic structure evolution
                              │       └── GenesisFieldNetwork   - Complete system
                              ├── examples/
                              │   └── demo.py              # Runnable demonstrations
                              └── README.md
                              ```

                              ---

                              ## Quick Start

                              ```python
                              import numpy as np
                              from genesis_field_network import GenesisFieldNetwork

                              # Create a GFN (not a neural network)
                              gfn = GenesisFieldNetwork(
                                  input_dim=2,
                                  output_dim=1,
                                  manifold_dim=16,
                                  num_fields=32,
                                  num_harmonics=8,
                              )

                              # XOR problem
                              X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
                              Y = np.array([[0], [1], [1], [0]], dtype=float)

                              # Train via phase adaptation (no backpropagation)
                              history = gfn.train(X, Y, epochs=100)

                              # Predict
                              predictions = gfn.predict(X)
                              print(predictions)

                              # Inspect the living network
                              state = gfn.get_state_summary()
                              print(f"Fields: {state['num_fields']}, Morphs: {state['morph_count']}")
                              ```

                              ---

                              ## Run the Demo

                              ```bash
                              cd Genesis-Field-Network
                              python examples/demo.py
                              ```

                              The demo includes:
                              1. **XOR Problem** - Classic non-linear benchmark
                              2. 2. **Sine Wave Regression** - Continuous function approximation
                                 3. 3. **Multi-class Classification** - Pattern recognition with 3 classes
                                   
                                    4. ---
                                   
                                    5. ## Requirements
                                   
                                    6. - Python 3.8+
                                       - - NumPy
                                        
                                         - No deep learning frameworks needed. No GPU required. Just pure computation.
                                        
                                         - ```bash
                                           pip install numpy
                                           ```

                                           ---

                                           ## How It Works (Technical)

                                           ### Forward Pass (Field Excitation)
                                           1. Input is projected as excitation energy onto field elements
                                           2. 2. Fields exchange energy through resonance channels over multiple steps
                                              3. 3. System reaches equilibrium via interference dynamics
                                                 4. 4. Output is read from the final energy state of all fields
                                                   
                                                    5. ### Learning (Phase Synchronization)
                                                    6. 1. **Dissonance** is measured between output and target (both magnitude and spectral)
                                                       2. 2. Each field adjusts its **phases** toward resonant neighbors (Kuramoto model)
                                                          3. 3. **Frequencies** shift based on local dissonance gradients
                                                             4. 4. **Curvatures** adapt based on energy flow directions
                                                                5. 5. **Amplitudes** redistribute toward resonance-contributing harmonics
                                                                  
                                                                   6. ### Topology Evolution
                                                                   7. After each learning step, the network structure may change:
                                                                   8. - Fields with >0.95 resonance correlation → merged
                                                                      - - Fields with high complexity × energy → split
                                                                        - - Persistent high dissonance → new fields spawned
                                                                          - - Near-zero energy fields → dissolved
                                                                           
                                                                            - ---

                                                                            ## Disclaimer

                                                                            This is an experimental research prototype exploring fundamentally new computational paradigms. It is not intended to compete with production deep learning systems in performance. The goal is to investigate whether computation can emerge from field resonance dynamics rather than from traditional neural network primitives.

                                                                            ---

                                                                            ## License

                                                                            MIT
