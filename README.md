# Genesis Field Network (GFN)

> **This is an early-stage experimental prototype.** It is not intended for production use and makes no claims of superiority over established methods. The project exists solely to explore whether an alternative computational substrate — one not based on conventional neural network primitives — can exhibit learning behavior at all.
>
> ## Overview
>
> GFN is a small research experiment that investigates field-based resonance dynamics as a potential substrate for computation. It replaces several standard building blocks of deep learning with physics-inspired alternatives:
>
> - **Neurons** are replaced by continuous oscillatory field elements defined over a manifold.
> - - **Weight matrices** are replaced by dynamically computed resonance coupling between fields.
>   - - **Backpropagation** is replaced by local phase synchronization inspired by the Kuramoto model.
>     - - **Fixed architectures** are replaced by a topology that may merge, split, spawn, or dissolve field elements during training.
>      
>       - Whether these substitutions lead to any practical advantage remains an open and likely skeptical question. This repository simply provides a runnable reference implementation for further investigation.
>      
>       - ## Structure
>      
>       - ```
> genesis_field_network/
>   __init__.py
>   core.py            # FieldElement, ResonanceCoupler, PhaseAdapter,
>                      # TopologicalMorpher, GenesisFieldNetwork
> examples/
>   demo.py            # XOR, sine regression, 3-class classification
> requirements.txt     # numpy
> ```
>
> ## Usage
>
> ```bash
> pip install numpy
> python examples/demo.py
> ```
>
> ## Limitations
>
> - Convergence is slow and not guaranteed.
> - - Computational cost scales poorly (O(n²) field coupling per step).
>   - - The approach has not been validated on any meaningful benchmark.
>     - - Many design choices are ad hoc and lack theoretical justification.
>      
>       - ## License
>      
>       - MIT
