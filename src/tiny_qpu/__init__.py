"""
tiny-qpu: A minimal, fast quantum computing library.

Features:
- Fast imports (<500ms vs Qiskit's 5+ seconds)
- Fluent API: Circuit(2).h(0).cx(0,1).measure_all()
- Educational mode: see state after each gate
- Practical apps: QRNG, QAOA, BB84
- ASCII visualization: circuits, states, Bloch sphere

Quick Start:
    >>> from tiny_qpu import Circuit
    >>> qc = Circuit(2).h(0).cx(0, 1).measure_all()
    >>> result = qc.run(shots=1000)
    >>> print(result.counts)  # {'00': ~500, '11': ~500}

Visualization:
    >>> from tiny_qpu import Circuit, draw_circuit, visualize
    >>> qc = Circuit(2).h(0).cx(0, 1)
    >>> print(draw_circuit(qc))  # ASCII circuit diagram
    >>> visualize(qc)  # Step-by-step execution
"""
__version__ = "1.0.0"
__author__ = "SK Biswas"

# Core components
from .core import Circuit, StateVector, SimulatorResult, gates

# Visualization
from .visualization import (
    draw_circuit,
    show_state,
    show_counts,
    show_bloch,
    visualize,
    CircuitVisualizer,
    StateVisualizer,
    BlochSphere,
    ExecutionVisualizer,
)

# Make apps accessible
from . import apps

__all__ = [
    # Core
    'Circuit',
    'StateVector', 
    'SimulatorResult',
    'gates',
    # Visualization
    'draw_circuit',
    'show_state',
    'show_counts',
    'show_bloch',
    'visualize',
    'CircuitVisualizer',
    'StateVisualizer',
    'BlochSphere',
    'ExecutionVisualizer',
    # Submodules
    'apps',
]
