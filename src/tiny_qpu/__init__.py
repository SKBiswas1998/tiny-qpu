"""
tiny-qpu: A minimal, fast quantum computing library.

Features:
- Fast imports (<100ms vs Qiskit's 5+ seconds)
- Fluent API: Circuit(2).h(0).cx(0,1).measure_all()
- Educational mode: see state after each gate
- Practical apps: QRNG, QAOA, VQE, BB84

Quick Start:
    >>> from tiny_qpu import Circuit
    >>> qc = Circuit(2).h(0).cx(0, 1).measure_all()
    >>> result = qc.run(shots=1000)
    >>> print(result.counts)  # {'00': ~500, '11': ~500}

Applications:
    >>> from tiny_qpu.apps import QRNG
    >>> qrng = QRNG()
    >>> print(qrng.random_bytes(32).hex())  # 256-bit random key
"""
__version__ = "1.0.0"
__author__ = "SK Biswas"

# Core components
from .core import Circuit, StateVector, SimulatorResult, gates

# Make apps accessible
from . import apps

__all__ = [
    # Core
    'Circuit',
    'StateVector', 
    'SimulatorResult',
    'gates',
    # Submodules
    'apps',
]
