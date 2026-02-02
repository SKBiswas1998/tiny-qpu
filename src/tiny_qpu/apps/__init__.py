"""
Practical quantum applications.

- QRNG: Quantum Random Number Generator
- QAOA: Quantum Approximate Optimization (MaxCut)
- BB84: Quantum Key Distribution
- VQE: Variational Quantum Eigensolver (Molecular simulation)
"""
from .qrng import QRNG, random_bits, random_bytes, random_int, random_float
from .qaoa import QAOA, QAOAResult, solve_maxcut
from .bb84 import BB84, BB84Result
from .vqe import (
    VQE, VQEResult, 
    Hamiltonian, PauliString, 
    MolecularHamiltonian,
    calculate_h2_ground_state
)

__all__ = [
    # QRNG
    'QRNG', 'random_bits', 'random_bytes', 'random_int', 'random_float',
    # QAOA
    'QAOA', 'QAOAResult', 'solve_maxcut',
    # BB84
    'BB84', 'BB84Result',
    # VQE
    'VQE', 'VQEResult', 'Hamiltonian', 'PauliString',
    'MolecularHamiltonian', 'calculate_h2_ground_state',
]
