"""
Practical quantum applications.

- QRNG: Quantum Random Number Generator
- QAOA: Quantum Approximate Optimization (MaxCut)
- BB84: Quantum Key Distribution
- VQE: Variational Quantum Eigensolver (coming soon)
"""
from .qrng import QRNG, random_bits, random_bytes, random_int, random_float
from .qaoa import QAOA, QAOAResult, solve_maxcut
from .bb84 import BB84, BB84Result

__all__ = [
    # QRNG
    'QRNG',
    'random_bits', 
    'random_bytes',
    'random_int',
    'random_float',
    # QAOA
    'QAOA',
    'QAOAResult',
    'solve_maxcut',
    # BB84
    'BB84',
    'BB84Result',
]
