"""
Quantum gradient computation for variational algorithms.

Provides three methods for computing gradients of quantum expectation values:
- Parameter-shift rule (exact, hardware-compatible)
- Adjoint differentiation (fast, simulator-optimal)
- Finite-difference (universal fallback)

Quick start:
    >>> from tiny_qpu.gradients import Hamiltonian, expectation_and_gradient
    >>> H = Hamiltonian({"ZZ": -1.0, "XI": 0.5})
    >>> val, grad = expectation_and_gradient(ansatz, H, params)
"""

from .hamiltonian import (
    Hamiltonian,
    transverse_field_ising,
    heisenberg_xyz,
    molecular_hydrogen,
)
from .differentiation import (
    expectation,
    gradient,
    expectation_and_gradient,
)

__all__ = [
    "Hamiltonian",
    "transverse_field_ising",
    "heisenberg_xyz",
    "molecular_hydrogen",
    "expectation",
    "gradient",
    "expectation_and_gradient",
]
