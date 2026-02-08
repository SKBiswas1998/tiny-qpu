"""
Quantum chemistry interface for molecular simulation.

Integrates PySCF electronic structure with tiny-qpu's quantum simulator
to enable VQE on real molecular systems.

Pipeline: Molecular geometry → PySCF HF → Jordan-Wigner → Hamiltonian → VQE

Requires PySCF: pip install pyscf

Quick start:
    >>> from tiny_qpu.chemistry import Molecule, hydrogen
    >>> mol = hydrogen(0.74)
    >>> H = mol.hamiltonian()
    >>> print(f"HF energy: {mol.hf_energy:.4f} Ha")
    >>> print(f"Qubits: {H.n_qubits}, Terms: {H.n_terms}")
"""

from .molecular import (
    Molecule,
    hydrogen,
    lithium_hydride,
    water,
    bond_dissociation_curve,
)
from .transforms import (
    jordan_wigner,
)

__all__ = [
    "Molecule",
    "hydrogen",
    "lithium_hydride",
    "water",
    "bond_dissociation_curve",
    "jordan_wigner",
]
