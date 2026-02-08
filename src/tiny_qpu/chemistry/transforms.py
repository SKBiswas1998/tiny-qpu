"""
Fermion-to-qubit transformations for quantum chemistry.

Implements the Jordan-Wigner transformation which maps fermionic
creation/annihilation operators to Pauli strings:

    a†_p = (1/2)(X_p - iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
    a_p  = (1/2)(X_p + iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0

This enables mapping molecular electronic Hamiltonians (expressed in
second quantization) to qubit operators compatible with VQE.
"""

import numpy as np
from typing import Dict, List, Tuple


# ─── Pauli algebra ───────────────────────────────────────────────────────

_PAULI_MULT = {
    ("I", "I"): ("I", 1), ("I", "X"): ("X", 1), ("I", "Y"): ("Y", 1), ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1), ("X", "X"): ("I", 1), ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1), ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1), ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1), ("Z", "X"): ("Y", 1j), ("Z", "Y"): ("X", -1j), ("Z", "Z"): ("I", 1),
}

# A "QubitOp" is a list of (pauli_string, complex_coefficient) pairs
QubitOp = List[Tuple[str, complex]]


def _multiply_pauli_strings(a: str, b: str) -> Tuple[str, complex]:
    """Multiply two Pauli strings, returning (result_string, phase)."""
    result = []
    phase = 1.0 + 0j
    for ca, cb in zip(a, b):
        r, p = _PAULI_MULT[(ca, cb)]
        result.append(r)
        phase *= p
    return "".join(result), phase


def _qubit_op_collect(ops: QubitOp) -> Dict[str, complex]:
    """Collect QubitOp into a dictionary, summing coefficients."""
    result: Dict[str, complex] = {}
    for pauli, coeff in ops:
        result[pauli] = result.get(pauli, 0.0) + coeff
    return result


def _qubit_op_multiply(a: QubitOp, b: QubitOp) -> QubitOp:
    """Multiply two QubitOps (distribute and multiply each pair)."""
    result = []
    for pa, ca in a:
        for pb, cb in b:
            prod, phase = _multiply_pauli_strings(pa, pb)
            result.append((prod, ca * cb * phase))
    return result


# ─── Jordan-Wigner mapping of individual operators ──────────────────────

def _creation_op_jw(p: int, n_qubits: int) -> QubitOp:
    """
    Jordan-Wigner representation of a†_p.

    a†_p = (1/2)(X_p - iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
    """
    x_part = list("I" * n_qubits)
    x_part[p] = "X"
    for k in range(p):
        x_part[k] = "Z"

    y_part = list("I" * n_qubits)
    y_part[p] = "Y"
    for k in range(p):
        y_part[k] = "Z"

    return [("".join(x_part), 0.5), ("".join(y_part), -0.5j)]


def _annihilation_op_jw(p: int, n_qubits: int) -> QubitOp:
    """
    Jordan-Wigner representation of a_p.

    a_p = (1/2)(X_p + iY_p) ⊗ Z_{p-1} ⊗ ... ⊗ Z_0
    """
    x_part = list("I" * n_qubits)
    x_part[p] = "X"
    for k in range(p):
        x_part[k] = "Z"

    y_part = list("I" * n_qubits)
    y_part[p] = "Y"
    for k in range(p):
        y_part[k] = "Z"

    return [("".join(x_part), 0.5), ("".join(y_part), 0.5j)]


# ─── Main transform ─────────────────────────────────────────────────────

def jordan_wigner(one_body: np.ndarray, two_body: np.ndarray,
                  nuclear_repulsion: float = 0.0,
                  threshold: float = 1e-10) -> Dict[str, float]:
    """
    Transform a molecular Hamiltonian from second quantization to qubit form.

    The electronic Hamiltonian:
        H = Σ_{pq} h_{pq} a†_p a_q
          + (1/2) Σ_{pqrs} g_{pqrs} a†_p a†_r a_s a_q
          + E_nuc

    where g_{pqrs} are in chemist notation (pq|rs), is mapped to
    a sum of Pauli strings using Jordan-Wigner.

    Parameters
    ----------
    one_body : np.ndarray
        One-electron integrals h_{pq}, shape (n_spin_orbitals, n_spin_orbitals).
    two_body : np.ndarray
        Two-electron integrals in chemist notation (pq|rs),
        shape (n, n, n, n).
    nuclear_repulsion : float
        Nuclear repulsion energy (constant offset).
    threshold : float
        Drop Pauli terms with |coefficient| < threshold.

    Returns
    -------
    dict
        Mapping of Pauli strings to real coefficients.
    """
    n_qubits = one_body.shape[0]
    all_terms: QubitOp = []

    # Nuclear repulsion → identity
    identity = "I" * n_qubits
    all_terms.append((identity, nuclear_repulsion))

    # Cache JW representations of creation/annihilation operators
    create_cache = {p: _creation_op_jw(p, n_qubits) for p in range(n_qubits)}
    annihil_cache = {p: _annihilation_op_jw(p, n_qubits) for p in range(n_qubits)}

    # One-body: Σ h_{pq} a†_p a_q
    for p in range(n_qubits):
        for q in range(n_qubits):
            coeff = one_body[p, q]
            if abs(coeff) < threshold:
                continue
            op = _qubit_op_multiply(create_cache[p], annihil_cache[q])
            for pauli, c in op:
                all_terms.append((pauli, coeff * c))

    # Two-body: (1/2) Σ g_{pqrs} a†_p a†_r a_s a_q
    # Chemist notation (pq|rs) with normal-ordered operators
    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coeff = 0.5 * two_body[p, q, r, s]
                    if abs(coeff) < threshold:
                        continue
                    # a†_p a†_r a_s a_q = (a†_p)(a†_r)(a_s)(a_q)
                    op = create_cache[p]
                    op = _qubit_op_multiply(op, create_cache[r])
                    op = _qubit_op_multiply(op, annihil_cache[s])
                    op = _qubit_op_multiply(op, annihil_cache[q])
                    for pauli, c in op:
                        all_terms.append((pauli, coeff * c))

    # Collect and convert to real
    collected = _qubit_op_collect(all_terms)
    real_terms = {}
    for pauli_str, coeff in collected.items():
        real_coeff = float(np.real(coeff))
        if abs(real_coeff) >= threshold:
            real_terms[pauli_str] = real_coeff

    return real_terms
