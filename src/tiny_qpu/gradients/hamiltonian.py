"""
Hamiltonian representation for quantum expectation values.

A Hamiltonian is a weighted sum of Pauli strings:
    H = c₁ P₁ + c₂ P₂ + ... + cₙ Pₙ

where each Pᵢ is a tensor product of Pauli operators {I, X, Y, Z}
and cᵢ are real coefficients.

Example:
    H₂ molecule at 0.74 Å bond length:
    H = -0.8105 II + 0.1716 ZI - 0.2228 IZ + 0.1209 ZZ + 0.0454 XX

Usage:
    >>> H = Hamiltonian({"II": -0.8105, "ZI": 0.1716, "IZ": -0.2228,
    ...                   "ZZ": 0.1209, "XX": 0.0454})
    >>> energy = H.expectation(statevector)
"""

import numpy as np
from typing import Dict, Optional, Union, List, Tuple


# Pauli matrices (2x2)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MAP = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


class Hamiltonian:
    """
    Pauli-string Hamiltonian for computing quantum expectation values.

    Stores a Hamiltonian as a dictionary of Pauli strings → coefficients.
    Efficiently computes ⟨ψ|H|ψ⟩ by applying each Pauli tensor product
    to the statevector and summing weighted inner products.

    Parameters
    ----------
    terms : dict
        Mapping of Pauli strings to real coefficients.
        Example: {"ZZ": -1.0, "XI": 0.5, "IX": 0.5}
        All strings must have the same length (number of qubits).

    Attributes
    ----------
    n_qubits : int
        Number of qubits inferred from Pauli string length.
    n_terms : int
        Number of Pauli terms in the Hamiltonian.
    """

    def __init__(self, terms: Dict[str, float]):
        if not terms:
            raise ValueError("Hamiltonian must have at least one term")

        # Validate and normalize
        self._terms = {}
        lengths = set()
        for pauli_str, coeff in terms.items():
            pauli_str = pauli_str.upper()
            lengths.add(len(pauli_str))
            if not all(c in "IXYZ" for c in pauli_str):
                raise ValueError(
                    f"Invalid Pauli string '{pauli_str}': "
                    "only I, X, Y, Z allowed"
                )
            if abs(coeff) > 1e-15:  # skip zero terms
                self._terms[pauli_str] = float(coeff)

        if len(lengths) > 1:
            raise ValueError(
                f"All Pauli strings must have the same length, "
                f"got lengths {lengths}"
            )

        self._n_qubits = lengths.pop() if lengths else 0

    @property
    def terms(self) -> Dict[str, float]:
        """Return copy of Pauli string → coefficient mapping."""
        return dict(self._terms)

    @property
    def n_qubits(self) -> int:
        """Number of qubits this Hamiltonian acts on."""
        return self._n_qubits

    @property
    def n_terms(self) -> int:
        """Number of non-zero Pauli terms."""
        return len(self._terms)

    def expectation(self, statevector: np.ndarray) -> float:
        """
        Compute ⟨ψ|H|ψ⟩ efficiently without building the full matrix.

        For each Pauli string P with coefficient c, computes:
            c * ⟨ψ|P|ψ⟩ = c * Re(ψ† · (P|ψ⟩))

        This is O(n_terms × 2^n) instead of O(2^2n) for full matrix.

        Parameters
        ----------
        statevector : np.ndarray
            Complex statevector of shape (2^n,).

        Returns
        -------
        float
            Real expectation value ⟨ψ|H|ψ⟩.
        """
        sv = np.asarray(statevector, dtype=complex).ravel()
        n = self._n_qubits
        expected_dim = 2 ** n
        if sv.shape[0] != expected_dim:
            raise ValueError(
                f"Statevector dimension {sv.shape[0]} doesn't match "
                f"{n}-qubit Hamiltonian (expected {expected_dim})"
            )

        total = 0.0
        for pauli_str, coeff in self._terms.items():
            # Apply Pauli string to statevector
            psi = _apply_pauli_string(sv, pauli_str, n)
            # Inner product ⟨ψ|P|ψ⟩
            total += coeff * np.real(np.vdot(sv, psi))
        return float(total)

    def matrix(self) -> np.ndarray:
        """
        Build the full 2^n × 2^n Hamiltonian matrix.

        Useful for small systems (≤ 10 qubits) and validation.
        For larger systems, use expectation() which avoids materializing
        the full matrix.

        Returns
        -------
        np.ndarray
            Hermitian matrix of shape (2^n, 2^n).
        """
        dim = 2 ** self._n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for pauli_str, coeff in self._terms.items():
            H += coeff * _pauli_string_matrix(pauli_str)
        return H

    def ground_state_energy(self) -> float:
        """
        Compute exact ground state energy via diagonalization.

        Only practical for small systems (≤ ~14 qubits).

        Returns
        -------
        float
            Minimum eigenvalue of H.
        """
        eigenvalues = np.linalg.eigvalsh(self.matrix())
        return float(eigenvalues[0])

    def ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Compute ground state energy and corresponding eigenvector.

        Returns
        -------
        tuple of (float, np.ndarray)
            (ground state energy, ground state vector)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix())
        return float(eigenvalues[0]), eigenvectors[:, 0]

    def __add__(self, other: "Hamiltonian") -> "Hamiltonian":
        """Add two Hamiltonians."""
        if not isinstance(other, Hamiltonian):
            return NotImplemented
        if self._n_qubits != other._n_qubits:
            raise ValueError("Cannot add Hamiltonians with different qubit counts")
        terms = dict(self._terms)
        for pauli_str, coeff in other._terms.items():
            terms[pauli_str] = terms.get(pauli_str, 0.0) + coeff
        return Hamiltonian(terms)

    def __mul__(self, scalar: float) -> "Hamiltonian":
        """Multiply Hamiltonian by a scalar."""
        return Hamiltonian({p: c * scalar for p, c in self._terms.items()})

    def __rmul__(self, scalar: float) -> "Hamiltonian":
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        terms_str = " + ".join(
            f"{c:+.4f} {p}" for p, c in self._terms.items()
        )
        return f"Hamiltonian({self._n_qubits}q, {self.n_terms} terms): {terms_str}"

    def __str__(self) -> str:
        lines = [f"Hamiltonian on {self._n_qubits} qubits ({self.n_terms} terms):"]
        for pauli_str, coeff in sorted(self._terms.items()):
            lines.append(f"  {coeff:+10.6f}  {pauli_str}")
        return "\n".join(lines)


# ─── Standard Hamiltonians ───────────────────────────────────────────

def transverse_field_ising(n_qubits: int, J: float = 1.0,
                           h: float = 1.0) -> Hamiltonian:
    """
    Transverse-field Ising model: H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ

    Parameters
    ----------
    n_qubits : int
        Number of spins (open boundary conditions).
    J : float
        Coupling strength.
    h : float
        Transverse field strength.
    """
    terms = {}
    for i in range(n_qubits - 1):
        pauli = "I" * i + "ZZ" + "I" * (n_qubits - i - 2)
        terms[pauli] = -J
    for i in range(n_qubits):
        pauli = "I" * i + "X" + "I" * (n_qubits - i - 1)
        terms[pauli] = -h
    return Hamiltonian(terms)


def heisenberg_xyz(n_qubits: int, Jx: float = 1.0, Jy: float = 1.0,
                   Jz: float = 1.0) -> Hamiltonian:
    """
    Heisenberg XYZ model: H = Σ (Jx XᵢXᵢ₊₁ + Jy YᵢYᵢ₊₁ + Jz ZᵢZᵢ₊₁)
    """
    terms = {}
    for i in range(n_qubits - 1):
        for pauli_char, J in [("X", Jx), ("Y", Jy), ("Z", Jz)]:
            pauli = "I" * i + pauli_char * 2 + "I" * (n_qubits - i - 2)
            terms[pauli] = J
    return Hamiltonian(terms)


def molecular_hydrogen(bond_length: float = 0.74) -> Hamiltonian:
    """
    H₂ molecule Hamiltonian in STO-3G basis (2-qubit, Jordan-Wigner).

    Coefficients from pre-computed classical quantum chemistry.
    Supports bond lengths: 0.5, 0.6, 0.7, 0.74, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0 Å.
    """
    # Pre-computed STO-3G coefficients for H₂ at various bond lengths
    _H2_DATA = {
        0.5:  {"II": -0.3858, "ZI":  0.3968, "IZ": -0.3968, "ZZ":  0.0112, "XX":  0.1811},
        0.6:  {"II": -0.5589, "ZI":  0.3158, "IZ": -0.3158, "ZZ":  0.0490, "XX":  0.1393},
        0.7:  {"II": -0.7028, "ZI":  0.2467, "IZ": -0.2467, "ZZ":  0.0797, "XX":  0.1044},
        0.74: {"II": -0.8105, "ZI":  0.1716, "IZ": -0.2228, "ZZ":  0.1209, "XX":  0.0454},
        0.8:  {"II": -0.8084, "ZI":  0.1896, "IZ": -0.1896, "ZZ":  0.1009, "XX":  0.0771},
        0.9:  {"II": -0.8765, "ZI":  0.1429, "IZ": -0.1429, "ZZ":  0.1147, "XX":  0.0570},
        1.0:  {"II": -0.9153, "ZI":  0.1048, "IZ": -0.1048, "ZZ":  0.1232, "XX":  0.0418},
        1.2:  {"II": -0.9453, "ZI":  0.0477, "IZ": -0.0477, "ZZ":  0.1303, "XX":  0.0222},
        1.5:  {"II": -0.9459, "ZI":  0.0078, "IZ": -0.0078, "ZZ":  0.1300, "XX":  0.0081},
        2.0:  {"II": -0.9366, "ZI": -0.0084, "IZ":  0.0084, "ZZ":  0.1269, "XX":  0.0014},
    }

    if bond_length not in _H2_DATA:
        available = sorted(_H2_DATA.keys())
        raise ValueError(
            f"Bond length {bond_length} Å not available. "
            f"Choose from: {available}"
        )
    return Hamiltonian(_H2_DATA[bond_length])


# ─── Internal utilities ──────────────────────────────────────────────

def _apply_pauli_string(sv: np.ndarray, pauli_str: str,
                        n_qubits: int) -> np.ndarray:
    """
    Apply a Pauli string (tensor product of Paulis) to a statevector.

    Uses efficient qubit-by-qubit application instead of building
    the full 2^n × 2^n tensor product matrix.

    For each non-identity Pauli, reshapes the statevector into a
    tensor and contracts with the 2×2 Pauli on the appropriate axis.
    """
    result = sv.copy()
    for qubit_idx, pauli_char in enumerate(pauli_str):
        if pauli_char == "I":
            continue
        pauli = PAULI_MAP[pauli_char]
        # Reshape into tensor: (2, 2, ..., 2) with n_qubits axes
        shape = [2] * n_qubits
        result = result.reshape(shape)
        # Apply 2x2 Pauli to the qubit_idx-th axis
        result = np.tensordot(pauli, result, axes=([1], [qubit_idx]))
        # Move the contracted axis back to its original position
        result = np.moveaxis(result, 0, qubit_idx)
        result = result.reshape(-1)
    return result


def _pauli_string_matrix(pauli_str: str) -> np.ndarray:
    """Build the full tensor product matrix for a Pauli string."""
    result = PAULI_MAP[pauli_str[0]]
    for char in pauli_str[1:]:
        result = np.kron(result, PAULI_MAP[char])
    return result
