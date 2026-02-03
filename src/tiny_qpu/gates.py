"""
Quantum gate definitions.

All gates are represented as unitary matrices (numpy arrays).
Parameterized gates are callables that return matrices.

Gate categories:
    - Single-qubit: I, X, Y, Z, H, S, T, Sdg, Tdg, SX
    - Rotations: Rx, Ry, Rz, P (phase), U3 (universal)
    - Two-qubit: CNOT/CX, CZ, SWAP, iSWAP, ECR
    - Controlled: CP, CRx, CRy, CRz, CCX (Toffoli), CSWAP (Fredkin)
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

# Type alias
Matrix = ndarray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT2_INV = 1.0 / np.sqrt(2.0)

# ---------------------------------------------------------------------------
# Single-qubit fixed gates
# ---------------------------------------------------------------------------

I = np.eye(2, dtype=np.complex128)
"""Identity gate."""

X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
"""Pauli-X (NOT) gate."""

Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
"""Pauli-Y gate."""

Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
"""Pauli-Z gate."""

H = np.array([[1, 1], [1, -1]], dtype=np.complex128) * _SQRT2_INV
"""Hadamard gate."""

S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
"""S (phase) gate: sqrt(Z)."""

Sdg = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
"""S-dagger gate."""

T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
"""T gate: sqrt(S)."""

Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
"""T-dagger gate."""

SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) * 0.5
"""sqrt(X) gate."""

# ---------------------------------------------------------------------------
# Single-qubit parameterized gates
# ---------------------------------------------------------------------------

def Rx(theta: float) -> Matrix:
    """Rotation around X-axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def Ry(theta: float) -> Matrix:
    """Rotation around Y-axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def Rz(phi: float) -> Matrix:
    """Rotation around Z-axis by angle phi."""
    return np.array(
        [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]],
        dtype=np.complex128,
    )


def P(lam: float) -> Matrix:
    """Phase gate: diagonal with entries [1, exp(i*lam)]."""
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)


def U3(theta: float, phi: float, lam: float) -> Matrix:
    """
    Universal single-qubit gate (IBM U3 convention).

    U3(θ, φ, λ) = [[cos(θ/2), -e^(iλ) sin(θ/2)],
                    [e^(iφ) sin(θ/2), e^(i(φ+λ)) cos(θ/2)]]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def U2(phi: float, lam: float) -> Matrix:
    """U2 gate: U3(pi/2, phi, lam)."""
    return U3(np.pi / 2, phi, lam)


def U1(lam: float) -> Matrix:
    """U1 gate: U3(0, 0, lam) = Phase gate."""
    return P(lam)


# ---------------------------------------------------------------------------
# Two-qubit fixed gates (4x4 matrices)
# ---------------------------------------------------------------------------

CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)
"""Controlled-NOT (CX) gate."""
CX = CNOT  # alias

CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
"""Controlled-Z gate."""

SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)
"""SWAP gate."""

iSWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)
"""iSWAP gate."""

ECR = np.array(
    [
        [0, 0, _SQRT2_INV, 1j * _SQRT2_INV],
        [0, 0, 1j * _SQRT2_INV, _SQRT2_INV],
        [_SQRT2_INV, -1j * _SQRT2_INV, 0, 0],
        [-1j * _SQRT2_INV, _SQRT2_INV, 0, 0],
    ],
    dtype=np.complex128,
)
"""Echoed cross-resonance gate (IBM native 2Q gate)."""

# ---------------------------------------------------------------------------
# Two-qubit parameterized gates
# ---------------------------------------------------------------------------

def CP(lam: float) -> Matrix:
    """Controlled-Phase gate."""
    return np.diag([1, 1, 1, np.exp(1j * lam)]).astype(np.complex128)


def CRx(theta: float) -> Matrix:
    """Controlled-Rx gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -1j * s],
            [0, 0, -1j * s, c],
        ],
        dtype=np.complex128,
    )


def CRy(theta: float) -> Matrix:
    """Controlled-Ry gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c],
        ],
        dtype=np.complex128,
    )


def CRz(phi: float) -> Matrix:
    """Controlled-Rz gate."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j * phi / 2), 0],
            [0, 0, 0, np.exp(1j * phi / 2)],
        ],
        dtype=np.complex128,
    )


def Rxx(theta: float) -> Matrix:
    """Ising XX coupling gate."""
    c = np.cos(theta / 2)
    s = 1j * np.sin(theta / 2)
    return np.array(
        [[c, 0, 0, -s], [0, c, -s, 0], [0, -s, c, 0], [-s, 0, 0, c]],
        dtype=np.complex128,
    )


def Ryy(theta: float) -> Matrix:
    """Ising YY coupling gate."""
    c = np.cos(theta / 2)
    s = 1j * np.sin(theta / 2)
    return np.array(
        [[c, 0, 0, s], [0, c, -s, 0], [0, -s, c, 0], [s, 0, 0, c]],
        dtype=np.complex128,
    )


def Rzz(theta: float) -> Matrix:
    """Ising ZZ coupling gate."""
    e_neg = np.exp(-1j * theta / 2)
    e_pos = np.exp(1j * theta / 2)
    return np.diag([e_neg, e_pos, e_pos, e_neg]).astype(np.complex128)


# ---------------------------------------------------------------------------
# Three-qubit gates (8x8 matrices)
# ---------------------------------------------------------------------------

CCX = np.eye(8, dtype=np.complex128)
CCX[6, 6] = 0
CCX[7, 7] = 0
CCX[6, 7] = 1
CCX[7, 6] = 1
"""Toffoli (CCX) gate."""
Toffoli = CCX  # alias

CSWAP = np.eye(8, dtype=np.complex128)
CSWAP[5, 5] = 0
CSWAP[6, 6] = 0
CSWAP[5, 6] = 1
CSWAP[6, 5] = 1
"""Fredkin (CSWAP) gate."""
Fredkin = CSWAP  # alias


# ---------------------------------------------------------------------------
# Gate metadata registry
# ---------------------------------------------------------------------------

GATE_REGISTRY: dict[str, dict] = {
    # Fixed single-qubit
    "i": {"matrix": I, "n_qubits": 1, "n_params": 0},
    "x": {"matrix": X, "n_qubits": 1, "n_params": 0},
    "y": {"matrix": Y, "n_qubits": 1, "n_params": 0},
    "z": {"matrix": Z, "n_qubits": 1, "n_params": 0},
    "h": {"matrix": H, "n_qubits": 1, "n_params": 0},
    "s": {"matrix": S, "n_qubits": 1, "n_params": 0},
    "sdg": {"matrix": Sdg, "n_qubits": 1, "n_params": 0},
    "t": {"matrix": T, "n_qubits": 1, "n_params": 0},
    "tdg": {"matrix": Tdg, "n_qubits": 1, "n_params": 0},
    "sx": {"matrix": SX, "n_qubits": 1, "n_params": 0},
    # Parameterized single-qubit
    "rx": {"factory": Rx, "n_qubits": 1, "n_params": 1},
    "ry": {"factory": Ry, "n_qubits": 1, "n_params": 1},
    "rz": {"factory": Rz, "n_qubits": 1, "n_params": 1},
    "p": {"factory": P, "n_qubits": 1, "n_params": 1},
    "u3": {"factory": U3, "n_qubits": 1, "n_params": 3},
    "u2": {"factory": U2, "n_qubits": 1, "n_params": 2},
    "u1": {"factory": U1, "n_qubits": 1, "n_params": 1},
    # Fixed two-qubit
    "cx": {"matrix": CNOT, "n_qubits": 2, "n_params": 0},
    "cnot": {"matrix": CNOT, "n_qubits": 2, "n_params": 0},
    "cz": {"matrix": CZ, "n_qubits": 2, "n_params": 0},
    "swap": {"matrix": SWAP, "n_qubits": 2, "n_params": 0},
    "iswap": {"matrix": iSWAP, "n_qubits": 2, "n_params": 0},
    "ecr": {"matrix": ECR, "n_qubits": 2, "n_params": 0},
    # Parameterized two-qubit
    "cp": {"factory": CP, "n_qubits": 2, "n_params": 1},
    "crx": {"factory": CRx, "n_qubits": 2, "n_params": 1},
    "cry": {"factory": CRy, "n_qubits": 2, "n_params": 1},
    "crz": {"factory": CRz, "n_qubits": 2, "n_params": 1},
    "rxx": {"factory": Rxx, "n_qubits": 2, "n_params": 1},
    "ryy": {"factory": Ryy, "n_qubits": 2, "n_params": 1},
    "rzz": {"factory": Rzz, "n_qubits": 2, "n_params": 1},
    # Three-qubit
    "ccx": {"matrix": CCX, "n_qubits": 3, "n_params": 0},
    "toffoli": {"matrix": CCX, "n_qubits": 3, "n_params": 0},
    "cswap": {"matrix": CSWAP, "n_qubits": 3, "n_params": 0},
    "fredkin": {"matrix": CSWAP, "n_qubits": 3, "n_params": 0},
}


def get_matrix(name: str, params: tuple[float, ...] = ()) -> Matrix:
    """
    Look up a gate matrix by name, with optional parameters.

    Parameters
    ----------
    name : str
        Gate name (case-insensitive).
    params : tuple of float
        Parameters for parameterized gates.

    Returns
    -------
    numpy.ndarray
        Unitary matrix for the gate.

    Raises
    ------
    KeyError
        If gate name is not found.
    ValueError
        If wrong number of parameters provided.
    """
    key = name.lower()
    if key not in GATE_REGISTRY:
        raise KeyError(f"Unknown gate: '{name}'. Available: {sorted(GATE_REGISTRY.keys())}")

    info = GATE_REGISTRY[key]
    n_params = info["n_params"]

    if n_params == 0:
        if params:
            raise ValueError(f"Gate '{name}' takes no parameters, got {len(params)}")
        return info["matrix"]
    else:
        if len(params) != n_params:
            raise ValueError(
                f"Gate '{name}' requires {n_params} parameter(s), got {len(params)}"
            )
        return info["factory"](*params)


def is_unitary(m, tol=1e-9):
    import numpy as np
    product = m @ m.conj().T
    return np.allclose(product, np.eye(len(m)), atol=tol)

