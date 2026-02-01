"""
Quantum gates as numpy arrays.

All gates are unitary matrices. Single-qubit gates are 2x2,
two-qubit gates are 4x4, three-qubit gates are 8x8.
"""
import numpy as np
from typing import Callable
from functools import lru_cache


# =============================================================================
# SINGLE-QUBIT GATES (2x2 matrices)
# =============================================================================

# Pauli gates
I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Hadamard
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

# Phase gates
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
S_DAG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
T_DAG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)

# Square root gates
SX = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np.complex128) / 2  # √X


# =============================================================================
# ROTATION GATES (parametric)
# =============================================================================

def Rx(theta: float) -> np.ndarray:
    """Rotation around X-axis: Rx(θ) = exp(-iθX/2)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def Ry(theta: float) -> np.ndarray:
    """Rotation around Y-axis: Ry(θ) = exp(-iθY/2)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def Rz(theta: float) -> np.ndarray:
    """Rotation around Z-axis: Rz(θ) = exp(-iθZ/2)"""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)


def P(phi: float) -> np.ndarray:
    """Phase gate: P(φ)|1⟩ = e^(iφ)|1⟩"""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)


def U3(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    General single-qubit unitary with 3 Euler angles.
    U3(θ,φ,λ) = Rz(φ)Ry(θ)Rz(λ)
    
    Any single-qubit gate can be expressed as U3.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=np.complex128)


# =============================================================================
# TWO-QUBIT GATES (4x4 matrices)
# =============================================================================

# CNOT (CX) - controlled NOT
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)
CX = CNOT  # Alias

# CZ - controlled Z
CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=np.complex128)

# CY - controlled Y
CY = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, -1j],
    [0, 0, 1j, 0]
], dtype=np.complex128)

# SWAP
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.complex128)

# iSWAP
ISWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=np.complex128)

# √SWAP
SQSWAP = np.array([
    [1, 0, 0, 0],
    [0, 0.5*(1+1j), 0.5*(1-1j), 0],
    [0, 0.5*(1-1j), 0.5*(1+1j), 0],
    [0, 0, 0, 1]
], dtype=np.complex128)


def CRz(theta: float) -> np.ndarray:
    """Controlled Rz rotation."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)


def CRx(theta: float) -> np.ndarray:
    """Controlled Rx rotation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, -1j * s],
        [0, 0, -1j * s, c]
    ], dtype=np.complex128)


def CRy(theta: float) -> np.ndarray:
    """Controlled Ry rotation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, -s],
        [0, 0, s, c]
    ], dtype=np.complex128)


def CP(phi: float) -> np.ndarray:
    """Controlled phase gate."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * phi)]
    ], dtype=np.complex128)


def RZZ(theta: float) -> np.ndarray:
    """ZZ interaction gate: exp(-i θ/2 ZZ)"""
    return np.diag([
        np.exp(-1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(-1j * theta / 2)
    ]).astype(np.complex128)


def RXX(theta: float) -> np.ndarray:
    """XX interaction gate: exp(-i θ/2 XX)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, 0, 0, -1j*s],
        [0, c, -1j*s, 0],
        [0, -1j*s, c, 0],
        [-1j*s, 0, 0, c]
    ], dtype=np.complex128)


def RYY(theta: float) -> np.ndarray:
    """YY interaction gate: exp(-i θ/2 YY)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, 0, 0, 1j*s],
        [0, c, -1j*s, 0],
        [0, -1j*s, c, 0],
        [1j*s, 0, 0, c]
    ], dtype=np.complex128)


# =============================================================================
# THREE-QUBIT GATES (8x8 matrices)
# =============================================================================

# Toffoli (CCNOT, CCX)
TOFFOLI = np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0]
], dtype=np.complex128)
CCX = TOFFOLI
CCNOT = TOFFOLI

# Fredkin (CSWAP)
FREDKIN = np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1]
], dtype=np.complex128)
CSWAP = FREDKIN


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_unitary(gate: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is unitary: U†U = I"""
    n = gate.shape[0]
    product = gate.conj().T @ gate
    return np.allclose(product, np.eye(n), atol=tol)


def gate_to_matrix(name: str, *params) -> np.ndarray:
    """Get gate matrix by name with optional parameters."""
    gates = {
        'I': I, 'X': X, 'Y': Y, 'Z': Z,
        'H': H, 'S': S, 'T': T, 'SX': SX,
        'CNOT': CNOT, 'CX': CX, 'CZ': CZ, 'CY': CY,
        'SWAP': SWAP, 'ISWAP': ISWAP,
        'TOFFOLI': TOFFOLI, 'CCX': CCX, 'CSWAP': CSWAP,
    }
    
    param_gates = {
        'RX': Rx, 'RY': Ry, 'RZ': Rz, 'P': P,
        'CRX': CRx, 'CRY': CRy, 'CRZ': CRz, 'CP': CP,
        'RXX': RXX, 'RYY': RYY, 'RZZ': RZZ,
        'U3': U3,
    }
    
    name = name.upper()
    if name in gates:
        return gates[name]
    elif name in param_gates:
        return param_gates[name](*params)
    else:
        raise ValueError(f"Unknown gate: {name}")
