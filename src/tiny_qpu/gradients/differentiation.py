"""
Quantum gradient computation: parameter-shift, adjoint, and finite-difference.

Three methods for computing ∂⟨ψ(θ)|H|ψ(θ)⟩/∂θ:

1. **Parameter-shift rule** (default for hardware):
   Exact analytic gradient via two circuit evaluations per parameter:
   ∂f/∂θᵢ = [f(θᵢ + π/2) − f(θᵢ − π/2)] / 2

2. **Adjoint differentiation** (default for simulators):
   Single forward + backward pass through the circuit. O(P) statevector
   operations instead of O(P) full circuit evaluations. ~100× faster
   for many parameters.

3. **Finite-difference** (universal fallback):
   Central difference: ∂f/∂θᵢ ≈ [f(θᵢ + ε) − f(θᵢ − ε)] / (2ε)

Usage with scipy.optimize.minimize:
    >>> from tiny_qpu.gradients import Hamiltonian, expectation_and_gradient
    >>> from scipy.optimize import minimize
    >>>
    >>> H = Hamiltonian({"ZZ": -1.0, "XI": 0.5})
    >>> def ansatz(params):
    ...     qc = Circuit(2)
    ...     qc.ry(params[0], 0).ry(params[1], 1).cx(0, 1)
    ...     return qc
    >>> result = minimize(
    ...     lambda p: expectation_and_gradient(ansatz, H, p),
    ...     x0=[0.1, 0.2], method="L-BFGS-B", jac=True
    ... )
"""

import numpy as np
from typing import Callable, Optional, Tuple, Union, List
from .hamiltonian import Hamiltonian, _apply_pauli_string


# ─── Gate generator registry ────────────────────────────────────────
# Maps gate names to their generator coefficient r such that
# ∂U(θ)/∂θ = -i r G U(θ) where G is the generator.
# For the parameter-shift rule: shift = π/(4r)
# Standard Pauli rotation gates have r = 1/2, so shift = π/2.

GENERATOR_COEFF = {
    "rx": 0.5, "ry": 0.5, "rz": 0.5,
    "crx": 0.5, "cry": 0.5, "crz": 0.5,
    "u1": 1.0, "p": 1.0,  # phase gates
    "rxx": 0.5, "ryy": 0.5, "rzz": 0.5,
    "cp": 1.0,  # controlled phase
}


# ─── Public API ──────────────────────────────────────────────────────

def expectation(circuit_fn: Callable, hamiltonian: Hamiltonian,
                params: np.ndarray) -> float:
    """
    Compute ⟨ψ(θ)|H|ψ(θ)⟩.

    Parameters
    ----------
    circuit_fn : callable
        Function params → Circuit. Must return a circuit object with
        a .statevector() method.
    hamiltonian : Hamiltonian
        Observable to measure.
    params : array-like
        Parameter values θ.

    Returns
    -------
    float
        Expectation value.
    """
    params = np.asarray(params, dtype=float)
    sv = _get_statevector(circuit_fn(params))
    return hamiltonian.expectation(sv)


def gradient(circuit_fn: Callable, hamiltonian: Hamiltonian,
             params: np.ndarray, method: str = "best") -> np.ndarray:
    """
    Compute ∇_θ ⟨ψ(θ)|H|ψ(θ)⟩.

    Parameters
    ----------
    circuit_fn : callable
        Function params → Circuit.
    hamiltonian : Hamiltonian
        Observable to measure.
    params : array-like
        Parameter values θ.
    method : str
        Gradient method: "param_shift", "adjoint", "finite_diff", or
        "best" (auto-selects adjoint for simulators).

    Returns
    -------
    np.ndarray
        Gradient vector of shape (len(params),).
    """
    params = np.asarray(params, dtype=float)
    method = _resolve_method(method)

    if method == "param_shift":
        return _param_shift_gradient(circuit_fn, hamiltonian, params)
    elif method == "adjoint":
        return _adjoint_gradient(circuit_fn, hamiltonian, params)
    elif method == "finite_diff":
        return _finite_diff_gradient(circuit_fn, hamiltonian, params)
    else:
        raise ValueError(f"Unknown method '{method}'. "
                         "Choose from: param_shift, adjoint, finite_diff, best")


def expectation_and_gradient(circuit_fn: Callable, hamiltonian: Hamiltonian,
                              params: np.ndarray, method: str = "best"
                              ) -> Tuple[float, np.ndarray]:
    """
    Compute both ⟨ψ(θ)|H|ψ(θ)⟩ and ∇_θ⟨ψ(θ)|H|ψ(θ)⟩ in one call.

    Designed for scipy.optimize.minimize(..., jac=True):

        result = minimize(
            lambda p: expectation_and_gradient(ansatz, H, p),
            x0=params0, method="L-BFGS-B", jac=True
        )

    Parameters
    ----------
    circuit_fn : callable
        Function params → Circuit.
    hamiltonian : Hamiltonian
        Observable to measure.
    params : array-like
        Parameter values θ.
    method : str
        Gradient method (see gradient()).

    Returns
    -------
    tuple of (float, np.ndarray)
        (expectation_value, gradient_vector)
    """
    params = np.asarray(params, dtype=float)
    method = _resolve_method(method)

    if method == "adjoint":
        return _adjoint_expectation_and_gradient(circuit_fn, hamiltonian, params)
    else:
        # For param_shift and finite_diff, compute separately
        val = expectation(circuit_fn, hamiltonian, params)
        grad = gradient(circuit_fn, hamiltonian, params, method=method)
        return val, grad


# ─── Parameter-Shift Rule ───────────────────────────────────────────

def _param_shift_gradient(circuit_fn: Callable, hamiltonian: Hamiltonian,
                          params: np.ndarray, shift: float = np.pi / 2
                          ) -> np.ndarray:
    """
    Exact analytic gradient via the parameter-shift rule.

    For each parameter θᵢ with generator coefficient r = 1/2:
        ∂f/∂θᵢ = [f(θᵢ + π/2) − f(θᵢ − π/2)] / 2

    Cost: 2P circuit evaluations for P parameters.
    Advantage: exact, works on real hardware.
    """
    n_params = len(params)
    grad = np.zeros(n_params)

    for i in range(n_params):
        # Shift parameter i forward
        params_plus = params.copy()
        params_plus[i] += shift

        # Shift parameter i backward
        params_minus = params.copy()
        params_minus[i] -= shift

        # Evaluate circuit at shifted parameters
        sv_plus = _get_statevector(circuit_fn(params_plus))
        sv_minus = _get_statevector(circuit_fn(params_minus))

        f_plus = hamiltonian.expectation(sv_plus)
        f_minus = hamiltonian.expectation(sv_minus)

        # Parameter-shift formula (r = 1/2 for standard rotation gates)
        grad[i] = (f_plus - f_minus) / (2 * np.sin(shift))

    return grad


# ─── Adjoint Differentiation ────────────────────────────────────────

def _adjoint_gradient(circuit_fn: Callable, hamiltonian: Hamiltonian,
                      params: np.ndarray) -> np.ndarray:
    """
    Adjoint differentiation — the fast path for statevector simulators.

    Algorithm:
    1. Forward pass: apply all gates to get |ψ(θ)⟩
    2. Compute |λ⟩ = H|ψ(θ)⟩
    3. Backward pass: for each gate (reverse order):
       a. Unapply gate from both |ψ⟩ and |λ⟩
       b. If gate is parameterized at θᵢ:
          grad[i] += 2 Re(⟨λ|∂U/∂θᵢ|ψ_partial⟩)

    Cost: 1 forward + 1 backward pass = O(G) gate applications
    where G = number of gates. Compare to O(P × G) for param-shift.
    """
    _, grad = _adjoint_expectation_and_gradient(circuit_fn, hamiltonian, params)
    return grad


def _adjoint_expectation_and_gradient(circuit_fn: Callable,
                                       hamiltonian: Hamiltonian,
                                       params: np.ndarray
                                       ) -> Tuple[float, np.ndarray]:
    """
    Combined expectation + gradient via adjoint method.

    Returns both values from a single forward-backward pass.
    """
    n_params = len(params)
    grad = np.zeros(n_params)

    # Build circuit and extract gate sequence
    circuit = circuit_fn(params)
    gate_sequence = _extract_gate_sequence(circuit)

    # Forward pass: compute |ψ⟩ = U_G ... U_2 U_1 |0⟩
    n_qubits = _get_n_qubits(circuit)
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[0] = 1.0  # |00...0⟩

    for gate_name, gate_matrix, qubits, param_idx in gate_sequence:
        psi = _apply_gate(psi, gate_matrix, qubits, n_qubits)

    # Expectation value
    exp_val = hamiltonian.expectation(psi)

    # Compute |λ⟩ = H|ψ⟩
    lam = np.zeros_like(psi)
    for pauli_str, coeff in hamiltonian.terms.items():
        lam += coeff * _apply_pauli_string(psi, pauli_str, n_qubits)

    # Backward pass: unapply gates in reverse
    # At each step for gate g:
    #   |ψ⟩ = U_g...U_1|0⟩  (state including gate g)
    #   |λ⟩ = U_{g+1}†...U_G† H |ψ_full⟩  (adjoint propagated past later gates)
    #
    # 1. Unapply gate from |ψ⟩: |ψ⟩ ← U_g†|ψ⟩  → now |ψ_{<g}⟩
    # 2. If parameterized: grad[i] += 2 Re(⟨λ|dU_g/dθ|ψ_{<g}⟩)
    # 3. Unapply gate from |λ⟩: |λ⟩ ← U_g†|λ⟩  → prepares for next iteration
    for gate_name, gate_matrix, qubits, param_idx in reversed(gate_sequence):
        gate_dag = gate_matrix.conj().T

        # Step 1: unapply from |ψ⟩ to get state BEFORE this gate
        psi = _apply_gate(psi, gate_dag, qubits, n_qubits)

        # Step 2: compute gradient contribution if parameterized
        if param_idx is not None:
            dU = _gate_derivative_matrix(gate_name, gate_matrix, params, param_idx)
            if dU is not None:
                dpsi = _apply_gate(psi.copy(), dU, qubits, n_qubits)
                grad[param_idx] += 2.0 * np.real(np.vdot(lam, dpsi))

        # Step 3: unapply from |λ⟩ to prepare for next iteration
        lam = _apply_gate(lam, gate_dag, qubits, n_qubits)

    return exp_val, grad


# ─── Finite Difference ──────────────────────────────────────────────

def _finite_diff_gradient(circuit_fn: Callable, hamiltonian: Hamiltonian,
                          params: np.ndarray, epsilon: float = 1e-7
                          ) -> np.ndarray:
    """
    Central finite-difference gradient (universal fallback).

    ∂f/∂θᵢ ≈ [f(θᵢ + ε) − f(θᵢ − ε)] / (2ε)

    Cost: 2P circuit evaluations.
    Disadvantage: approximate (O(ε²) error).
    Advantage: works for any parameterized gate, not just Pauli rotations.
    """
    n_params = len(params)
    grad = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += epsilon

        params_minus = params.copy()
        params_minus[i] -= epsilon

        sv_plus = _get_statevector(circuit_fn(params_plus))
        sv_minus = _get_statevector(circuit_fn(params_minus))

        f_plus = hamiltonian.expectation(sv_plus)
        f_minus = hamiltonian.expectation(sv_minus)

        grad[i] = (f_plus - f_minus) / (2 * epsilon)

    return grad


# ─── Internal utilities ─────────────────────────────────────────────

def _resolve_method(method: str) -> str:
    """Resolve 'best' to the optimal method for the environment."""
    if method == "best":
        return "adjoint"  # Best for statevector simulators
    return method


def _get_statevector(circuit) -> np.ndarray:
    """
    Extract statevector from a circuit object.

    Supports multiple circuit APIs:
    - circuit.statevector() — tiny-qpu Circuit
    - circuit.run() — legacy API
    - numpy array passthrough
    """
    if isinstance(circuit, np.ndarray):
        return circuit

    # Try the standard API first
    if hasattr(circuit, "statevector"):
        sv = circuit.statevector()
        if isinstance(sv, np.ndarray):
            return sv
        # SimulationResult object
        if hasattr(sv, "statevector"):
            return np.asarray(sv.statevector)
        return np.asarray(sv)

    # Legacy: run() returns statevector directly
    if hasattr(circuit, "run"):
        result = circuit.run()
        if isinstance(result, np.ndarray):
            return result
        if hasattr(result, "statevector"):
            return np.asarray(result.statevector)
        return np.asarray(result)

    raise TypeError(
        f"Cannot extract statevector from {type(circuit).__name__}. "
        "Circuit must have .statevector() or .run() method."
    )


def _get_n_qubits(circuit) -> int:
    """Get number of qubits from circuit."""
    if hasattr(circuit, "n_qubits"):
        return circuit.n_qubits
    if hasattr(circuit, "num_qubits"):
        n = circuit.num_qubits
        return n() if callable(n) else n
    raise AttributeError("Cannot determine number of qubits from circuit")


def _extract_gate_sequence(circuit) -> List[Tuple]:
    """
    Extract ordered gate sequence from a circuit for adjoint differentiation.

    Returns list of (gate_name, unitary_matrix, qubit_indices, param_index_or_None).

    This works with the tiny-qpu Circuit that stores instructions as
    Instruction(name, qubits, params, matrix) objects.
    """
    gates = []

    # Try tiny-qpu Circuit API: circuit._instructions or circuit.instructions
    instructions = None
    if hasattr(circuit, "_instructions"):
        instructions = circuit._instructions
    elif hasattr(circuit, "instructions"):
        instructions = circuit.instructions
        if callable(instructions):
            instructions = instructions()

    if instructions is not None:
        for instr in instructions:
            name = instr.name if hasattr(instr, "name") else str(instr)
            qubits = instr.qubits if hasattr(instr, "qubits") else []
            matrix = instr.matrix if hasattr(instr, "matrix") else None
            param_idx = instr.param_idx if hasattr(instr, "param_idx") else None

            # For parameterized gates, try to find the param_idx from the params
            if matrix is not None and name.lower() not in ("measure", "barrier"):
                gates.append((name.lower(), np.asarray(matrix, dtype=complex),
                             list(qubits), param_idx))
    return gates


def _apply_gate(sv: np.ndarray, gate_matrix: np.ndarray,
                qubits: List[int], n_qubits: int) -> np.ndarray:
    """
    Apply a gate unitary to specific qubits of a statevector.

    Uses tensor contraction (einsum) for efficiency.
    """
    n_gate_qubits = len(qubits)

    if n_gate_qubits == 1:
        return _apply_single_qubit_gate(sv, gate_matrix, qubits[0], n_qubits)
    elif n_gate_qubits == 2:
        return _apply_two_qubit_gate(sv, gate_matrix, qubits, n_qubits)
    else:
        # General case: build full unitary and multiply
        return _apply_general_gate(sv, gate_matrix, qubits, n_qubits)


def _apply_single_qubit_gate(sv: np.ndarray, gate: np.ndarray,
                              qubit: int, n_qubits: int) -> np.ndarray:
    """Apply single-qubit gate via tensor contraction."""
    shape = [2] * n_qubits
    sv = sv.reshape(shape)
    sv = np.tensordot(gate, sv, axes=([1], [qubit]))
    sv = np.moveaxis(sv, 0, qubit)
    return sv.reshape(-1)


def _apply_two_qubit_gate(sv: np.ndarray, gate: np.ndarray,
                           qubits: List[int], n_qubits: int) -> np.ndarray:
    """Apply two-qubit gate via tensor contraction."""
    shape = [2] * n_qubits
    sv = sv.reshape(shape)
    gate = gate.reshape(2, 2, 2, 2)

    # Contract on the two qubit axes
    q0, q1 = qubits
    sv = np.tensordot(gate, sv, axes=([2, 3], [q0, q1]))
    # Move contracted axes back
    # After tensordot, new axes are at positions 0, 1
    # We need to move them to q0, q1
    perm = list(range(n_qubits))
    # The two new axes are at the front (0, 1), rest shifted by 2
    # Build the correct permutation
    remaining = [i for i in range(n_qubits) if i not in qubits]
    # Current order: [gate_out_0, gate_out_1, remaining...]
    # Desired order: each axis in its original position
    target = [None] * n_qubits
    target[q0] = 0
    target[q1] = 1
    r_idx = 2
    for i in range(n_qubits):
        if target[i] is None:
            target[i] = r_idx
            r_idx += 1
    # Invert permutation
    inv_perm = [0] * n_qubits
    for i, t in enumerate(target):
        inv_perm[t] = i

    sv = sv.transpose(inv_perm)
    return sv.reshape(-1)


def _apply_general_gate(sv: np.ndarray, gate: np.ndarray,
                        qubits: List[int], n_qubits: int) -> np.ndarray:
    """Apply multi-qubit gate via full matrix multiplication (fallback)."""
    dim = 2 ** n_qubits
    n_gate = len(qubits)

    # Build the full unitary by tensoring with identity on other qubits
    # This is the slow path — only used for 3+ qubit gates
    full_gate = np.eye(dim, dtype=complex)

    for i in range(dim):
        for j in range(dim):
            # Extract the gate-qubit bits from i and j
            i_bits = [(i >> (n_qubits - 1 - q)) & 1 for q in qubits]
            j_bits = [(j >> (n_qubits - 1 - q)) & 1 for q in qubits]

            # Check if non-gate qubits match
            match = True
            for q in range(n_qubits):
                if q not in qubits:
                    if ((i >> (n_qubits - 1 - q)) & 1) != ((j >> (n_qubits - 1 - q)) & 1):
                        match = False
                        break

            if match:
                gi = sum(b << (n_gate - 1 - k) for k, b in enumerate(i_bits))
                gj = sum(b << (n_gate - 1 - k) for k, b in enumerate(j_bits))
                full_gate[i, j] = gate[gi, gj]
            else:
                full_gate[i, j] = 0.0

    return full_gate @ sv


def _gate_derivative_matrix(gate_name: str, gate_matrix: np.ndarray,
                            params: np.ndarray, param_idx: int
                            ) -> Optional[np.ndarray]:
    """
    Compute the derivative matrix dU/dθ for parameterized gates.

    For Pauli rotation gates Rp(θ) = exp(-iθP/2):
        dRp/dθ = -i/2 P Rp(θ)

    For phase gate P(θ) = diag(1, e^{iθ}):
        dP/dθ = diag(0, i e^{iθ})
    """
    name = gate_name.lower()
    dim = gate_matrix.shape[0]

    if name in ("rx", "crx"):
        # dRx/dθ = -i/2 X Rx(θ) for single qubit part
        if dim == 2:
            X = np.array([[0, 1], [1, 0]], dtype=complex)
            return -0.5j * X @ gate_matrix
        else:  # controlled version
            dU = np.zeros_like(gate_matrix)
            X = np.array([[0, 1], [1, 0]], dtype=complex)
            # Derivative only in the |1⟩⟨1| control block
            sub = gate_matrix[2:4, 2:4]
            dU[2:4, 2:4] = -0.5j * X @ sub
            return dU

    elif name in ("ry", "cry"):
        if dim == 2:
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            return -0.5j * Y @ gate_matrix
        else:
            dU = np.zeros_like(gate_matrix)
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            sub = gate_matrix[2:4, 2:4]
            dU[2:4, 2:4] = -0.5j * Y @ sub
            return dU

    elif name in ("rz", "crz"):
        if dim == 2:
            Z = np.array([[1, 0], [0, -1]], dtype=complex)
            return -0.5j * Z @ gate_matrix
        else:
            dU = np.zeros_like(gate_matrix)
            Z = np.array([[1, 0], [0, -1]], dtype=complex)
            sub = gate_matrix[2:4, 2:4]
            dU[2:4, 2:4] = -0.5j * Z @ sub
            return dU

    elif name in ("u1", "p", "cp"):
        # Phase gate: P(θ) = diag(1, e^{iθ})
        dU = np.zeros_like(gate_matrix)
        if dim == 2:
            dU[1, 1] = 1j * gate_matrix[1, 1]
        else:  # controlled phase
            dU[3, 3] = 1j * gate_matrix[3, 3]
        return dU

    elif name in ("rxx", "ryy", "rzz"):
        # Ising coupling gates
        pauli_map = {"rxx": "XX", "ryy": "YY", "rzz": "ZZ"}
        P = np.array([[0, 1], [1, 0]], dtype=complex)  # default X
        if "y" in name:
            P = np.array([[0, -1j], [1j, 0]], dtype=complex)
        elif "z" in name:
            P = np.array([[1, 0], [0, -1]], dtype=complex)
        PP = np.kron(P, P)
        return -0.5j * PP @ gate_matrix

    else:
        # Unknown parameterized gate — cannot compute analytic derivative
        return None
