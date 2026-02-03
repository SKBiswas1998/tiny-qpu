"""
Density matrix simulation backend.

Supports mixed states and noise channels (Kraus operators).
Memory: O(4^n) — 2x the exponent of statevector.
Practical limit: ~15 qubits on standard hardware.

This backend enables:
    - Noisy circuit simulation
    - Mixed state preparation
    - Quantum channel application
    - Partial trace and subsystem analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy import ndarray

from tiny_qpu.circuit import Circuit, Instruction


@dataclass
class DensityMatrixResult:
    """Result of density matrix simulation."""

    density_matrix: ndarray
    counts: dict[int, int] | None
    n_qubits: int
    shots: int = 0

    def probabilities(self) -> dict[int, float]:
        """Diagonal elements of density matrix (measurement probabilities)."""
        diag = np.real(np.diag(self.density_matrix))
        return {i: float(p) for i, p in enumerate(diag) if p > 1e-10}

    def purity(self) -> float:
        """Tr(ρ²) — 1.0 for pure states, 1/d for maximally mixed."""
        return float(np.real(np.trace(self.density_matrix @ self.density_matrix)))

    def fidelity_to_pure(self, state: ndarray) -> float:
        """Fidelity F = ⟨ψ|ρ|ψ⟩ with a pure state."""
        return float(np.real(state.conj() @ self.density_matrix @ state))

    def von_neumann_entropy(self) -> float:
        """Von Neumann entropy S = -Tr(ρ log ρ)."""
        eigenvalues = np.linalg.eigvalsh(self.density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    def partial_trace(self, keep_qubits: list[int]) -> ndarray:
        """
        Compute partial trace, keeping only specified qubits.

        Parameters
        ----------
        keep_qubits : list[int]
            Qubit indices to keep.

        Returns
        -------
        ndarray
            Reduced density matrix.
        """
        n = self.n_qubits
        rho = self.density_matrix.reshape([2] * (2 * n))

        trace_qubits = [q for q in range(n) if q not in keep_qubits]
        # Trace out qubits by contracting bra and ket indices
        for q in sorted(trace_qubits, reverse=True):
            rho = np.trace(rho, axis1=q, axis2=q + n - (n - rho.ndim // 2))
            # Adjust: after each trace, dimensions reduce

        # Simpler approach: use einsum
        rho = self.density_matrix.reshape([2] * (2 * n))

        # Build index lists
        bra_indices = list(range(n))
        ket_indices = list(range(n, 2 * n))

        # For traced-out qubits, make bra and ket indices the same
        result_bra = []
        result_ket = []
        next_idx = 2 * n
        contraction = list(range(2 * n))  # default: all distinct

        for q in trace_qubits:
            contraction[q + n] = contraction[q]  # ket index = bra index → trace

        # Build einsum
        in_indices = contraction
        out_indices = [contraction[q] for q in keep_qubits] + [
            contraction[q + n] for q in keep_qubits
        ]

        k = len(keep_qubits)
        reduced = np.einsum(rho, in_indices, out_indices)
        return reduced.reshape(2**k, 2**k)

    def bitstring_counts(self) -> dict[str, int]:
        """Get counts as bitstring keys."""
        if self.counts is None:
            return {}
        return {
            format(k, f"0{self.n_qubits}b"): v
            for k, v in sorted(self.counts.items())
        }


# ---------------------------------------------------------------------------
# Noise channels as Kraus operators
# ---------------------------------------------------------------------------

def depolarizing_channel(p: float) -> list[ndarray]:
    """
    Single-qubit depolarizing channel Kraus operators.

    ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

    Parameters
    ----------
    p : float
        Error probability (0 to 1).
    """
    from tiny_qpu.gates import I, X, Y, Z
    return [
        np.sqrt(1 - p) * I,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]


def amplitude_damping_channel(gamma: float) -> list[ndarray]:
    """
    Amplitude damping channel (T1 relaxation).

    Parameters
    ----------
    gamma : float
        Damping probability (0 to 1).
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [K0, K1]


def phase_damping_channel(gamma: float) -> list[ndarray]:
    """
    Phase damping channel (T2 dephasing without energy loss).

    Parameters
    ----------
    gamma : float
        Dephasing probability (0 to 1).
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)
    return [K0, K1]


def bit_flip_channel(p: float) -> list[ndarray]:
    """Bit-flip channel: ρ → (1-p)ρ + pXρX."""
    from tiny_qpu.gates import I, X
    return [np.sqrt(1 - p) * I, np.sqrt(p) * X]


def phase_flip_channel(p: float) -> list[ndarray]:
    """Phase-flip channel: ρ → (1-p)ρ + pZρZ."""
    from tiny_qpu.gates import I, Z
    return [np.sqrt(1 - p) * I, np.sqrt(p) * Z]


def thermal_relaxation_channel(t1: float, t2: float, gate_time: float) -> list[ndarray]:
    """
    Combined T1/T2 thermal relaxation channel.

    Parameters
    ----------
    t1 : float
        T1 relaxation time (μs).
    t2 : float
        T2 dephasing time (μs). Must satisfy t2 <= 2*t1.
    gate_time : float
        Gate execution time (μs).
    """
    if t2 > 2 * t1:
        raise ValueError(f"T2 ({t2}) must be <= 2*T1 ({2*t1})")

    p_reset = 1 - np.exp(-gate_time / t1)
    p_phase = 1 - np.exp(-gate_time / t2) if t2 > 0 else 1.0

    # Combine amplitude damping and additional dephasing
    gamma_ad = p_reset
    gamma_pd = max(0, p_phase - p_reset / 2)

    kraus_ops = []
    ad_ops = amplitude_damping_channel(gamma_ad)
    pd_ops = phase_damping_channel(gamma_pd)

    for ad in ad_ops:
        for pd in pd_ops:
            combined = ad @ pd
            if np.linalg.norm(combined) > 1e-10:
                kraus_ops.append(combined)

    return kraus_ops


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

@dataclass
class NoiseModel:
    """
    Configuration for circuit noise.

    Specify error channels per gate type or globally.
    """
    gate_errors: dict[str, list[ndarray]] = None  # gate_name → Kraus ops
    measurement_error: ndarray | None = None  # 2×2 confusion matrix
    idle_noise: list[ndarray] | None = None  # noise during idle qubits

    def __post_init__(self) -> None:
        if self.gate_errors is None:
            self.gate_errors = {}

    def add_gate_error(self, gate_name: str, kraus_ops: list[ndarray]) -> None:
        """Add a noise channel to a specific gate type."""
        self.gate_errors[gate_name] = kraus_ops

    def add_all_qubit_error(self, kraus_ops: list[ndarray]) -> None:
        """Add noise channel to all single-qubit gates."""
        single_qubit_gates = [
            "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx",
            "rx", "ry", "rz", "p", "u3", "u2", "u1",
        ]
        for gate in single_qubit_gates:
            self.gate_errors[gate] = kraus_ops


# ---------------------------------------------------------------------------
# DensityMatrixBackend
# ---------------------------------------------------------------------------

class DensityMatrixBackend:
    """
    Density matrix quantum simulator with noise support.

    Parameters
    ----------
    noise_model : NoiseModel, optional
        Noise configuration. If None, runs ideal simulation.
    seed : int | None
        Random seed for measurement sampling.

    Example
    -------
    >>> from tiny_qpu import Circuit, DensityMatrixBackend
    >>> from tiny_qpu.backends.density_matrix import NoiseModel, depolarizing_channel
    >>> noise = NoiseModel()
    >>> noise.add_gate_error("cx", depolarizing_channel(0.01))
    >>> backend = DensityMatrixBackend(noise_model=noise)
    >>> qc = Circuit(2)
    >>> qc.h(0).cx(0, 1)
    >>> result = backend.run(qc)
    >>> print(f"Purity: {result.purity():.4f}")
    """

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
    ) -> None:
        self.noise_model = noise_model or NoiseModel()
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        circuit: Circuit,
        shots: int = 0,
        initial_state: ndarray | None = None,
    ) -> DensityMatrixResult:
        """
        Simulate circuit using density matrix formalism.

        Parameters
        ----------
        circuit : Circuit
            Circuit to simulate.
        shots : int
            Number of measurement shots.
        initial_state : ndarray, optional
            Initial density matrix or state vector.
        """
        n = circuit.n_qubits
        dim = 2**n

        # Initialize density matrix
        if initial_state is not None:
            if initial_state.ndim == 1:
                rho = np.outer(initial_state, initial_state.conj())
            else:
                rho = initial_state.copy()
        else:
            rho = np.zeros((dim, dim), dtype=np.complex128)
            rho[0, 0] = 1.0

        # Apply circuit
        for inst in circuit.instructions:
            if inst.name in ("barrier", "measure"):
                continue

            matrix = inst.matrix()
            rho = self._apply_unitary(rho, matrix, inst.qubits, n)

            # Apply noise if configured
            gate_key = inst.name.lower()
            if gate_key in self.noise_model.gate_errors:
                kraus_ops = self.noise_model.gate_errors[gate_key]
                for qubit in inst.qubits:
                    rho = self._apply_kraus(rho, kraus_ops, (qubit,), n)

        # Sample
        counts = None
        if shots > 0:
            probs = np.real(np.diag(rho))
            probs = np.maximum(probs, 0)  # numerical safety
            probs /= probs.sum()
            outcomes = self._rng.choice(dim, size=shots, p=probs)
            unique, counts_arr = np.unique(outcomes, return_counts=True)
            counts = dict(zip(unique.tolist(), counts_arr.tolist()))

        return DensityMatrixResult(
            density_matrix=rho, counts=counts, n_qubits=n, shots=shots
        )

    def _apply_unitary(
        self, rho: ndarray, U: ndarray, qubits: tuple[int, ...], n: int
    ) -> ndarray:
        """Apply ρ → U ρ U† using tensor contraction."""
        dim = 2**n
        k = len(qubits)

        if k == 1:
            return self._apply_single_unitary(rho, U, qubits[0], n)

        # General case: build full operator via tensor product embedding
        # For small gate count this is efficient enough
        full_U = self._embed_gate(U, qubits, n)
        return full_U @ rho @ full_U.conj().T

    def _apply_single_unitary(
        self, rho: ndarray, U: ndarray, qubit: int, n: int
    ) -> ndarray:
        """Efficient single-qubit unitary on density matrix."""
        dim = 2**n
        # ρ → U_q ρ U_q†
        # Reshape rho as tensor, apply U to both bra and ket on qubit axis
        rho_tensor = rho.reshape([2] * (2 * n))

        indices_in = list("abcdefghijklmnopqrst"[:2 * n])
        # Apply U to ket (first n indices)
        u_out = "Y"
        u_in = indices_in[qubit]
        result1 = indices_in.copy()
        result1[qubit] = u_out

        einsum1 = f"{u_out}{u_in},{''.join(indices_in)}->{''.join(result1)}"
        rho_tensor = np.einsum(einsum1, U, rho_tensor)

        # Apply U† to bra (last n indices)
        indices_in = result1
        udg_out = "Z"
        udg_in = indices_in[n + qubit]
        result2 = indices_in.copy()
        result2[n + qubit] = udg_out

        einsum2 = f"{udg_out}{udg_in},{''.join(indices_in)}->{''.join(result2)}"
        rho_tensor = np.einsum(einsum2, U.conj(), rho_tensor)

        return rho_tensor.reshape(dim, dim)

    def _apply_kraus(
        self, rho: ndarray, kraus_ops: list[ndarray], qubits: tuple[int, ...], n: int
    ) -> ndarray:
        """Apply Kraus channel: ρ → Σ_k K_k ρ K_k†."""
        new_rho = np.zeros_like(rho)
        for K in kraus_ops:
            full_K = self._embed_gate(K, qubits, n) if len(qubits) == 1 else K
            if len(qubits) == 1:
                new_rho += full_K @ rho @ full_K.conj().T
            else:
                full_K = self._embed_gate(K, qubits, n)
                new_rho += full_K @ rho @ full_K.conj().T
        return new_rho

    def _embed_gate(
        self, gate: ndarray, qubits: tuple[int, ...], n: int
    ) -> ndarray:
        """Embed a k-qubit gate into the full 2^n × 2^n space."""
        dim = 2**n
        k = len(qubits)

        if k == 1:
            # Fast path for single qubit
            q = qubits[0]
            ops = []
            for i in range(n):
                ops.append(gate if i == q else np.eye(2, dtype=np.complex128))
            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return result

        # General multi-qubit embedding
        # Build permutation to move target qubits to front
        perm = list(qubits) + [i for i in range(n) if i not in qubits]
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i

        # Construct gate ⊗ I_{remaining}
        remaining = n - k
        full = np.kron(gate, np.eye(2**remaining, dtype=np.complex128))

        # Permutation matrix
        P = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            bits = [(i >> (n - 1 - j)) & 1 for j in range(n)]
            permuted_bits = [bits[p] for p in perm]
            j = sum(b << (n - 1 - idx) for idx, b in enumerate(permuted_bits))
            P[j, i] = 1.0

        return P.T @ full @ P
