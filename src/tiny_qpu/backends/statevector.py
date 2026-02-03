"""
Efficient statevector simulation backend.

Uses gate-by-gate state updates via tensor reshaping instead of full
matrix multiplication. This gives O(2^n) per gate rather than O(2^(2n)),
which is the key performance insight from the research.

Memory: ~16 bytes * 2^n (complex128) per state.
    20 qubits = 16 MB, 25 qubits = 512 MB, 30 qubits = 16 GB.

Performance can be further enhanced with:
    - Numba JIT (10-50x on hot loops) — Phase 2
    - Gate fusion for 14+ qubits — Phase 2
    - CuPy GPU backend — Phase 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import ndarray

from tiny_qpu.circuit import Circuit, Instruction, Parameter


@dataclass
class SimulationResult:
    """
    Result of a quantum circuit simulation.

    Attributes
    ----------
    statevector : ndarray
        Final state vector (complex128, length 2^n).
    counts : dict[int, int] | None
        Measurement counts if circuit was measured. Keys are integer
        representations of bitstrings.
    n_qubits : int
        Number of qubits.
    shots : int
        Number of measurement shots (0 if no measurement).
    """

    statevector: ndarray
    counts: dict[int, int] | None
    n_qubits: int
    shots: int = 0

    def probabilities(self) -> dict[int, float]:
        """
        Get measurement probabilities from the statevector.

        Returns
        -------
        dict[int, float]
            Mapping from basis state (int) to probability.
            Only includes states with probability > 1e-10.
        """
        probs = np.abs(self.statevector) ** 2
        return {i: float(p) for i, p in enumerate(probs) if p > 1e-10}

    def probabilities_array(self) -> ndarray:
        """Get probability array for all basis states."""
        return np.abs(self.statevector) ** 2

    def bitstring_counts(self) -> dict[str, int]:
        """
        Get counts as bitstring keys (e.g., '01', '10').

        Returns
        -------
        dict[str, int]
            Measurement results with bitstring keys.
        """
        if self.counts is None:
            return {}
        return {
            format(k, f"0{self.n_qubits}b"): v
            for k, v in sorted(self.counts.items())
        }

    def expectation(self, observable: ndarray) -> complex:
        """
        Compute expectation value <ψ|O|ψ>.

        Parameters
        ----------
        observable : ndarray
            Hermitian operator as a 2^n × 2^n matrix.

        Returns
        -------
        complex
            Expectation value.
        """
        return complex(self.statevector.conj() @ observable @ self.statevector)

    def entropy(self, qubits: list[int] | None = None) -> float:
        """
        Compute von Neumann entropy of the reduced density matrix.

        Parameters
        ----------
        qubits : list[int], optional
            Subsystem qubits. If None, returns 0 (pure state).

        Returns
        -------
        float
            Von Neumann entropy in nats.
        """
        if qubits is None or len(qubits) == self.n_qubits:
            return 0.0  # Pure state

        # Construct reduced density matrix
        psi = self.statevector.reshape([2] * self.n_qubits)
        remaining = [q for q in range(self.n_qubits) if q not in qubits]

        # Full density matrix of subsystem
        rho = np.tensordot(psi, psi.conj(), axes=(remaining, remaining))
        rho = rho.reshape(2 ** len(qubits), 2 ** len(qubits))

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))


class StatevectorBackend:
    """
    Efficient statevector quantum simulator.

    Applies gates via tensor reshaping for O(2^n) per-gate cost
    instead of O(2^(2n)) full matrix multiplication.

    Parameters
    ----------
    seed : int | None
        Random seed for measurement sampling.

    Example
    -------
    >>> from tiny_qpu import Circuit, StatevectorBackend
    >>> qc = Circuit(2)
    >>> qc.h(0).cx(0, 1)
    >>> result = StatevectorBackend(seed=42).run(qc, shots=1000)
    >>> print(result.bitstring_counts())
    {'00': 503, '11': 497}
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        circuit: Circuit,
        shots: int = 0,
        initial_state: ndarray | None = None,
    ) -> SimulationResult:
        """
        Simulate a quantum circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to simulate. Must not have unbound parameters.
        shots : int
            Number of measurement shots. 0 = no sampling (return statevector).
        initial_state : ndarray, optional
            Initial state vector. Defaults to |0...0⟩.

        Returns
        -------
        SimulationResult
            Simulation results including statevector and optional counts.
        """
        n = circuit.n_qubits

        # Initialize state
        if initial_state is not None:
            state = np.array(initial_state, dtype=np.complex128).copy()
            if state.shape != (2**n,):
                raise ValueError(
                    f"Initial state shape {state.shape} != expected ({2**n},)"
                )
        else:
            state = np.zeros(2**n, dtype=np.complex128)
            state[0] = 1.0

        # Track which qubits have been measured and their results
        measured_qubits: dict[int, int | None] = {}
        classical_register = [0] * max(circuit.n_clbits, 1)

        # Apply gates
        for inst in circuit.instructions:
            if inst.name == "barrier":
                continue
            elif inst.name == "measure":
                # Mid-circuit measurement: collapse the state
                qubit = inst.qubits[0]
                clbit = inst.classical_bits[0] if inst.classical_bits else 0
                state, outcome = self._measure_qubit(state, n, qubit)
                if clbit < len(classical_register):
                    classical_register[clbit] = outcome
                measured_qubits[qubit] = outcome
            else:
                matrix = inst.matrix()
                state = self._apply_gate(state, matrix, inst.qubits, n)

        # Sample if requested
        counts = None
        has_measurements = any(
            inst.name == "measure" for inst in circuit.instructions
        )

        if shots > 0:
            if has_measurements:
                # Re-run with sampling
                counts = self._sample_with_measurements(circuit, n, shots, initial_state)
            else:
                # Sample from final statevector
                counts = self._sample(state, n, shots)

        return SimulationResult(
            statevector=state,
            counts=counts,
            n_qubits=n,
            shots=shots,
        )

    def _apply_gate(
        self,
        state: ndarray,
        gate_matrix: ndarray,
        qubits: tuple[int, ...],
        n_qubits: int,
    ) -> ndarray:
        """
        Apply a gate to specific qubits via tensor contraction.

        This is the core performance method. Instead of constructing the
        full 2^n × 2^n operator matrix and multiplying, we:
        1. Reshape state into a rank-n tensor (2×2×...×2)
        2. Reshape gate into appropriate tensor
        3. Contract along the target qubit axes
        4. Reshape back to vector

        Cost: O(2^n × 4^k) where k = number of gate qubits
        vs O(2^(2n)) for full matrix multiply.
        """
        n_gate_qubits = len(qubits)

        if n_gate_qubits == 1:
            return self._apply_single_qubit_gate(state, gate_matrix, qubits[0], n_qubits)
        elif n_gate_qubits == 2:
            return self._apply_two_qubit_gate(state, gate_matrix, qubits, n_qubits)
        else:
            return self._apply_multi_qubit_gate(state, gate_matrix, qubits, n_qubits)

    def _apply_single_qubit_gate(
        self, state: ndarray, gate: ndarray, qubit: int, n: int
    ) -> ndarray:
        """
        Apply single-qubit gate using reshape + tensordot.

        Reshape state to (2^a, 2, 2^b) where a = qubit index,
        b = n - qubit - 1. Contract gate along axis 1.
        """
        shape = [2] * n
        state = state.reshape(shape)

        # Use einsum for clean single-qubit application
        # Contract gate[i,j] with state[..., j, ...] along qubit axis
        axes = list(range(n))
        # Build einsum string
        state_indices = list("abcdefghijklmnopqrst"[:n])
        gate_out = "z"
        gate_in = state_indices[qubit]
        result_indices = state_indices.copy()
        result_indices[qubit] = gate_out

        einsum_str = (
            f"{gate_out}{gate_in},{''.join(state_indices)}->{''.join(result_indices)}"
        )
        state = np.einsum(einsum_str, gate, state)
        return state.reshape(2**n)

    def _apply_two_qubit_gate(
        self, state: ndarray, gate: ndarray, qubits: tuple[int, ...], n: int
    ) -> ndarray:
        """
        Apply two-qubit gate using reshape + einsum.

        Gate is 4×4, reshaped to (2,2,2,2). Contract with state tensor
        along the two target qubit axes.
        """
        q0, q1 = qubits
        gate_tensor = gate.reshape(2, 2, 2, 2)  # [out0, out1, in0, in1]

        shape = [2] * n
        state = state.reshape(shape)

        state_indices = list("abcdefghijklmnopqrst"[:n])
        out0, out1 = "y", "z"
        in0, in1 = state_indices[q0], state_indices[q1]

        result_indices = state_indices.copy()
        result_indices[q0] = out0
        result_indices[q1] = out1

        einsum_str = (
            f"{out0}{out1}{in0}{in1},"
            f"{''.join(state_indices)}->{''.join(result_indices)}"
        )
        state = np.einsum(einsum_str, gate_tensor, state)
        return state.reshape(2**n)

    def _apply_multi_qubit_gate(
        self, state: ndarray, gate: ndarray, qubits: tuple[int, ...], n: int
    ) -> ndarray:
        """
        Apply k-qubit gate using general einsum contraction.
        """
        k = len(qubits)
        gate_tensor = gate.reshape([2] * (2 * k))  # [out0,..,outk-1, in0,..,ink-1]

        shape = [2] * n
        state = state.reshape(shape)

        state_indices = list("abcdefghijklmnopqrst"[:n])
        # Assign output indices for gate qubits
        out_labels = list("ABCDEFGHIJ"[:k])
        in_labels = [state_indices[q] for q in qubits]

        result_indices = state_indices.copy()
        for i, q in enumerate(qubits):
            result_indices[q] = out_labels[i]

        gate_str = "".join(out_labels) + "".join(in_labels)
        einsum_str = f"{gate_str},{''.join(state_indices)}->{''.join(result_indices)}"
        state = np.einsum(einsum_str, gate_tensor, state)
        return state.reshape(2**n)

    def _measure_qubit(
        self, state: ndarray, n: int, qubit: int
    ) -> tuple[ndarray, int]:
        """
        Perform projective measurement on a single qubit.

        Returns the collapsed state and the measurement outcome.
        """
        probs = np.abs(state) ** 2

        # Probability of measuring 0 on this qubit
        mask_0 = np.zeros(2**n, dtype=bool)
        for i in range(2**n):
            if not (i >> (n - 1 - qubit)) & 1:
                mask_0[i] = True

        p0 = np.sum(probs[mask_0])
        outcome = 0 if self._rng.random() < p0 else 1

        # Collapse
        new_state = np.zeros_like(state)
        if outcome == 0:
            new_state[mask_0] = state[mask_0]
            norm = np.sqrt(p0)
        else:
            mask_1 = ~mask_0
            new_state[mask_1] = state[mask_1]
            norm = np.sqrt(1 - p0)

        if norm > 1e-15:
            new_state /= norm

        return new_state, outcome

    def _sample(self, state: ndarray, n: int, shots: int) -> dict[int, int]:
        """Sample measurement outcomes from statevector."""
        probs = np.abs(state) ** 2
        # Normalize to handle floating point
        probs /= probs.sum()
        outcomes = self._rng.choice(2**n, size=shots, p=probs)
        unique, counts_arr = np.unique(outcomes, return_counts=True)
        return dict(zip(unique.tolist(), counts_arr.tolist()))

    def _sample_with_measurements(
        self,
        circuit: Circuit,
        n: int,
        shots: int,
        initial_state: ndarray | None,
    ) -> dict[int, int]:
        """Re-run circuit multiple times for proper measurement statistics."""
        all_counts: dict[int, int] = {}

        for _ in range(shots):
            # Fresh state each shot
            if initial_state is not None:
                state = initial_state.copy()
            else:
                state = np.zeros(2**n, dtype=np.complex128)
                state[0] = 1.0

            classical_bits = [0] * max(circuit.n_clbits, 1)

            for inst in circuit.instructions:
                if inst.name == "barrier":
                    continue
                elif inst.name == "measure":
                    qubit = inst.qubits[0]
                    clbit = inst.classical_bits[0] if inst.classical_bits else 0
                    state, outcome = self._measure_qubit(state, n, qubit)
                    if clbit < len(classical_bits):
                        classical_bits[clbit] = outcome
                else:
                    matrix = inst.matrix()
                    state = self._apply_gate(state, matrix, inst.qubits, n)

            # Convert classical register to integer
            result_int = 0
            for i, bit in enumerate(classical_bits[:circuit.n_clbits]):
                result_int |= bit << (circuit.n_clbits - 1 - i)

            all_counts[result_int] = all_counts.get(result_int, 0) + 1

        return all_counts

    # -- Utility methods ----------------------------------------------------

    def statevector(self, circuit: Circuit) -> ndarray:
        """Convenience: run circuit and return just the state vector."""
        return self.run(circuit).statevector

    def expectation_value(
        self, circuit: Circuit, observable: ndarray
    ) -> float:
        """
        Compute ⟨ψ|O|ψ⟩ for a circuit and Hermitian observable.

        Parameters
        ----------
        circuit : Circuit
            Circuit preparing the state |ψ⟩.
        observable : ndarray
            Hermitian operator as 2^n × 2^n matrix.

        Returns
        -------
        float
            Real part of the expectation value.
        """
        result = self.run(circuit)
        return result.expectation(observable).real

    def fidelity(self, state1: ndarray, state2: ndarray) -> float:
        """Compute state fidelity |⟨ψ₁|ψ₂⟩|²."""
        return float(np.abs(np.vdot(state1, state2)) ** 2)
