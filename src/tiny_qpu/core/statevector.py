"""
Optimized state vector simulator using tensor contractions.

Key insight: Never build full 2^n x 2^n gate matrices.
Instead, reshape state to (2,2,...,2) tensor and apply gates directly.
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SimulatorResult:
    """Result of circuit execution."""
    counts: Dict[str, int]
    statevector: Optional[np.ndarray] = None
    
    def most_frequent(self) -> str:
        return max(self.counts, key=self.counts.get)
    
    def probability(self, state: str) -> float:
        total = sum(self.counts.values())
        return self.counts.get(state, 0) / total


class StateVector:
    """
    Optimized quantum state vector using tensor representation.
    
    Supports up to 26 qubits (limited by einsum indices).
    Memory: 2^n * 16 bytes (complex128)
    """
    
    def __init__(self, num_qubits: int):
        if num_qubits > 26:
            raise ValueError("Maximum 26 qubits supported")
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self._data = np.zeros(self.dim, dtype=np.complex128)
        self._data[0] = 1.0
    
    def reset(self) -> None:
        self._data.fill(0)
        self._data[0] = 1.0
    
    @property
    def tensor(self) -> np.ndarray:
        return self._data.reshape([2] * self.num_qubits)
    
    @tensor.setter
    def tensor(self, value: np.ndarray) -> None:
        self._data = value.reshape(self.dim)
    
    @property 
    def vector(self) -> np.ndarray:
        return self._data.copy()
    
    def apply_single_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply single-qubit gate using tensor contraction."""
        tensor = self.tensor
        tensor = np.moveaxis(tensor, qubit, 0)
        new_shape = tensor.shape
        tensor = tensor.reshape(2, -1)
        tensor = gate @ tensor
        tensor = tensor.reshape(new_shape)
        tensor = np.moveaxis(tensor, 0, qubit)
        self.tensor = tensor
    
    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        """Apply two-qubit gate using tensor contraction."""
        tensor = self.tensor
        gate_tensor = gate.reshape(2, 2, 2, 2)
        
        n = self.num_qubits
        state_in = list(range(n))
        state_out = list(range(n))
        gate_idx = [n, n+1, n+2, n+3]
        
        state_in[qubit1] = n+2
        state_in[qubit2] = n+3
        state_out[qubit1] = n
        state_out[qubit2] = n+1
        
        tensor = np.einsum(gate_tensor, gate_idx, tensor, state_in, state_out, optimize=True)
        self.tensor = tensor
    
    def apply_three_qubit_gate(self, gate: np.ndarray, q0: int, q1: int, q2: int) -> None:
        """Apply three-qubit gate (e.g., Toffoli, Fredkin)."""
        tensor = self.tensor
        gate_tensor = gate.reshape(2, 2, 2, 2, 2, 2)
        
        n = self.num_qubits
        state_in = list(range(n))
        state_out = list(range(n))
        gate_idx = [n, n+1, n+2, n+3, n+4, n+5]
        
        state_in[q0] = n+3
        state_in[q1] = n+4
        state_in[q2] = n+5
        state_out[q0] = n
        state_out[q1] = n+1
        state_out[q2] = n+2
        
        tensor = np.einsum(gate_tensor, gate_idx, tensor, state_in, state_out, optimize=True)
        self.tensor = tensor
    
    def measure_qubit(self, qubit: int) -> int:
        """
        Measure a single qubit, collapse state, return result.
        
        Optimized with NumPy vectorization.
        """
        # Reshape to tensor to easily sum over the qubit axis
        tensor = self.tensor
        probs = np.abs(tensor) ** 2
        
        # Sum probabilities for qubit=0 and qubit=1
        # Move target qubit to first axis, then sum over all other axes
        probs = np.moveaxis(probs, qubit, 0)
        prob_0 = probs[0].sum()
        prob_1 = probs[1].sum()
        
        # Normalize
        total = prob_0 + prob_1
        if total > 1e-15:
            prob_0 /= total
            prob_1 /= total
        else:
            prob_0, prob_1 = 0.5, 0.5
        
        # Sample result
        result = 1 if np.random.random() < prob_1 else 0
        
        # Collapse: zero out inconsistent amplitudes
        tensor = np.moveaxis(self.tensor, qubit, 0)
        tensor[1 - result] = 0
        tensor = np.moveaxis(tensor, 0, qubit)
        
        # Renormalize
        norm = np.linalg.norm(tensor)
        if norm > 1e-15:
            tensor = tensor / norm
        
        self.tensor = tensor
        return result
    
    def probabilities(self) -> np.ndarray:
        return np.abs(self._data) ** 2
    
    def sample(self, shots: int = 1024) -> Dict[str, int]:
        """Sample measurement outcomes efficiently."""
        probs = self.probabilities()
        
        # Normalize to handle numerical errors
        probs = probs / probs.sum()
        
        indices = np.random.choice(self.dim, size=shots, p=probs)
        
        # Count using numpy
        unique, counts_arr = np.unique(indices, return_counts=True)
        
        counts = {}
        for idx, count in zip(unique, counts_arr):
            bitstring = format(idx, f'0{self.num_qubits}b')
            counts[bitstring] = int(count)
        
        return counts
    
    def expectation(self, observable: np.ndarray) -> float:
        return np.real(np.vdot(self._data, observable @ self._data))
    
    def __repr__(self) -> str:
        return f"StateVector(qubits={self.num_qubits}, dim={self.dim})"
