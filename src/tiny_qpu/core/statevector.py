"""
Optimized state vector simulator using tensor contractions.

Key insight: Never build full 2^n x 2^n gate matrices.
Instead, reshape state to (2,2,...,2) tensor and apply gates directly.
This reduces gate application from O(4^n) to O(2^n).
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SimulatorResult:
    """Result of circuit execution."""
    counts: Dict[str, int]
    statevector: Optional[np.ndarray] = None
    
    def most_frequent(self) -> str:
        """Return the most frequently measured state."""
        return max(self.counts, key=self.counts.get)
    
    def probability(self, state: str) -> float:
        """Return probability of a specific state."""
        total = sum(self.counts.values())
        return self.counts.get(state, 0) / total


class StateVector:
    """
    Optimized quantum state vector using tensor representation.
    
    Memory usage: 2^n * 16 bytes (complex128)
    - 10 qubits: 16 KB
    - 15 qubits: 512 KB  
    - 20 qubits: 16 MB
    - 25 qubits: 512 MB
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self._data = np.zeros(self.dim, dtype=np.complex128)
        self._data[0] = 1.0  # |00...0⟩
    
    def reset(self) -> None:
        """Reset to |00...0⟩ state."""
        self._data.fill(0)
        self._data[0] = 1.0
    
    @property
    def tensor(self) -> np.ndarray:
        """Return state as (2,2,...,2) tensor for gate application."""
        return self._data.reshape([2] * self.num_qubits)
    
    @tensor.setter
    def tensor(self, value: np.ndarray) -> None:
        """Set state from tensor."""
        self._data = value.reshape(self.dim)
    
    @property 
    def vector(self) -> np.ndarray:
        """Return flat state vector."""
        return self._data.copy()
    
    def apply_single_gate(self, gate: np.ndarray, qubit: int) -> None:
        """
        Apply single-qubit gate using tensor contraction.
        
        This is O(2^n) instead of O(4^n) for matrix multiplication.
        """
        # Reshape to tensor
        tensor = self.tensor
        
        # Apply gate via einsum (contracts gate with target qubit axis)
        # Move target qubit to first axis, apply gate, move back
        tensor = np.moveaxis(tensor, qubit, 0)
        new_shape = tensor.shape
        tensor = tensor.reshape(2, -1)
        tensor = gate @ tensor
        tensor = tensor.reshape(new_shape)
        tensor = np.moveaxis(tensor, 0, qubit)
        
        self.tensor = tensor
    
    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        """
        Apply two-qubit gate using tensor contraction.
        
        gate: 4x4 matrix reshaped to (2,2,2,2)
        """
        tensor = self.tensor
        gate_tensor = gate.reshape(2, 2, 2, 2)
        
        # Use einsum for efficient contraction
        # This contracts qubits at positions qubit1, qubit2 with gate
        axes = list(range(self.num_qubits))
        
        # Build einsum string dynamically
        input_axes = ''.join(chr(ord('a') + i) for i in range(self.num_qubits))
        q1_char = chr(ord('a') + qubit1)
        q2_char = chr(ord('a') + qubit2)
        
        # Gate indices: new_q1, new_q2, old_q1, old_q2
        gate_str = f"ijkl"  # i,j are new; k,l are old
        
        # Output: replace old qubit chars with new ones
        output_axes = input_axes.replace(q1_char, 'i').replace(q2_char, 'j')
        
        # Contract: gate[i,j,k,l] * state[...k...l...] -> state[...i...j...]
        subscripts = f"{gate_str},{input_axes.replace(q1_char, 'k').replace(q2_char, 'l')}->{output_axes}"
        
        tensor = np.einsum(subscripts, gate_tensor, tensor, optimize=True)
        self.tensor = tensor
    
    def measure_qubit(self, qubit: int) -> int:
        """
        Measure a single qubit, collapse state, return result.
        """
        # Work with flat vector for measurement
        probs_0 = 0.0
        probs_1 = 0.0
        
        # Calculate probabilities
        for i in range(self.dim):
            prob = np.abs(self._data[i]) ** 2
            if (i >> qubit) & 1:
                probs_1 += prob
            else:
                probs_0 += prob
        
        # Normalize probabilities (handle numerical errors)
        total = probs_0 + probs_1
        if total > 0:
            probs_0 /= total
            probs_1 /= total
        else:
            probs_0, probs_1 = 0.5, 0.5
        
        # Sample
        result = np.random.choice([0, 1], p=[probs_0, probs_1])
        
        # Collapse: zero out amplitudes inconsistent with measurement
        for i in range(self.dim):
            if ((i >> qubit) & 1) != result:
                self._data[i] = 0
        
        # Renormalize
        norm = np.linalg.norm(self._data)
        if norm > 1e-15:
            self._data /= norm
        
        return result
    
    def probabilities(self) -> np.ndarray:
        """Return measurement probabilities for all basis states."""
        return np.abs(self._data) ** 2
    
    def sample(self, shots: int = 1024) -> Dict[str, int]:
        """Sample measurement outcomes."""
        probs = self.probabilities()
        indices = np.random.choice(self.dim, size=shots, p=probs)
        
        counts = {}
        for idx in indices:
            bitstring = format(idx, f'0{self.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def expectation(self, observable: np.ndarray) -> float:
        """Calculate expectation value <ψ|O|ψ>."""
        return np.real(np.vdot(self._data, observable @ self._data))
    
    def __repr__(self) -> str:
        return f"StateVector(qubits={self.num_qubits}, dim={self.dim})"
