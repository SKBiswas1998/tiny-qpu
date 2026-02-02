"""
Quantum Noise Simulator
========================
Simulates real hardware noise using density matrix representation.

Noise Models:
- Depolarizing: Random Pauli errors (quantum "static")
- Amplitude Damping: Energy loss (T1 decay)  
- Phase Damping: Phase loss (T2 dephasing)
- Bit Flip: Classical bit errors
- Phase Flip: Phase errors
- Readout Error: Measurement mistakes
- Thermal Relaxation: Combined T1/T2 decay

Usage:
    from tiny_qpu.noise import NoiseModel, depolarizing, amplitude_damping
    
    noise = NoiseModel()
    noise.add_all_qubit_error(depolarizing(0.01))
    noise.add_readout_error(0.02)
    
    qc = Circuit(2).h(0).cx(0,1).measure_all()
    result = noise.run(qc, shots=1000)
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from ..core import Circuit
from ..core.statevector import SimulatorResult


class QuantumChannel:
    """
    Quantum noise channel defined by Kraus operators.
    
    A channel E(rho) = sum_i K_i rho K_i^dag where sum_i K_i^dag K_i = I
    """
    
    def __init__(self, kraus_ops: List[np.ndarray], name: str = "channel"):
        self.kraus_ops = [np.array(k, dtype=np.complex128) for k in kraus_ops]
        self.name = name
        self._validate()
    
    def _validate(self):
        total = sum(k.conj().T @ k for k in self.kraus_ops)
        dim = self.kraus_ops[0].shape[0]
        if not np.allclose(total, np.eye(dim), atol=1e-6):
            raise ValueError(f"Kraus operators don't sum to I")
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        result = np.zeros_like(rho)
        for k in self.kraus_ops:
            result += k @ rho @ k.conj().T
        return result
    
    @property
    def num_qubits(self) -> int:
        dim = self.kraus_ops[0].shape[0]
        n = 0
        while (1 << n) < dim:
            n += 1
        return n
    
    def __repr__(self):
        return f"QuantumChannel('{self.name}', {len(self.kraus_ops)} Kraus ops)"


# =============================================================================
# Pre-built Noise Channels
# =============================================================================

def depolarizing(p: float) -> QuantumChannel:
    """
    Depolarizing channel: with probability p, replace qubit with maximally mixed state.
    
    E(rho) = (1-p)rho + (p/3)(X rho X + Y rho Y + Z rho Z)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    kraus = [
        np.sqrt(1 - 3*p/4) * I,
        np.sqrt(p/4) * X,
        np.sqrt(p/4) * Y,
        np.sqrt(p/4) * Z,
    ]
    return QuantumChannel(kraus, f"depolarizing(p={p})")


def amplitude_damping(gamma: float) -> QuantumChannel:
    """
    Amplitude damping: models energy dissipation (T1 decay).
    |1> decays to |0> with probability gamma.
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma must be in [0,1], got {gamma}")
    
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    
    return QuantumChannel([K0, K1], f"amplitude_damping(g={gamma})")


def phase_damping(gamma: float) -> QuantumChannel:
    """
    Phase damping: models dephasing without energy loss (T2 process).
    Off-diagonal elements decay by sqrt(1-gamma).
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma must be in [0,1], got {gamma}")
    
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)
    
    return QuantumChannel([K0, K1], f"phase_damping(g={gamma})")


def bit_flip(p: float) -> QuantumChannel:
    """Bit flip channel: X gate applied with probability p."""
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return QuantumChannel([np.sqrt(1-p)*I, np.sqrt(p)*X], f"bit_flip(p={p})")


def phase_flip(p: float) -> QuantumChannel:
    """Phase flip channel: Z gate applied with probability p."""
    I = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return QuantumChannel([np.sqrt(1-p)*I, np.sqrt(p)*Z], f"phase_flip(p={p})")


def depolarizing_2q(p: float) -> QuantumChannel:
    """Two-qubit depolarizing channel."""
    if not 0 <= p <= 1:
        raise ValueError(f"Probability must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    paulis_1q = [I, X, Y, Z]
    kraus = []
    for i, p1 in enumerate(paulis_1q):
        for j, p2 in enumerate(paulis_1q):
            if i == 0 and j == 0:
                kraus.append(np.sqrt(1 - 15*p/16) * np.kron(p1, p2))
            else:
                kraus.append(np.sqrt(p/16) * np.kron(p1, p2))
    
    return QuantumChannel(kraus, f"depolarizing_2q(p={p})")


def thermal_relaxation(t1: float, t2: float, gate_time: float) -> QuantumChannel:
    """
    Thermal relaxation combining T1 and T2 processes.
    Models real hardware where T2 <= 2*T1.
    """
    if t2 > 2 * t1:
        raise ValueError(f"T2 ({t2}) cannot exceed 2*T1 ({2*t1})")
    
    p_reset = 1 - np.exp(-gate_time / t1)
    
    if t2 < 2 * t1:
        t_phi = 1.0 / (1.0/t2 - 1.0/(2*t1))
        p_phase = 1 - np.exp(-gate_time / t_phi)
    else:
        p_phase = 0.0
    
    ad = amplitude_damping(p_reset)
    pd = phase_damping(p_phase)
    
    kraus = []
    for a in ad.kraus_ops:
        for ph in pd.kraus_ops:
            k = ph @ a
            if np.linalg.norm(k) > 1e-12:
                kraus.append(k)
    
    return QuantumChannel(kraus, f"thermal_relaxation(T1={t1}, T2={t2})")


# =============================================================================
# Density Matrix Simulator
# =============================================================================

class DensityMatrix:
    """
    Density matrix representation for mixed quantum states.
    Supports noise simulation via quantum channels.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self._rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self._rho[0, 0] = 1.0
    
    @classmethod
    def from_statevector(cls, state: np.ndarray) -> 'DensityMatrix':
        """Create density matrix from pure state |psi><psi|."""
        n = int(np.log2(len(state)))
        dm = cls(n)
        dm._rho = np.outer(state, state.conj())
        return dm
    
    @property
    def matrix(self) -> np.ndarray:
        return self._rho.copy()
    
    def purity(self) -> float:
        """Tr(rho^2) - 1 for pure, 1/d for maximally mixed."""
        return float(np.real(np.trace(self._rho @ self._rho)))
    
    def entropy(self) -> float:
        """Von Neumann entropy: -Tr(rho log2 rho)"""
        eigenvalues = np.linalg.eigvalsh(self._rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    
    def fidelity_to_pure(self, pure_state: np.ndarray) -> float:
        """Fidelity F = <psi|rho|psi>."""
        return float(np.real(pure_state.conj() @ self._rho @ pure_state))
    
    def apply_gate(self, gate: np.ndarray, qubits: List[int]) -> None:
        """Apply unitary gate: rho -> U rho U^dag"""
        U = self._expand_gate(gate, qubits)
        self._rho = U @ self._rho @ U.conj().T
    
    def apply_channel(self, channel: QuantumChannel, qubits: List[int]) -> None:
        """Apply noise channel to specific qubits."""
        new_rho = np.zeros_like(self._rho)
        for kraus in channel.kraus_ops:
            K = self._expand_gate(kraus, qubits)
            new_rho += K @ self._rho @ K.conj().T
        self._rho = new_rho
    
    def _expand_gate(self, gate: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Expand gate on specific qubits to full system."""
        n = self.num_qubits
        gate_n = len(qubits)
        gate_dim = 2 ** gate_n
        
        full = np.zeros((self.dim, self.dim), dtype=np.complex128)
        gate_2d = gate.reshape(gate_dim, gate_dim)
        
        for i in range(self.dim):
            for j in range(self.dim):
                bits_i = [(i >> (n-1-k)) & 1 for k in range(n)]
                bits_j = [(j >> (n-1-k)) & 1 for k in range(n)]
                
                # Non-target qubits must match
                match = True
                for k in range(n):
                    if k not in qubits and bits_i[k] != bits_j[k]:
                        match = False
                        break
                
                if match:
                    # Get target qubit indices
                    gi = sum(bits_i[qubits[k]] << (gate_n-1-k) for k in range(gate_n))
                    gj = sum(bits_j[qubits[k]] << (gate_n-1-k) for k in range(gate_n))
                    full[i, j] = gate_2d[gi, gj]
        
        return full
    
    def probabilities(self) -> np.ndarray:
        return np.real(np.diag(self._rho))
    
    def sample(self, shots: int = 1024) -> Dict[str, int]:
        probs = self.probabilities()
        probs = np.abs(probs)
        probs = probs / probs.sum()
        
        indices = np.random.choice(self.dim, size=shots, p=probs)
        unique, count_arr = np.unique(indices, return_counts=True)
        
        counts = {}
        for idx, count in zip(unique, count_arr):
            bitstring = format(idx, f'0{self.num_qubits}b')
            counts[bitstring] = int(count)
        return counts
    
    def partial_trace(self, keep_qubits: List[int]) -> 'DensityMatrix':
        """Trace out qubits NOT in keep_qubits."""
        n = self.num_qubits
        trace_out = [q for q in range(n) if q not in keep_qubits]
        n_keep = len(keep_qubits)
        n_trace = len(trace_out)
        dim_keep = 2 ** n_keep
        
        reduced = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        
        for i in range(dim_keep):
            for j in range(dim_keep):
                bits_i = [(i >> (n_keep-1-k)) & 1 for k in range(n_keep)]
                bits_j = [(j >> (n_keep-1-k)) & 1 for k in range(n_keep)]
                
                for t in range(2**n_trace):
                    trace_bits = [(t >> (n_trace-1-k)) & 1 for k in range(n_trace)]
                    
                    full_i = [0] * n
                    full_j = [0] * n
                    for k, q in enumerate(keep_qubits):
                        full_i[q] = bits_i[k]
                        full_j[q] = bits_j[k]
                    for k, q in enumerate(trace_out):
                        full_i[q] = trace_bits[k]
                        full_j[q] = trace_bits[k]
                    
                    idx_i = sum(b << (n-1-pos) for pos, b in enumerate(full_i))
                    idx_j = sum(b << (n-1-pos) for pos, b in enumerate(full_j))
                    reduced[i, j] += self._rho[idx_i, idx_j]
        
        result = DensityMatrix(n_keep)
        result._rho = reduced
        return result
    
    def concurrence(self) -> float:
        """Concurrence (entanglement measure) for 2-qubit state."""
        if self.num_qubits != 2:
            raise ValueError("Concurrence only defined for 2-qubit states")
        
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        YY = np.kron(Y, Y)
        rho_tilde = YY @ self._rho.conj() @ YY
        R = self._rho @ rho_tilde
        eigenvalues = np.sort(np.real(np.sqrt(np.abs(np.linalg.eigvals(R)))))[::-1]
        return float(max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3]))
    
    def __repr__(self):
        return f"DensityMatrix(qubits={self.num_qubits}, purity={self.purity():.4f})"


# =============================================================================
# Noise Model
# =============================================================================

class NoiseModel:
    """
    Configurable noise model for quantum circuits.
    
    Example:
        noise = NoiseModel()
        noise.add_all_qubit_error(depolarizing(0.01))
        noise.add_gate_error('cx', depolarizing_2q(0.02))
        noise.add_readout_error(0.03)
        result = noise.run(circuit, shots=1000)
    """
    
    def __init__(self):
        self._single_qubit_errors: List[QuantumChannel] = []
        self._two_qubit_errors: List[QuantumChannel] = []
        self._gate_errors: Dict[str, QuantumChannel] = {}
        self._readout_error: float = 0.0
        self._qubit_errors: Dict[int, List[QuantumChannel]] = {}
    
    def add_all_qubit_error(self, channel: QuantumChannel) -> 'NoiseModel':
        """Add noise applied after every single-qubit gate."""
        if channel.num_qubits == 1:
            self._single_qubit_errors.append(channel)
        elif channel.num_qubits == 2:
            self._two_qubit_errors.append(channel)
        return self
    
    def add_gate_error(self, gate_name: str, channel: QuantumChannel) -> 'NoiseModel':
        """Add noise for a specific gate type."""
        self._gate_errors[gate_name.upper()] = channel
        return self
    
    def add_qubit_error(self, qubit: int, channel: QuantumChannel) -> 'NoiseModel':
        """Add noise for a specific qubit."""
        if qubit not in self._qubit_errors:
            self._qubit_errors[qubit] = []
        self._qubit_errors[qubit].append(channel)
        return self
    
    def add_readout_error(self, p: float) -> 'NoiseModel':
        """Add measurement readout error probability."""
        self._readout_error = p
        return self
    
    @staticmethod
    def from_backend(t1: float = 50e3, t2: float = 70e3,
                     gate_time_1q: float = 50, gate_time_2q: float = 300,
                     readout_error: float = 0.02) -> 'NoiseModel':
        """
        Create noise model mimicking real hardware.
        
        Default values approximate IBM superconducting qubits:
            T1 = 50 us, T2 = 70 us, 1q gate = 50 ns, 2q gate = 300 ns
        
        Args:
            t1: T1 relaxation time (ns)
            t2: T2 dephasing time (ns)
            gate_time_1q: Single-qubit gate duration (ns)
            gate_time_2q: Two-qubit gate duration (ns)
            readout_error: Measurement error probability
        """
        noise = NoiseModel()
        noise.add_all_qubit_error(thermal_relaxation(t1, t2, gate_time_1q))
        noise.add_gate_error('CX', depolarizing_2q(0.01))
        noise.add_readout_error(readout_error)
        return noise
    
    def run(self, circuit: Circuit, shots: int = 1024,
            seed: Optional[int] = None) -> SimulatorResult:
        """Run circuit with noise using density matrix simulation."""
        if seed is not None:
            np.random.seed(seed)
        
        from ..core import gates as g
        
        dm = DensityMatrix(circuit.num_qubits)
        
        gate_map = {
            'H': g.H, 'X': g.X, 'Y': g.Y, 'Z': g.Z,
            'S': g.S, 'T': g.T, 'SDG': g.S_DAG, 'TDG': g.T_DAG,
            'SX': g.SX, 'CX': g.CNOT, 'CY': g.CY, 'CZ': g.CZ,
            'SWAP': g.SWAP, 'CCX': g.TOFFOLI,
        }
        
        for op in circuit._operations:
            if op.name == 'MEASURE' or op.name == 'BARRIER':
                continue
            
            if op.name in gate_map:
                gate_matrix = gate_map[op.name]
            elif op.name == 'RX':
                gate_matrix = g.Rx(op.params[0])
            elif op.name == 'RY':
                gate_matrix = g.Ry(op.params[0])
            elif op.name == 'RZ':
                gate_matrix = g.Rz(op.params[0])
            elif op.name == 'P':
                gate_matrix = g.P(op.params[0])
            elif op.name == 'CRZ':
                gate_matrix = g.CRz(op.params[0])
            elif op.name == 'CP':
                gate_matrix = g.CP(op.params[0])
            elif op.name == 'RXX':
                gate_matrix = g.RXX(op.params[0])
            elif op.name == 'RYY':
                gate_matrix = g.RYY(op.params[0])
            elif op.name == 'RZZ':
                gate_matrix = g.RZZ(op.params[0])
            else:
                continue
            
            dm.apply_gate(gate_matrix, op.qubits)
            
            # Apply noise after gate
            # Gate-specific errors (e.g., CX depolarizing)
            if op.name in self._gate_errors:
                dm.apply_channel(self._gate_errors[op.name], op.qubits)
            
            # 2-qubit correlated errors
            if len(op.qubits) == 2:
                for error in self._two_qubit_errors:
                    dm.apply_channel(error, op.qubits)
            
            # Single-qubit errors on EVERY qubit involved (matches real hardware)
            for qubit in op.qubits:
                for error in self._single_qubit_errors:
                    dm.apply_channel(error, [qubit])
                if qubit in self._qubit_errors:
                    for error in self._qubit_errors[qubit]:
                        dm.apply_channel(error, [qubit])
        
        # Sample with readout error
        counts = dm.sample(shots)
        
        if self._readout_error > 0:
            noisy_counts: Dict[str, int] = {}
            for bitstring, count in counts.items():
                for _ in range(count):
                    noisy_bits = list(bitstring)
                    for i in range(len(noisy_bits)):
                        if np.random.random() < self._readout_error:
                            noisy_bits[i] = '1' if noisy_bits[i] == '0' else '0'
                    noisy_str = ''.join(noisy_bits)
                    noisy_counts[noisy_str] = noisy_counts.get(noisy_str, 0) + 1
            counts = noisy_counts
        
        return SimulatorResult(counts=counts)
    
    def __repr__(self):
        parts = []
        if self._single_qubit_errors:
            parts.append(f"{len(self._single_qubit_errors)} 1q errors")
        if self._two_qubit_errors:
            parts.append(f"{len(self._two_qubit_errors)} 2q errors")
        if self._gate_errors:
            parts.append(f"{len(self._gate_errors)} gate errors")
        if self._readout_error > 0:
            parts.append(f"readout={self._readout_error:.1%}")
        return f"NoiseModel({', '.join(parts) or 'clean'})"

