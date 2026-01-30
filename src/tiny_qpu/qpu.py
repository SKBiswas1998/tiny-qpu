"""
Quantum Processing Unit - Main simulator class.

Inspired by tiny-gpu: https://github.com/adam-maj/tiny-gpu
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

@dataclass
class QPUResult:
    """Result of QPU execution."""
    counts: Dict[str, int]
    classical_register: List[int]


class QubitRegister:
    """Quantum state register - holds the state vector."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.reset()
    
    def reset(self):
        """Reset to |00...0⟩ state."""
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0
    
    @property
    def state(self):
        return self._state.copy()
    
    def _qubit_to_bit(self, qubit: int) -> int:
        """Convert qubit index to bit position (handles ordering)."""
        return self.num_qubits - 1 - qubit
    
    def apply_single_gate(self, gate: np.ndarray, qubit: int):
        """Apply single-qubit gate."""
        ops = [np.eye(2, dtype=np.complex128) for _ in range(self.num_qubits)]
        ops[qubit] = gate
        full_gate = ops[0]
        for op in ops[1:]:
            full_gate = np.kron(full_gate, op)
        self._state = full_gate @ self._state
    
    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        ctrl_bit = self._qubit_to_bit(control)
        tgt_bit = self._qubit_to_bit(target)
        
        new_state = np.zeros_like(self._state)
        for i in range(self.dim):
            if (i >> ctrl_bit) & 1:
                j = i ^ (1 << tgt_bit)
                new_state[j] += self._state[i]
            else:
                new_state[i] += self._state[i]
        self._state = new_state
    
    def apply_cz(self, control: int, target: int):
        """Apply CZ gate."""
        ctrl_bit = self._qubit_to_bit(control)
        tgt_bit = self._qubit_to_bit(target)
        
        for i in range(self.dim):
            if ((i >> ctrl_bit) & 1) and ((i >> tgt_bit) & 1):
                self._state[i] *= -1
    
    def measure(self, qubit: int) -> int:
        """Measure a qubit, collapse state, return result."""
        bit_pos = self._qubit_to_bit(qubit)
        
        prob_0 = sum(np.abs(self._state[i])**2 
                     for i in range(self.dim) 
                     if not (i >> bit_pos) & 1)
        
        result = np.random.choice([0, 1], p=[prob_0, 1 - prob_0])
        
        for i in range(self.dim):
            if ((i >> bit_pos) & 1) != result:
                self._state[i] = 0
        
        norm = np.linalg.norm(self._state)
        if norm > 1e-10:
            self._state /= norm
        
        return result


class QPU:
    """
    Quantum Processing Unit simulator.
    
    Example:
        >>> qpu = QPU(num_qubits=2)
        >>> qpu.load_program("programs/bell_state.qasm")
        >>> result = qpu.run(shots=1000)
        >>> print(result.counts)
    """
    
    GATES = {
        'H': np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
        'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        'S': np.array([[1, 0], [0, 1j]], dtype=np.complex128),
        'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
    }
    
    def __init__(self, num_qubits: int = 2, num_classical: Optional[int] = None):
        self.num_qubits = num_qubits
        self.num_classical = num_classical or num_qubits
        self.qreg = QubitRegister(num_qubits)
        self.creg = [0] * self.num_classical
        self.program: List[Dict] = []
    
    def reset(self):
        self.qreg.reset()
        self.creg = [0] * self.num_classical
    
    def load_program(self, filepath: str):
        with open(filepath, 'r') as f:
            self.load_source(f.read())
    
    def load_source(self, source: str):
        self.program = []
        for line in source.split('\n'):
            line = line.split('#')[0].strip()
            if not line or line.startswith('.'):
                continue
            inst = self._parse_instruction(line)
            if inst:
                self.program.append(inst)
    
    def _parse_instruction(self, line: str) -> Optional[Dict]:
        parts = line.replace(',', ' ').split()
        op = parts[0].upper()
        
        def parse_qubit(s):
            m = re.search(r'q\[(\d+)\]', s)
            return int(m.group(1)) if m else int(s)
        
        def parse_classical(s):
            m = re.search(r'c\[(\d+)\]', s)
            return int(m.group(1)) if m else int(s)
        
        if op in ['H', 'X', 'Y', 'Z', 'S', 'T']:
            return {'op': op, 'qubit': parse_qubit(parts[1])}
        elif op in ['CNOT', 'CX']:
            return {'op': 'CNOT', 'control': parse_qubit(parts[1]), 'target': parse_qubit(parts[2])}
        elif op == 'CZ':
            return {'op': 'CZ', 'control': parse_qubit(parts[1]), 'target': parse_qubit(parts[2])}
        elif op == 'MEASURE':
            return {'op': 'MEASURE', 'qubit': parse_qubit(parts[1]), 'classical': parse_classical(parts[2])}
        elif op == 'RESET':
            return {'op': 'RESET', 'qubit': parse_qubit(parts[1])}
        elif op == 'BARRIER':
            return {'op': 'BARRIER'}
        return None
    
    def _execute(self, inst: Dict):
        op = inst['op']
        
        if op in self.GATES:
            self.qreg.apply_single_gate(self.GATES[op], inst['qubit'])
        elif op == 'CNOT':
            self.qreg.apply_cnot(inst['control'], inst['target'])
        elif op == 'CZ':
            self.qreg.apply_cz(inst['control'], inst['target'])
        elif op == 'MEASURE':
            result = self.qreg.measure(inst['qubit'])
            self.creg[inst['classical']] = result
        elif op == 'RESET':
            result = self.qreg.measure(inst['qubit'])
            if result == 1:
                self.qreg.apply_single_gate(self.GATES['X'], inst['qubit'])
    
    def run(self, shots: int = 1024, seed: Optional[int] = None) -> QPUResult:
        if seed is not None:
            np.random.seed(seed)
        
        counts: Dict[str, int] = {}
        
        for _ in range(shots):
            self.reset()
            for inst in self.program:
                self._execute(inst)
            
            bitstring = ''.join(str(b) for b in reversed(self.creg))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return QPUResult(counts=counts, classical_register=self.creg.copy())
    
    def __repr__(self):
        return f"QPU(qubits={self.num_qubits}, classical={self.num_classical})"
