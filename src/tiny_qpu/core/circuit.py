"""
Quantum Circuit - the main user-facing API.

Supports fluent/chained API: Circuit(2).h(0).cx(0,1).measure_all()
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from .statevector import StateVector, SimulatorResult
from . import gates


@dataclass
class Operation:
    """A quantum operation (gate or measurement)."""
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    classical: Optional[int] = None  # For measurement


class Circuit:
    """
    Quantum Circuit with fluent API.
    
    Example:
        >>> qc = Circuit(2)
        >>> qc.h(0).cx(0, 1).measure_all()
        >>> result = qc.run(shots=1000)
        >>> print(result.counts)  # {'00': 500, '11': 500}
    
    Or use context manager for educational mode:
        >>> with Circuit(2, educational=True) as qc:
        ...     qc.h(0)  # Prints state after each gate
        ...     qc.cx(0, 1)
    """
    
    def __init__(self, num_qubits: int, num_classical: Optional[int] = None, 
                 educational: bool = False):
        """
        Create a quantum circuit.
        
        Args:
            num_qubits: Number of qubits
            num_classical: Number of classical bits (default: num_qubits)
            educational: If True, print state after each gate
        """
        self.num_qubits = num_qubits
        self.num_classical = num_classical or num_qubits
        self.educational = educational
        self._operations: List[Operation] = []
        self._state: Optional[StateVector] = None
        
    # =========================================================================
    # SINGLE-QUBIT GATES
    # =========================================================================
    
    def i(self, qubit: int) -> 'Circuit':
        """Identity gate (no-op)."""
        self._add_gate('I', (qubit,))
        return self
    
    def x(self, qubit: int) -> 'Circuit':
        """Pauli-X (NOT) gate."""
        self._add_gate('X', (qubit,))
        return self
    
    def y(self, qubit: int) -> 'Circuit':
        """Pauli-Y gate."""
        self._add_gate('Y', (qubit,))
        return self
    
    def z(self, qubit: int) -> 'Circuit':
        """Pauli-Z gate."""
        self._add_gate('Z', (qubit,))
        return self
    
    def h(self, qubit: int) -> 'Circuit':
        """Hadamard gate."""
        self._add_gate('H', (qubit,))
        return self
    
    def s(self, qubit: int) -> 'Circuit':
        """S (√Z) gate."""
        self._add_gate('S', (qubit,))
        return self
    
    def sdg(self, qubit: int) -> 'Circuit':
        """S† gate."""
        self._add_gate('SDG', (qubit,))
        return self
    
    def t(self, qubit: int) -> 'Circuit':
        """T (π/8) gate."""
        self._add_gate('T', (qubit,))
        return self
    
    def tdg(self, qubit: int) -> 'Circuit':
        """T† gate."""
        self._add_gate('TDG', (qubit,))
        return self
    
    def sx(self, qubit: int) -> 'Circuit':
        """√X gate."""
        self._add_gate('SX', (qubit,))
        return self
    
    def rx(self, theta: float, qubit: int) -> 'Circuit':
        """Rotation around X-axis."""
        self._add_gate('RX', (qubit,), (theta,))
        return self
    
    def ry(self, theta: float, qubit: int) -> 'Circuit':
        """Rotation around Y-axis."""
        self._add_gate('RY', (qubit,), (theta,))
        return self
    
    def rz(self, theta: float, qubit: int) -> 'Circuit':
        """Rotation around Z-axis."""
        self._add_gate('RZ', (qubit,), (theta,))
        return self
    
    def p(self, phi: float, qubit: int) -> 'Circuit':
        """Phase gate."""
        self._add_gate('P', (qubit,), (phi,))
        return self
    
    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> 'Circuit':
        """General single-qubit unitary."""
        self._add_gate('U3', (qubit,), (theta, phi, lam))
        return self
    
    # =========================================================================
    # TWO-QUBIT GATES
    # =========================================================================
    
    def cx(self, control: int, target: int) -> 'Circuit':
        """CNOT (controlled-X) gate."""
        self._add_gate('CX', (control, target))
        return self
    
    def cnot(self, control: int, target: int) -> 'Circuit':
        """Alias for cx."""
        return self.cx(control, target)
    
    def cy(self, control: int, target: int) -> 'Circuit':
        """Controlled-Y gate."""
        self._add_gate('CY', (control, target))
        return self
    
    def cz(self, control: int, target: int) -> 'Circuit':
        """Controlled-Z gate."""
        self._add_gate('CZ', (control, target))
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'Circuit':
        """SWAP gate."""
        self._add_gate('SWAP', (qubit1, qubit2))
        return self
    
    def iswap(self, qubit1: int, qubit2: int) -> 'Circuit':
        """iSWAP gate."""
        self._add_gate('ISWAP', (qubit1, qubit2))
        return self
    
    def crx(self, theta: float, control: int, target: int) -> 'Circuit':
        """Controlled Rx rotation."""
        self._add_gate('CRX', (control, target), (theta,))
        return self
    
    def cry(self, theta: float, control: int, target: int) -> 'Circuit':
        """Controlled Ry rotation."""
        self._add_gate('CRY', (control, target), (theta,))
        return self
    
    def crz(self, theta: float, control: int, target: int) -> 'Circuit':
        """Controlled Rz rotation."""
        self._add_gate('CRZ', (control, target), (theta,))
        return self
    
    def cp(self, phi: float, control: int, target: int) -> 'Circuit':
        """Controlled phase gate."""
        self._add_gate('CP', (control, target), (phi,))
        return self
    
    def rxx(self, theta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """XX interaction gate."""
        self._add_gate('RXX', (qubit1, qubit2), (theta,))
        return self
    
    def ryy(self, theta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """YY interaction gate."""
        self._add_gate('RYY', (qubit1, qubit2), (theta,))
        return self
    
    def rzz(self, theta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """ZZ interaction gate."""
        self._add_gate('RZZ', (qubit1, qubit2), (theta,))
        return self
    
    # =========================================================================
    # THREE-QUBIT GATES
    # =========================================================================
    
    def ccx(self, control1: int, control2: int, target: int) -> 'Circuit':
        """Toffoli (CCX) gate."""
        self._add_gate('CCX', (control1, control2, target))
        return self
    
    def toffoli(self, control1: int, control2: int, target: int) -> 'Circuit':
        """Alias for ccx."""
        return self.ccx(control1, control2, target)
    
    def cswap(self, control: int, target1: int, target2: int) -> 'Circuit':
        """Fredkin (CSWAP) gate."""
        self._add_gate('CSWAP', (control, target1, target2))
        return self
    
    def fredkin(self, control: int, target1: int, target2: int) -> 'Circuit':
        """Alias for cswap."""
        return self.cswap(control, target1, target2)
    
    # =========================================================================
    # MEASUREMENT
    # =========================================================================
    
    def measure(self, qubit: int, classical: Optional[int] = None) -> 'Circuit':
        """Measure a qubit into a classical bit."""
        if classical is None:
            classical = qubit
        self._operations.append(Operation('MEASURE', (qubit,), classical=classical))
        return self
    
    def measure_all(self) -> 'Circuit':
        """Measure all qubits."""
        for i in range(self.num_qubits):
            self.measure(i, i)
        return self
    
    # =========================================================================
    # SPECIAL OPERATIONS
    # =========================================================================
    
    def barrier(self) -> 'Circuit':
        """Add a barrier (visual separator, no effect on simulation)."""
        self._operations.append(Operation('BARRIER', ()))
        return self
    
    def reset(self, qubit: int) -> 'Circuit':
        """Reset qubit to |0⟩."""
        self._operations.append(Operation('RESET', (qubit,)))
        return self
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    def run(self, shots: int = 1024, seed: Optional[int] = None) -> SimulatorResult:
        """
        Execute the circuit.
        
        Args:
            shots: Number of measurement samples
            seed: Random seed for reproducibility
            
        Returns:
            SimulatorResult with counts and optional statevector
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Check if measurements are only at the end (can optimize)
        has_measurements = any(op.name == 'MEASURE' for op in self._operations)
        
        if has_measurements:
            # Check if all measurements are at the end (no mid-circuit measurement)
            found_measure = False
            mid_circuit_measure = False
            for op in self._operations:
                if op.name == 'MEASURE':
                    found_measure = True
                elif found_measure and op.name not in ('MEASURE', 'BARRIER'):
                    mid_circuit_measure = True
                    break
            
            if mid_circuit_measure:
                # Must run shot-by-shot for mid-circuit measurements
                counts: Dict[str, int] = {}
                for _ in range(shots):
                    state = StateVector(self.num_qubits)
                    classical = [0] * self.num_classical
                    self._execute_operations(state, classical)
                    bitstring = ''.join(str(b) for b in reversed(classical))
                    counts[bitstring] = counts.get(bitstring, 0) + 1
                return SimulatorResult(counts=counts)
            else:
                # Optimized: run circuit once, sample from statevector
                state = StateVector(self.num_qubits)
                # Execute all non-measurement operations
                for op in self._operations:
                    if op.name != 'MEASURE':
                        self._apply_operation(state, op, [0] * self.num_classical)
                # Sample from final statevector
                return SimulatorResult(
                    counts=state.sample(shots),
                    statevector=state.vector
                )
        else:
            # No measurements - return statevector directly
            state = StateVector(self.num_qubits)
            self._execute_operations(state, [0] * self.num_classical)
            return SimulatorResult(
                counts=state.sample(shots),
                statevector=state.vector
            )
    
    def statevector(self) -> np.ndarray:
        """Get the statevector (without measurements)."""
        state = StateVector(self.num_qubits)
        classical = [0] * self.num_classical
        
        for op in self._operations:
            if op.name == 'MEASURE':
                continue  # Skip measurements for statevector
            self._apply_operation(state, op, classical)
        
        return state.vector
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _add_gate(self, name: str, qubits: Tuple[int, ...], 
                  params: Tuple[float, ...] = ()) -> None:
        """Add a gate operation."""
        # Validate qubits
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Qubit {q} is out of range. "
                    f"Circuit has {self.num_qubits} qubits (0 to {self.num_qubits-1})."
                )
        
        self._operations.append(Operation(name, qubits, params))
    
    def _execute_operations(self, state: StateVector, classical: List[int]) -> None:
        """Execute all operations on the given state."""
        for op in self._operations:
            self._apply_operation(state, op, classical)
            
            if self.educational and op.name not in ('MEASURE', 'BARRIER'):
                self._print_state(state, op)
    
    def _apply_operation(self, state: StateVector, op: Operation, 
                         classical: List[int]) -> None:
        """Apply a single operation."""
        if op.name == 'BARRIER':
            return
        
        if op.name == 'MEASURE':
            result = state.measure_qubit(op.qubits[0])
            if op.classical is not None:
                classical[op.classical] = result
            return
        
        if op.name == 'RESET':
            result = state.measure_qubit(op.qubits[0])
            if result == 1:
                state.apply_single_gate(gates.X, op.qubits[0])
            return
        
        # Get gate matrix
        gate_matrix = self._get_gate_matrix(op)
        
        # Apply based on number of qubits
        if len(op.qubits) == 1:
            state.apply_single_gate(gate_matrix, op.qubits[0])
        elif len(op.qubits) == 2:
            state.apply_two_qubit_gate(gate_matrix, op.qubits[0], op.qubits[1])
        elif len(op.qubits) == 3:
            state.apply_three_qubit_gate(gate_matrix, op.qubits[0], op.qubits[1], op.qubits[2])
        else:
            raise NotImplementedError(f"4+ qubit gates not yet supported: {op.name}")
    
    def _get_gate_matrix(self, op: Operation) -> np.ndarray:
        """Get the gate matrix for an operation."""
        name = op.name.upper()
        
        # Non-parametric gates
        simple_gates = {
            'I': gates.I, 'X': gates.X, 'Y': gates.Y, 'Z': gates.Z,
            'H': gates.H, 'S': gates.S, 'SDG': gates.S_DAG,
            'T': gates.T, 'TDG': gates.T_DAG, 'SX': gates.SX,
            'CX': gates.CX, 'CNOT': gates.CNOT, 'CY': gates.CY, 'CZ': gates.CZ,
            'SWAP': gates.SWAP, 'ISWAP': gates.ISWAP,
            'CCX': gates.CCX, 'TOFFOLI': gates.TOFFOLI,
            'CSWAP': gates.CSWAP, 'FREDKIN': gates.FREDKIN,
        }
        
        if name in simple_gates:
            return simple_gates[name]
        
        # Parametric gates
        param_gates = {
            'RX': gates.Rx, 'RY': gates.Ry, 'RZ': gates.Rz, 'P': gates.P,
            'CRX': gates.CRx, 'CRY': gates.CRy, 'CRZ': gates.CRz, 'CP': gates.CP,
            'RXX': gates.RXX, 'RYY': gates.RYY, 'RZZ': gates.RZZ,
            'U3': gates.U3,
        }
        
        if name in param_gates:
            return param_gates[name](*op.params)
        
        raise ValueError(f"Unknown gate: {op.name}")
    
    def _print_state(self, state: StateVector, op: Operation) -> None:
        """Print state for educational mode."""
        params_str = f"({', '.join(f'{p:.3f}' for p in op.params)})" if op.params else ""
        qubits_str = ', '.join(str(q) for q in op.qubits)
        print(f"\nAfter {op.name}{params_str} on qubit(s) [{qubits_str}]:")
        
        vec = state.vector
        for i, amp in enumerate(vec):
            if np.abs(amp) > 1e-10:
                bitstring = format(i, f'0{self.num_qubits}b')
                prob = np.abs(amp)**2
                print(f"  |{bitstring}⟩: {amp: .4f}  (prob: {prob:.2%})")
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def __len__(self) -> int:
        """Return number of operations."""
        return len(self._operations)
    
    def depth(self) -> int:
        """Return circuit depth (layers of parallel gates)."""
        if not self._operations:
            return 0
        
        qubit_depths = [0] * self.num_qubits
        for op in self._operations:
            if op.name in ('BARRIER', 'MEASURE'):
                continue
            max_depth = max(qubit_depths[q] for q in op.qubits)
            for q in op.qubits:
                qubit_depths[q] = max_depth + 1
        
        return max(qubit_depths)
    
    def __repr__(self) -> str:
        return f"Circuit(qubits={self.num_qubits}, ops={len(self._operations)})"
    
    def __enter__(self) -> 'Circuit':
        """Context manager entry for educational mode."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        pass


