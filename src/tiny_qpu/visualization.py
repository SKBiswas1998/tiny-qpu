"""
Quantum circuit and state visualization.

Features:
- ASCII circuit diagrams (no dependencies)
- State vector bar charts
- Measurement histograms
- Bloch sphere (requires matplotlib)
- Step-by-step execution visualization
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


class CircuitDrawer:
    """
    Draw quantum circuits as ASCII art.
    
    Example output:
        q0: ──H──●──────
                 │      
        q1: ─────X──M───
    """
    
    # Gate symbols
    GATE_SYMBOLS = {
        'H': 'H', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
        'S': 'S', 'T': 'T', 'SDG': 'S†', 'TDG': 'T†',
        'SX': '√X',
        'RX': 'Rx', 'RY': 'Ry', 'RZ': 'Rz', 'P': 'P',
        'U3': 'U',
        'MEASURE': 'M', 'RESET': 'R', 'BARRIER': '░',
    }
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.columns: List[List[str]] = []
        
    def add_single_gate(self, gate: str, qubit: int, params: tuple = ()) -> None:
        """Add a single-qubit gate."""
        col = ['─'] * self.num_qubits
        symbol = self.GATE_SYMBOLS.get(gate.upper(), gate[0])
        if params:
            # Add angle for rotation gates
            angle = params[0]
            if abs(angle - np.pi) < 0.01:
                symbol = f'{symbol}π'
            elif abs(angle - np.pi/2) < 0.01:
                symbol = f'{symbol}π/2'
            else:
                symbol = f'{symbol}{angle:.1f}'
        col[qubit] = f'[{symbol}]'
        self.columns.append(col)
    
    def add_cx(self, control: int, target: int) -> None:
        """Add CNOT gate."""
        col = ['│' if min(control, target) < i < max(control, target) else '─' 
               for i in range(self.num_qubits)]
        col[control] = '●'
        col[target] = '⊕'
        self.columns.append(col)
    
    def add_cz(self, control: int, target: int) -> None:
        """Add CZ gate."""
        col = ['│' if min(control, target) < i < max(control, target) else '─' 
               for i in range(self.num_qubits)]
        col[control] = '●'
        col[target] = '●'
        self.columns.append(col)
    
    def add_swap(self, q1: int, q2: int) -> None:
        """Add SWAP gate."""
        col = ['│' if min(q1, q2) < i < max(q1, q2) else '─' 
               for i in range(self.num_qubits)]
        col[q1] = '✕'
        col[q2] = '✕'
        self.columns.append(col)
    
    def add_barrier(self) -> None:
        """Add barrier."""
        col = ['░'] * self.num_qubits
        self.columns.append(col)
    
    def add_measure(self, qubit: int) -> None:
        """Add measurement."""
        col = ['─'] * self.num_qubits
        col[qubit] = '[M]'
        self.columns.append(col)
    
    def draw(self) -> str:
        """Generate ASCII circuit diagram."""
        if not self.columns:
            return "Empty circuit"
        
        lines = []
        for q in range(self.num_qubits):
            line = f'q{q}: '
            for col in self.columns:
                cell = col[q]
                if cell == '─':
                    line += '───'
                elif cell == '│':
                    line += ' │ '
                elif cell == '●':
                    line += '─●─'
                elif cell == '⊕':
                    line += '─⊕─'
                elif cell == '✕':
                    line += '─✕─'
                elif cell == '░':
                    line += ' ░ '
                elif cell.startswith('['):
                    # Gate with brackets
                    line += cell.center(5, '─')
                else:
                    line += f'─{cell}─'
            line += '───'
            lines.append(line)
        
        return '\n'.join(lines)


class StateVisualizer:
    """
    Visualize quantum states.
    """
    
    @staticmethod
    def amplitudes_ascii(statevector: np.ndarray, num_qubits: int, 
                         threshold: float = 0.01) -> str:
        """
        Display state vector amplitudes as ASCII bar chart.
        """
        lines = []
        lines.append("State Vector:")
        lines.append("─" * 50)
        
        for i, amp in enumerate(statevector):
            prob = np.abs(amp) ** 2
            if prob < threshold:
                continue
            
            bitstring = format(i, f'0{num_qubits}b')
            magnitude = np.abs(amp)
            phase = np.angle(amp)
            
            bar_width = int(prob * 40)
            bar = '█' * bar_width
            
            if abs(phase) < 0.01:
                phase_str = ''
            elif abs(phase - np.pi) < 0.01:
                phase_str = ' (π)'
            elif abs(phase + np.pi) < 0.01:
                phase_str = ' (-π)'
            else:
                phase_str = f' ({phase:.2f})'
            
            lines.append(f"|{bitstring}⟩: {bar:40s} {magnitude:.3f}{phase_str} ({prob*100:.1f}%)")
        
        return '\n'.join(lines)
    
    @staticmethod
    def probabilities_ascii(statevector: np.ndarray, num_qubits: int,
                           threshold: float = 0.01) -> str:
        """Display probabilities only."""
        lines = []
        lines.append("Probabilities:")
        lines.append("─" * 50)
        
        probs = np.abs(statevector) ** 2
        for i, prob in enumerate(probs):
            if prob < threshold:
                continue
            
            bitstring = format(i, f'0{num_qubits}b')
            bar_width = int(prob * 40)
            bar = '█' * bar_width
            
            lines.append(f"|{bitstring}⟩: {bar:40s} {prob*100:5.1f}%")
        
        return '\n'.join(lines)
    
    @staticmethod
    def counts_ascii(counts: Dict[str, int], total: Optional[int] = None) -> str:
        """Display measurement counts as histogram."""
        if total is None:
            total = sum(counts.values())
        
        lines = []
        lines.append("Measurement Results:")
        lines.append("─" * 50)
        
        for bitstring in sorted(counts.keys()):
            count = counts[bitstring]
            prob = count / total
            bar_width = int(prob * 40)
            bar = '█' * bar_width
            
            lines.append(f"|{bitstring}⟩: {bar:40s} {count:4d} ({prob*100:5.1f}%)")
        
        return '\n'.join(lines)


class BlochSphere:
    """
    Bloch sphere visualization for single-qubit states.
    """
    
    @staticmethod
    def state_to_bloch(alpha: complex, beta: complex) -> Tuple[float, float, float]:
        """Convert qubit amplitudes to Bloch sphere coordinates."""
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        alpha, beta = alpha/norm, beta/norm
        
        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = np.abs(alpha)**2 - np.abs(beta)**2
        
        return (float(x), float(y), float(z))
    
    @staticmethod
    def bloch_to_angles(x: float, y: float, z: float) -> Tuple[float, float]:
        """Convert Bloch coordinates to spherical angles (θ, φ)."""
        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)
        return (float(theta), float(phi))
    
    @staticmethod
    def ascii_bloch(alpha: complex, beta: complex) -> str:
        """ASCII representation of Bloch sphere with state marked."""
        x, y, z = BlochSphere.state_to_bloch(alpha, beta)
        theta, phi = BlochSphere.bloch_to_angles(x, y, z)
        
        lines = []
        lines.append("Bloch Sphere:")
        lines.append("─" * 40)
        lines.append(f"  Coordinates: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        lines.append(f"  Angles: θ={theta:.3f} rad, φ={phi:.3f} rad")
        lines.append("")
        
        grid_size = 11
        center = grid_size // 2
        
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        
        for angle in np.linspace(0, 2*np.pi, 40):
            gx = int(center + center * 0.9 * np.cos(angle))
            gy = int(center + center * 0.9 * np.sin(angle))
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                grid[gy][gx] = '·'
        
        for i in range(grid_size):
            if grid[center][i] == ' ':
                grid[center][i] = '─'
            if grid[i][center] == ' ':
                grid[i][center] = '│'
        grid[center][center] = '┼'
        
        state_gx = int(center + center * 0.9 * x)
        state_gy = int(center - center * 0.9 * y)
        if 0 <= state_gx < grid_size and 0 <= state_gy < grid_size:
            if z > 0.3:
                grid[state_gy][state_gx] = '●'
            elif z < -0.3:
                grid[state_gy][state_gx] = '○'
            else:
                grid[state_gy][state_gx] = '◐'
        
        lines.append("     +y")
        for row in grid:
            lines.append("    " + ''.join(row))
        lines.append("  -x     +x")
        lines.append("     -y")
        lines.append("")
        lines.append("  ● = above equator (z>0)")
        lines.append("  ○ = below equator (z<0)")
        lines.append("  ◐ = near equator")
        
        return '\n'.join(lines)


class CircuitVisualizer:
    """Unified visualization for circuits."""
    
    def __init__(self, circuit):
        self.circuit = circuit
        self._drawer = CircuitDrawer(circuit.num_qubits)
        self._build_diagram()
    
    def _build_diagram(self) -> None:
        """Build the circuit diagram from operations."""
        for op in self.circuit._operations:
            name = op.name.upper()
            
            if name in ('H', 'X', 'Y', 'Z', 'S', 'T', 'SDG', 'TDG', 'SX'):
                self._drawer.add_single_gate(name, op.qubits[0])
            elif name in ('RX', 'RY', 'RZ', 'P'):
                self._drawer.add_single_gate(name, op.qubits[0], op.params)
            elif name in ('CX', 'CNOT'):
                self._drawer.add_cx(op.qubits[0], op.qubits[1])
            elif name == 'CZ':
                self._drawer.add_cz(op.qubits[0], op.qubits[1])
            elif name == 'SWAP':
                self._drawer.add_swap(op.qubits[0], op.qubits[1])
            elif name == 'BARRIER':
                self._drawer.add_barrier()
            elif name == 'MEASURE':
                self._drawer.add_measure(op.qubits[0])
            else:
                self._drawer.add_single_gate(name, op.qubits[0])
    
    def draw(self) -> str:
        """Return ASCII circuit diagram."""
        return self._drawer.draw()
    
    def print(self) -> None:
        """Print the circuit diagram."""
        print(self.draw())


class ExecutionVisualizer:
    """Visualize circuit execution step by step."""
    
    def __init__(self, circuit, delay: float = 0.5):
        self.circuit = circuit
        self.delay = delay
    
    def run(self, clear_screen: bool = False) -> None:
        """Execute circuit with visualization after each gate."""
        import time
        
        # Import from package (absolute imports)
        from tiny_qpu.core import StateVector
        from tiny_qpu.core import gates as g
        
        state = StateVector(self.circuit.num_qubits)
        step = 0
        
        print("=" * 60)
        print("Circuit Execution Visualizer")
        print("=" * 60)
        print(f"\nCircuit ({self.circuit.num_qubits} qubits, {len(self.circuit._operations)} operations):")
        print(CircuitVisualizer(self.circuit).draw())
        print("\n" + "=" * 60)
        print("Initial State:")
        print(StateVisualizer.amplitudes_ascii(state.vector, self.circuit.num_qubits))
        
        if self.delay > 0:
            time.sleep(self.delay)
        
        for op in self.circuit._operations:
            if op.name == 'MEASURE':
                continue
            if op.name == 'BARRIER':
                continue
            
            step += 1
            
            # Apply the operation
            self._apply_op(state, op, g)
            
            if clear_screen:
                print("\033[H\033[J", end="")
            
            print("\n" + "-" * 60)
            params_str = f"({', '.join(f'{p:.2f}' for p in op.params)})" if op.params else ""
            qubits_str = ', '.join(str(q) for q in op.qubits)
            print(f"Step {step}: {op.name}{params_str} on qubit(s) [{qubits_str}]")
            print("-" * 60)
            print(StateVisualizer.amplitudes_ascii(state.vector, self.circuit.num_qubits))
            
            if self.circuit.num_qubits == 1:
                print()
                print(BlochSphere.ascii_bloch(state.vector[0], state.vector[1]))
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        print("\n" + "=" * 60)
        print("Execution Complete!")
        print("=" * 60)
    
    def _apply_op(self, state, op, g) -> None:
        """Apply an operation to the state."""
        name = op.name.upper()
        
        single_gates = {
            'I': g.I, 'X': g.X, 'Y': g.Y, 'Z': g.Z,
            'H': g.H, 'S': g.S, 'T': g.T,
            'SDG': g.S_DAG, 'TDG': g.T_DAG, 'SX': g.SX,
        }
        
        if name in single_gates:
            state.apply_single_gate(single_gates[name], op.qubits[0])
        elif name == 'RX':
            state.apply_single_gate(g.Rx(op.params[0]), op.qubits[0])
        elif name == 'RY':
            state.apply_single_gate(g.Ry(op.params[0]), op.qubits[0])
        elif name == 'RZ':
            state.apply_single_gate(g.Rz(op.params[0]), op.qubits[0])
        elif name == 'P':
            state.apply_single_gate(g.P(op.params[0]), op.qubits[0])
        elif name in ('CX', 'CNOT'):
            state.apply_two_qubit_gate(g.CNOT, op.qubits[0], op.qubits[1])
        elif name == 'CZ':
            state.apply_two_qubit_gate(g.CZ, op.qubits[0], op.qubits[1])
        elif name == 'SWAP':
            state.apply_two_qubit_gate(g.SWAP, op.qubits[0], op.qubits[1])


# Convenience functions
def draw_circuit(circuit) -> str:
    """Draw a circuit as ASCII."""
    return CircuitVisualizer(circuit).draw()


def show_state(statevector: np.ndarray, num_qubits: int) -> str:
    """Show state vector as ASCII bar chart."""
    return StateVisualizer.amplitudes_ascii(statevector, num_qubits)


def show_counts(counts: Dict[str, int]) -> str:
    """Show measurement counts as histogram."""
    return StateVisualizer.counts_ascii(counts)


def show_bloch(alpha: complex, beta: complex) -> str:
    """Show single-qubit state on Bloch sphere."""
    return BlochSphere.ascii_bloch(alpha, beta)


def visualize(circuit, delay: float = 0.5) -> None:
    """Run circuit with step-by-step visualization."""
    ExecutionVisualizer(circuit, delay).run()
