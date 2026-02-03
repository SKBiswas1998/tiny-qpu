"""
Quantum circuit representation.

Provides a builder-style API for constructing quantum circuits with
support for parameterized gates, measurement, and barriers.

Example
-------
>>> from tiny_qpu import Circuit, Parameter
>>> theta = Parameter("theta")
>>> qc = Circuit(2)
>>> qc.h(0)
>>> qc.ry(theta, 1)
>>> qc.cx(0, 1)
>>> qc.measure_all()
>>> bound = qc.bind({theta: 0.5})
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from tiny_qpu import gates as g


# ---------------------------------------------------------------------------
# Parameter — symbolic placeholder for variational circuits
# ---------------------------------------------------------------------------

class Parameter:
    """
    Symbolic parameter for parameterized quantum circuits.

    Parameters
    ----------
    name : str
        Human-readable parameter name.

    Example
    -------
    >>> theta = Parameter("theta")
    >>> phi = Parameter("phi")
    """

    __slots__ = ("name", "_id")
    _counter = 0

    def __init__(self, name: str) -> None:
        self.name = name
        Parameter._counter += 1
        self._id = Parameter._counter

    def __repr__(self) -> str:
        return f"Parameter('{self.name}')"

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parameter):
            return self._id == other._id
        return NotImplemented


# ---------------------------------------------------------------------------
# Instruction — a single operation in the circuit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instruction:
    """A single gate operation applied to specific qubits."""
    name: str
    qubits: tuple[int, ...]
    params: tuple[Any, ...] = ()  # float or Parameter
    classical_bits: tuple[int, ...] = ()  # for measurements

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def is_parameterized(self) -> bool:
        return any(isinstance(p, Parameter) for p in self.params)

    def bind(self, param_map: dict[Parameter, float]) -> Instruction:
        """Return a new Instruction with parameters resolved."""
        if not self.is_parameterized:
            return self
        new_params = tuple(
            param_map[p] if isinstance(p, Parameter) else p for p in self.params
        )
        return Instruction(
            name=self.name,
            qubits=self.qubits,
            params=new_params,
            classical_bits=self.classical_bits,
        )

    def matrix(self) -> np.ndarray:
        """Get the unitary matrix. Raises if unresolved parameters remain."""
        if self.is_parameterized:
            raise ValueError(
                f"Cannot get matrix: gate '{self.name}' has unbound parameters "
                f"{[p for p in self.params if isinstance(p, Parameter)]}"
            )
        if self.name == "measure":
            raise ValueError("Measurement has no unitary matrix.")
        if self.name == "barrier":
            raise ValueError("Barrier has no unitary matrix.")
        return g.get_matrix(self.name, self.params)


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

class Circuit:
    """
    Quantum circuit with n_qubits quantum bits and optional classical bits.

    Supports builder-style gate application, parameterized gates,
    measurement, and serialization to/from OpenQASM.

    Parameters
    ----------
    n_qubits : int
        Number of quantum bits.
    n_clbits : int, optional
        Number of classical bits. Defaults to 0 (auto-allocated on measure).
    name : str, optional
        Circuit name for display/QASM export.
    """

    def __init__(
        self, n_qubits: int, n_clbits: int = 0, name: str = "circuit"
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"Need at least 1 qubit, got {n_qubits}")
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.name = name
        self._instructions: list[Instruction] = []
        self._parameters: set[Parameter] = set()

    # -- Properties ---------------------------------------------------------

    @property
    def instructions(self) -> list[Instruction]:
        """List of instructions in the circuit."""
        return list(self._instructions)

    @property
    def parameters(self) -> set[Parameter]:
        """Set of unbound parameters in the circuit."""
        return set(self._parameters)

    @property
    def depth(self) -> int:
        """Circuit depth (longest path through any qubit)."""
        if not self._instructions:
            return 0
        qubit_depth = [0] * self.n_qubits
        for inst in self._instructions:
            if inst.name in ("measure", "barrier"):
                continue
            max_d = max(qubit_depth[q] for q in inst.qubits)
            for q in inst.qubits:
                qubit_depth[q] = max_d + 1
        return max(qubit_depth) if qubit_depth else 0

    @property
    def num_gates(self) -> int:
        """Total number of gates (excluding measurements and barriers)."""
        return sum(
            1 for inst in self._instructions if inst.name not in ("measure", "barrier")
        )

    @property
    def is_parameterized(self) -> bool:
        return len(self._parameters) > 0

    # -- Internal helpers ---------------------------------------------------

    def _validate_qubits(self, qubits: Sequence[int]) -> None:
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(
                    f"Qubit {q} out of range for {self.n_qubits}-qubit circuit"
                )
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Duplicate qubits in {qubits}")

    def _add(self, name: str, qubits: tuple[int, ...], params: tuple = ()) -> Circuit:
        """Add an instruction and return self for chaining."""
        self._validate_qubits(qubits)
        inst = Instruction(name=name, qubits=qubits, params=params)
        self._instructions.append(inst)
        for p in params:
            if isinstance(p, Parameter):
                self._parameters.add(p)
        return self

    # -- Single-qubit gates -------------------------------------------------

    def i(self, qubit: int) -> Circuit:
        """Identity gate."""
        return self._add("i", (qubit,))

    def x(self, qubit: int) -> Circuit:
        """Pauli-X gate."""
        return self._add("x", (qubit,))

    def y(self, qubit: int) -> Circuit:
        """Pauli-Y gate."""
        return self._add("y", (qubit,))

    def z(self, qubit: int) -> Circuit:
        """Pauli-Z gate."""
        return self._add("z", (qubit,))

    def h(self, qubit: int) -> Circuit:
        """Hadamard gate."""
        return self._add("h", (qubit,))

    def s(self, qubit: int) -> Circuit:
        """S gate."""
        return self._add("s", (qubit,))

    def sdg(self, qubit: int) -> Circuit:
        """S-dagger gate."""
        return self._add("sdg", (qubit,))

    def t(self, qubit: int) -> Circuit:
        """T gate."""
        return self._add("t", (qubit,))

    def tdg(self, qubit: int) -> Circuit:
        """T-dagger gate."""
        return self._add("tdg", (qubit,))

    def sx(self, qubit: int) -> Circuit:
        """sqrt(X) gate."""
        return self._add("sx", (qubit,))

    # -- Parameterized single-qubit gates -----------------------------------

    def rx(self, theta: float | Parameter, qubit: int) -> Circuit:
        """Rotation around X-axis."""
        return self._add("rx", (qubit,), (theta,))

    def ry(self, theta: float | Parameter, qubit: int) -> Circuit:
        """Rotation around Y-axis."""
        return self._add("ry", (qubit,), (theta,))

    def rz(self, phi: float | Parameter, qubit: int) -> Circuit:
        """Rotation around Z-axis."""
        return self._add("rz", (qubit,), (phi,))

    def p(self, lam: float | Parameter, qubit: int) -> Circuit:
        """Phase gate."""
        return self._add("p", (qubit,), (lam,))

    def u3(
        self,
        theta: float | Parameter,
        phi: float | Parameter,
        lam: float | Parameter,
        qubit: int,
    ) -> Circuit:
        """Universal single-qubit gate (U3)."""
        return self._add("u3", (qubit,), (theta, phi, lam))

    def u2(self, phi: float | Parameter, lam: float | Parameter, qubit: int) -> Circuit:
        """U2 gate."""
        return self._add("u2", (qubit,), (phi, lam))

    def u1(self, lam: float | Parameter, qubit: int) -> Circuit:
        """U1 gate."""
        return self._add("u1", (qubit,), (lam,))

    # -- Two-qubit gates ----------------------------------------------------

    def cx(self, control: int, target: int) -> Circuit:
        """Controlled-NOT (CNOT) gate."""
        return self._add("cx", (control, target))

    def cnot(self, control: int, target: int) -> Circuit:
        """Alias for cx."""
        return self.cx(control, target)

    def cz(self, q0: int, q1: int) -> Circuit:
        """Controlled-Z gate."""
        return self._add("cz", (q0, q1))

    def swap(self, q0: int, q1: int) -> Circuit:
        """SWAP gate."""
        return self._add("swap", (q0, q1))

    def iswap(self, q0: int, q1: int) -> Circuit:
        """iSWAP gate."""
        return self._add("iswap", (q0, q1))

    def ecr(self, q0: int, q1: int) -> Circuit:
        """Echoed cross-resonance gate."""
        return self._add("ecr", (q0, q1))

    def cp(self, lam: float | Parameter, q0: int, q1: int) -> Circuit:
        """Controlled-Phase gate."""
        return self._add("cp", (q0, q1), (lam,))

    def crx(self, theta: float | Parameter, control: int, target: int) -> Circuit:
        """Controlled-Rx gate."""
        return self._add("crx", (control, target), (theta,))

    def cry(self, theta: float | Parameter, control: int, target: int) -> Circuit:
        """Controlled-Ry gate."""
        return self._add("cry", (control, target), (theta,))

    def crz(self, phi: float | Parameter, control: int, target: int) -> Circuit:
        """Controlled-Rz gate."""
        return self._add("crz", (control, target), (phi,))

    def rxx(self, theta: float | Parameter, q0: int, q1: int) -> Circuit:
        """Ising XX coupling gate."""
        return self._add("rxx", (q0, q1), (theta,))

    def ryy(self, theta: float | Parameter, q0: int, q1: int) -> Circuit:
        """Ising YY coupling gate."""
        return self._add("ryy", (q0, q1), (theta,))

    def rzz(self, theta: float | Parameter, q0: int, q1: int) -> Circuit:
        """Ising ZZ coupling gate."""
        return self._add("rzz", (q0, q1), (theta,))

    # -- Three-qubit gates --------------------------------------------------

    def ccx(self, c0: int, c1: int, target: int) -> Circuit:
        """Toffoli (CCX) gate."""
        return self._add("ccx", (c0, c1, target))

    def toffoli(self, c0: int, c1: int, target: int) -> Circuit:
        """Alias for ccx."""
        return self.ccx(c0, c1, target)

    def cswap(self, control: int, q0: int, q1: int) -> Circuit:
        """Fredkin (CSWAP) gate."""
        return self._add("cswap", (control, q0, q1))

    def fredkin(self, control: int, q0: int, q1: int) -> Circuit:
        """Alias for cswap."""
        return self.cswap(control, q0, q1)

    # -- Measurement --------------------------------------------------------

    def measure(self, qubit: int, clbit: int | None = None) -> Circuit:
        """
        Add a measurement on a qubit.

        Parameters
        ----------
        qubit : int
            Qubit to measure.
        clbit : int, optional
            Classical bit to store the result. Auto-allocated if not given.
        """
        self._validate_qubits((qubit,))
        if clbit is None:
            clbit = self.n_clbits
            self.n_clbits += 1
        inst = Instruction(
            name="measure", qubits=(qubit,), classical_bits=(clbit,)
        )
        self._instructions.append(inst)
        return self

    def measure_all(self) -> Circuit:
        """Add measurements on all qubits."""
        start_clbit = self.n_clbits
        self.n_clbits = start_clbit + self.n_qubits
        for i in range(self.n_qubits):
            self.measure(i, start_clbit + i)
        return self

    # -- Barriers -----------------------------------------------------------

    def barrier(self, *qubits: int) -> Circuit:
        """Add a barrier (visual/optimization boundary)."""
        if not qubits:
            qubits = tuple(range(self.n_qubits))
        self._validate_qubits(qubits)
        self._instructions.append(Instruction(name="barrier", qubits=qubits))
        return self

    # -- Parameter binding --------------------------------------------------

    def bind(self, param_map: dict[Parameter, float]) -> Circuit:
        """
        Return a new circuit with parameters bound to concrete values.

        Parameters
        ----------
        param_map : dict
            Mapping from Parameter objects to float values.

        Returns
        -------
        Circuit
            New circuit with bound parameters.
        """
        new_circuit = Circuit(self.n_qubits, self.n_clbits, self.name)
        for inst in self._instructions:
            bound_inst = inst.bind(param_map)
            new_circuit._instructions.append(bound_inst)
            for p in bound_inst.params:
                if isinstance(p, Parameter):
                    new_circuit._parameters.add(p)
        return new_circuit

    # -- Composition --------------------------------------------------------

    def compose(self, other: Circuit, qubit_map: dict[int, int] | None = None) -> Circuit:
        """
        Append another circuit to this one.

        Parameters
        ----------
        other : Circuit
            Circuit to append.
        qubit_map : dict, optional
            Mapping from other's qubits to this circuit's qubits.
        """
        if qubit_map is None:
            if other.n_qubits > self.n_qubits:
                raise ValueError(
                    f"Cannot compose {other.n_qubits}-qubit circuit onto "
                    f"{self.n_qubits}-qubit circuit without qubit_map"
                )
            qubit_map = {i: i for i in range(other.n_qubits)}

        for inst in other._instructions:
            mapped_qubits = tuple(qubit_map[q] for q in inst.qubits)
            new_inst = Instruction(
                name=inst.name,
                qubits=mapped_qubits,
                params=inst.params,
                classical_bits=inst.classical_bits,
            )
            self._instructions.append(new_inst)
            for p in inst.params:
                if isinstance(p, Parameter):
                    self._parameters.add(p)
        return self

    def inverse(self) -> Circuit:
        """Return the inverse (adjoint) circuit."""
        inv = Circuit(self.n_qubits, name=f"{self.name}_inv")
        for inst in reversed(self._instructions):
            if inst.name in ("measure", "barrier"):
                continue
            # For parameterized gates, negate parameters
            if inst.name in ("rx", "ry", "rz", "p", "u1"):
                new_params = tuple(-p if not isinstance(p, Parameter) else p for p in inst.params)
                inv._instructions.append(
                    Instruction(name=inst.name, qubits=inst.qubits, params=new_params)
                )
            elif inst.name == "s":
                inv._instructions.append(Instruction(name="sdg", qubits=inst.qubits))
            elif inst.name == "sdg":
                inv._instructions.append(Instruction(name="s", qubits=inst.qubits))
            elif inst.name == "t":
                inv._instructions.append(Instruction(name="tdg", qubits=inst.qubits))
            elif inst.name == "tdg":
                inv._instructions.append(Instruction(name="t", qubits=inst.qubits))
            else:
                # Self-inverse gates (X, Y, Z, H, CX, CZ, SWAP, CCX, CSWAP)
                inv._instructions.append(inst)
            for p in inst.params:
                if isinstance(p, Parameter):
                    inv._parameters.add(p)
        return inv

    # -- Copy ---------------------------------------------------------------

    def copy(self) -> Circuit:
        """Return a deep copy of this circuit."""
        return copy.deepcopy(self)

    # -- Display ------------------------------------------------------------

    def __repr__(self) -> str:
        params_str = f", params={sorted(p.name for p in self._parameters)}" if self._parameters else ""
        return (
            f"Circuit(n_qubits={self.n_qubits}, n_clbits={self.n_clbits}, "
            f"depth={self.depth}, gates={self.num_gates}{params_str})"
        )

    def draw(self, style: str = "text") -> str:
        """
        Draw the circuit as ASCII art.

        Parameters
        ----------
        style : str
            Drawing style. Currently only 'text' supported.

        Returns
        -------
        str
            ASCII representation of the circuit.
        """
        if style != "text":
            raise ValueError(f"Unknown style '{style}'. Supported: 'text'")

        # Build per-qubit timelines
        lines: list[list[str]] = [[] for _ in range(self.n_qubits)]
        max_widths: list[int] = []

        for inst in self._instructions:
            if inst.name == "barrier":
                # Add barrier marker to involved qubits
                col_idx = len(max_widths)
                for q in inst.qubits:
                    lines[q].append("│")
                for q in range(self.n_qubits):
                    if q not in inst.qubits:
                        lines[q].append("─")
                max_widths.append(1)
                continue

            # Build gate label
            if inst.name == "measure":
                label = "M"
            elif inst.params:
                param_strs = []
                for p in inst.params:
                    if isinstance(p, Parameter):
                        param_strs.append(p.name)
                    else:
                        param_strs.append(f"{p:.2f}" if isinstance(p, float) else str(p))
                label = f"{inst.name}({','.join(param_strs)})"
            else:
                label = inst.name.upper()

            width = len(label) + 2  # for brackets

            if len(inst.qubits) == 1:
                q = inst.qubits[0]
                lines[q].append(f"[{label}]")
                for other in range(self.n_qubits):
                    if other != q:
                        lines[other].append("─" * width)
            else:
                # Multi-qubit: show on first qubit, dots on others
                q0 = inst.qubits[0]
                lines[q0].append(f"[{label}]")
                for idx, q in enumerate(inst.qubits[1:], 1):
                    marker = f"  {'●' if idx < len(inst.qubits) else '○'}"
                    lines[q].append(marker.ljust(width))
                for other in range(self.n_qubits):
                    if other not in inst.qubits:
                        lines[other].append("─" * width)

            max_widths.append(width)

        # Render
        result = []
        for q in range(self.n_qubits):
            prefix = f"q{q}: "
            content = "─".join(lines[q]) if lines[q] else ""
            result.append(f"{prefix}──{content}──")

        return "\n".join(result)

    # -- QASM export --------------------------------------------------------

    def to_qasm(self) -> str:
        """Export circuit as OpenQASM 2.0 string."""
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.n_qubits}];",
        ]
        if self.n_clbits > 0:
            lines.append(f"creg c[{self.n_clbits}];")

        for inst in self._instructions:
            if inst.name == "barrier":
                qubits_str = ",".join(f"q[{q}]" for q in inst.qubits)
                lines.append(f"barrier {qubits_str};")
            elif inst.name == "measure":
                lines.append(
                    f"measure q[{inst.qubits[0]}] -> c[{inst.classical_bits[0]}];"
                )
            elif inst.params:
                params_str = ",".join(str(p) for p in inst.params)
                qubits_str = ",".join(f"q[{q}]" for q in inst.qubits)
                lines.append(f"{inst.name}({params_str}) {qubits_str};")
            else:
                qubits_str = ",".join(f"q[{q}]" for q in inst.qubits)
                lines.append(f"{inst.name} {qubits_str};")

        return "\n".join(lines) + "\n"

    # --- backward-compatibility shims for legacy API ---
    @property
    def num_qubits(self):
        return self.n_qubits

    def statevector(self):
        from .backends.statevector import StatevectorBackend
        return StatevectorBackend().run(self).statevector

    def run(self, shots=1024, seed=None):
        from .backends.statevector import StatevectorBackend
        from types import SimpleNamespace
        result = StatevectorBackend(seed=seed).run(self, shots=shots)
        counts = result.bitstring_counts()
        return SimpleNamespace(counts=counts)
