"""
Stabilizer (Clifford) simulation backend using the CHP algorithm.

Simulates Clifford circuits in O(n²) time per gate and O(n³) per measurement,
where n is the number of qubits. This enables simulation of **1,000+ qubits**
for circuits restricted to Clifford gates (H, S, CNOT, X, Y, Z, CZ, SWAP).

Based on the Gottesman–Knill theorem: any circuit composed entirely of
  - state preparation in the computational basis,
  - Clifford gates (H, S, CNOT and their compositions),
  - measurements in the computational basis,
can be efficiently simulated classically.

The state is represented by a stabilizer tableau: a set of 2n Pauli generators
(n stabilizers + n destabilizers) stored as binary vectors over GF(2).

Tableau layout (2n rows × (2n+1) columns):
  Row i (0 ≤ i < n):   destabilizer i
  Row i (n ≤ i < 2n):  stabilizer i-n
  Each row: [x₀ ... xₙ₋₁ | z₀ ... zₙ₋₁ | r]

  xⱼ=1, zⱼ=0 → X on qubit j
  xⱼ=0, zⱼ=1 → Z on qubit j
  xⱼ=1, zⱼ=1 → Y on qubit j
  xⱼ=0, zⱼ=0 → I on qubit j
  r ∈ {0, 1}  → phase (+1 or -1)

Reference:
    Aaronson, Gottesman, "Improved simulation of stabilizer circuits",
    PRA 70, 052328 (2004). arXiv:quant-ph/0406196

Usage:
    >>> from tiny_qpu.backends.stabilizer import StabilizerBackend
    >>> sim = StabilizerBackend(n_qubits=100)
    >>> sim.h(0)
    >>> sim.cx(0, 1)          # Bell pair on qubits 0,1
    >>> sim.measure(0)        # Random: 0 or 1
    >>> sim.measure(1)        # Same as qubit 0 (entangled!)
"""

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union


class StabilizerTableau:
    """
    Binary symplectic tableau for stabilizer state tracking.

    Stores 2n Pauli generators as binary vectors over GF(2).
    The first n rows are destabilizers; the last n rows are stabilizers.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. No exponential memory — cost is O(n²).
    """

    __slots__ = ("n", "x", "z", "r")

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        # x[i, j] and z[i, j] are the X and Z components of generator i on qubit j
        # Using uint8 for speed; boolean would also work
        self.x = np.zeros((2 * n_qubits, n_qubits), dtype=np.uint8)
        self.z = np.zeros((2 * n_qubits, n_qubits), dtype=np.uint8)
        self.r = np.zeros(2 * n_qubits, dtype=np.uint8)  # phase bits

        # Initial state |0...0⟩:
        #   Destabilizer i = Xᵢ
        #   Stabilizer i   = Zᵢ
        for i in range(n_qubits):
            self.x[i, i] = 1          # destabilizer i = Xᵢ
            self.z[n_qubits + i, i] = 1  # stabilizer i = Zᵢ

    def copy(self) -> "StabilizerTableau":
        """Return a deep copy of the tableau."""
        t = StabilizerTableau.__new__(StabilizerTableau)
        t.n = self.n
        t.x = self.x.copy()
        t.z = self.z.copy()
        t.r = self.r.copy()
        return t

    def _rowmult(self, h: int, i: int) -> None:
        """
        Left-multiply row h by row i: row_h ← row_i × row_h.

        Updates the phase using the symplectic inner product rule:
        when multiplying Pauli P_a ⊗ P_b, the phase contribution is
        determined by the commutation relations of single-qubit Paulis.
        """
        n = self.n
        # Compute phase: sum of g(x_i, z_i, x_h, z_h) over all qubits
        # g gives the power of i when P1 * P2 → i^g * P3
        phase = self._row_phase(i, h)
        self.r[h] = phase
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    def _row_phase(self, i: int, h: int) -> int:
        """
        Compute the phase when row_i is left-multiplied onto row_h.

        Returns the new phase bit (0 or 1) for the product row.
        """
        # g(x1,z1, x2,z2) gives the exponent of i:
        #   I*P → 0,  X*X → 0, X*Y → 1, X*Z → 3,
        #   Y*X → 3, Y*Y → 0, Y*Z → 1,
        #   Z*X → 1, Z*Y → 3, Z*Z → 0
        xi, zi = self.x[i], self.z[i]
        xh, zh = self.x[h], self.z[h]

        # Vectorized g computation
        # For each qubit, compute g(xi, zi, xh, zh) mod 4
        g = np.zeros(self.n, dtype=np.int32)

        # Case: xi=1, zi=0 (X):  g = zh * (2*xh - 1) → zh=0:0, xh=0,zh=1:-1→3, xh=1,zh=1:1
        mask_x = (xi == 1) & (zi == 0)
        g[mask_x] = zh[mask_x] * (2 * xh[mask_x] - 1)

        # Case: xi=0, zi=1 (Z):  g = xh * (1 - 2*zh) → xh=0:0, xh=1,zh=0:1, xh=1,zh=1:-1→3
        mask_z = (xi == 0) & (zi == 1)
        g[mask_z] = xh[mask_z] * (1 - 2 * zh[mask_z])

        # Case: xi=1, zi=1 (Y):  g = zh - xh → xh=0,zh=0:0, xh=1,zh=0:-1→3, xh=0,zh=1:1, xh=1,zh=1:0
        mask_y = (xi == 1) & (zi == 1)
        g[mask_y] = zh[mask_y].astype(np.int32) - xh[mask_y].astype(np.int32)

        # Case: xi=0, zi=0 (I): g = 0 (already set)

        total = (2 * int(self.r[i]) + 2 * int(self.r[h]) + int(np.sum(g))) % 4
        return 0 if total == 0 else (1 if total == 2 else 0)

    def stabilizer_str(self, row: int) -> str:
        """
        Return a human-readable Pauli string for the given tableau row.

        Parameters
        ----------
        row : int
            Row index (0 to 2n-1).

        Returns
        -------
        str
            e.g., "+XZI", "-YIZ"
        """
        sign = "+" if self.r[row] == 0 else "-"
        pauli = []
        for j in range(self.n):
            xi, zi = self.x[row, j], self.z[row, j]
            if xi == 0 and zi == 0:
                pauli.append("I")
            elif xi == 1 and zi == 0:
                pauli.append("X")
            elif xi == 0 and zi == 1:
                pauli.append("Z")
            else:
                pauli.append("Y")
        return sign + "".join(pauli)

    def stabilizers(self) -> List[str]:
        """Return list of stabilizer generator strings."""
        return [self.stabilizer_str(self.n + i) for i in range(self.n)]

    def destabilizers(self) -> List[str]:
        """Return list of destabilizer generator strings."""
        return [self.stabilizer_str(i) for i in range(self.n)]


class StabilizerBackend:
    """
    Clifford circuit simulator using the stabilizer tableau formalism.

    Supports: H, S, Sdg, X, Y, Z, CNOT/CX, CZ, SWAP, and computational
    basis measurement. Scales to 1,000+ qubits with O(n²) memory.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (no practical upper limit from memory).
    seed : int, optional
        Random seed for measurement outcomes.

    Example
    -------
    >>> sim = StabilizerBackend(n_qubits=1000)
    >>> sim.h(0)
    >>> for i in range(999):
    ...     sim.cx(i, i + 1)     # 1000-qubit GHZ state
    >>> sim.measure(0)            # Random: 0 or 1
    >>> sim.measure(999)          # Always matches qubit 0
    """

    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be ≥ 1, got {n_qubits}")
        self._tableau = StabilizerTableau(n_qubits)
        self._n = n_qubits
        self._rng = np.random.default_rng(seed)
        self._gate_count = 0
        self._measurement_count = 0

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def tableau(self) -> StabilizerTableau:
        """Access the underlying tableau (for inspection/education)."""
        return self._tableau

    @property
    def gate_count(self) -> int:
        return self._gate_count

    @property
    def measurement_count(self) -> int:
        return self._measurement_count

    def copy(self) -> "StabilizerBackend":
        """Return a deep copy of the simulator state."""
        new = StabilizerBackend.__new__(StabilizerBackend)
        new._tableau = self._tableau.copy()
        new._n = self._n
        new._rng = np.random.default_rng()
        new._gate_count = self._gate_count
        new._measurement_count = self._measurement_count
        return new

    # ─── Single-qubit Clifford gates ─────────────────────────────────

    def _validate_qubit(self, q: int, name: str = "qubit") -> None:
        if not (0 <= q < self._n):
            raise ValueError(f"{name} {q} out of range [0, {self._n})")

    def h(self, q: int) -> "StabilizerBackend":
        """
        Hadamard gate on qubit q.

        H: X↔Z (swaps X and Z components, with phase correction).
        """
        self._validate_qubit(q)
        t = self._tableau
        for i in range(2 * self._n):
            t.r[i] ^= t.x[i, q] & t.z[i, q]
            t.x[i, q], t.z[i, q] = t.z[i, q], t.x[i, q]
        self._gate_count += 1
        return self

    def s(self, q: int) -> "StabilizerBackend":
        """
        S (phase) gate on qubit q.

        S: Z→Z, X→Y (Z absorbs X with phase).
        """
        self._validate_qubit(q)
        t = self._tableau
        for i in range(2 * self._n):
            t.r[i] ^= t.x[i, q] & t.z[i, q]
            t.z[i, q] ^= t.x[i, q]
        self._gate_count += 1
        return self

    def sdg(self, q: int) -> "StabilizerBackend":
        """S† (inverse phase) gate. S†= S³ = ZS."""
        self.s(q)
        self.z_gate(q)
        self._gate_count -= 1  # Count as one gate
        return self

    def x_gate(self, q: int) -> "StabilizerBackend":
        """
        Pauli X gate on qubit q.

        X anticommutes with Z, commutes with X.
        """
        self._validate_qubit(q)
        t = self._tableau
        for i in range(2 * self._n):
            t.r[i] ^= t.z[i, q]
        self._gate_count += 1
        return self

    def y_gate(self, q: int) -> "StabilizerBackend":
        """
        Pauli Y gate on qubit q.

        Y anticommutes with both X and Z.
        """
        self._validate_qubit(q)
        t = self._tableau
        for i in range(2 * self._n):
            t.r[i] ^= t.x[i, q] ^ t.z[i, q]
        self._gate_count += 1
        return self

    def z_gate(self, q: int) -> "StabilizerBackend":
        """
        Pauli Z gate on qubit q.

        Z anticommutes with X, commutes with Z.
        """
        self._validate_qubit(q)
        t = self._tableau
        for i in range(2 * self._n):
            t.r[i] ^= t.x[i, q]
        self._gate_count += 1
        return self

    # ─── Two-qubit Clifford gates ────────────────────────────────────

    def cx(self, control: int, target: int) -> "StabilizerBackend":
        """
        CNOT (controlled-X) gate.

        CX: Xc→XcXt, Zt→ZcZt (X propagates forward, Z propagates backward).
        """
        self._validate_qubit(control, "control")
        self._validate_qubit(target, "target")
        if control == target:
            raise ValueError("Control and target must differ")
        t = self._tableau
        a, b = control, target
        for i in range(2 * self._n):
            t.r[i] ^= t.x[i, a] & t.z[i, b] & (t.x[i, b] ^ t.z[i, a] ^ 1)
            t.x[i, b] ^= t.x[i, a]
            t.z[i, a] ^= t.z[i, b]
        self._gate_count += 1
        return self

    def cnot(self, control: int, target: int) -> "StabilizerBackend":
        """Alias for cx()."""
        return self.cx(control, target)

    def cz(self, q1: int, q2: int) -> "StabilizerBackend":
        """
        Controlled-Z gate. CZ = H(target) · CX · H(target).

        Symmetric: CZ(a,b) = CZ(b,a).
        """
        self.h(q2)
        self.cx(q1, q2)
        self.h(q2)
        self._gate_count -= 2  # Count as one gate
        return self

    def swap(self, q1: int, q2: int) -> "StabilizerBackend":
        """SWAP gate. SWAP = CX(a,b) · CX(b,a) · CX(a,b)."""
        self.cx(q1, q2)
        self.cx(q2, q1)
        self.cx(q1, q2)
        self._gate_count -= 2  # Count as one gate
        return self

    # ─── Measurement ─────────────────────────────────────────────────

    def measure(self, q: int, force: Optional[int] = None) -> int:
        """
        Measure qubit q in the computational basis.

        Parameters
        ----------
        q : int
            Qubit to measure.
        force : int, optional
            Force outcome to 0 or 1 (for testing/education). Only valid
            when the outcome is truly random (non-deterministic).

        Returns
        -------
        int
            Measurement outcome (0 or 1).
        """
        self._validate_qubit(q)
        t = self._tableau
        n = self._n
        self._measurement_count += 1

        # Step 1: Find if any stabilizer anticommutes with Z_q
        # A stabilizer anticommutes with Z_q iff it has X_q or Y_q component,
        # i.e., x[row, q] == 1 for some stabilizer row.
        p = None
        for i in range(n, 2 * n):
            if t.x[i, q] == 1:
                p = i
                break

        if p is not None:
            # RANDOM outcome — some stabilizer anticommutes with Z_q
            return self._measure_random(q, p, force)
        else:
            # DETERMINISTIC outcome — all stabilizers commute with Z_q
            return self._measure_deterministic(q)

    def _measure_random(self, q: int, p: int, force: Optional[int]) -> int:
        """Handle measurement when outcome is random (anticommuting stabilizer exists)."""
        t = self._tableau
        n = self._n

        # For every other row that anticommutes with Z_q, multiply by row p
        for i in range(2 * n):
            if i != p and t.x[i, q] == 1:
                t._rowmult(i, p)

        # Move stabilizer p to its destabilizer slot
        dest_row = p - n
        t.x[dest_row] = t.x[p].copy()
        t.z[dest_row] = t.z[p].copy()
        t.r[dest_row] = t.r[p]

        # Set stabilizer p to ±Z_q
        t.x[p] = 0
        t.z[p] = 0
        t.z[p, q] = 1

        if force is not None:
            outcome = int(force)
        else:
            outcome = int(self._rng.integers(2))

        t.r[p] = outcome
        return outcome

    def _measure_deterministic(self, q: int) -> int:
        """Handle measurement when outcome is deterministic."""
        t = self._tableau
        n = self._n

        # No stabilizer anticommutes — outcome is determined.
        # Compute by multiplying destabilizers that anticommute with Z_q.
        # Use a scratch row approach.

        # Find which destabilizers anticommute with Z_q
        scratch_r = 0
        scratch_x = np.zeros(n, dtype=np.uint8)
        scratch_z = np.zeros(n, dtype=np.uint8)

        for i in range(n):
            if t.x[i, q] == 1:
                # Multiply scratch by stabilizer i (not destabilizer!)
                # We use stabilizer row n+i
                si = n + i
                # Compute phase of product
                xi, zi = t.x[si], t.z[si]

                g = np.zeros(n, dtype=np.int32)
                mask_x = (xi == 1) & (zi == 0)
                g[mask_x] = scratch_z[mask_x] * (2 * scratch_x[mask_x] - 1)
                mask_z = (xi == 0) & (zi == 1)
                g[mask_z] = scratch_x[mask_z] * (1 - 2 * scratch_z[mask_z])
                mask_y = (xi == 1) & (zi == 1)
                g[mask_y] = scratch_z[mask_y].astype(np.int32) - scratch_x[mask_y].astype(np.int32)

                total = (2 * int(t.r[si]) + 2 * scratch_r + int(np.sum(g))) % 4
                scratch_r = 0 if total == 0 else (1 if total == 2 else 0)
                scratch_x ^= xi
                scratch_z ^= zi

        return int(scratch_r)

    def measure_all(self, force: Optional[List[int]] = None) -> str:
        """
        Measure all qubits and return a bitstring.

        Parameters
        ----------
        force : list of int, optional
            Force outcomes for each qubit.

        Returns
        -------
        str
            Bitstring of measurement outcomes, e.g., "010".
        """
        bits = []
        for q in range(self._n):
            f = force[q] if force is not None else None
            bits.append(str(self.measure(q, force=f)))
        return "".join(bits)

    # ─── State queries ───────────────────────────────────────────────

    def is_deterministic(self, q: int) -> bool:
        """
        Check if measuring qubit q would give a deterministic result.

        Returns True if no stabilizer anticommutes with Z_q.
        """
        self._validate_qubit(q)
        t = self._tableau
        for i in range(self._n, 2 * self._n):
            if t.x[i, q] == 1:
                return False
        return True

    def get_deterministic_outcome(self, q: int) -> Optional[int]:
        """
        Return the deterministic outcome if it exists, else None.

        Useful for verifying entanglement: if qubit is entangled with
        measured qubits, its outcome becomes deterministic.
        """
        if not self.is_deterministic(q):
            return None
        # Simulate the deterministic measurement without modifying state
        return self._measure_deterministic(q)

    def stabilizers(self) -> List[str]:
        """Return the stabilizer generators as Pauli strings."""
        return self._tableau.stabilizers()

    def destabilizers(self) -> List[str]:
        """Return the destabilizer generators as Pauli strings."""
        return self._tableau.destabilizers()

    def is_entangled(self, q1: int, q2: int) -> bool:
        """
        Heuristic check for entanglement between two qubits.

        Combines two checks:
        1. If both qubits have deterministic measurement outcomes, they are
           in a product state (not entangled).
        2. Otherwise, checks if any stabilizer generator acts non-trivially
           on both qubits (indicating possible entanglement).

        Not a complete entanglement witness, but catches common cases
        (Bell pairs, GHZ states, etc.) and correctly identifies product states
        after measurement collapse.
        """
        self._validate_qubit(q1)
        self._validate_qubit(q2)

        # If both qubits are deterministic, they're in a product state
        if self.is_deterministic(q1) and self.is_deterministic(q2):
            return False

        t = self._tableau
        for i in range(self._n, 2 * self._n):
            nontrivial_q1 = (t.x[i, q1] | t.z[i, q1]) == 1
            nontrivial_q2 = (t.x[i, q2] | t.z[i, q2]) == 1
            if nontrivial_q1 and nontrivial_q2:
                return True
        return False

    # ─── Batch operations ────────────────────────────────────────────

    def sample(self, shots: int = 1024) -> Dict[str, int]:
        """
        Sample measurement outcomes by running the circuit multiple times.

        Creates a fresh copy for each shot to avoid state collapse issues.

        Parameters
        ----------
        shots : int
            Number of measurement samples.

        Returns
        -------
        dict
            Bitstring counts, e.g., {"000": 510, "111": 514}.
        """
        counts: Dict[str, int] = {}
        for _ in range(shots):
            sim_copy = self.copy()
            result = sim_copy.measure_all()
            counts[result] = counts.get(result, 0) + 1
        return counts

    def apply_circuit(self, operations: List[Tuple]) -> "StabilizerBackend":
        """
        Apply a sequence of Clifford operations.

        Parameters
        ----------
        operations : list of tuples
            Each tuple is (gate_name, qubit(s)), e.g.:
            [("h", 0), ("cx", 0, 1), ("s", 1)]

        Returns
        -------
        self
        """
        for op in operations:
            name = op[0].lower()
            if name == "h":
                self.h(op[1])
            elif name in ("s", "phase"):
                self.s(op[1])
            elif name in ("sdg", "s_dagger"):
                self.sdg(op[1])
            elif name == "x":
                self.x_gate(op[1])
            elif name == "y":
                self.y_gate(op[1])
            elif name == "z":
                self.z_gate(op[1])
            elif name in ("cx", "cnot"):
                self.cx(op[1], op[2])
            elif name == "cz":
                self.cz(op[1], op[2])
            elif name == "swap":
                self.swap(op[1], op[2])
            else:
                raise ValueError(
                    f"Gate '{name}' is not a Clifford gate. "
                    f"StabilizerBackend supports: h, s, sdg, x, y, z, cx, cz, swap"
                )
        return self

    # ─── State preparation helpers ───────────────────────────────────

    def prepare_bell_pair(self, q1: int, q2: int) -> "StabilizerBackend":
        """Prepare Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 on qubits q1, q2."""
        self.h(q1)
        self.cx(q1, q2)
        return self

    def prepare_ghz(self, qubits: Optional[Sequence[int]] = None) -> "StabilizerBackend":
        """
        Prepare GHZ state (|00...0⟩ + |11...1⟩)/√2 on given qubits.

        Parameters
        ----------
        qubits : sequence of int, optional
            Qubits to entangle. Default: all qubits.
        """
        if qubits is None:
            qubits = list(range(self._n))
        if len(qubits) < 2:
            raise ValueError("GHZ requires ≥ 2 qubits")
        self.h(qubits[0])
        for i in range(1, len(qubits)):
            self.cx(qubits[0], qubits[i])
        return self

    # ─── Conversion & comparison ─────────────────────────────────────

    def to_statevector(self) -> np.ndarray:
        """
        Convert stabilizer state to a full statevector (for small n only).

        This defeats the purpose of the stabilizer formalism but is useful
        for validation against the statevector backend.

        Warning: Exponential memory! Only use for n ≤ ~20.

        Returns
        -------
        np.ndarray
            Complex statevector of shape (2^n,).
        """
        n = self._n
        if n > 20:
            raise ValueError(
                f"to_statevector() with {n} qubits would require "
                f"{2**n * 16 / 1e9:.1f} GB of memory. Use n ≤ 20."
            )
        dim = 2 ** n

        # Build the projector P = Π (I + S_i) / 2 as a full matrix,
        # then extract a non-zero column (any column in the image is the state).
        proj = np.eye(dim, dtype=complex)
        for s in range(n):
            row = n + s
            pauli_mat = self._stabilizer_to_matrix(row)
            proj = proj @ (np.eye(dim, dtype=complex) + pauli_mat) / 2.0

        # Find a non-zero column of the projector
        for col in range(dim):
            state = proj[:, col]
            norm = np.linalg.norm(state)
            if norm > 1e-10:
                return state / norm

        # Should never reach here for a valid stabilizer state
        raise RuntimeError("Failed to reconstruct statevector from tableau")

    def _stabilizer_to_matrix(self, row: int) -> np.ndarray:
        """Build the full matrix for a stabilizer generator (for validation)."""
        t = self._tableau
        n = self._n
        sign = (-1.0) ** t.r[row]

        # Build tensor product of single-qubit Paulis
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        mat = np.array([1.0], dtype=complex)
        for j in range(n):
            xj, zj = t.x[row, j], t.z[row, j]
            if xj == 0 and zj == 0:
                p = I
            elif xj == 1 and zj == 0:
                p = X
            elif xj == 0 and zj == 1:
                p = Z
            else:
                p = Y
            mat = np.kron(mat, p)

        return sign * mat

    def __repr__(self) -> str:
        return (
            f"StabilizerBackend(n_qubits={self._n}, "
            f"gates={self._gate_count}, "
            f"measurements={self._measurement_count})"
        )

    def __str__(self) -> str:
        lines = [f"StabilizerBackend({self._n} qubits)"]
        lines.append("Stabilizers:")
        for s in self.stabilizers():
            lines.append(f"  {s}")
        return "\n".join(lines)
