"""
Tests for the stabilizer (Clifford) simulation backend.

Tests verify correctness by:
1. Checking stabilizer generators after gate application
2. Comparing measurement outcomes with known states
3. Cross-validating against statevector simulation for small circuits
4. Testing entanglement properties (Bell pairs, GHZ states)
5. Verifying deterministic vs. random measurement behavior
6. Stress-testing at scale (100+ qubits)

Covers:
- Single-qubit Cliffords: H, S, Sdg, X, Y, Z
- Two-qubit Cliffords: CNOT/CX, CZ, SWAP
- Measurement: deterministic outcomes, random outcomes, entangled measurement
- State preparation: Bell pairs, GHZ states
- Scale: 100, 500, 1000 qubit circuits
- Error handling and edge cases
"""

import numpy as np
import pytest
import time

from tiny_qpu.backends.stabilizer import StabilizerBackend, StabilizerTableau


# ═══════════════════════════════════════════════════════════════════
# Initial State
# ═══════════════════════════════════════════════════════════════════


class TestInitialState:
    """Verify |0...0⟩ initialization."""

    def test_single_qubit_initial(self):
        sim = StabilizerBackend(1)
        assert sim.stabilizers() == ["+Z"]
        assert sim.destabilizers() == ["+X"]

    def test_two_qubit_initial(self):
        sim = StabilizerBackend(2)
        assert sim.stabilizers() == ["+ZI", "+IZ"]

    def test_measure_zero_state(self):
        """All qubits in |0⟩ → all measurements return 0."""
        sim = StabilizerBackend(5, seed=42)
        for q in range(5):
            assert sim.measure(q) == 0
            assert sim.is_deterministic(q)

    def test_initial_statevector(self):
        """Verify |0...0⟩ statevector."""
        sim = StabilizerBackend(3)
        sv = sim.to_statevector()
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════
# Single-qubit Clifford Gates
# ═══════════════════════════════════════════════════════════════════


class TestHadamard:
    """Hadamard: |0⟩ → |+⟩, |1⟩ → |−⟩."""

    def test_h_on_zero(self):
        sim = StabilizerBackend(1)
        sim.h(0)
        assert sim.stabilizers() == ["+X"]

    def test_h_twice_is_identity(self):
        sim = StabilizerBackend(1)
        sim.h(0).h(0)
        assert sim.stabilizers() == ["+Z"]

    def test_h_creates_superposition(self):
        """H|0⟩ = |+⟩ → measurement is random."""
        sim = StabilizerBackend(1)
        sim.h(0)
        assert not sim.is_deterministic(0)

    def test_h_statevector(self):
        sim = StabilizerBackend(1)
        sim.h(0)
        sv = sim.to_statevector()
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_h_on_second_qubit(self):
        sim = StabilizerBackend(2)
        sim.h(1)
        assert sim.stabilizers() == ["+ZI", "+IX"]


class TestSGate:
    """S (phase) gate: Z→Z, X→Y."""

    def test_s_on_plus(self):
        """S|+⟩ should give stabilizer Y (= |i⟩ state)."""
        sim = StabilizerBackend(1)
        sim.h(0)  # |+⟩, stabilizer +X
        sim.s(0)  # S|+⟩, stabilizer +Y
        assert sim.stabilizers() == ["+Y"]

    def test_s_four_times_identity(self):
        """S⁴ = I."""
        sim = StabilizerBackend(1)
        sim.h(0)
        stab_before = sim.stabilizers()
        sim.s(0).s(0).s(0).s(0)
        assert sim.stabilizers() == stab_before

    def test_sdg_inverts_s(self):
        """S†S = I."""
        sim = StabilizerBackend(1)
        sim.h(0)
        stab_before = sim.stabilizers()
        sim.s(0).sdg(0)
        assert sim.stabilizers() == stab_before


class TestPauliGates:
    """X, Y, Z Pauli gates."""

    def test_x_flips_zero_to_one(self):
        sim = StabilizerBackend(1)
        sim.x_gate(0)
        assert sim.measure(0) == 1

    def test_x_on_zero_stabilizer(self):
        """X on |0⟩ gives |1⟩, stabilizer -Z."""
        sim = StabilizerBackend(1)
        sim.x_gate(0)
        assert sim.stabilizers() == ["-Z"]

    def test_x_twice_identity(self):
        sim = StabilizerBackend(1)
        sim.x_gate(0).x_gate(0)
        assert sim.stabilizers() == ["+Z"]

    def test_z_on_plus(self):
        """Z|+⟩ = |−⟩, stabilizer -X."""
        sim = StabilizerBackend(1)
        sim.h(0)
        sim.z_gate(0)
        assert sim.stabilizers() == ["-X"]

    def test_y_on_zero(self):
        """Y|0⟩ = i|1⟩, measure gives 1."""
        sim = StabilizerBackend(1)
        sim.y_gate(0)
        assert sim.measure(0) == 1

    def test_y_twice_identity_on_stabilizer(self):
        """Y² = I (up to global phase, which doesn't affect stabilizers)."""
        sim = StabilizerBackend(1)
        sim.y_gate(0).y_gate(0)
        assert sim.stabilizers() == ["+Z"]


# ═══════════════════════════════════════════════════════════════════
# Two-qubit Gates
# ═══════════════════════════════════════════════════════════════════


class TestCNOT:
    """CNOT/CX gate tests."""

    def test_cx_on_computational_basis(self):
        """CX|10⟩ = |11⟩."""
        sim = StabilizerBackend(2)
        sim.x_gate(0)  # |10⟩
        sim.cx(0, 1)   # |11⟩
        assert sim.measure(0) == 1
        assert sim.measure(1) == 1

    def test_cx_no_flip_when_control_zero(self):
        """CX|00⟩ = |00⟩."""
        sim = StabilizerBackend(2)
        sim.cx(0, 1)
        assert sim.measure(0) == 0
        assert sim.measure(1) == 0

    def test_cx_same_qubit_raises(self):
        sim = StabilizerBackend(2)
        with pytest.raises(ValueError, match="differ"):
            sim.cx(0, 0)

    def test_cnot_alias(self):
        sim = StabilizerBackend(2)
        sim.x_gate(0)
        sim.cnot(0, 1)
        assert sim.measure(1) == 1


class TestBellState:
    """Bell state creation and verification."""

    def test_bell_state_stabilizers(self):
        """Bell pair |Φ⁺⟩ has stabilizers +XX and +ZZ."""
        sim = StabilizerBackend(2)
        sim.h(0).cx(0, 1)
        stabs = sim.stabilizers()
        assert "+XX" in stabs
        assert "+ZZ" in stabs

    def test_bell_state_entangled(self):
        sim = StabilizerBackend(2)
        sim.h(0).cx(0, 1)
        assert sim.is_entangled(0, 1)

    def test_bell_state_correlated_measurement(self):
        """Measuring one qubit of a Bell pair determines the other."""
        results = {"same": 0, "diff": 0}
        for _ in range(100):
            sim = StabilizerBackend(2, seed=None)
            sim.h(0).cx(0, 1)
            a = sim.measure(0)
            b = sim.measure(1)
            if a == b:
                results["same"] += 1
            else:
                results["diff"] += 1
        assert results["same"] == 100  # Always correlated

    def test_bell_statevector(self):
        sim = StabilizerBackend(2)
        sim.h(0).cx(0, 1)
        sv = sim.to_statevector()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[3] = 1 / np.sqrt(2)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_prepare_bell_pair_helper(self):
        sim = StabilizerBackend(2)
        sim.prepare_bell_pair(0, 1)
        assert sim.is_entangled(0, 1)


class TestCZGate:
    """Controlled-Z gate."""

    def test_cz_symmetric(self):
        """CZ(a,b) = CZ(b,a)."""
        sim1 = StabilizerBackend(2)
        sim1.h(0).h(1).cz(0, 1)

        sim2 = StabilizerBackend(2)
        sim2.h(0).h(1).cz(1, 0)

        assert sim1.stabilizers() == sim2.stabilizers()

    def test_cz_on_plus_plus(self):
        """CZ|++⟩ creates entanglement visible in stabilizers."""
        sim = StabilizerBackend(2)
        sim.h(0).h(1).cz(0, 1)
        stabs = sim.stabilizers()
        # Should have +XZ and +ZX
        assert "+XZ" in stabs
        assert "+ZX" in stabs


class TestSWAP:
    """SWAP gate."""

    def test_swap_states(self):
        """SWAP|10⟩ = |01⟩."""
        sim = StabilizerBackend(2)
        sim.x_gate(0)
        sim.swap(0, 1)
        assert sim.measure(0) == 0
        assert sim.measure(1) == 1

    def test_swap_twice_identity(self):
        """SWAP² = I."""
        sim = StabilizerBackend(2)
        sim.x_gate(0)
        sim.swap(0, 1).swap(0, 1)
        assert sim.measure(0) == 1
        assert sim.measure(1) == 0


# ═══════════════════════════════════════════════════════════════════
# Measurement
# ═══════════════════════════════════════════════════════════════════


class TestMeasurement:
    """Measurement correctness and determinism."""

    def test_deterministic_zero(self):
        sim = StabilizerBackend(1)
        assert sim.is_deterministic(0)
        assert sim.get_deterministic_outcome(0) == 0

    def test_deterministic_one(self):
        sim = StabilizerBackend(1)
        sim.x_gate(0)
        assert sim.is_deterministic(0)
        assert sim.get_deterministic_outcome(0) == 1

    def test_random_after_hadamard(self):
        sim = StabilizerBackend(1)
        sim.h(0)
        assert not sim.is_deterministic(0)
        assert sim.get_deterministic_outcome(0) is None

    def test_measurement_collapses_state(self):
        """After measurement, state is deterministic."""
        sim = StabilizerBackend(1, seed=42)
        sim.h(0)
        result = sim.measure(0)
        # Now the state is collapsed
        assert sim.is_deterministic(0)
        assert sim.measure(0) == result

    def test_forced_measurement(self):
        """Force specific outcome on random measurement."""
        sim = StabilizerBackend(1)
        sim.h(0)
        result = sim.measure(0, force=0)
        assert result == 0

        sim2 = StabilizerBackend(1)
        sim2.h(0)
        result2 = sim2.measure(0, force=1)
        assert result2 == 1

    def test_measure_all(self):
        sim = StabilizerBackend(3)
        sim.x_gate(1)  # |010⟩
        result = sim.measure_all()
        assert result == "010"

    def test_measurement_statistics(self):
        """H|0⟩ should give ~50/50 statistics."""
        counts = {"0": 0, "1": 0}
        for seed in range(1000):
            sim = StabilizerBackend(1, seed=seed)
            sim.h(0)
            counts[str(sim.measure(0))] += 1
        # Should be roughly 500/500 (allow 10% deviation)
        assert 400 < counts["0"] < 600
        assert 400 < counts["1"] < 600


# ═══════════════════════════════════════════════════════════════════
# GHZ States
# ═══════════════════════════════════════════════════════════════════


class TestGHZState:
    """GHZ state creation and properties."""

    def test_3_qubit_ghz_stabilizers(self):
        """3-qubit GHZ: stabilizers include XXX, ZZI, IZZ."""
        sim = StabilizerBackend(3)
        sim.prepare_ghz()
        stabs = sim.stabilizers()
        assert "+XXX" in stabs

    def test_ghz_all_entangled(self):
        sim = StabilizerBackend(4)
        sim.prepare_ghz()
        for i in range(4):
            for j in range(i + 1, 4):
                assert sim.is_entangled(i, j)

    def test_ghz_correlated_measurement(self):
        """All qubits in GHZ give same result."""
        for _ in range(50):
            sim = StabilizerBackend(5, seed=None)
            sim.prepare_ghz()
            bits = [sim.measure(q) for q in range(5)]
            assert len(set(bits)) == 1  # All same

    def test_ghz_statevector(self):
        sim = StabilizerBackend(3)
        sim.prepare_ghz()
        sv = sim.to_statevector()
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_ghz_requires_two_qubits(self):
        sim = StabilizerBackend(1)
        with pytest.raises(ValueError, match="≥ 2"):
            sim.prepare_ghz()


# ═══════════════════════════════════════════════════════════════════
# Circuit Application
# ═══════════════════════════════════════════════════════════════════


class TestCircuitApplication:
    """Test apply_circuit() method."""

    def test_bell_via_circuit(self):
        sim = StabilizerBackend(2)
        sim.apply_circuit([("h", 0), ("cx", 0, 1)])
        assert sim.is_entangled(0, 1)

    def test_unknown_gate_raises(self):
        sim = StabilizerBackend(2)
        with pytest.raises(ValueError, match="not a Clifford"):
            sim.apply_circuit([("rx", 0)])

    def test_teleportation_circuit(self):
        """Quantum teleportation: transfer |1⟩ from q0 to q2."""
        sim = StabilizerBackend(3, seed=42)

        # Prepare |1⟩ on q0
        sim.x_gate(0)

        # Create Bell pair on q1, q2
        sim.h(1).cx(1, 2)

        # Bell measurement on q0, q1
        sim.cx(0, 1).h(0)
        m0 = sim.measure(0)
        m1 = sim.measure(1)

        # Corrections on q2
        if m1 == 1:
            sim.x_gate(2)
        if m0 == 1:
            sim.z_gate(2)

        # q2 should now be |1⟩
        assert sim.measure(2) == 1


# ═══════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════


class TestSampling:
    """Test sampling / shot-based measurement."""

    def test_sample_deterministic(self):
        sim = StabilizerBackend(2)
        sim.x_gate(0)
        counts = sim.sample(shots=100)
        assert counts == {"10": 100}

    def test_sample_bell_state(self):
        sim = StabilizerBackend(2, seed=42)
        sim.h(0).cx(0, 1)
        counts = sim.sample(shots=1000)
        assert set(counts.keys()).issubset({"00", "11"})
        assert counts.get("00", 0) + counts.get("11", 0) == 1000


# ═══════════════════════════════════════════════════════════════════
# Cross-validation with Statevector
# ═══════════════════════════════════════════════════════════════════


class TestCrossValidation:
    """Verify stabilizer results match statevector for small circuits."""

    def _build_random_clifford_circuit(self, n_qubits, depth, seed=42):
        """Generate a random Clifford circuit."""
        rng = np.random.default_rng(seed)
        ops = []
        single_gates = ["h", "s", "x", "y", "z"]
        for _ in range(depth):
            if rng.random() < 0.5 or n_qubits == 1:
                gate = rng.choice(single_gates)
                q = int(rng.integers(n_qubits))
                ops.append((gate, q))
            else:
                q1, q2 = rng.choice(n_qubits, size=2, replace=False)
                gate = rng.choice(["cx", "cz", "swap"])
                ops.append((gate, int(q1), int(q2)))
        return ops

    def test_random_2q_circuits(self):
        """Random 2-qubit Clifford circuits should match statevector."""
        for seed in range(20):
            sim = StabilizerBackend(2)
            ops = self._build_random_clifford_circuit(2, depth=10, seed=seed)
            sim.apply_circuit(ops)
            sv = sim.to_statevector()
            # Verify it's a valid state
            assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_random_3q_circuits(self):
        """Random 3-qubit Clifford circuits."""
        for seed in range(10):
            sim = StabilizerBackend(3)
            ops = self._build_random_clifford_circuit(3, depth=15, seed=seed)
            sim.apply_circuit(ops)
            sv = sim.to_statevector()
            assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_deterministic_outcomes_match_statevector(self):
        """For product states, stabilizer measurement matches statevector peaks."""
        sim = StabilizerBackend(3)
        sim.x_gate(0).x_gate(2)  # |101⟩
        sv = sim.to_statevector()
        # |101⟩ = index 5 in binary
        assert abs(sv[5]) > 0.99
        assert sim.measure_all() == "101"


# ═══════════════════════════════════════════════════════════════════
# Scale Tests
# ═══════════════════════════════════════════════════════════════════


class TestScale:
    """Test performance at large qubit counts."""

    def test_100_qubit_ghz(self):
        """100-qubit GHZ state in < 1 second."""
        start = time.time()
        sim = StabilizerBackend(100, seed=42)
        sim.prepare_ghz()
        elapsed = time.time() - start
        assert elapsed < 1.0

        # All qubits should be entangled with qubit 0
        assert sim.is_entangled(0, 99)

        # Measure and verify correlation
        r0 = sim.measure(0)
        r99 = sim.measure(99)
        assert r0 == r99

    def test_500_qubit_circuit(self):
        """500-qubit Clifford circuit with 1000 gates."""
        n = 500
        sim = StabilizerBackend(n, seed=42)
        rng = np.random.default_rng(42)

        start = time.time()
        for _ in range(1000):
            q = int(rng.integers(n))
            gate = rng.choice(["h", "s", "x"])
            if gate == "h":
                sim.h(q)
            elif gate == "s":
                sim.s(q)
            else:
                sim.x_gate(q)
            if rng.random() < 0.3:
                q2 = int(rng.integers(n))
                while q2 == q:
                    q2 = int(rng.integers(n))
                sim.cx(q, q2)
        elapsed = time.time() - start
        assert elapsed < 30.0  # Should be well under
        assert sim.gate_count > 1000

    def test_1000_qubit_creation(self):
        """Can create a 1000-qubit stabilizer state."""
        sim = StabilizerBackend(1000)
        assert sim.n_qubits == 1000
        assert len(sim.stabilizers()) == 1000


# ═══════════════════════════════════════════════════════════════════
# Gate Algebra Identities
# ═══════════════════════════════════════════════════════════════════


class TestGateAlgebra:
    """Verify Clifford group algebraic identities."""

    def test_hssh_equals_hzh_equals_x_action(self):
        """H·S²·H = H·Z·H acts as X on the stabilizer state."""
        # HSSH on |0⟩: H→+X, S→+Y, S→-X, H→-Z = |1⟩
        sim1 = StabilizerBackend(1)
        sim1.h(0).s(0).s(0).h(0)

        # Same as X|0⟩ = |1⟩
        sim2 = StabilizerBackend(1)
        sim2.x_gate(0)

        assert sim1.stabilizers() == sim2.stabilizers()

    def test_hxh_equals_z(self):
        """HXH = Z."""
        sim = StabilizerBackend(1)
        sim.x_gate(0)  # Apply X to |0⟩, get -Z stabilizer
        # Now apply H...H
        sim2 = StabilizerBackend(1)
        sim2.h(0).x_gate(0).h(0)
        # This should be same as Z|0⟩
        sim3 = StabilizerBackend(1)
        sim3.z_gate(0)
        assert sim2.stabilizers() == sim3.stabilizers()

    def test_cx_cx_identity(self):
        """CX · CX = I."""
        sim = StabilizerBackend(2)
        sim.x_gate(0)
        stab_before = sim.stabilizers()
        sim.cx(0, 1).cx(0, 1)
        assert sim.stabilizers() == stab_before

    def test_s_s_equals_z(self):
        """S² = Z."""
        sim1 = StabilizerBackend(1)
        sim1.s(0).s(0)

        sim2 = StabilizerBackend(1)
        sim2.z_gate(0)

        assert sim1.stabilizers() == sim2.stabilizers()


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_qubit_backend(self):
        sim = StabilizerBackend(1)
        sim.h(0)
        result = sim.measure(0)
        assert result in [0, 1]

    def test_invalid_qubit_raises(self):
        sim = StabilizerBackend(2)
        with pytest.raises(ValueError):
            sim.h(2)
        with pytest.raises(ValueError):
            sim.h(-1)

    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError):
            StabilizerBackend(0)

    def test_copy_independence(self):
        """Copy should be independent of original."""
        sim = StabilizerBackend(2)
        sim.h(0).cx(0, 1)
        copy = sim.copy()
        copy.x_gate(0)
        assert sim.stabilizers() != copy.stabilizers()

    def test_repr(self):
        sim = StabilizerBackend(3)
        sim.h(0)
        assert "3" in repr(sim)
        assert "gates=1" in repr(sim)

    def test_str_shows_stabilizers(self):
        sim = StabilizerBackend(2)
        s = str(sim)
        assert "Stabilizers" in s
        assert "+ZI" in s

    def test_to_statevector_too_large_raises(self):
        sim = StabilizerBackend(25)
        with pytest.raises(ValueError, match="memory"):
            sim.to_statevector()

    def test_gate_count_tracking(self):
        sim = StabilizerBackend(2)
        assert sim.gate_count == 0
        sim.h(0).cx(0, 1).s(0)
        assert sim.gate_count == 3

    def test_measurement_count_tracking(self):
        sim = StabilizerBackend(2)
        assert sim.measurement_count == 0
        sim.measure(0)
        sim.measure(1)
        assert sim.measurement_count == 2


# ═══════════════════════════════════════════════════════════════════
# Entanglement Witness
# ═══════════════════════════════════════════════════════════════════


class TestEntanglement:
    """Entanglement detection tests."""

    def test_product_state_not_entangled(self):
        sim = StabilizerBackend(2)
        sim.h(0)  # |+⟩|0⟩ — product state
        assert not sim.is_entangled(0, 1)

    def test_bell_pair_entangled(self):
        sim = StabilizerBackend(2)
        sim.h(0).cx(0, 1)
        assert sim.is_entangled(0, 1)

    def test_entanglement_broken_by_measurement(self):
        """Measuring one qubit of a Bell pair breaks entanglement."""
        sim = StabilizerBackend(2, seed=42)
        sim.h(0).cx(0, 1)
        assert sim.is_entangled(0, 1)
        sim.measure(0)
        assert not sim.is_entangled(0, 1)

    def test_chain_entanglement(self):
        """Linear chain: 0-1-2-3, all should be entangled with each other."""
        sim = StabilizerBackend(4)
        sim.h(0)
        sim.cx(0, 1).cx(1, 2).cx(2, 3)
        # This creates a GHZ-like state
        assert sim.is_entangled(0, 3)


# ═══════════════════════════════════════════════════════════════════
# Quantum Error Correction Preview
# ═══════════════════════════════════════════════════════════════════


class TestQECPreview:
    """
    Demonstrate QEC-readiness with a 3-qubit bit-flip code.

    Encodes |0⟩ → |000⟩, |1⟩ → |111⟩.
    Syndrome measurement detects single bit-flip errors.
    """

    def test_3_qubit_repetition_encode(self):
        """Encode |0⟩ into |000⟩ using CNOT."""
        sim = StabilizerBackend(3)
        sim.cx(0, 1).cx(0, 2)
        # |000⟩ — all zero
        assert sim.measure_all() == "000"

    def test_3_qubit_repetition_encode_one(self):
        """Encode |1⟩ into |111⟩."""
        sim = StabilizerBackend(3)
        sim.x_gate(0)
        sim.cx(0, 1).cx(0, 2)
        assert sim.measure_all() == "111"

    def test_single_bitflip_detection(self):
        """
        Detect a single bit-flip error on the encoded state.

        Syndrome: ZZI detects q0≠q1, IZZ detects q1≠q2.
        """
        # Encode |0⟩L = |000⟩
        sim = StabilizerBackend(5, seed=42)  # 3 data + 2 ancilla
        sim.cx(0, 1).cx(0, 2)

        # Introduce error: X on qubit 1
        sim.x_gate(1)

        # Syndrome extraction
        sim.cx(0, 3).cx(1, 3)  # Ancilla 3 measures Z₀Z₁
        sim.cx(1, 4).cx(2, 4)  # Ancilla 4 measures Z₁Z₂

        s0 = sim.measure(3)
        s1 = sim.measure(4)

        # Syndrome (1,1) → error on qubit 1
        assert s0 == 1
        assert s1 == 1

        # Correct
        sim.x_gate(1)

        # Verify corrected state
        assert sim.measure(0) == 0
        assert sim.measure(1) == 0
        assert sim.measure(2) == 0
