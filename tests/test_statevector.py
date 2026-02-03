"""Tests for statevector simulation backend."""

import numpy as np
import pytest

from tiny_qpu import Circuit, StatevectorBackend, Parameter
from tiny_qpu import gates as g


@pytest.fixture
def backend():
    return StatevectorBackend(seed=42)


# ---------------------------------------------------------------------------
# Basic state preparation
# ---------------------------------------------------------------------------

def test_zero_state(backend):
    """|0⟩ state for single qubit."""
    qc = Circuit(1)
    result = backend.run(qc)
    np.testing.assert_allclose(result.statevector, [1, 0], atol=1e-12)


def test_multi_qubit_zero_state(backend):
    """|00...0⟩ for n qubits."""
    qc = Circuit(4)
    result = backend.run(qc)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 1.0
    np.testing.assert_allclose(result.statevector, expected, atol=1e-12)


def test_x_gate_flips_to_one(backend):
    qc = Circuit(1)
    qc.x(0)
    sv = backend.statevector(qc)
    np.testing.assert_allclose(sv, [0, 1], atol=1e-12)


def test_hadamard_superposition(backend):
    qc = Circuit(1)
    qc.h(0)
    sv = backend.statevector(qc)
    expected = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Bell states
# ---------------------------------------------------------------------------

def test_bell_state_phi_plus(backend):
    """|Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    sv = backend.statevector(qc)
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-12)


def test_bell_state_psi_plus(backend):
    """|Ψ+⟩ = (|01⟩ + |10⟩)/√2"""
    qc = Circuit(2)
    qc.x(1).h(0).cx(0, 1)  # |01⟩ → H⊗I → CX → |Ψ+⟩
    sv = backend.statevector(qc)
    expected = np.array([0, 1, 1, 0]) / np.sqrt(2)
    np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-12)


# ---------------------------------------------------------------------------
# GHZ states
# ---------------------------------------------------------------------------

def test_ghz_3_qubit(backend):
    """GHZ = (|000⟩ + |111⟩)/√2"""
    qc = Circuit(3)
    qc.h(0).cx(0, 1).cx(0, 2)
    sv = backend.statevector(qc)
    assert abs(sv[0]) ** 2 == pytest.approx(0.5, abs=1e-10)
    assert abs(sv[7]) ** 2 == pytest.approx(0.5, abs=1e-10)
    assert sum(abs(sv[i]) ** 2 for i in range(1, 7)) == pytest.approx(0, abs=1e-10)


# ---------------------------------------------------------------------------
# Gate correctness
# ---------------------------------------------------------------------------

def test_z_gate_phase(backend):
    """Z|+⟩ = |−⟩"""
    qc = Circuit(1)
    qc.h(0).z(0)
    sv = backend.statevector(qc)
    expected = np.array([1, -1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-12)


def test_s_gate(backend):
    """S|+⟩ should give specific phase."""
    qc = Circuit(1)
    qc.h(0).s(0)
    sv = backend.statevector(qc)
    expected = np.array([1, 1j]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-12)


def test_cnot_no_flip_on_zero_control(backend):
    """CNOT with control=|0⟩ doesn't flip target."""
    qc = Circuit(2)
    qc.cx(0, 1)  # control=0, target=1
    sv = backend.statevector(qc)
    np.testing.assert_allclose(sv, [1, 0, 0, 0], atol=1e-12)


def test_cnot_flips_on_one_control(backend):
    """CNOT with control=|1⟩ flips target."""
    qc = Circuit(2)
    qc.x(0).cx(0, 1)  # |10⟩ → |11⟩
    sv = backend.statevector(qc)
    np.testing.assert_allclose(sv, [0, 0, 0, 1], atol=1e-12)


def test_swap_gate(backend):
    qc = Circuit(2)
    qc.x(0).swap(0, 1)  # |10⟩ → |01⟩
    sv = backend.statevector(qc)
    np.testing.assert_allclose(sv, [0, 1, 0, 0], atol=1e-12)


def test_toffoli_gate(backend):
    """CCX only flips when both controls are |1⟩."""
    # |110⟩ → |111⟩
    qc = Circuit(3)
    qc.x(0).x(1).ccx(0, 1, 2)
    sv = backend.statevector(qc)
    np.testing.assert_allclose(sv, [0, 0, 0, 0, 0, 0, 0, 1], atol=1e-12)

    # |100⟩ → |100⟩ (only one control)
    qc2 = Circuit(3)
    qc2.x(0).ccx(0, 1, 2)
    sv2 = backend.statevector(qc2)
    np.testing.assert_allclose(sv2, [0, 0, 0, 0, 1, 0, 0, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# Rotation gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2, np.pi, 2*np.pi])
def test_rx_rotation(backend, angle):
    """Rx rotation preserves norm."""
    qc = Circuit(1)
    qc.rx(angle, 0)
    sv = backend.statevector(qc)
    assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-12)


def test_ry_creates_superposition(backend):
    """Ry(π/2)|0⟩ = (|0⟩ + |1⟩)/√2."""
    qc = Circuit(1)
    qc.ry(np.pi / 2, 0)
    sv = backend.statevector(qc)
    np.testing.assert_allclose(np.abs(sv) ** 2, [0.5, 0.5], atol=1e-12)


def test_rz_phase_only(backend):
    """Rz only changes phase, not probabilities from |0⟩."""
    qc = Circuit(1)
    qc.rz(np.pi / 3, 0)
    sv = backend.statevector(qc)
    np.testing.assert_allclose(np.abs(sv) ** 2, [1, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# Parameterized circuits
# ---------------------------------------------------------------------------

def test_parameterized_circuit(backend):
    theta = Parameter("theta")
    qc = Circuit(1)
    qc.ry(theta, 0)

    bound = qc.bind({theta: np.pi})
    sv = backend.statevector(bound)
    # Ry(π)|0⟩ = |1⟩
    np.testing.assert_allclose(np.abs(sv) ** 2, [0, 1], atol=1e-12)


def test_multiple_parameters(backend):
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = Circuit(2)
    qc.ry(theta, 0).ry(phi, 1).cx(0, 1)

    bound = qc.bind({theta: np.pi / 2, phi: 0})
    sv = backend.statevector(bound)
    assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def test_measurement_sampling(backend):
    """Bell state should give ~50/50 |00⟩ and |11⟩."""
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    result = backend.run(qc, shots=10000)
    assert result.counts is not None
    total = sum(result.counts.values())
    assert total == 10000
    # Should only have |00⟩ (0) and |11⟩ (3)
    for key in result.counts:
        assert key in (0, 3), f"Unexpected outcome: {key}"


def test_bitstring_counts(backend):
    qc = Circuit(2)
    qc.x(1)  # |01⟩
    result = backend.run(qc, shots=100)
    bs = result.bitstring_counts()
    assert "01" in bs
    assert bs["01"] == 100


def test_deterministic_state_sampling(backend):
    """|1⟩ should always measure 1."""
    qc = Circuit(1)
    qc.x(0)
    result = backend.run(qc, shots=100)
    assert result.counts == {1: 100}


# ---------------------------------------------------------------------------
# Probabilities and expectation values
# ---------------------------------------------------------------------------

def test_probabilities(backend):
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    result = backend.run(qc)
    probs = result.probabilities()
    assert 0 in probs and 3 in probs
    assert probs[0] == pytest.approx(0.5, abs=1e-10)
    assert probs[3] == pytest.approx(0.5, abs=1e-10)


def test_expectation_value(backend):
    """⟨0|Z|0⟩ = 1, ⟨1|Z|1⟩ = -1."""
    qc0 = Circuit(1)
    assert backend.expectation_value(qc0, g.Z) == pytest.approx(1.0)

    qc1 = Circuit(1)
    qc1.x(0)
    assert backend.expectation_value(qc1, g.Z) == pytest.approx(-1.0)


def test_expectation_hadamard_z(backend):
    """⟨+|Z|+⟩ = 0."""
    qc = Circuit(1)
    qc.h(0)
    assert backend.expectation_value(qc, g.Z) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

def test_fidelity_same_state(backend):
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    sv = backend.statevector(qc)
    assert backend.fidelity(sv, sv) == pytest.approx(1.0)


def test_fidelity_orthogonal_states(backend):
    sv0 = np.array([1, 0], dtype=np.complex128)
    sv1 = np.array([0, 1], dtype=np.complex128)
    assert backend.fidelity(sv0, sv1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_custom_initial_state(backend):
    """Start from |1⟩ instead of |0⟩."""
    qc = Circuit(1)
    qc.h(0)
    initial = np.array([0, 1], dtype=np.complex128)
    result = backend.run(qc, initial_state=initial)
    # H|1⟩ = |−⟩ = (|0⟩ - |1⟩)/√2
    expected = np.array([1, -1]) / np.sqrt(2)
    np.testing.assert_allclose(result.statevector, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def test_entropy_product_state(backend):
    """Product state has zero entanglement entropy."""
    qc = Circuit(2)
    qc.h(0)  # |+⟩|0⟩ — product state
    result = backend.run(qc)
    entropy = result.entropy([0])
    assert entropy == pytest.approx(0.0, abs=1e-10)


def test_entropy_bell_state(backend):
    """Bell state has maximal entanglement entropy = ln(2)."""
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    result = backend.run(qc)
    entropy = result.entropy([0])
    assert entropy == pytest.approx(np.log(2), abs=1e-10)


# ---------------------------------------------------------------------------
# Normalization preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_normalization_random_circuit(backend, n_qubits):
    """Random circuit should preserve state normalization."""
    rng = np.random.default_rng(123)
    qc = Circuit(n_qubits)
    for _ in range(20):
        q = rng.integers(n_qubits)
        gate = rng.choice(["h", "x", "y", "z", "s", "t"])
        getattr(qc, gate)(q)
        if n_qubits >= 2:
            q0, q1 = rng.choice(n_qubits, size=2, replace=False)
            qc.cx(int(q0), int(q1))

    sv = backend.statevector(qc)
    assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Multi-qubit scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [6, 8, 10])
def test_larger_circuits(backend, n):
    """Verify simulator works for larger qubit counts."""
    qc = Circuit(n)
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    sv = backend.statevector(qc)
    assert sv.shape == (2**n,)
    assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)
