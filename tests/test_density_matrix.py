"""Tests for density matrix backend and noise channels."""

import numpy as np
import pytest

from tiny_qpu import Circuit, DensityMatrixBackend
from tiny_qpu.backends.density_matrix import (
    NoiseModel,
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    bit_flip_channel,
    phase_flip_channel,
    thermal_relaxation_channel,
)


@pytest.fixture
def backend():
    return DensityMatrixBackend(seed=42)


# ---------------------------------------------------------------------------
# Ideal simulation
# ---------------------------------------------------------------------------

def test_zero_state(backend):
    qc = Circuit(1)
    result = backend.run(qc)
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    np.testing.assert_allclose(result.density_matrix, expected, atol=1e-12)


def test_x_gate(backend):
    qc = Circuit(1)
    qc.x(0)
    result = backend.run(qc)
    expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    np.testing.assert_allclose(result.density_matrix, expected, atol=1e-12)


def test_bell_state_dm(backend):
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    result = backend.run(qc)

    # Verify purity (pure state)
    assert result.purity() == pytest.approx(1.0, abs=1e-10)

    # Verify probabilities
    probs = result.probabilities()
    assert probs[0] == pytest.approx(0.5, abs=1e-10)
    assert probs[3] == pytest.approx(0.5, abs=1e-10)


def test_purity_pure_state(backend):
    qc = Circuit(1)
    qc.h(0)
    result = backend.run(qc)
    assert result.purity() == pytest.approx(1.0, abs=1e-10)


def test_von_neumann_entropy_pure(backend):
    qc = Circuit(1)
    result = backend.run(qc)
    assert result.von_neumann_entropy() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Noise channels — Kraus operator validity
# ---------------------------------------------------------------------------

def _verify_kraus_completeness(kraus_ops):
    """Verify Σ K†K = I (completeness relation)."""
    dim = kraus_ops[0].shape[0]
    total = sum(K.conj().T @ K for K in kraus_ops)
    np.testing.assert_allclose(total, np.eye(dim), atol=1e-10)


def test_depolarizing_completeness():
    for p in [0, 0.01, 0.1, 0.5, 1.0]:
        _verify_kraus_completeness(depolarizing_channel(p))


def test_amplitude_damping_completeness():
    for gamma in [0, 0.01, 0.5, 1.0]:
        _verify_kraus_completeness(amplitude_damping_channel(gamma))


def test_phase_damping_completeness():
    for gamma in [0, 0.01, 0.5, 1.0]:
        _verify_kraus_completeness(phase_damping_channel(gamma))


def test_bit_flip_completeness():
    for p in [0, 0.1, 0.5]:
        _verify_kraus_completeness(bit_flip_channel(p))


def test_phase_flip_completeness():
    for p in [0, 0.1, 0.5]:
        _verify_kraus_completeness(phase_flip_channel(p))


def test_thermal_relaxation_completeness():
    ops = thermal_relaxation_channel(t1=50, t2=70, gate_time=0.1)
    _verify_kraus_completeness(ops)


def test_thermal_relaxation_invalid_t2():
    with pytest.raises(ValueError):
        thermal_relaxation_channel(t1=50, t2=200, gate_time=0.1)


# ---------------------------------------------------------------------------
# Noisy simulation
# ---------------------------------------------------------------------------

def test_depolarizing_reduces_purity():
    """Depolarizing noise should reduce state purity."""
    noise = NoiseModel()
    noise.add_gate_error("h", depolarizing_channel(0.1))

    noisy = DensityMatrixBackend(noise_model=noise, seed=42)
    ideal = DensityMatrixBackend(seed=42)

    qc = Circuit(1)
    qc.h(0)

    r_noisy = noisy.run(qc)
    r_ideal = ideal.run(qc)

    assert r_noisy.purity() < r_ideal.purity()


def test_amplitude_damping_decays_to_ground():
    """Strong amplitude damping should push state toward |0⟩."""
    noise = NoiseModel()
    noise.add_gate_error("x", amplitude_damping_channel(0.9))

    backend = DensityMatrixBackend(noise_model=noise, seed=42)
    qc = Circuit(1)
    qc.x(0)  # Prepare |1⟩, then damping pushes toward |0⟩

    result = backend.run(qc)
    probs = result.probabilities()
    # With 90% damping, most probability should be at |0⟩
    assert probs.get(0, 0) > 0.5


def test_no_noise_matches_ideal():
    """DensityMatrix with no noise should match pure state simulation."""
    noise = NoiseModel()  # empty
    backend = DensityMatrixBackend(noise_model=noise, seed=42)

    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    result = backend.run(qc)
    assert result.purity() == pytest.approx(1.0, abs=1e-10)


def test_noisy_sampling():
    """Noisy backend should still produce valid measurement counts."""
    noise = NoiseModel()
    noise.add_gate_error("h", depolarizing_channel(0.05))

    backend = DensityMatrixBackend(noise_model=noise, seed=42)
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    result = backend.run(qc, shots=1000)

    assert result.counts is not None
    assert sum(result.counts.values()) == 1000


# ---------------------------------------------------------------------------
# Partial trace
# ---------------------------------------------------------------------------

def test_partial_trace_bell_state(backend):
    """Partial trace of Bell state should give maximally mixed state."""
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    result = backend.run(qc)

    reduced = result.partial_trace([0])
    expected = np.eye(2) / 2  # maximally mixed
    np.testing.assert_allclose(reduced, expected, atol=1e-10)


def test_partial_trace_product_state(backend):
    """Partial trace of product state should give pure reduced state."""
    qc = Circuit(2)
    qc.h(0)  # |+⟩|0⟩
    result = backend.run(qc)

    reduced = result.partial_trace([1])
    # Qubit 1 is |0⟩, so reduced should be |0⟩⟨0|
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    np.testing.assert_allclose(reduced, expected, atol=1e-10)
