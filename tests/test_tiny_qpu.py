"""
Comprehensive test suite for tiny-qpu.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, 'src')

from tiny_qpu import Circuit, gates
from tiny_qpu.apps import QRNG, QAOA, BB84, VQE, MolecularHamiltonian
from tiny_qpu.noise import (
    NoiseModel, DensityMatrix, depolarizing, amplitude_damping,
    phase_damping, bit_flip, depolarizing_2q, thermal_relaxation
)
from tiny_qpu.error_correction import BitFlipCode, ShorCode, SteaneCode, compare_codes
from tiny_qpu.algorithms import shor_factor


# ==================== Core Circuit Tests ====================

class TestCircuit:
    def test_bell_state(self):
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = qc.run(shots=1000, seed=42)
        assert '00' in result.counts and '11' in result.counts
        assert result.counts.get('01', 0) == 0

    def test_ghz_state(self):
        qc = Circuit(3).h(0).cx(0, 1).cx(1, 2)
        state = qc.statevector()
        expected = np.zeros(8); expected[0] = expected[7] = 1/np.sqrt(2)
        assert np.allclose(state, expected)

    def test_x_gate(self):
        state = Circuit(1).x(0).statevector()
        assert np.allclose(state, [0, 1])

    def test_hadamard(self):
        state = Circuit(1).h(0).statevector()
        assert np.allclose(state, [1/np.sqrt(2), 1/np.sqrt(2)])

    def test_swap(self):
        state = Circuit(2).x(0).swap(0, 1).statevector()
        assert np.allclose(state, [0, 1, 0, 0])

    def test_toffoli(self):
        state = Circuit(3).x(0).x(1).ccx(0, 1, 2).statevector()
        assert np.allclose(np.abs(state[7]), 1)

    def test_depth(self):
        qc = Circuit(3).h(0).h(1).cx(0, 2).h(2)
        assert qc.depth() == 3

    def test_invalid_qubit(self):
        with pytest.raises(ValueError):
            Circuit(2).h(5)

    def test_20_qubits(self):
        qc = Circuit(20)
        for i in range(20): qc.h(i)
        qc.measure_all()
        result = qc.run(shots=10)
        assert sum(result.counts.values()) == 10


# ==================== Gate Tests ====================

class TestGates:
    def test_unitarity(self):
        for g in [gates.X, gates.Y, gates.Z, gates.H, gates.S, gates.T, gates.CNOT]:
            assert gates.is_unitary(g)

    def test_pauli_algebra(self):
        assert np.allclose(gates.X @ gates.Y, 1j * gates.Z)
        assert np.allclose(gates.Y @ gates.Z, 1j * gates.X)
        assert np.allclose(gates.Z @ gates.X, 1j * gates.Y)

    def test_rotations(self):
        assert np.allclose(gates.Rx(2*np.pi), -gates.I)
        assert np.allclose(gates.Ry(2*np.pi), -gates.I)
        assert np.allclose(gates.Rz(2*np.pi), -gates.I)


# ==================== QRNG Tests ====================

class TestQRNG:
    def test_bits(self):
        bits = QRNG().random_bits(100)
        assert len(bits) == 100
        assert 30 < sum(bits) < 70

    def test_bytes(self):
        data = QRNG().random_bytes(32)
        assert len(data) == 32

    def test_int_range(self):
        for _ in range(50):
            assert 0 <= QRNG().random_int(0, 10) < 10

    def test_float_range(self):
        for _ in range(50):
            assert 0 <= QRNG().random_float() < 1


# ==================== QAOA Tests ====================

class TestQAOA:
    def test_triangle(self):
        result = QAOA([(0,1),(1,2),(2,0)], p=1).optimize(shots=512, seed=42)
        assert result.cut_value() == 2

    def test_square(self):
        result = QAOA([(0,1),(1,2),(2,3),(3,0)], p=2).optimize(shots=512, seed=42)
        assert result.cut_value() == 4


# ==================== BB84 Tests ====================

class TestBB84:
    def test_secure(self):
        result = BB84(128).run(with_eavesdropper=False, seed=42)
        assert result.error_rate < 0.05

    def test_eavesdropper(self):
        result = BB84(128).run(with_eavesdropper=True, seed=42)
        assert result.eavesdropper_detected


# ==================== VQE Tests ====================

class TestVQE:
    def test_h2_converges(self):
        h2 = MolecularHamiltonian.H2(0.735)
        vqe = VQE(h2, depth=3)
        result = vqe.run(maxiter=200, seed=42)
        exact = h2.exact_ground_state()
        assert abs(result.energy - exact) < 0.001


# ==================== Noise Simulator Tests ====================

class TestNoise:
    def test_depolarizing_channel(self):
        ch = depolarizing(0.1)
        assert len(ch.kraus_ops) == 4

    def test_noisy_bell(self):
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        noise = NoiseModel()
        noise.add_all_qubit_error(depolarizing(0.1))
        result = noise.run(qc, shots=1000, seed=42)
        # Should have some errors (01, 10 states)
        assert len(result.counts) > 2

    def test_density_matrix_pure(self):
        dm = DensityMatrix(1)
        assert abs(dm.purity() - 1.0) < 1e-10

    def test_density_matrix_mixed(self):
        dm = DensityMatrix(1)
        dm.apply_channel(depolarizing(1.0), [0])  # Fully depolarize
        assert dm.purity() < 0.6

    def test_bell_concurrence(self):
        dm = DensityMatrix(2)
        dm.apply_gate(gates.H, [0])
        dm.apply_gate(gates.CNOT, [0, 1])
        assert abs(dm.concurrence() - 1.0) < 1e-6

    def test_partial_trace(self):
        dm = DensityMatrix(2)
        dm.apply_gate(gates.H, [0])
        dm.apply_gate(gates.CNOT, [0, 1])
        reduced = dm.partial_trace([0])
        assert abs(reduced.purity() - 0.5) < 1e-6

    def test_thermal_relaxation(self):
        ch = thermal_relaxation(50, 70, 10)
        assert len(ch.kraus_ops) > 0

    def test_hardware_noise_model(self):
        noise = NoiseModel.from_backend()
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = noise.run(qc, shots=100, seed=42)
        assert sum(result.counts.values()) == 100


# ==================== Error Correction Tests ====================

class TestErrorCorrection:
    def test_bit_flip_improvement(self):
        result = BitFlipCode().demonstrate(error_rate=0.05, shots=5000, seed=42)
        assert result.logical_error_rate < result.physical_error_rate

    def test_shor_code(self):
        result = ShorCode().demonstrate(error_rate=0.01, shots=5000, seed=42)
        assert result.logical_error_rate is not None

    def test_steane_code(self):
        result = SteaneCode().demonstrate(error_rate=0.01, shots=5000, seed=42)
        assert result.logical_error_rate is not None

    def test_circuit_correction(self):
        bf = BitFlipCode()
        # No error - should get |00000>
        result = bf.demonstrate_circuit(error_qubit=None, shots=100, seed=42)
        assert result.counts.get('00000', 0) == 100


# ==================== Shor's Algorithm Tests ====================

class TestShor:
    def test_factor_15(self):
        result = shor_factor(15, seed=42)
        assert result.success
        assert set(result.factors) == {3, 5}

    def test_factor_21(self):
        result = shor_factor(21, seed=42)
        assert result.success
        assert set(result.factors) == {3, 7}

    def test_factor_91(self):
        result = shor_factor(91, seed=42)
        assert result.success
        assert set(result.factors) == {7, 13}

    def test_even_number(self):
        result = shor_factor(14, seed=42)
        assert result.factors == (2, 7)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
