"""Comprehensive tests for tiny-qpu."""
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from tiny_qpu import Circuit, gates
from tiny_qpu.apps import QRNG, QAOA, BB84


class TestCircuit:
    """Test Circuit class."""
    
    def test_bell_state(self):
        """Bell state should produce 00 and 11 with equal probability."""
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = qc.run(shots=1000, seed=42)
        assert set(result.counts.keys()).issubset({'00', '11'})
        
    def test_ghz_state(self):
        """GHZ state on 3 qubits."""
        qc = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
        result = qc.run(shots=1000, seed=42)
        assert set(result.counts.keys()).issubset({'000', '111'})
    
    def test_x_gate_flips(self):
        """X gate should flip |0⟩ to |1⟩."""
        qc = Circuit(1).x(0).measure_all()
        result = qc.run(shots=100, seed=42)
        assert result.counts.get('1', 0) == 100
    
    def test_hadamard_superposition(self):
        """H gate creates superposition."""
        qc = Circuit(1).h(0).measure_all()
        result = qc.run(shots=1000, seed=42)
        # Should be roughly 50/50
        assert 400 < result.counts.get('0', 0) < 600
        assert 400 < result.counts.get('1', 0) < 600
    
    def test_circuit_depth(self):
        """Test circuit depth calculation."""
        qc = Circuit(3).h(0).h(1).cx(0, 2).h(2)
        assert qc.depth() == 3
    
    def test_invalid_qubit_raises(self):
        """Accessing invalid qubit should raise error."""
        qc = Circuit(2)
        with pytest.raises(ValueError):
            qc.h(5)  # Only qubits 0, 1 exist


class TestGates:
    """Test quantum gates."""
    
    def test_pauli_gates_are_unitary(self):
        """Pauli gates should be unitary."""
        for gate in [gates.X, gates.Y, gates.Z]:
            assert gates.is_unitary(gate)
    
    def test_pauli_squared_is_identity(self):
        """X², Y², Z² = I."""
        assert np.allclose(gates.X @ gates.X, gates.I)
        assert np.allclose(gates.Y @ gates.Y, gates.I)
        assert np.allclose(gates.Z @ gates.Z, gates.I)
    
    def test_hadamard_squared_is_identity(self):
        """H² = I."""
        assert np.allclose(gates.H @ gates.H, gates.I)
    
    def test_rotation_gates(self):
        """Test rotation gates at specific angles."""
        # Rx(π) = -iX
        assert np.allclose(gates.Rx(np.pi), -1j * gates.X)
        # Ry(π) = -iY  
        assert np.allclose(gates.Ry(np.pi), -1j * gates.Y)
        # Rz(π) = -iZ
        assert np.allclose(gates.Rz(np.pi), -1j * gates.Z)


class TestQRNG:
    """Test Quantum Random Number Generator."""
    
    def test_random_bits_length(self):
        """Should generate correct number of bits."""
        qrng = QRNG()
        bits = qrng.random_bits(100)
        assert len(bits) == 100
        assert all(b in [0, 1] for b in bits)
    
    def test_random_bytes_length(self):
        """Should generate correct number of bytes."""
        qrng = QRNG()
        data = qrng.random_bytes(32)
        assert len(data) == 32
        assert isinstance(data, bytes)
    
    def test_random_int_range(self):
        """Random int should be in range."""
        qrng = QRNG()
        for _ in range(100):
            n = qrng.random_int(10, 20)
            assert 10 <= n < 20
    
    def test_random_float_range(self):
        """Random float should be in [0, 1)."""
        qrng = QRNG()
        for _ in range(100):
            f = qrng.random_float()
            assert 0.0 <= f < 1.0
    
    def test_uuid_format(self):
        """UUID should have correct format."""
        qrng = QRNG()
        uuid = qrng.random_uuid4()
        parts = uuid.split('-')
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


class TestQAOA:
    """Test QAOA MaxCut solver."""
    
    def test_triangle_maxcut(self):
        """Triangle has max cut of 2."""
        edges = [(0, 1), (1, 2), (2, 0)]
        qaoa = QAOA(edges, p=1)
        result = qaoa.optimize(shots=256, seed=42)
        assert result.cut_value() == 2
    
    def test_random_graph(self):
        """Random graph should work."""
        qaoa = QAOA.random_graph(5, edge_prob=0.5, seed=42)
        result = qaoa.optimize(shots=256, seed=42)
        assert result.cut_value() >= 0


class TestBB84:
    """Test BB84 QKD protocol."""
    
    def test_secure_channel_low_error(self):
        """Secure channel should have low error rate."""
        bb84 = BB84(key_length=64)
        result = bb84.run(with_eavesdropper=False, seed=42)
        assert result.error_rate < 0.05
        assert not result.eavesdropper_detected
    
    def test_eavesdropper_detected(self):
        """Eavesdropper should cause high error rate."""
        bb84 = BB84(key_length=64)
        result = bb84.run(with_eavesdropper=True, seed=42)
        assert result.error_rate > 0.15
        assert result.eavesdropper_detected
    
    def test_key_generated(self):
        """Key should be generated."""
        bb84 = BB84(key_length=64)
        result = bb84.run(seed=42)
        assert len(result.key) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
