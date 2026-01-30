"""Tests for tiny-qpu."""
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from tiny_qpu import QPU

class TestQPU:
    def test_bell_state(self):
        """Bell state should give 50/50 |00⟩ and |11⟩."""
        qpu = QPU(num_qubits=2)
        qpu.load_program("programs/bell_state.qasm")
        result = qpu.run(shots=1000, seed=42)
        
        # Should only have 00 and 11
        assert set(result.counts.keys()).issubset({'00', '11'})
        
        # Should be roughly 50/50
        total = sum(result.counts.values())
        for count in result.counts.values():
            ratio = count / total
            assert 0.35 < ratio < 0.65
    
    def test_grover_finds_target(self):
        """Grover should find |11⟩ with high probability."""
        qpu = QPU(num_qubits=2)
        qpu.load_program("programs/grover_2qubit.qasm")
        result = qpu.run(shots=1000, seed=42)
        
        total = sum(result.counts.values())
        assert result.counts.get('11', 0) / total > 0.8
    
    def test_deutsch_jozsa_balanced(self):
        """Deutsch-Jozsa with CNOT oracle should return 1 (balanced)."""
        qpu = QPU(num_qubits=2, num_classical=1)
        qpu.load_program("programs/deutsch_jozsa.qasm")
        result = qpu.run(shots=100, seed=42)
        
        # Should always measure 1 for balanced oracle
        assert result.counts.get('1', 0) == 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
