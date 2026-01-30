# bell_state.qasm - Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
.qubits 2
.classical 2

H q[0]           # Put q[0] in superposition
CNOT q[0], q[1]  # Entangle q[0] and q[1]
MEASURE q[0], c[0]
MEASURE q[1], c[1]
