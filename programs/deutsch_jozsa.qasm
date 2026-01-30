# deutsch_jozsa.qasm - Determine if oracle is constant or balanced
.qubits 2
.classical 1

# Initialize
X q[1]
H q[0]
H q[1]

# Oracle (balanced function: CNOT)
CNOT q[0], q[1]

# Final Hadamard
H q[0]

# Measure - 0 means constant, 1 means balanced
MEASURE q[0], c[0]
