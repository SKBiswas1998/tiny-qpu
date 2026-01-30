# teleportation.qasm - Quantum teleportation protocol
.qubits 3
.classical 3

# Prepare state to teleport on q[0]
H q[0]

# Create Bell pair between q[1] and q[2]
H q[1]
CNOT q[1], q[2]

# Bell measurement
CNOT q[0], q[1]
H q[0]
MEASURE q[0], c[0]
MEASURE q[1], c[1]

# Measure teleported state
MEASURE q[2], c[2]
