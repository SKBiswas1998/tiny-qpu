# grover_2qubit.qasm - Search for |11⟩
.qubits 2
.classical 2

# Initialize superposition
H q[0]
H q[1]

# Oracle: mark |11⟩ with phase flip
CZ q[0], q[1]

# Diffusion operator
H q[0]
H q[1]
X q[0]
X q[1]
CZ q[0], q[1]
X q[0]
X q[1]
H q[0]
H q[1]

# Measure
MEASURE q[0], c[0]
MEASURE q[1], c[1]
