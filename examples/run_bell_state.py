"""Example: Run Bell state on tiny-qpu."""
import sys
sys.path.insert(0, 'src')

from tiny_qpu import QPU

print("=" * 50)
print("tiny-qpu: Bell State Example")
print("=" * 50)

qpu = QPU(num_qubits=2)
qpu.load_program("programs/bell_state.qasm")
result = qpu.run(shots=1000)

print("\nMeasurement Results:")
for state, count in sorted(result.counts.items()):
    print(f"  |{state}⟩: {count:4d} ({100*count/1000:5.1f}%)")

print("\nExpected: ~50% |00⟩ and ~50% |11⟩ (entangled!)")
