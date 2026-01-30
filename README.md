# tiny-qpu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal Quantum Processing Unit simulator. Inspired by [tiny-gpu](https://github.com/adam-maj/tiny-gpu).

## Architecture
```
┌─────────────────────────────────────────────────┐
│                    tiny-qpu                      │
├─────────────────────────────────────────────────┤
│  Program Memory → Decoder → Scheduler           │
│                       ↓                          │
│              Gate Execution Unit                 │
│                       ↓                          │
│                Qubit Register                    │
│   |ψ⟩ = α₀|00⟩ + α₁|01⟩ + α₂|10⟩ + α₃|11⟩      │
│                       ↓                          │
│              Measurement Unit                    │
│                       ↓                          │
│             Classical Register                   │
└─────────────────────────────────────────────────┘
```

## Instruction Set Architecture (ISA)

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| H | `H q[n]` | Hadamard gate |
| X | `X q[n]` | Pauli-X (NOT) gate |
| Y | `Y q[n]` | Pauli-Y gate |
| Z | `Z q[n]` | Pauli-Z gate |
| CNOT | `CNOT q[c], q[t]` | Controlled-NOT |
| CZ | `CZ q[c], q[t]` | Controlled-Z |
| MEASURE | `MEASURE q[n], c[m]` | Measure qubit to classical bit |
| RESET | `RESET q[n]` | Reset qubit to \|0⟩ |

## Quick Start
```python
from tiny_qpu import QPU

qpu = QPU(num_qubits=2)
qpu.load_program("programs/bell_state.qasm")
result = qpu.run(shots=1000)
print(result.counts)  # {'00': 503, '11': 497}
```

## Example: Bell State
```asm
# bell_state.qasm
.qubits 2
.classical 2

H q[0]
CNOT q[0], q[1]
MEASURE q[0], c[0]
MEASURE q[1], c[1]
```

## From GPU to QPU

This project maps classical GPU concepts to quantum:

| GPU | QPU |
|-----|-----|
| Thread registers | Qubit register (state vector) |
| ALU | Gate execution unit |
| SIMD execution | Quantum parallelism via superposition |
| Memory controller | Measurement unit |

## References

- [tiny-gpu](https://github.com/adam-maj/tiny-gpu) - Inspiration
- Nielsen & Chuang (2010). Quantum Computation and Quantum Information

## License

MIT License

---
*Capstone project of the Quantum Computing Portfolio by SK Biswas*
