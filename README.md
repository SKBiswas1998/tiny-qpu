# tiny-qpu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A minimal, fast quantum computing library with practical applications.

**Why tiny-qpu?**
- ⚡ **Fast**: <500ms import (vs Qiskit's 5+ seconds)
- 🎯 **Simple**: Fluent API - quantum circuits in 3 lines, not 30
- 🔧 **Practical**: Real applications (QRNG, QAOA, BB84), not just demos
- 📚 **Educational**: See quantum state evolution after each gate

## Installation
```bash
pip install tiny-qpu
```

Or from source:
```bash
git clone https://github.com/SKBiswas1998/tiny-qpu.git
cd tiny-qpu
pip install -e .
```

## Quick Start

### As a Library
```python
from tiny_qpu import Circuit

# Create a Bell state in 3 lines
qc = Circuit(2).h(0).cx(0, 1).measure_all()
result = qc.run(shots=1000)
print(result.counts)  # {'00': ~500, '11': ~500}
```

### From Command Line
```bash
# Generate quantum random numbers
tiny-qpu qrng --bytes 32 --hex

# Solve MaxCut optimization
tiny-qpu maxcut --random 6

# Run BB84 key distribution demo
tiny-qpu bb84 --demo

# Run Bell state demo
tiny-qpu run bell
```

## Applications

### 🎲 Quantum Random Number Generator (QRNG)

Generate true random numbers using quantum superposition:
```python
from tiny_qpu.apps import QRNG

qrng = QRNG()

# Random bytes (for cryptographic keys)
key = qrng.random_bytes(32)
print(f"256-bit key: {key.hex()}")

# Random integers
dice = qrng.random_int(1, 7)  # Roll a die

# Random UUID
uuid = qrng.random_uuid4()
```

### 📊 QAOA MaxCut Solver

Solve graph optimization problems using the Quantum Approximate Optimization Algorithm:
```python
from tiny_qpu.apps import QAOA, solve_maxcut

# Define a graph (edges)
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

# Solve MaxCut
result = solve_maxcut(edges, p=2)
print(f"Best partition: {result.bitstring}")
print(f"Cut value: {result.cut_value()}")

# Or use the full API
qaoa = QAOA(edges, p=2)
result = qaoa.optimize(shots=1024)
```

### 🔐 BB84 Quantum Key Distribution

Simulate quantum cryptography with eavesdropper detection:
```python
from tiny_qpu.apps import BB84

# Generate a shared secret key
bb84 = BB84(key_length=256)
result = bb84.run()

print(f"Key: {result.key.hex()}")
print(f"Error rate: {result.error_rate:.2%}")

# Simulate with eavesdropper (Eve)
result = bb84.run(with_eavesdropper=True)
if result.eavesdropper_detected:
    print("⚠️ Eavesdropper detected!")
```

## Educational Mode

See the quantum state after each gate:
```python
from tiny_qpu import Circuit

# Educational mode shows state evolution
with Circuit(2, educational=True) as qc:
    qc.h(0)      # Shows superposition
    qc.cx(0, 1)  # Shows entanglement
```

Output:
```
After H on qubit(s) [0]:
  |00⟩:  0.7071  (prob: 50.00%)
  |10⟩:  0.7071  (prob: 50.00%)

After CX on qubit(s) [0, 1]:
  |00⟩:  0.7071  (prob: 50.00%)
  |11⟩:  0.7071  (prob: 50.00%)
```

## Supported Gates

### Single-Qubit Gates
| Gate | Method | Description |
|------|--------|-------------|
| I | `.i(q)` | Identity |
| X | `.x(q)` | Pauli-X (NOT) |
| Y | `.y(q)` | Pauli-Y |
| Z | `.z(q)` | Pauli-Z |
| H | `.h(q)` | Hadamard |
| S | `.s(q)` | S gate (√Z) |
| T | `.t(q)` | T gate (π/8) |
| Rx | `.rx(θ, q)` | X rotation |
| Ry | `.ry(θ, q)` | Y rotation |
| Rz | `.rz(θ, q)` | Z rotation |

### Two-Qubit Gates
| Gate | Method | Description |
|------|--------|-------------|
| CNOT | `.cx(c, t)` | Controlled-X |
| CZ | `.cz(c, t)` | Controlled-Z |
| SWAP | `.swap(q1, q2)` | Swap qubits |
| CRz | `.crz(θ, c, t)` | Controlled Rz |
| RZZ | `.rzz(θ, q1, q2)` | ZZ interaction |

### Three-Qubit Gates
| Gate | Method | Description |
|------|--------|-------------|
| Toffoli | `.ccx(c1, c2, t)` | CCX (AND gate) |
| Fredkin | `.cswap(c, t1, t2)` | Controlled SWAP |

## Architecture
```
┌─────────────────────────────────────────────────┐
│                    tiny-qpu                      │
├─────────────────────────────────────────────────┤
│  Circuit API    →  StateVector  →  Measurement  │
│  (fluent/chain)    (tensor ops)    (sampling)   │
├─────────────────────────────────────────────────┤
│                  Applications                    │
│   QRNG  │  QAOA (MaxCut)  │  BB84 (QKD)        │
├─────────────────────────────────────────────────┤
│                      CLI                         │
│  tiny-qpu qrng | maxcut | bb84 | run            │
└─────────────────────────────────────────────────┘
```

## Performance

| Qubits | Memory | tiny-qpu | Qiskit |
|--------|--------|----------|--------|
| 10 | 16 KB | ✅ | ✅ |
| 15 | 512 KB | ✅ | ✅ |
| 20 | 16 MB | ✅ | ✅ |
| 25 | 512 MB | ✅ | ✅ |

Import time comparison:
- **tiny-qpu**: ~350ms
- **Qiskit**: ~5-10 seconds

## Comparison with Other Frameworks

| Feature | tiny-qpu | Qiskit | Cirq |
|---------|----------|--------|------|
| Import time | <500ms | 5-10s | 2-3s |
| Dependencies | 2 | 50+ | 20+ |
| Learning curve | Low | High | Medium |
| Built-in QRNG | ✅ | ❌ | ❌ |
| Built-in QAOA | ✅ | Plugin | Plugin |
| Built-in BB84 | ✅ | ❌ | ❌ |
| Educational mode | ✅ | ❌ | ❌ |
| Hardware backends | ❌ | ✅ | ✅ |

**tiny-qpu is ideal for:**
- Rapid prototyping
- Education and learning
- Embedded applications
- When you don't need real hardware

## References

- [Quantum Computing: An Applied Approach](https://link.springer.com/book/10.1007/978-3-030-23922-0)
- [QAOA Original Paper](https://arxiv.org/abs/1411.4028)
- [BB84 Protocol](https://en.wikipedia.org/wiki/BB84)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Part of the Quantum Computing Portfolio by SK Biswas*

**GitHub**: https://github.com/SKBiswas1998/tiny-qpu
