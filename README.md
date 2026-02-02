# tiny-qpu 🔬

A complete quantum processing unit simulator built from scratch in Python.
No Qiskit. No Cirq. Just NumPy and linear algebra.

[![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Qubits](https://img.shields.io/badge/qubits-up%20to%2020-purple)]()

## What's Inside

**Core Simulator**
- Statevector simulation up to 20 qubits
- 20+ quantum gates (Pauli, Clifford, rotation, controlled, 3-qubit)
- Optimized measurement: 7000x speedup via vectorized sampling
- Circuit depth analysis and ASCII visualization

**Applications**
- **QRNG** — Quantum random number generator (bits, bytes, ints, floats)
- **QAOA** — Approximate optimization for MaxCut problems
- **BB84** — Quantum key distribution with eavesdropper detection
- **VQE** — Variational eigensolver for H₂ molecular ground state

**Advanced**
- **Shor's Algorithm** — Integer factorization via quantum period finding (QPE)
- **Noise Simulator** — Density matrix simulation with depolarizing, amplitude damping, phase damping, thermal relaxation, and readout errors
- **Error Correction** — Bit flip [[3,1,1]], phase flip, Shor [[9,1,3]], and Steane [[7,1,3]] codes
- **Quantum Fourier Transform** — Full QFT and inverse QFT circuits

## Quick Start
```bash
pip install -e .
```
```python
from tiny_qpu import Circuit

# Bell state
qc = Circuit(2).h(0).cx(0, 1).measure_all()
result = qc.run(shots=1000)
print(result.counts)  # {'00': ~500, '11': ~500}
```

## CLI
```bash
tiny-qpu run "H 0; CX 0 1; MEASURE 0; MEASURE 1" --shots 1000
tiny-qpu qrng --bits 256
tiny-qpu qaoa --graph triangle --rounds 2
tiny-qpu bb84 --key-bits 128 --eavesdrop
tiny-qpu vqe --molecule h2 --bond-length 0.735
```

## Factor Integers with Shor's Algorithm
```python
from tiny_qpu.algorithms import shor_factor

result = shor_factor(15, seed=42)
print(result)  # Shor: 15 = 5 x 3 (a=8, r=4, attempts=1)

result = shor_factor(91, seed=42)
print(result)  # Shor: 91 = 13 x 7 (a=71, r=12, attempts=2)
```

## Simulate Real Hardware Noise
```python
from tiny_qpu import Circuit
from tiny_qpu.noise import NoiseModel, depolarizing, depolarizing_2q

noise = NoiseModel()
noise.add_all_qubit_error(depolarizing(0.01))
noise.add_gate_error('CX', depolarizing_2q(0.02))
noise.add_readout_error(0.03)

qc = Circuit(2).h(0).cx(0, 1).measure_all()

clean = qc.run(shots=1000)          # {'00': 500, '11': 500}
noisy = noise.run(qc, shots=1000)   # {'00': 471, '01': 40, '10': 46, '11': 443}
```

Or use a preset hardware model:
```python
noise = NoiseModel.from_backend(t1=50e3, t2=70e3, readout_error=0.02)
```

## Quantum Error Correction
```python
from tiny_qpu.error_correction import BitFlipCode, compare_codes

result = BitFlipCode().demonstrate(error_rate=0.05, shots=10000)
print(f"Physical error: {result.physical_error_rate:.4f}")
print(f"Logical error:  {result.logical_error_rate:.4f}")
print(f"Improvement:    {result.improvement:.1f}x")

# Compare all codes
compare_codes(error_rates=[0.01, 0.05, 0.10])
```

## VQE Molecular Simulation
```python
from tiny_qpu.apps import VQE, MolecularHamiltonian

h2 = MolecularHamiltonian.H2(bond_length=0.735)  # Angstroms
vqe = VQE(h2, depth=3)
result = vqe.run(maxiter=200)

print(f"Ground state energy: {result.energy:.6f} Ha")
print(f"Exact energy:        {h2.exact_ground_state():.6f} Ha")
print(f"Error:               {abs(result.energy - h2.exact_ground_state()):.6f} Ha")
```

## Architecture
```
tiny_qpu/
├── core/              # Statevector engine, gates, circuits
│   ├── statevector.py # Tensor-based simulation (up to 20 qubits)
│   ├── gates.py       # 20+ quantum gates with unitarity checks
│   └── circuit.py     # Fluent API with optimized measurement
├── apps/              # Quantum applications
│   ├── qrng.py        # Quantum random number generation
│   ├── qaoa.py        # Combinatorial optimization
│   ├── bb84.py        # Quantum cryptography
│   └── vqe.py         # Variational quantum eigensolver
├── algorithms/        # Famous quantum algorithms
│   └── __init__.py    # Shor's factoring, QPE, QFT
├── noise/             # Hardware noise simulation
│   └── __init__.py    # Density matrices, quantum channels, noise models
├── error_correction/  # QEC codes
│   └── __init__.py    # Bit flip, phase flip, Shor, Steane codes
├── cli/               # Command-line interface
└── visualization.py   # ASCII circuit diagrams
```

## Performance

| Benchmark | Result |
|-----------|--------|
| 20-qubit Hadamard | < 1s |
| 10k shots (10 qubits) | 0.002s |
| QAOA 5-node graph | 0.04s |
| Shor factor(15) | 0.10s |
| VQE H₂ ground state | 0.18s |
| Factor(91) | ~10s |

## Tests
```bash
python -m pytest tests/ -v  # 37 tests, all passing
```

## Built Without

No Qiskit. No Cirq. No PennyLane. Just:
- **NumPy** for linear algebra
- **SciPy** for VQE optimization
- Pure Python for everything else
