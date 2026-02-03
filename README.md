# tiny-qpu 🔬

A complete quantum processing unit simulator built from scratch in Python.
No Qiskit. No Cirq. Just NumPy and linear algebra.

[![Tests](https://img.shields.io/badge/tests-353%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Qubits](https://img.shields.io/badge/qubits-up%20to%2020-purple)]()
[![Chemical Accuracy](https://img.shields.io/badge/chemical%20accuracy-100%25-green)]()
[![OpenQASM](https://img.shields.io/badge/OpenQASM-2.0-orange)]()
[![Gates](https://img.shields.io/badge/gates-35%2B-blueviolet)]()

## What's Inside

**Core Simulator Engine**
- Statevector simulation up to 20 qubits with gate-by-gate tensor contraction
- Density matrix backend for noisy simulation with 6 quantum channels
- 35+ quantum gates: Pauli, Clifford, rotation, controlled, Ising, 3-qubit
- Pure-Python OpenQASM 2.0 parser (zero dependencies, regex tokenizer)
- Parameterized circuits with symbolic parameters and binding
- Optimized measurement: 7000x speedup via vectorized sampling
- Circuit depth analysis, inverse, composition, and ASCII visualization

**Quantum Applications**
- **QRNG** — Quantum random number generator
- **QAOA** — Approximate optimization for MaxCut
- **BB84** — Quantum key distribution with eavesdropper detection
- **VQE** — Variational eigensolver for molecular ground states
- **Shor's Algorithm** — Integer factorization via QPE

**Molecular Chemistry Benchmark Suite**
- Pre-computed Hamiltonians: H₂, HeH⁺, LiH, H₄ (Jordan-Wigner mapped)
- Multiple ansatze: hardware-efficient, Ry-linear, UCCSD-inspired
- Noise-aware benchmarking with depolarizing, amplitude damping, thermal relaxation
- CSV/JSON export for publication-ready results
- 100% chemical accuracy (<1 kcal/mol) on all molecules in clean simulation

**Advanced**
- **Noise Simulator** — Density matrix simulation with 7 quantum channels
- **Error Correction** — Bit flip, phase flip, Shor [[9,1,3]], Steane [[7,1,3]] codes
- **Quantum Fourier Transform** — Full QFT and inverse QFT circuits

## Architecture

![Architecture](diagrams/architecture.png)

The simulator is built in four layers: a **Circuit Builder** with OpenQASM 2.0 parsing at the top, a **Gate Library** (35+ gates) with parameter system and instruction IR in the middle, dual **Simulation Backends** (statevector O(2ⁿ) and density matrix O(4ⁿ)), and structured **Result Objects** at the output.

## Gate Library

![Gate Coverage](diagrams/gate_coverage.png)

35+ gates across 5 categories: single-qubit fixed (I, X, Y, Z, H, S, Sdg, T, Tdg, SX), single-qubit rotation (Rx, Ry, Rz, P, U1, U2, U3), two-qubit fixed (CNOT, CZ, SWAP, iSWAP, ECR), two-qubit parameterized (CP, CRx, CRy, CRz, Rxx, Ryy, Rzz), and three-qubit (CCX/Toffoli, CSWAP/Fredkin).

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

## OpenQASM 2.0 Support

Parse and execute standard OpenQASM circuits with zero external dependencies:

```python
from tiny_qpu.qasm import parse_qasm
from tiny_qpu.backends.statevector import StatevectorBackend

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

circuit = parse_qasm(qasm)
backend = StatevectorBackend()
result = backend.run(circuit, shots=1000)
print(result.bitstring_counts())  # {'00': ~500, '11': ~500}
```

Supports all standard gates, parameterized expressions (`pi/4`, `2*pi`, `-pi/2`), custom gate definitions, multiple quantum/classical registers, barriers, and measurements.

## Density Matrix & Noise Simulation

```python
from tiny_qpu import Circuit
from tiny_qpu.backends.density_matrix import DensityMatrixBackend, NoiseModel
from tiny_qpu.backends.density_matrix import depolarizing, amplitude_damping

# Build noise model
noise = NoiseModel()
noise.add_all_qubit_error(depolarizing(0.01))

# Simulate with noise
qc = Circuit(2).h(0).cx(0, 1)
backend = DensityMatrixBackend(noise_model=noise)
result = backend.run(qc)

print(f"Purity: {result.purity():.4f}")         # < 1.0 (mixed state)
print(f"Entropy: {result.von_neumann_entropy():.4f}")  # > 0 (information loss)
```

![Noise Channels](diagrams/noise_channels.png)

Six noise channels available: depolarizing, amplitude damping, phase damping, bit flip, phase flip, and thermal relaxation. Each preserves trace and positivity (completely positive trace-preserving maps).

## Molecular Chemistry Benchmarks

```python
from tiny_qpu.benchmark import ChemistryBenchmark
from tiny_qpu.benchmark.molecules import MoleculeLibrary

# List available molecules
MoleculeLibrary.list_molecules()

# Run quick benchmark (H2 + HeH+ at equilibrium)
bench = ChemistryBenchmark(seed=42)
suite = bench.run_quick()
print(suite.summary())

# Full benchmark: all molecules, multiple ansatze, noise levels
suite = bench.run_full()

# Noise sweep: how does error scale with hardware noise?
suite = bench.run_noise_sweep('H2', noise_levels=[0, 0.01, 0.05, 0.10])

# Export for papers
ChemistryBenchmark.export_csv(suite, "results.csv")
```

### Available Molecules

| Molecule | Qubits | Terms | Equilibrium Energy | Description |
|----------|--------|-------|--------------------|-------------|
| H₂ | 2 | 6 | -1.0967 Ha | Simplest molecular benchmark |
| HeH⁺ | 2 | 6 | -2.1543 Ha | Heteronuclear, astrochemistry |
| LiH | 4 | 21 | -1.3343 Ha | Key VQE benchmark (active space) |
| H₄ | 4 | 21 | -1.8193 Ha | Strong electron correlation |

### Benchmark Results (Clean Simulation)

| Molecule | Ansatz | Depth | Error (mHa) | Chemical Accuracy | Time |
|----------|--------|-------|-------------|-------------------|------|
| H₂ | hardware_efficient | 2 | 0.00 | ✓ | 0.9s |
| HeH⁺ | hardware_efficient | 2 | 0.00 | ✓ | 0.8s |
| LiH | hardware_efficient | 2 | 0.00 | ✓ | 3.2s |
| H₄ | ry_linear | 3 | 0.00 | ✓ | 3.7s |

## CLI

```bash
tiny-qpu run bell --shots 1000
tiny-qpu qrng --bits 256
tiny-qpu qaoa --graph triangle --rounds 2
tiny-qpu bb84 --key-bits 128 --eavesdrop
tiny-qpu vqe --molecule h2 --bond-length 0.735

# Benchmarks
tiny-qpu benchmark --list
tiny-qpu benchmark --quick
tiny-qpu benchmark --molecule H2 --noise-sweep
tiny-qpu benchmark --full --export results.csv
```

## Factor Integers with Shor's Algorithm

```python
from tiny_qpu.algorithms import shor_factor

result = shor_factor(15, seed=42)
print(result)  # Shor: 15 = 5 x 3 (a=8, r=4, attempts=1)
```

## Simulate Real Hardware Noise

```python
from tiny_qpu.noise import NoiseModel, depolarizing

noise = NoiseModel()
noise.add_all_qubit_error(depolarizing(0.01))
noise.add_readout_error(0.03)

qc = Circuit(2).h(0).cx(0, 1).measure_all()
noisy = noise.run(qc, shots=1000)
```

## Project Structure

```
tiny_qpu/
├── circuit.py             # Circuit builder with parameters & QASM export
├── gates.py               # 35+ gate library with GATE_REGISTRY
├── backends/
│   ├── statevector.py     # O(2^n) statevector simulation engine
│   └── density_matrix.py  # O(4^n) density matrix with 6 noise channels
├── qasm/
│   └── parser.py          # Pure-Python OpenQASM 2.0 parser
├── core/                  # Legacy statevector engine
├── apps/                  # QRNG, QAOA, BB84, VQE
├── algorithms/            # Shor's factoring, QPE, QFT
├── benchmark/             # Chemistry benchmark suite
│   ├── molecules.py       # H2, HeH+, LiH, H4 Hamiltonians
│   └── __init__.py        # Benchmark runner + export
├── noise/                 # Legacy noise module
├── error_correction/      # Bit flip, Shor, Steane codes
├── cli/                   # Command-line interface
└── visualization.py       # ASCII circuit diagrams
```

## Visualizations

### Bell State Measurement

![Bell State](diagrams/bell_state.png)

10,000-shot measurement of the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, showing near-perfect 50/50 distribution between |00⟩ and |11⟩ outcomes with zero probability for |01⟩ and |10⟩.

### Simulation Performance

![Performance](diagrams/performance.png)

Statevector simulation scaling from 2 to 20 qubits, confirming the expected exponential O(2ⁿ) growth. A 20-qubit circuit (1M amplitudes) completes in under 1 second.

### Test Coverage

![Test Summary](diagrams/test_summary.png)

### Potential Energy Surfaces

All four molecules showing characteristic potential wells with equilibrium geometries:

![Potential Energy Surfaces](docs/images/pes_curves.png)

### Molecule Benchmark Overview

Accuracy, runtime, and qubit comparison across all molecules:

![Molecule Overview](docs/images/molecule_overview.png)

### Noise Analysis

Error growth and fidelity decay under depolarizing noise for H2:

![Noise Analysis](docs/images/noise_H2.png)

### Animations

**VQE Optimization** — Watch the variational optimizer converge to the ground state energy:

![VQE Optimization](docs/images/vqe_optimization.gif)

**Bloch Sphere** — Qubit state evolution through quantum gates (H, X, Z, T, Y):

![Bloch Sphere](docs/images/bloch_sphere.gif)

**H2 Potential Energy Surface Scan** — Bond length sweep revealing the energy minimum:

![PES Scan](docs/images/pes_H2.gif)

**Noise Degradation** — How depolarizing noise destroys quantum state fidelity:

![Noise Degradation](docs/images/noise_degradation.gif)

## Performance

| Benchmark | Result |
|-----------|--------|
| 20-qubit Hadamard | < 1s |
| 10k shots (10 qubits) | 0.002s |
| Shor factor(15) | 0.10s |
| VQE H₂ ground state | 0.9s |
| Full benchmark (4 molecules) | ~40s |
| 353 tests | 1.2s |

## Tests

```bash
python -m pytest tests/ -v  # 353 tests, all passing
```

89 comprehensive tests covering 11 categories: gate algebra, state preparation, circuit construction, statevector accuracy, measurement statistics, density matrix simulation, QASM round-trip, cross-backend consistency, edge cases, performance scaling, and quantum information theory (no-cloning theorem, quantum teleportation).

Plus 264 unit tests across gates, circuit builder, statevector backend, density matrix backend, QASM parser, and legacy API compatibility.

## Built Without

No Qiskit. No Cirq. No PennyLane. Just:
- **NumPy** for linear algebra
- **SciPy** for VQE optimization
- **matplotlib** for diagram generation (optional)
- Pure Python for everything else
