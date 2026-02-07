<p align="center">
  <img src="assets/tiny-qpu-logo-512.png" alt="tiny-qpu" width="200"/>
</p>

<h1 align="center">tiny-qpu</h1>

<p align="center">
  <em>A quantum processing unit simulator — Python library, CLI tool, and native desktop application.</em>
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
  <a href="#tests"><img src="https://img.shields.io/badge/tests-386%20passing-brightgreen.svg" alt="Tests"/></a>
  <a href="#gate-library"><img src="https://img.shields.io/badge/gates-35+-purple.svg" alt="35+ Gates"/></a>
  <a href="#openqasm-20"><img src="https://img.shields.io/badge/OpenQASM-2.0-orange.svg" alt="OpenQASM 2.0"/></a>
  <a href="#quantum-lab-desktop-app"><img src="https://img.shields.io/badge/Quantum%20Lab-Desktop%20App-00d4ff.svg" alt="Quantum Lab"/></a>
  <a href="#windows-installer"><img src="https://img.shields.io/badge/Windows-Installer-0078d4.svg" alt="Windows Installer"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/></a>
</p>

---

## What's New (February 7, 2026)

### Phase 2: Interactive Quantum Lab — Desktop Application

- **Native Windows Desktop App** — Built with pywebview + Flask. No browser needed. Real Windows window with title bar, taskbar icon, minimize/maximize/close.
- **Visual Circuit Builder** — Click to place 20 quantum gates on up to 8 qubits. Drag to reorder. Undo with Ctrl+Z.
- **Bloch Sphere Visualization** — Real-time 3D rendering of single-qubit quantum states.
- **Step-by-Step Execution** — Execute one gate at a time, watch the quantum state evolve through each operation.
- **8 Preset Circuits** — Bell State, GHZ, Quantum Teleportation, Deutsch-Jozsa, QFT, Grover's Search, Phase Kickback, Bit-Flip Error Correction.
- **OpenQASM Import/Export** — Load circuits from QASM files, export your designs to standard format.
- **In-App Help System** — Press `?` for a 5-tab reference covering Quick Start, Gates, Shortcuts, Concepts, and About.
- **Custom Logo & Branding** — Atom-orbital Bloch sphere logo in SVG, ICO (multi-size), and PNG formats.
- **16-Page PDF Manual** — Dark-themed manual covering every feature, gate reference with matrices, quantum concepts glossary.
- **Windows Installer** — Inno Setup installer with Start Menu shortcuts, desktop icon, and uninstaller. Also distributable as standalone .exe (33 MB).
- **Automated Build Pipeline** — Single command (`python build_installer.py`) generates icons, builds the exe, creates the manual, and compiles the installer.

---

## What is tiny-qpu?

tiny-qpu is a quantum computing toolkit that works at three levels:

**As a Python library** — build quantum circuits programmatically, run simulations with statevector or density matrix backends, explore algorithms like Shor's factoring, Grover's search, VQE, QAOA, and BB84 quantum key distribution.

**As a desktop application** — a native Windows app with a visual circuit builder, Bloch sphere visualization, step-by-step execution, and 8 preset quantum circuits. No Python required.

**As a learning tool** — 16-page manual, in-app help, step-by-step mode, and educational presets that walk you through fundamental quantum phenomena.

```python
from tiny_qpu.circuit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
result = qc.simulate(shots=1000)
print(result.counts)  # {'00': ~500, '11': ~500}
```

---

## Quantum Lab (Desktop App)

<p align="center">
  <strong>Build quantum circuits visually. Watch quantum states evolve in real time.</strong>
</p>

The Interactive Quantum Lab is a native Windows desktop application. Double-click `tiny-qpu.exe` and start building quantum circuits immediately.

### Features

| Feature | Description |
|---------|-------------|
| **Circuit Builder** | Click to place gates on qubits. Visual grid with wire connections. |
| **20 Quantum Gates** | H, X, Y, Z, S, T, √X, Rx, Ry, Rz, P, CX, CZ, SWAP, CRz, CCX, CSWAP, Measure |
| **Bloch Sphere** | Real-time 3D visualization of single-qubit states |
| **Step Mode** | Execute gate-by-gate, watch probabilities change at each step |
| **8 Presets** | Bell, GHZ, Teleportation, Deutsch-Jozsa, QFT, Grover, Phase Kickback, Bit-Flip Code |
| **QASM I/O** | Import and export circuits in OpenQASM 2.0 format |
| **Help Modal** | Press `?` for built-in reference (gates, shortcuts, concepts) |
| **Keyboard Shortcuts** | R=Run, S=Step, C=Clear, H/X/Y/M=Quick gate, Ctrl+Z=Undo |

### Design

Deep-space aesthetic with electric cyan and warm amber accents. Three-panel layout: gate palette (left), circuit builder (center), results panel (right) with Bloch sphere, probability bars, histogram, and amplitude table.

### Launch Options

```bash
# Native desktop window (default)
python tiny_qpu_launcher.py

# Browser mode
python tiny_qpu_launcher.py --browser

# Custom port
python tiny_qpu_launcher.py --port 9000

# Or just double-click the exe
.\dist\tiny-qpu.exe
```

---

## Installation

### From Source (library + dashboard)

```bash
git clone https://github.com/SKBiswas1998/tiny-qpu.git
cd tiny-qpu
pip install -e .
```

### Desktop App Dependencies

```bash
pip install flask pywebview     # Required for Quantum Lab
pip install reportlab           # Optional: PDF manual generation
pip install scipy matplotlib    # Optional: visualizations
```

### Windows Installer

Download `tiny-qpu-setup.exe` from [Releases](https://github.com/SKBiswas1998/tiny-qpu/releases). Run the installer — creates Start Menu shortcut, desktop icon, and uninstaller. No Python needed.

---

## Core Simulator Engine

### Circuit Builder API

```python
from tiny_qpu.circuit import QuantumCircuit

# Build a 3-qubit circuit
qc = QuantumCircuit(3)
qc.h(0)                    # Hadamard on qubit 0
qc.cx(0, 1)                # CNOT: control=0, target=1
qc.ccx(0, 1, 2)            # Toffoli gate
qc.rx(0.5, 0)              # Rx rotation by 0.5 radians
qc.measure_all()

result = qc.simulate(shots=1024)
print(result.counts)
print(result.statevector)
```

### OpenQASM 2.0

```python
from tiny_qpu.qasm import parse_qasm

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

circuit = parse_qasm(qasm)
result = circuit.simulate(shots=1000)
```

### Dual Simulation Backends

```python
# Statevector backend (pure states, fast, up to ~20 qubits)
result = qc.simulate(backend='statevector', shots=1000)

# Density matrix backend (mixed states, noise simulation, up to ~15 qubits)
result = qc.simulate(backend='density_matrix', shots=1000)
```

---

## Gate Library (35+)

| Category | Gates |
|----------|-------|
| **Single-qubit fixed** | H, X, Y, Z, S, S†, T, T†, √X, I |
| **Single-qubit rotation** | Rx(θ), Ry(θ), Rz(θ), P(θ), U1(λ), U2(φ,λ), U3(θ,φ,λ) |
| **Two-qubit** | CX (CNOT), CY, CZ, CH, SWAP, CRx, CRy, CRz, CP, iSWAP, √SWAP |
| **Three-qubit** | CCX (Toffoli), CSWAP (Fredkin), CCZ |

---

## Quantum Applications

### Quantum Random Number Generator (QRNG)
```python
from applications.qrng import QuantumRNG
rng = QuantumRNG()
random_bits = rng.generate(256)         # 256 quantum random bits
random_int = rng.random_int(1, 100)     # Random integer in range
```

### QAOA (Quantum Approximate Optimization)
```python
from applications.qaoa import QAOAOptimizer
optimizer = QAOAOptimizer(graph, p=2)
solution = optimizer.optimize()          # MaxCut approximation
```

### BB84 Quantum Key Distribution
```python
from applications.bb84 import BB84Protocol
alice, bob = BB84Protocol.run(key_length=128)
```

### VQE (Variational Quantum Eigensolver)
```python
from applications.vqe import VQERunner
energy = VQERunner(hamiltonian).run()    # Ground state energy
```

### Shor's Factoring Algorithm
```python
from algorithms.shor import ShorFactoring
factors = ShorFactoring(15).factor()     # Returns (3, 5)
```

---

## Molecular Chemistry Benchmarks

Ground state energy calculations using VQE:

| Molecule | Qubits | Bond Length (Å) | Energy (Hartree) |
|----------|--------|-----------------|-------------------|
| H₂       | 2      | 0.735           | -1.137            |
| LiH      | 4      | 1.546           | -7.882            |
| BeH₂     | 6      | 1.326           | -15.835           |
| H₂O      | 8      | 0.958           | -75.012           |

---

## Visualizations

### Static Plots
![Potential Energy Surfaces](docs/images/pes_curves.png)
![Molecule Overview](docs/images/molecule_overview.png)
![Noise Analysis](docs/images/noise_H2.png)

### Animations
![VQE Optimization](docs/images/vqe_optimization.gif)
![Bloch Sphere](docs/images/bloch_sphere.gif)
![PES Scan](docs/images/pes_H2.gif)
![Noise Degradation](docs/images/noise_degradation.gif)

---

## Project Structure

```
tiny-qpu/
├── src/tiny_qpu/
│   ├── circuit.py                 # QuantumCircuit fluent API
│   ├── gates.py                   # 35+ gate matrix definitions
│   ├── backends/
│   │   ├── statevector.py         # Pure state simulation
│   │   └── density_matrix.py      # Mixed state + noise channels
│   ├── qasm/
│   │   └── parser.py              # OpenQASM 2.0 parser
│   └── dashboard/
│       ├── __init__.py            # Package init with launch()
│       ├── server.py              # Flask backend (8 REST endpoints)
│       └── templates/
│           └── index.html         # Self-contained UI (70 KB)
├── algorithms/
│   └── shor.py                    # Shor's factoring algorithm
├── applications/
│   ├── qrng.py                    # Quantum random number generator
│   ├── qaoa.py                    # QAOA optimizer
│   ├── bb84.py                    # BB84 QKD protocol
│   └── vqe.py                     # Variational eigensolver
├── benchmarks/
│   └── molecules.py               # H₂, LiH, BeH₂, H₂O benchmarks
├── assets/
│   ├── tiny-qpu-logo.ico          # Windows icon (16-256px)
│   ├── tiny-qpu-logo.svg          # Vector logo
│   ├── tiny-qpu-logo.png          # 256px PNG
│   ├── tiny-qpu-logo-512.png      # 512px PNG
│   ├── installer_sidebar.bmp      # Inno Setup wizard image
│   └── installer_small.bmp        # Inno Setup header image
├── docs/images/                    # Visualization outputs
├── tests/                          # 386 tests
│   ├── test_dashboard.py          # 33 dashboard API tests
│   └── ...                        # Core simulator tests
├── tiny_qpu_launcher.py           # Desktop app entry point
├── tiny_qpu_manual.pdf            # 16-page dark-themed manual
├── build_installer.py             # Automated build pipeline
├── create_icon.py                 # Logo/icon asset generator
├── generate_manual.py             # PDF manual generator
├── patch_help_modal.py            # In-app help system patcher
├── tiny-qpu-setup.iss             # Inno Setup installer script
└── .gitignore
```

---

## Building the Desktop App

### One Command Build

```bash
pip install pyinstaller pillow pywebview flask numpy
python build_installer.py
```

The build pipeline automatically:

1. **Generates logo assets** — ICO (multi-size), PNG, BMP installer images from Python (no external tools)
2. **Builds `tiny-qpu.exe`** — Native Windows app via PyInstaller with embedded icon, `--windowed` (no console)
3. **Creates PDF manual** — 16-page dark-themed manual via reportlab
4. **Compiles Windows installer** — via Inno Setup (if installed), creates `installer_output/tiny-qpu-setup.exe`

### Build Outputs

| File | Size | Description |
|------|------|-------------|
| `dist/tiny-qpu.exe` | 33 MB | Standalone native Windows app |
| `tiny_qpu_manual.pdf` | 28 KB | 16-page user manual |
| `assets/tiny-qpu-logo.ico` | 23 KB | Multi-size Windows icon |
| `installer_output/tiny-qpu-setup.exe` | ~34 MB | Windows installer (with Inno Setup) |

### Windows Installer

The installer provides the full Windows experience:

- Welcome screen with tiny-qpu branding
- Install location selection
- Start Menu shortcuts (app + manual + uninstall)
- Desktop shortcut with custom icon
- "Launch after install" checkbox
- Clean uninstall via Add/Remove Programs

To build the installer, install [Inno Setup](https://jrsoftware.org/isdl.php) and re-run `python build_installer.py`.

---

## Dashboard API

The Quantum Lab backend exposes 8 REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the Quantum Lab UI |
| `/api/simulate` | POST | Run a quantum circuit simulation |
| `/api/step` | POST | Execute a single gate step |
| `/api/bloch` | POST | Get Bloch sphere coordinates |
| `/api/presets` | GET | List 8 preset circuits |
| `/api/presets/<name>` | GET | Load a specific preset |
| `/api/qasm/import` | POST | Parse OpenQASM to circuit |
| `/api/qasm/export` | POST | Export circuit to OpenQASM |

---

## Tests

```bash
pytest tests/ -v
```

**386 tests** across all modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| Core simulator (statevector) | 150+ | Circuit, gates, measurement |
| Core simulator (density matrix) | 80+ | Noise channels, mixed states |
| OpenQASM parser | 30+ | Parsing, gate mapping, edge cases |
| Applications (QRNG, QAOA, BB84, VQE) | 50+ | Algorithm correctness |
| Shor's algorithm | 10+ | Factoring verification |
| Molecular benchmarks | 20+ | Energy accuracy |
| Dashboard API | 33 | All 8 endpoints, presets, QASM I/O |

---

## Performance

| Metric | Value |
|--------|-------|
| Full test suite (386 tests) | ~1.5s |
| 10-qubit circuit, 100 gates | <50ms |
| Bell state, 1000 shots | <5ms |
| Max practical qubits (statevector) | ~20 |
| Max practical qubits (density matrix) | ~15 |
| Dashboard response time | <100ms |

---

## Development History

| Phase | Date | What Was Built |
|-------|------|----------------|
| **Phase 1** | Feb 3, 2026 | Core simulator engine: circuit builder, 35+ gates, statevector & density matrix backends, OpenQASM 2.0 parser. Tests: 41 → 353. |
| **Phase 2** | Feb 7, 2026 | Interactive Quantum Lab: Flask dashboard (8 endpoints, 63 KB frontend), 33 tests. Strategic vision defined. |
| **Phase 2.1** | Feb 7, 2026 | Desktop app: pywebview native window, Bloch sphere, step mode, 8 presets, QASM I/O, in-app help modal. |
| **Phase 2.2** | Feb 7, 2026 | Packaging: custom logo (SVG/ICO/PNG), 16-page PDF manual, PyInstaller exe (33 MB), Inno Setup Windows installer, automated build pipeline. |

---

## Built With

- **Python 3.10+** — Core language
- **NumPy** — Linear algebra engine
- **Flask** — Dashboard REST API
- **pywebview** — Native Windows rendering (Edge WebView2)
- **PyInstaller** — Executable packaging
- **Pillow** — Logo/icon generation
- **reportlab** — PDF manual creation
- **Inno Setup** — Windows installer compiler

---

## Roadmap

- [ ] Parameter-shift gradients for VQE/QAOA optimization
- [ ] PySCF integration for real molecular Hamiltonians
- [ ] Zero-noise extrapolation (ZNE) error mitigation
- [ ] Qiskit circuit import compatibility
- [ ] PyPI packaging (`pip install tiny-qpu`)
- [ ] CLI tool: `tiny-qpu serve`, `tiny-qpu demo`, `tiny-qpu benchmark`
- [ ] Educational mode with step-by-step explanations
- [ ] Linux and macOS desktop app support
- [ ] GPU-accelerated simulation backend

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for learning. Useful for real quantum algorithms. Ships as native Windows software.</sub>
</p>
