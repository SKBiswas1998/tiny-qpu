## Stabilizer Simulator — Interactive Demo

The stabilizer backend uses the **CHP algorithm** (Aaronson & Gottesman, 2004) to simulate Clifford circuits
on a binary symplectic tableau instead of a full statevector. This means **O(n²) memory** instead of O(2ⁿ) —
enabling simulation of circuits that are physically impossible with conventional methods.

<p align="center">
  <img src="docs/images/stabilizer_interactive.png" alt="Interactive stabilizer simulator" width="820"/>
</p>

Build circuits gate-by-gate and watch the stabilizer tableau update in real time.
Each Pauli operator is color-coded: <b style="color:#e74c3c">X</b> <b style="color:#3498db">Z</b> <b style="color:#f39c12">Y</b> — making entanglement structure immediately visible.

### 1,000-Qubit GHZ State in Milliseconds

<p align="center">
  <img src="docs/images/stabilizer_scale.png" alt="1000-qubit scale test" width="820"/>
</p>

The same GHZ state that would require **10²⁹² GB** as a statevector runs in **~16 ms** on the stabilizer backend
using less than **4 MB** of memory. Measurements on the first and last qubit are always perfectly correlated —
verified across 1,000 qubits.

### Try It

Open `visualizations/stabilizer_demo.html` in any browser — no build step, no dependencies:

```
# From the repo root
start visualizations/stabilizer_demo.html    # Windows
open visualizations/stabilizer_demo.html     # macOS
xdg-open visualizations/stabilizer_demo.html # Linux
```

**Four tabs to explore:**

| Tab | What it does |
|-----|-------------|
| **Interactive** | Build circuits gate-by-gate, live tableau view, measure qubits |
| **Demos** | Bell pair, GHZ-4, quantum teleportation, QEC 3-bit code, cluster state |
| **Scale Test** | 10 → 2,000 qubit GHZ states with timing benchmarks |
| **Benchmark** | Statevector vs stabilizer performance comparison chart |

### Python Backend

The demo is powered by the same algorithm as the Python backend in `src/tiny_qpu/backends/stabilizer.py`:

```python
from tiny_qpu.backends.stabilizer import StabilizerBackend

# 1000-qubit GHZ state
sim = StabilizerBackend(1000)
sim.h(0)
for i in range(1, 1000):
    sim.cx(0, i)

# All qubits perfectly correlated
print(sim.stabilizers()[:3])
# ['+XXXXXXXXXXXX...', '+ZZIIIIIIIII...', '+IZZIIIIIIII...']

result = sim.measure(0)
print(sim.measure(999) == result)  # Always True
```

**Supported gates:** H, S, S†, X, Y, Z, CNOT, CZ, SWAP
**Test coverage:** 74 tests covering gates, measurement, cross-validation, scale, and gate algebra
