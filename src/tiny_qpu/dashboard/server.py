"""
tiny-qpu Dashboard Server.

A Flask application providing:
- REST API for quantum circuit simulation
- Interactive web-based circuit builder
- Live state visualization (Bloch sphere, amplitudes, histograms)

Usage:
    from tiny_qpu.dashboard import launch
    launch(port=8888)

    # Or via CLI:
    # tiny-qpu serve --port 8888
"""

from __future__ import annotations

import json
import webbrowser
import threading
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports for tiny-qpu core (graceful fallback for testing)
# ---------------------------------------------------------------------------

def _get_circuit_and_backend():
    """Import Circuit and StatevectorBackend from tiny-qpu."""
    from tiny_qpu.circuit import Circuit
    from tiny_qpu.backends.statevector import StatevectorBackend
    return Circuit, StatevectorBackend


def _get_qasm_parser():
    """Import QASM parser."""
    from tiny_qpu.qasm.parser import parse_qasm
    return parse_qasm


# ---------------------------------------------------------------------------
# Gate metadata for the frontend
# ---------------------------------------------------------------------------

GATE_CATALOG = [
    # Single-qubit fixed gates
    {"name": "h",   "label": "H",   "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "Hadamard — creates superposition", "color": "#00d4ff"},
    {"name": "x",   "label": "X",   "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "Pauli-X (NOT gate)", "color": "#ff6b6b"},
    {"name": "y",   "label": "Y",   "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "Pauli-Y gate", "color": "#51cf66"},
    {"name": "z",   "label": "Z",   "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "Pauli-Z (phase flip)", "color": "#845ef7"},
    {"name": "s",   "label": "S",   "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "S gate (√Z)", "color": "#fab005"},
    {"name": "t",   "label": "T",   "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "T gate (π/8)", "color": "#f783ac"},
    {"name": "sdg", "label": "S†",  "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "S-dagger gate", "color": "#fab005"},
    {"name": "tdg", "label": "T†",  "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "T-dagger gate", "color": "#f783ac"},
    {"name": "sx",  "label": "√X",  "category": "single", "n_qubits": 1, "n_params": 0,
     "description": "sqrt(X) gate", "color": "#ff6b6b"},

    # Rotation gates
    {"name": "rx",  "label": "Rx",  "category": "rotation", "n_qubits": 1, "n_params": 1,
     "description": "X-rotation by θ", "color": "#ff6b6b", "param_names": ["θ"]},
    {"name": "ry",  "label": "Ry",  "category": "rotation", "n_qubits": 1, "n_params": 1,
     "description": "Y-rotation by θ", "color": "#51cf66", "param_names": ["θ"]},
    {"name": "rz",  "label": "Rz",  "category": "rotation", "n_qubits": 1, "n_params": 1,
     "description": "Z-rotation by θ", "color": "#845ef7", "param_names": ["θ"]},
    {"name": "p",   "label": "P",   "category": "rotation", "n_qubits": 1, "n_params": 1,
     "description": "Phase gate P(θ)", "color": "#fab005", "param_names": ["θ"]},

    # Two-qubit gates
    {"name": "cx",   "label": "CX",   "category": "multi", "n_qubits": 2, "n_params": 0,
     "description": "CNOT (controlled-X)", "color": "#00d4ff"},
    {"name": "cz",   "label": "CZ",   "category": "multi", "n_qubits": 2, "n_params": 0,
     "description": "Controlled-Z", "color": "#845ef7"},
    {"name": "swap", "label": "SWAP", "category": "multi", "n_qubits": 2, "n_params": 0,
     "description": "Swap two qubits", "color": "#fab005"},
    {"name": "crz",  "label": "CRz",  "category": "multi", "n_qubits": 2, "n_params": 1,
     "description": "Controlled Rz(θ)", "color": "#845ef7", "param_names": ["θ"]},

    # Three-qubit gates
    {"name": "ccx",   "label": "CCX",   "category": "multi", "n_qubits": 3, "n_params": 0,
     "description": "Toffoli (CCNOT)", "color": "#00d4ff"},
    {"name": "cswap", "label": "CSWAP", "category": "multi", "n_qubits": 3, "n_params": 0,
     "description": "Fredkin (controlled SWAP)", "color": "#fab005"},

    # Measurement
    {"name": "measure", "label": "M", "category": "measure", "n_qubits": 1, "n_params": 0,
     "description": "Measure qubit", "color": "#868e96"},
]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _build_circuit(data: dict) -> Any:
    """Build a Circuit from the JSON circuit definition."""
    Circuit, _ = _get_circuit_and_backend()

    n_qubits = data.get("n_qubits", 2)
    gates = data.get("gates", [])

    qc = Circuit(n_qubits)

    for g in gates:
        name = g["name"].lower()
        qubits = g.get("qubits", [])
        params = g.get("params", [])

        if name == "measure":
            if hasattr(qc, 'measure'):
                qc.measure(qubits[0], qubits[0])
            continue

        # Map gate names to circuit methods
        method = getattr(qc, name, None)
        if method is None:
            # Try common aliases
            aliases = {"cnot": "cx", "ccnot": "ccx", "toffoli": "ccx", "fredkin": "cswap"}
            method = getattr(qc, aliases.get(name, ""), None)

        if method is None:
            continue

        # Call with appropriate argument order
        # tiny-qpu convention: params first, then qubits for parameterized gates
        # fixed gates: just qubits
        try:
            if params:
                method(*params, *qubits)
            else:
                method(*qubits)
        except Exception:
            # Fallback: try qubits first
            try:
                if params:
                    method(*qubits, *params)
                else:
                    method(*qubits)
            except Exception:
                pass

    return qc


def _simulate(data: dict) -> dict:
    """Run simulation and return comprehensive results."""
    _, StatevectorBackend = _get_circuit_and_backend()

    qc = _build_circuit(data)
    shots = data.get("shots", 1024)
    seed = data.get("seed", 42)

    backend = StatevectorBackend(seed=seed)

    # Check if circuit has measurements
    has_measure = any(g["name"].lower() == "measure" for g in data.get("gates", []))

    result = backend.run(qc, shots=shots)

    # Extract statevector
    sv = result.statevector
    n_qubits = data.get("n_qubits", 2)

    # Probabilities
    probs = np.abs(sv) ** 2

    # Format probabilities as bitstring -> probability
    prob_dict = {}
    for i, p in enumerate(probs):
        if p > 1e-10:
            bitstring = format(i, f'0{n_qubits}b')
            prob_dict[bitstring] = float(p)

    # Measurement counts
    counts = {}
    if hasattr(result, 'bitstring_counts'):
        try:
            counts = result.bitstring_counts()
        except Exception:
            pass
    if not counts and hasattr(result, 'counts') and result.counts:
        counts = {format(k, f'0{n_qubits}b'): v for k, v in result.counts.items()}
    if not counts:
        # Sample from probabilities
        rng = np.random.default_rng(seed)
        samples = rng.choice(len(probs), size=shots, p=probs)
        for s in samples:
            bs = format(s, f'0{n_qubits}b')
            counts[bs] = counts.get(bs, 0) + 1

    # Bloch sphere coordinates for each qubit
    bloch_coords = []
    for qubit in range(n_qubits):
        coords = _compute_bloch_coords(sv, n_qubits, qubit)
        bloch_coords.append(coords)

    # Statevector as [real, imag] pairs
    sv_list = [[float(c.real), float(c.imag)] for c in sv]

    # Amplitude details for display
    amplitudes = []
    for i, c in enumerate(sv):
        bitstring = format(i, f'0{n_qubits}b')
        amp = float(abs(c))
        phase = float(np.angle(c)) if amp > 1e-10 else 0.0
        prob = float(probs[i])
        amplitudes.append({
            "index": i,
            "bitstring": bitstring,
            "real": float(c.real),
            "imag": float(c.imag),
            "amplitude": amp,
            "phase": phase,
            "probability": prob,
        })

    # Circuit depth
    depth = getattr(qc, 'depth', len(data.get('gates', [])))

    return {
        "statevector": sv_list,
        "probabilities": prob_dict,
        "counts": counts,
        "bloch_coords": bloch_coords,
        "amplitudes": amplitudes,
        "n_qubits": n_qubits,
        "circuit_depth": depth,
        "shots": shots,
    }


def _compute_bloch_coords(statevector: np.ndarray, n_qubits: int, qubit: int) -> dict:
    """
    Compute Bloch sphere coordinates for a single qubit by partial trace.
    Returns {"x": float, "y": float, "z": float, "purity": float}.
    """
    dim = 2 ** n_qubits
    rho = np.outer(statevector, statevector.conj())

    # Partial trace to get single-qubit density matrix
    rho_qubit = _partial_trace_single(rho, n_qubits, qubit)

    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    x = float(np.trace(rho_qubit @ X).real)
    y = float(np.trace(rho_qubit @ Y).real)
    z = float(np.trace(rho_qubit @ Z).real)
    purity = float(np.trace(rho_qubit @ rho_qubit).real)

    return {"x": x, "y": y, "z": z, "purity": purity}


def _partial_trace_single(rho: np.ndarray, n_qubits: int, keep_qubit: int) -> np.ndarray:
    """Partial trace keeping only one qubit."""
    dim = 2 ** n_qubits
    rho_reshaped = rho.reshape([2] * (2 * n_qubits))

    # Trace over all qubits except keep_qubit
    trace_axes = []
    for q in range(n_qubits):
        if q != keep_qubit:
            trace_axes.append(q)

    # Contract pairs: axis q with axis q + n_qubits
    # We need to trace in the right order
    result = rho_reshaped
    offset = 0
    for q in sorted(trace_axes, reverse=True):
        result = np.trace(result, axis1=q - offset, axis2=q + n_qubits - 2 * offset)
        offset += 1

    return result.reshape(2, 2)


def _step_simulate(data: dict) -> list:
    """
    Step-by-step simulation: return state after each gate.
    Returns list of {gate_index, gate_name, statevector, probabilities, bloch_coords}.
    """
    _, StatevectorBackend = _get_circuit_and_backend()
    Circuit, _ = _get_circuit_and_backend()

    n_qubits = data.get("n_qubits", 2)
    gates = data.get("gates", [])

    steps = []

    # Initial state |0...0⟩
    sv = np.zeros(2 ** n_qubits, dtype=np.complex128)
    sv[0] = 1.0

    # Add initial state
    bloch_init = []
    for q in range(n_qubits):
        bloch_init.append(_compute_bloch_coords(sv, n_qubits, q))

    steps.append({
        "gate_index": -1,
        "gate_name": "Initial |0⟩⊗n",
        "statevector": [[float(c.real), float(c.imag)] for c in sv],
        "probabilities": {format(0, f'0{n_qubits}b'): 1.0},
        "bloch_coords": bloch_init,
    })

    # Apply gates one by one using backend
    for i, g in enumerate(gates):
        if g["name"].lower() == "measure":
            continue

        # Build circuit with gates up to index i
        partial_data = {"n_qubits": n_qubits, "gates": gates[:i + 1]}
        qc = _build_circuit(partial_data)

        backend = StatevectorBackend()
        result = backend.run(qc)
        sv = result.statevector

        probs = np.abs(sv) ** 2
        prob_dict = {}
        for idx, p in enumerate(probs):
            if p > 1e-10:
                prob_dict[format(idx, f'0{n_qubits}b')] = float(p)

        bloch = []
        for q in range(n_qubits):
            bloch.append(_compute_bloch_coords(sv, n_qubits, q))

        qubits_str = ",".join(str(q) for q in g.get("qubits", []))
        params_str = ""
        if g.get("params"):
            params_str = f"({','.join(f'{p:.3f}' for p in g['params'])})"

        steps.append({
            "gate_index": i,
            "gate_name": f"{g['name'].upper()}{params_str} on q[{qubits_str}]",
            "statevector": [[float(c.real), float(c.imag)] for c in sv],
            "probabilities": prob_dict,
            "bloch_coords": bloch,
        })

    return steps


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

def create_app() -> Any:
    """Create and configure the Flask application."""
    try:
        from flask import Flask, render_template, request, jsonify, send_from_directory
    except ImportError:
        raise ImportError(
            "Flask is required for the dashboard. Install it with:\n"
            "  pip install flask\n"
            "Or install tiny-qpu with dashboard extras:\n"
            "  pip install tiny-qpu[dashboard]"
        )

    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config['JSON_SORT_KEYS'] = False

    # ---- Routes ----

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/gates")
    def api_gates():
        return jsonify(GATE_CATALOG)

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        try:
            data = request.get_json()
            result = _simulate(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/step", methods=["POST"])
    def api_step():
        try:
            data = request.get_json()
            steps = _step_simulate(data)
            return jsonify({"steps": steps})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/parse-qasm", methods=["POST"])
    def api_parse_qasm():
        try:
            data = request.get_json()
            qasm_str = data.get("qasm", "")
            parse_qasm = _get_qasm_parser()
            circuit = parse_qasm(qasm_str)

            # Convert circuit to gate list
            gates = []
            n_qubits = circuit.n_qubits if hasattr(circuit, 'n_qubits') else 2

            if hasattr(circuit, '_instructions'):
                for inst in circuit._instructions:
                    gates.append({
                        "name": inst.name.lower(),
                        "qubits": list(inst.qubits),
                        "params": [float(p) for p in (inst.params or [])],
                    })
            elif hasattr(circuit, '_operations'):
                for op in circuit._operations:
                    gates.append({
                        "name": op.name.lower(),
                        "qubits": list(op.qubits),
                        "params": [float(p) for p in (op.params or [])],
                    })

            return jsonify({"n_qubits": n_qubits, "gates": gates})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/export-qasm", methods=["POST"])
    def api_export_qasm():
        try:
            data = request.get_json()
            qc = _build_circuit(data)
            qasm_str = qc.to_qasm() if hasattr(qc, 'to_qasm') else "// QASM export not available"
            return jsonify({"qasm": qasm_str})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/presets")
    def api_presets():
        """Return preset circuit examples."""
        presets = {
            "bell_state": {
                "name": "Bell State (Φ⁺)",
                "description": "Maximally entangled pair: |00⟩ + |11⟩",
                "n_qubits": 2,
                "gates": [
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "cx", "qubits": [0, 1], "params": []},
                ]
            },
            "ghz_3": {
                "name": "GHZ State (3 qubits)",
                "description": "Three-way entanglement: |000⟩ + |111⟩",
                "n_qubits": 3,
                "gates": [
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "cx", "qubits": [0, 1], "params": []},
                    {"name": "cx", "qubits": [0, 2], "params": []},
                ]
            },
            "superposition": {
                "name": "Uniform Superposition",
                "description": "All basis states equally likely",
                "n_qubits": 3,
                "gates": [
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "h", "qubits": [1], "params": []},
                    {"name": "h", "qubits": [2], "params": []},
                ]
            },
            "teleportation": {
                "name": "Quantum Teleportation",
                "description": "Teleport qubit 0 state to qubit 2",
                "n_qubits": 3,
                "gates": [
                    {"name": "rx", "qubits": [0], "params": [1.2]},
                    {"name": "h", "qubits": [1], "params": []},
                    {"name": "cx", "qubits": [1, 2], "params": []},
                    {"name": "cx", "qubits": [0, 1], "params": []},
                    {"name": "h", "qubits": [0], "params": []},
                ]
            },
            "deutsch_jozsa": {
                "name": "Deutsch-Jozsa",
                "description": "Determine if function is constant or balanced",
                "n_qubits": 3,
                "gates": [
                    {"name": "x", "qubits": [2], "params": []},
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "h", "qubits": [1], "params": []},
                    {"name": "h", "qubits": [2], "params": []},
                    {"name": "cx", "qubits": [0, 2], "params": []},
                    {"name": "cx", "qubits": [1, 2], "params": []},
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "h", "qubits": [1], "params": []},
                ]
            },
            "qft_2": {
                "name": "Quantum Fourier Transform",
                "description": "2-qubit QFT circuit",
                "n_qubits": 2,
                "gates": [
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "crz", "qubits": [1, 0], "params": [1.5708]},
                    {"name": "h", "qubits": [1], "params": []},
                    {"name": "swap", "qubits": [0, 1], "params": []},
                ]
            },
            "grover_2": {
                "name": "Grover's Search",
                "description": "Search for |11⟩ in 2-qubit space",
                "n_qubits": 2,
                "gates": [
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "h", "qubits": [1], "params": []},
                    {"name": "cz", "qubits": [0, 1], "params": []},
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "h", "qubits": [1], "params": []},
                    {"name": "x", "qubits": [0], "params": []},
                    {"name": "x", "qubits": [1], "params": []},
                    {"name": "cz", "qubits": [0, 1], "params": []},
                    {"name": "x", "qubits": [0], "params": []},
                    {"name": "x", "qubits": [1], "params": []},
                    {"name": "h", "qubits": [0], "params": []},
                    {"name": "h", "qubits": [1], "params": []},
                ]
            },
            "error_correction": {
                "name": "Bit-Flip Code",
                "description": "3-qubit repetition code protecting |1⟩",
                "n_qubits": 3,
                "gates": [
                    {"name": "x", "qubits": [0], "params": []},
                    {"name": "cx", "qubits": [0, 1], "params": []},
                    {"name": "cx", "qubits": [0, 2], "params": []},
                ]
            },
        }
        return jsonify(presets)

    return app


def launch(port: int = 8888, host: str = "127.0.0.1", debug: bool = False,
           open_browser: bool = True):
    """
    Launch the tiny-qpu Interactive Quantum Lab.

    Parameters
    ----------
    port : int
        Port to serve on (default 8888).
    host : str
        Host address (default localhost).
    debug : bool
        Enable Flask debug mode.
    open_browser : bool
        Automatically open browser.
    """
    app = create_app()

    url = f"http://{host}:{port}"
    print(f"""
╔══════════════════════════════════════════════════════╗
║         tiny-qpu ⚛ Interactive Quantum Lab          ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   Dashboard: {url:<39s} ║
║                                                      ║
║   Build circuits, visualize quantum states,          ║
║   and explore quantum computing — all in             ║
║   your browser.                                      ║
║                                                      ║
║   Press Ctrl+C to stop the server.                   ║
╚══════════════════════════════════════════════════════╝
""")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=debug)
