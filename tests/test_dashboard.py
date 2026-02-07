"""
Tests for the tiny-qpu Interactive Quantum Lab Dashboard.

Tests cover:
- Simulation API correctness
- Bloch sphere coordinate calculation
- Step-by-step simulation
- QASM import/export
- Preset circuits
- Flask endpoint responses
"""

import pytest
import json
import numpy as np
import sys
import os

# Ensure tiny_qpu is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tiny_qpu.dashboard.server import (
    create_app, _simulate, _compute_bloch_coords,
    _step_simulate, _build_circuit, _partial_trace_single,
    GATE_CATALOG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Simulation correctness
# ---------------------------------------------------------------------------

class TestSimulation:
    """Test quantum simulation through the dashboard API."""

    def test_bell_state(self):
        """Bell state should give 50/50 on |00⟩ and |11⟩."""
        data = {
            "n_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
            "shots": 1024,
        }
        result = _simulate(data)

        assert "00" in result["probabilities"]
        assert "11" in result["probabilities"]
        assert abs(result["probabilities"]["00"] - 0.5) < 0.01
        assert abs(result["probabilities"]["11"] - 0.5) < 0.01

    def test_ghz_state(self):
        """GHZ state: |000⟩ + |111⟩."""
        data = {
            "n_qubits": 3,
            "gates": [
                {"name": "h", "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
                {"name": "cx", "qubits": [0, 2], "params": []},
            ],
            "shots": 100,
        }
        result = _simulate(data)
        assert abs(result["probabilities"].get("000", 0) - 0.5) < 0.01
        assert abs(result["probabilities"].get("111", 0) - 0.5) < 0.01

    def test_x_gate(self):
        """X gate flips |0⟩ to |1⟩."""
        data = {
            "n_qubits": 1,
            "gates": [{"name": "x", "qubits": [0], "params": []}],
            "shots": 100,
        }
        result = _simulate(data)
        assert abs(result["probabilities"].get("1", 0) - 1.0) < 0.01

    def test_hadamard(self):
        """H gate creates equal superposition."""
        data = {
            "n_qubits": 1,
            "gates": [{"name": "h", "qubits": [0], "params": []}],
            "shots": 100,
        }
        result = _simulate(data)
        assert abs(result["probabilities"].get("0", 0) - 0.5) < 0.01
        assert abs(result["probabilities"].get("1", 0) - 0.5) < 0.01

    def test_rotation_gates(self):
        """Rx(π) should flip |0⟩ to |1⟩."""
        data = {
            "n_qubits": 1,
            "gates": [{"name": "rx", "qubits": [0], "params": [3.14159265]}],
            "shots": 100,
        }
        result = _simulate(data)
        assert abs(result["probabilities"].get("1", 0) - 1.0) < 0.02

    def test_ry_rotation(self):
        """Ry(π/2) from |0⟩ gives equal superposition."""
        data = {
            "n_qubits": 1,
            "gates": [{"name": "ry", "qubits": [0], "params": [1.5707963]}],
            "shots": 100,
        }
        result = _simulate(data)
        assert abs(result["probabilities"].get("0", 0) - 0.5) < 0.02

    def test_empty_circuit(self):
        """Empty circuit returns |0...0⟩."""
        data = {"n_qubits": 3, "gates": [], "shots": 100}
        result = _simulate(data)
        assert abs(result["probabilities"].get("000", 0) - 1.0) < 0.01

    def test_measurement_counts(self):
        """Counts should sum to shots."""
        data = {
            "n_qubits": 2,
            "gates": [{"name": "h", "qubits": [0], "params": []}],
            "shots": 2048,
        }
        result = _simulate(data)
        total = sum(result["counts"].values())
        assert total == 2048

    def test_amplitudes_present(self):
        """Amplitudes should be returned."""
        data = {
            "n_qubits": 2,
            "gates": [{"name": "h", "qubits": [0], "params": []}],
            "shots": 100,
        }
        result = _simulate(data)
        assert len(result["amplitudes"]) == 4  # 2^2 = 4

    def test_cz_gate(self):
        """CZ on |11⟩ should give phase flip."""
        data = {
            "n_qubits": 2,
            "gates": [
                {"name": "x", "qubits": [0], "params": []},
                {"name": "x", "qubits": [1], "params": []},
                {"name": "cz", "qubits": [0, 1], "params": []},
            ],
            "shots": 100,
        }
        result = _simulate(data)
        # CZ gives phase, state is still |11⟩ in terms of probability
        assert abs(result["probabilities"].get("11", 0) - 1.0) < 0.01

    def test_swap_gate(self):
        """SWAP should exchange qubit states."""
        data = {
            "n_qubits": 2,
            "gates": [
                {"name": "x", "qubits": [0], "params": []},
                {"name": "swap", "qubits": [0, 1], "params": []},
            ],
            "shots": 100,
        }
        result = _simulate(data)
        assert abs(result["probabilities"].get("01", 0) - 1.0) < 0.01

    def test_toffoli_gate(self):
        """Toffoli flips target when both controls are |1⟩."""
        data = {
            "n_qubits": 3,
            "gates": [
                {"name": "x", "qubits": [0], "params": []},
                {"name": "x", "qubits": [1], "params": []},
                {"name": "ccx", "qubits": [0, 1, 2], "params": []},
            ],
            "shots": 100,
        }
        result = _simulate(data)
        assert abs(result["probabilities"].get("111", 0) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Bloch sphere
# ---------------------------------------------------------------------------

class TestBlochSphere:
    """Test Bloch sphere coordinate calculations."""

    def test_zero_state(self):
        """The |0⟩ state should be at north pole (z=1)."""
        sv = np.array([1, 0], dtype=complex)
        coords = _compute_bloch_coords(sv, 1, 0)
        assert abs(coords["z"] - 1.0) < 1e-10
        assert abs(coords["x"]) < 1e-10
        assert abs(coords["y"]) < 1e-10
        assert abs(coords["purity"] - 1.0) < 1e-10

    def test_one_state(self):
        """The |1⟩ state should be at south pole (z=-1)."""
        sv = np.array([0, 1], dtype=complex)
        coords = _compute_bloch_coords(sv, 1, 0)
        assert abs(coords["z"] + 1.0) < 1e-10

    def test_plus_state(self):
        """The |+⟩ state should be on equator (x=1)."""
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        coords = _compute_bloch_coords(sv, 1, 0)
        assert abs(coords["x"] - 1.0) < 1e-10
        assert abs(coords["z"]) < 1e-10

    def test_entangled_bloch(self):
        """Entangled qubit should have purity ≈ 0.5 (maximally mixed)."""
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # Bell state
        coords = _compute_bloch_coords(sv, 2, 0)
        assert abs(coords["purity"] - 0.5) < 1e-10
        # Mixed state: Bloch vector near origin
        assert abs(coords["x"]) < 1e-10
        assert abs(coords["y"]) < 1e-10
        assert abs(coords["z"]) < 1e-10


# ---------------------------------------------------------------------------
# Step mode
# ---------------------------------------------------------------------------

class TestStepMode:
    """Test step-by-step simulation."""

    def test_step_count(self):
        """Should return n_gates + 1 steps (including initial state)."""
        data = {
            "n_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
        }
        steps = _step_simulate(data)
        assert len(steps) == 3  # initial + 2 gates

    def test_step_initial_state(self):
        """First step should be all-zero state."""
        data = {"n_qubits": 2, "gates": [{"name": "h", "qubits": [0], "params": []}]}
        steps = _step_simulate(data)
        assert steps[0]["gate_index"] == -1
        assert "00" in steps[0]["probabilities"]
        assert abs(steps[0]["probabilities"]["00"] - 1.0) < 1e-10

    def test_step_final_matches_full(self):
        """Last step should match full simulation."""
        data = {
            "n_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
            "shots": 100,
        }
        steps = _step_simulate(data)
        full = _simulate(data)

        # Compare probabilities
        for bs, prob in full["probabilities"].items():
            assert abs(steps[-1]["probabilities"].get(bs, 0) - prob) < 1e-10

    def test_step_bloch_present(self):
        """Each step should have Bloch coordinates."""
        data = {"n_qubits": 1, "gates": [{"name": "h", "qubits": [0], "params": []}]}
        steps = _step_simulate(data)
        for s in steps:
            assert "bloch_coords" in s
            assert len(s["bloch_coords"]) == 1


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------

class TestFlaskEndpoints:
    """Test all API endpoints."""

    def test_index(self, client):
        """Dashboard HTML should load."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"tiny-qpu" in resp.data

    def test_gates_endpoint(self, client):
        """Gates API should return gate catalog."""
        resp = client.get("/api/gates")
        data = json.loads(resp.data)
        assert len(data) > 10
        assert any(g["name"] == "h" for g in data)

    def test_presets_endpoint(self, client):
        """Presets API should return preset circuits."""
        resp = client.get("/api/presets")
        data = json.loads(resp.data)
        assert "bell_state" in data
        assert "ghz_3" in data

    def test_simulate_endpoint(self, client):
        """Simulate API should return valid results."""
        payload = {
            "n_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
            "shots": 100,
        }
        resp = client.post("/api/simulate",
            data=json.dumps(payload),
            content_type="application/json")
        data = json.loads(resp.data)
        assert "probabilities" in data
        assert "counts" in data
        assert "bloch_coords" in data

    def test_step_endpoint(self, client):
        """Step API should return step data."""
        payload = {
            "n_qubits": 2,
            "gates": [{"name": "h", "qubits": [0], "params": []}],
        }
        resp = client.post("/api/step",
            data=json.dumps(payload),
            content_type="application/json")
        data = json.loads(resp.data)
        assert "steps" in data

    def test_export_qasm(self, client):
        """QASM export should return valid QASM."""
        payload = {
            "n_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
        }
        resp = client.post("/api/export-qasm",
            data=json.dumps(payload),
            content_type="application/json")
        data = json.loads(resp.data)
        assert "OPENQASM" in data["qasm"]
        assert "h q[0]" in data["qasm"]

    def test_simulate_error(self, client):
        """Invalid input should return error."""
        resp = client.post("/api/simulate",
            data="not json",
            content_type="application/json")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Gate catalog
# ---------------------------------------------------------------------------

class TestGateCatalog:
    """Test gate catalog completeness."""

    def test_has_basic_gates(self):
        names = [g["name"] for g in GATE_CATALOG]
        for gate in ["h", "x", "y", "z", "s", "t", "cx", "cz", "swap", "ccx"]:
            assert gate in names, f"Missing gate: {gate}"

    def test_has_rotation_gates(self):
        names = [g["name"] for g in GATE_CATALOG]
        for gate in ["rx", "ry", "rz"]:
            assert gate in names, f"Missing rotation gate: {gate}"

    def test_gate_colors(self):
        """Every gate should have a color."""
        for g in GATE_CATALOG:
            assert "color" in g and g["color"].startswith("#")

    def test_gate_metadata(self):
        """Every gate should have required fields."""
        for g in GATE_CATALOG:
            assert "name" in g
            assert "label" in g
            assert "n_qubits" in g
            assert "n_params" in g


# ---------------------------------------------------------------------------
# Partial trace
# ---------------------------------------------------------------------------

class TestPartialTrace:
    """Test partial trace implementation."""

    def test_product_state(self):
        """Partial trace of |0⟩|1⟩ over qubit 1 should give |0⟩⟨0|."""
        sv = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        rho = np.outer(sv, sv.conj())
        reduced = _partial_trace_single(rho, 2, 0)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(reduced, expected, atol=1e-10)

    def test_bell_state_reduced(self):
        """Partial trace of Bell state should be maximally mixed."""
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(sv, sv.conj())
        reduced = _partial_trace_single(rho, 2, 0)
        expected = np.eye(2) / 2
        np.testing.assert_allclose(reduced, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
