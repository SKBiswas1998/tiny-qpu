"""Tests for OpenQASM 2.0 parser."""

import math

import numpy as np
import pytest

from tiny_qpu.qasm import parse_qasm, QasmParser
from tiny_qpu.qasm.parser import QasmParseError
from tiny_qpu import StatevectorBackend


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_minimal_circuit():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    """
    qc = parse_qasm(qasm)
    assert qc.n_qubits == 2
    assert qc.num_gates == 0


def test_single_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    h q[0];
    """
    qc = parse_qasm(qasm)
    assert qc.num_gates == 1
    assert qc.instructions[0].name == "h"


def test_bell_state():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[0];
    cx q[0],q[1];
    """
    qc = parse_qasm(qasm)
    assert qc.n_qubits == 2
    assert qc.num_gates == 2

    # Verify simulation
    backend = StatevectorBackend()
    sv = backend.statevector(qc)
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Parameterized gates
# ---------------------------------------------------------------------------

def test_rotation_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rx(1.5707963267948966) q[0];
    """
    qc = parse_qasm(qasm)
    assert qc.num_gates == 1
    assert qc.instructions[0].name == "rx"
    assert qc.instructions[0].params[0] == pytest.approx(math.pi / 2, abs=1e-10)


def test_pi_in_params():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rx(pi) q[0];
    """
    qc = parse_qasm(qasm)
    assert qc.instructions[0].params[0] == pytest.approx(math.pi)


def test_pi_arithmetic():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    rz(pi/2) q[0];
    """
    qc = parse_qasm(qasm)
    assert qc.instructions[0].params[0] == pytest.approx(math.pi / 2)


def test_u3_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u3(1.0,2.0,3.0) q[0];
    """
    qc = parse_qasm(qasm)
    assert len(qc.instructions[0].params) == 3
    assert qc.instructions[0].params == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Measurement and classical registers
# ---------------------------------------------------------------------------

def test_measurement():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    qc = parse_qasm(qasm)
    assert qc.n_clbits == 2
    measures = [i for i in qc.instructions if i.name == "measure"]
    assert len(measures) == 2


# ---------------------------------------------------------------------------
# Multiple registers
# ---------------------------------------------------------------------------

def test_multiple_qregs():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg a[2];
    qreg b[1];
    h a[0];
    cx a[0],b[0];
    """
    qc = parse_qasm(qasm)
    assert qc.n_qubits == 3
    assert qc.num_gates == 2
    # b[0] should map to qubit index 2
    assert qc.instructions[1].qubits == (0, 2)


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

def test_barrier():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[0];
    barrier q[0],q[1];
    cx q[0],q[1];
    """
    qc = parse_qasm(qasm)
    barriers = [i for i in qc.instructions if i.name == "barrier"]
    assert len(barriers) == 1
    assert barriers[0].qubits == (0, 1)


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

def test_single_line_comments():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    // This is a comment
    qreg q[1];
    h q[0]; // inline comment
    """
    qc = parse_qasm(qasm)
    assert qc.num_gates == 1


def test_multi_line_comments():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    /* multi
    line
    comment */
    qreg q[1];
    h q[0];
    """
    qc = parse_qasm(qasm)
    assert qc.num_gates == 1


# ---------------------------------------------------------------------------
# Gate coverage
# ---------------------------------------------------------------------------

def test_all_single_qubit_gates():
    gates = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"]
    for gate in gates:
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        {gate} q[0];
        """
        qc = parse_qasm(qasm)
        assert qc.num_gates == 1, f"Failed for gate {gate}"


def test_two_qubit_gates():
    for gate in ["cx", "cz", "swap"]:
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        {gate} q[0],q[1];
        """
        qc = parse_qasm(qasm)
        assert qc.num_gates == 1, f"Failed for gate {gate}"


def test_three_qubit_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    ccx q[0],q[1],q[2];
    """
    qc = parse_qasm(qasm)
    assert qc.num_gates == 1
    assert qc.instructions[0].qubits == (0, 1, 2)


def test_id_gate_maps_to_i():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    id q[0];
    """
    qc = parse_qasm(qasm)
    assert qc.instructions[0].name == "i"


# ---------------------------------------------------------------------------
# Round-trip: Circuit → QASM → Circuit
# ---------------------------------------------------------------------------

def test_roundtrip():
    """Parse a circuit, export to QASM, parse again, verify same result."""
    from tiny_qpu import Circuit

    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    qasm = qc.to_qasm()

    qc2 = parse_qasm(qasm)
    assert qc2.n_qubits == 2
    assert qc2.num_gates == 2

    # Both should produce the same statevector
    backend = StatevectorBackend()
    sv1 = backend.statevector(qc)
    sv2 = backend.statevector(qc2)
    np.testing.assert_allclose(sv1, sv2, atol=1e-12)


# ---------------------------------------------------------------------------
# Custom gate definitions
# ---------------------------------------------------------------------------

def test_custom_gate_definition():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    gate bell a,b {
        h a;
        cx a,b;
    }
    qreg q[2];
    bell q[0],q[1];
    """
    qc = parse_qasm(qasm)
    assert qc.n_qubits == 2
    assert qc.num_gates == 2  # expanded to h + cx

    backend = StatevectorBackend()
    sv = backend.statevector(qc)
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    foobar q[0];
    """
    with pytest.raises(QasmParseError, match="Unknown gate"):
        parse_qasm(qasm)
