"""Tests for Circuit class."""

import numpy as np
import pytest

from tiny_qpu.circuit import Circuit, Parameter, Instruction


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_empty_circuit():
    qc = Circuit(3)
    assert qc.n_qubits == 3
    assert qc.n_clbits == 0
    assert qc.depth == 0
    assert qc.num_gates == 0
    assert not qc.is_parameterized


def test_invalid_qubit_count():
    with pytest.raises(ValueError):
        Circuit(0)


def test_single_gate():
    qc = Circuit(1)
    qc.h(0)
    assert qc.num_gates == 1
    assert qc.depth == 1


def test_method_chaining():
    qc = Circuit(2)
    result = qc.h(0).cx(0, 1).measure_all()
    assert result is qc
    assert qc.num_gates == 2


def test_invalid_qubit_index():
    qc = Circuit(2)
    with pytest.raises(ValueError):
        qc.h(2)

    with pytest.raises(ValueError):
        qc.h(-1)


def test_duplicate_qubits():
    qc = Circuit(2)
    with pytest.raises(ValueError):
        qc.cx(0, 0)


# ---------------------------------------------------------------------------
# Gate coverage
# ---------------------------------------------------------------------------

def test_all_single_qubit_gates():
    qc = Circuit(1)
    for gate in ["i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"]:
        getattr(qc, gate)(0)
    assert qc.num_gates == 10


def test_all_parameterized_single_gates():
    qc = Circuit(1)
    qc.rx(0.5, 0).ry(0.5, 0).rz(0.5, 0).p(0.5, 0)
    qc.u3(0.1, 0.2, 0.3, 0).u2(0.1, 0.2, 0).u1(0.5, 0)
    assert qc.num_gates == 7


def test_all_two_qubit_gates():
    qc = Circuit(2)
    qc.cx(0, 1).cz(0, 1).swap(0, 1).iswap(0, 1).ecr(0, 1)
    assert qc.num_gates == 5


def test_three_qubit_gates():
    qc = Circuit(3)
    qc.ccx(0, 1, 2).cswap(0, 1, 2)
    assert qc.num_gates == 2


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def test_parameter_creation():
    theta = Parameter("theta")
    assert theta.name == "theta"
    assert repr(theta) == "Parameter('theta')"


def test_parameter_equality():
    a = Parameter("a")
    b = Parameter("b")
    a2 = a
    assert a == a2
    assert a != b
    assert hash(a) == hash(a2)
    assert hash(a) != hash(b)


def test_parameterized_circuit():
    theta = Parameter("theta")
    qc = Circuit(1)
    qc.rx(theta, 0)
    assert qc.is_parameterized
    assert theta in qc.parameters


def test_bind_parameters():
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = Circuit(2)
    qc.rx(theta, 0).ry(phi, 1).cx(0, 1)

    bound = qc.bind({theta: 0.5, phi: 1.0})
    assert not bound.is_parameterized
    assert bound.num_gates == 3

    # Original unchanged
    assert qc.is_parameterized


def test_instruction_matrix_with_params():
    theta = Parameter("theta")
    inst = Instruction("rx", (0,), (theta,))
    assert inst.is_parameterized

    with pytest.raises(ValueError):
        inst.matrix()

    bound = inst.bind({theta: np.pi})
    assert not bound.is_parameterized
    mat = bound.matrix()
    assert mat.shape == (2, 2)


# ---------------------------------------------------------------------------
# Circuit depth
# ---------------------------------------------------------------------------

def test_depth_parallel():
    """Parallel gates on different qubits have depth 1."""
    qc = Circuit(3)
    qc.h(0).h(1).h(2)
    assert qc.depth == 1


def test_depth_serial():
    """Serial gates on the same qubit have depth = num_gates."""
    qc = Circuit(1)
    qc.x(0).y(0).z(0)
    assert qc.depth == 3


def test_depth_mixed():
    qc = Circuit(3)
    qc.h(0)         # depth 1
    qc.cx(0, 1)     # depth 2
    qc.h(2)         # depth 1 (parallel)
    qc.cx(1, 2)     # depth 3
    assert qc.depth == 3


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def test_measure_single():
    qc = Circuit(2)
    qc.measure(0)
    assert qc.n_clbits == 1


def test_measure_all():
    qc = Circuit(3)
    qc.measure_all()
    assert qc.n_clbits == 3
    measures = [i for i in qc.instructions if i.name == "measure"]
    assert len(measures) == 3


def test_measure_with_explicit_clbit():
    qc = Circuit(2, 2)
    qc.measure(0, 0)
    qc.measure(1, 1)
    assert qc.n_clbits == 2


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

def test_barrier_all():
    qc = Circuit(3)
    qc.barrier()
    assert len(qc.instructions) == 1
    assert qc.instructions[0].name == "barrier"
    assert qc.instructions[0].qubits == (0, 1, 2)


def test_barrier_subset():
    qc = Circuit(3)
    qc.barrier(0, 2)
    assert qc.instructions[0].qubits == (0, 2)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_compose():
    qc1 = Circuit(2)
    qc1.h(0)

    qc2 = Circuit(2)
    qc2.cx(0, 1)

    qc1.compose(qc2)
    assert qc1.num_gates == 2


def test_compose_with_qubit_map():
    qc1 = Circuit(3)
    qc1.h(0)

    qc2 = Circuit(2)
    qc2.cx(0, 1)

    qc1.compose(qc2, {0: 1, 1: 2})
    inst = qc1.instructions[-1]
    assert inst.qubits == (1, 2)


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------

def test_inverse_basic():
    qc = Circuit(2)
    qc.h(0).s(0).t(1).cx(0, 1)

    inv = qc.inverse()
    assert inv.num_gates == 4
    # First instruction of inverse should be last of original (cx)
    assert inv.instructions[0].name == "cx"
    # S becomes Sdg
    assert any(i.name == "sdg" for i in inv.instructions)
    # T becomes Tdg
    assert any(i.name == "tdg" for i in inv.instructions)


def test_inverse_rotation():
    qc = Circuit(1)
    qc.rx(0.5, 0)
    inv = qc.inverse()
    assert inv.instructions[0].params == (-0.5,)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def test_copy_independent():
    qc = Circuit(2)
    qc.h(0)
    qc2 = qc.copy()
    qc2.x(1)
    assert qc.num_gates == 1
    assert qc2.num_gates == 2


# ---------------------------------------------------------------------------
# QASM export
# ---------------------------------------------------------------------------

def test_to_qasm_basic():
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    qasm = qc.to_qasm()
    assert "OPENQASM 2.0;" in qasm
    assert "qreg q[2];" in qasm
    assert "h q[0];" in qasm
    assert "cx q[0],q[1];" in qasm


def test_to_qasm_with_params():
    qc = Circuit(1)
    qc.rx(1.5, 0)
    qasm = qc.to_qasm()
    assert "rx(1.5) q[0];" in qasm


def test_to_qasm_with_measurement():
    qc = Circuit(2, 2)
    qc.h(0).cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qasm = qc.to_qasm()
    assert "creg c[2];" in qasm
    assert "measure q[0] -> c[0];" in qasm


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def test_draw_returns_string():
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    output = qc.draw()
    assert isinstance(output, str)
    assert "q0:" in output
    assert "q1:" in output


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

def test_repr():
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    r = repr(qc)
    assert "n_qubits=2" in r
    assert "gates=2" in r
