"""Tests for quantum gate definitions."""

import numpy as np
import pytest

from tiny_qpu import gates as g


# ---------------------------------------------------------------------------
# Unitarity tests — every gate must satisfy U†U = I
# ---------------------------------------------------------------------------

FIXED_GATES = [
    ("I", g.I), ("X", g.X), ("Y", g.Y), ("Z", g.Z), ("H", g.H),
    ("S", g.S), ("Sdg", g.Sdg), ("T", g.T), ("Tdg", g.Tdg), ("SX", g.SX),
    ("CNOT", g.CNOT), ("CZ", g.CZ), ("SWAP", g.SWAP), ("iSWAP", g.iSWAP),
    ("ECR", g.ECR), ("CCX", g.CCX), ("CSWAP", g.CSWAP),
]


@pytest.mark.parametrize("name,matrix", FIXED_GATES)
def test_fixed_gate_unitary(name, matrix):
    """Every fixed gate must be unitary: U†U = I."""
    dim = matrix.shape[0]
    product = matrix.conj().T @ matrix
    np.testing.assert_allclose(product, np.eye(dim), atol=1e-12, err_msg=f"{name} is not unitary")


@pytest.mark.parametrize("name,matrix", FIXED_GATES)
def test_fixed_gate_shape(name, matrix):
    """Gates must be square matrices of dimension 2^k."""
    assert matrix.shape[0] == matrix.shape[1]
    dim = matrix.shape[0]
    assert dim & (dim - 1) == 0, f"{name} dimension {dim} is not a power of 2"


PARAM_GATES_1 = [
    ("Rx", g.Rx), ("Ry", g.Ry), ("Rz", g.Rz), ("P", g.P), ("U1", g.U1),
]


@pytest.mark.parametrize("name,factory", PARAM_GATES_1)
@pytest.mark.parametrize("theta", [0, 0.5, np.pi, 2 * np.pi, -1.3])
def test_param_gate_unitary(name, factory, theta):
    """Parameterized single-param gates must be unitary for all angles."""
    mat = factory(theta)
    product = mat.conj().T @ mat
    np.testing.assert_allclose(product, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("theta", [0, 0.7, np.pi, -0.3])
def test_u3_unitary(theta):
    mat = g.U3(theta, 0.5, -0.2)
    product = mat.conj().T @ mat
    np.testing.assert_allclose(product, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi / 2])
def test_u2_unitary(theta):
    mat = g.U2(theta, 0.3)
    product = mat.conj().T @ mat
    np.testing.assert_allclose(product, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("factory", [g.CP, g.CRx, g.CRy, g.CRz, g.Rxx, g.Ryy, g.Rzz])
@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi, -0.5])
def test_two_qubit_param_unitary(factory, theta):
    mat = factory(theta)
    product = mat.conj().T @ mat
    np.testing.assert_allclose(product, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Gate algebra tests
# ---------------------------------------------------------------------------

def test_x_squared_is_identity():
    np.testing.assert_allclose(g.X @ g.X, g.I, atol=1e-12)


def test_y_squared_is_identity():
    np.testing.assert_allclose(g.Y @ g.Y, g.I, atol=1e-12)


def test_z_squared_is_identity():
    np.testing.assert_allclose(g.Z @ g.Z, g.I, atol=1e-12)


def test_h_squared_is_identity():
    np.testing.assert_allclose(g.H @ g.H, g.I, atol=1e-12)


def test_s_squared_is_z():
    np.testing.assert_allclose(g.S @ g.S, g.Z, atol=1e-12)


def test_t_squared_is_s():
    np.testing.assert_allclose(g.T @ g.T, g.S, atol=1e-12)


def test_sdg_is_s_inverse():
    np.testing.assert_allclose(g.S @ g.Sdg, g.I, atol=1e-12)


def test_tdg_is_t_inverse():
    np.testing.assert_allclose(g.T @ g.Tdg, g.I, atol=1e-12)


def test_xyz_anticommutation():
    """XY = iZ, YZ = iX, ZX = iY."""
    np.testing.assert_allclose(g.X @ g.Y, 1j * g.Z, atol=1e-12)
    np.testing.assert_allclose(g.Y @ g.Z, 1j * g.X, atol=1e-12)
    np.testing.assert_allclose(g.Z @ g.X, 1j * g.Y, atol=1e-12)


def test_rx_at_pi_is_neg_ix():
    """Rx(π) = -iX."""
    np.testing.assert_allclose(g.Rx(np.pi), -1j * g.X, atol=1e-12)


def test_ry_at_pi_is_neg_iy():
    """Ry(π) = -iY."""
    np.testing.assert_allclose(g.Ry(np.pi), -1j * g.Y, atol=1e-12)


def test_rz_at_pi_is_neg_iz():
    """Rz(π) = -iZ."""
    np.testing.assert_allclose(g.Rz(np.pi), -1j * g.Z, atol=1e-12)


def test_rotation_at_zero_is_identity():
    for factory in [g.Rx, g.Ry, g.Rz]:
        np.testing.assert_allclose(factory(0), g.I, atol=1e-12)


def test_swap_squared_is_identity():
    np.testing.assert_allclose(g.SWAP @ g.SWAP, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Gate registry
# ---------------------------------------------------------------------------

def test_get_matrix_fixed():
    np.testing.assert_allclose(g.get_matrix("h"), g.H, atol=1e-12)


def test_get_matrix_param():
    np.testing.assert_allclose(g.get_matrix("rx", (np.pi,)), g.Rx(np.pi), atol=1e-12)


def test_get_matrix_unknown():
    with pytest.raises(KeyError):
        g.get_matrix("nonexistent")


def test_get_matrix_wrong_params():
    with pytest.raises(ValueError):
        g.get_matrix("h", (1.0,))

    with pytest.raises(ValueError):
        g.get_matrix("rx")
