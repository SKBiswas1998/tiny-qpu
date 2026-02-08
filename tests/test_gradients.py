"""
Comprehensive tests for the quantum gradient computation module.

Tests cover:
- Hamiltonian construction, expectation values, algebra
- Standard Hamiltonians (Ising, Heisenberg, H₂)
- Parameter-shift gradient correctness
- Adjoint differentiation correctness
- Finite-difference gradient correctness
- Cross-method agreement (all three methods match)
- Known analytic gradient cases
- Multi-parameter circuits
- VQE optimization convergence
- Edge cases
"""

import numpy as np
import pytest
from typing import Callable

# Import the gradients module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tiny_qpu.gradients.hamiltonian import (
    Hamiltonian, transverse_field_ising, heisenberg_xyz,
    molecular_hydrogen, _apply_pauli_string, _pauli_string_matrix,
    PAULI_MAP,
)
from tiny_qpu.gradients.differentiation import (
    expectation, gradient, expectation_and_gradient,
    _param_shift_gradient, _adjoint_gradient, _finite_diff_gradient,
    _apply_single_qubit_gate, _apply_two_qubit_gate,
)


# ─── Minimal Circuit class for standalone testing ────────────────────
# This provides a lightweight Circuit that works without the full tiny-qpu
# install, while also being compatible with the gradient API.

class _Instruction:
    """Minimal instruction for testing."""
    def __init__(self, name, qubits, matrix, param_idx=None):
        self.name = name
        self.qubits = qubits
        self.matrix = matrix
        self.param_idx = param_idx


class MiniCircuit:
    """
    Minimal circuit class for testing gradients independently.

    Supports: rx, ry, rz, h, x, cx, and statevector simulation.
    """
    def __init__(self, n_qubits):
        self._n_qubits = n_qubits
        self._instructions = []

    @property
    def n_qubits(self):
        return self._n_qubits

    @property
    def num_qubits(self):
        return self._n_qubits

    def rx(self, theta, qubit):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        mat = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        self._instructions.append(_Instruction("rx", [qubit], mat))
        return self

    def ry(self, theta, qubit):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        mat = np.array([[c, -s], [s, c]], dtype=complex)
        self._instructions.append(_Instruction("ry", [qubit], mat))
        return self

    def rz(self, theta, qubit):
        mat = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
        self._instructions.append(_Instruction("rz", [qubit], mat))
        return self

    def h(self, qubit):
        mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._instructions.append(_Instruction("h", [qubit], mat))
        return self

    def x(self, qubit):
        mat = np.array([[0, 1], [1, 0]], dtype=complex)
        self._instructions.append(_Instruction("x", [qubit], mat))
        return self

    def z(self, qubit):
        mat = np.array([[1, 0], [0, -1]], dtype=complex)
        self._instructions.append(_Instruction("z", [qubit], mat))
        return self

    def cx(self, control, target):
        mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        self._instructions.append(_Instruction("cx", [control, target], mat))
        return self

    def statevector(self):
        """Simulate and return statevector."""
        sv = np.zeros(2 ** self._n_qubits, dtype=complex)
        sv[0] = 1.0
        for instr in self._instructions:
            if len(instr.qubits) == 1:
                sv = _apply_single_qubit_gate(
                    sv, instr.matrix, instr.qubits[0], self._n_qubits
                )
            elif len(instr.qubits) == 2:
                sv = _apply_two_qubit_gate(
                    sv, instr.matrix, instr.qubits, self._n_qubits
                )
        return sv


def _make_circuit_fn_with_param_tracking(build_fn):
    """
    Wrap a circuit-building function so each parameterized gate
    records which parameter index it uses (for adjoint method).
    """
    def tracked_fn(params):
        qc = build_fn(params)
        # Tag parameterized instructions with their param_idx
        param_counter = [0]
        for instr in qc._instructions:
            if instr.name.lower() in ("rx", "ry", "rz", "crx", "cry", "crz",
                                       "rxx", "ryy", "rzz", "u1", "p"):
                instr.param_idx = param_counter[0]
                param_counter[0] += 1
        return qc
    return tracked_fn


# ═══════════════════════════════════════════════════════════════════
# HAMILTONIAN TESTS
# ═══════════════════════════════════════════════════════════════════

class TestHamiltonian:
    """Tests for the Hamiltonian class."""

    def test_construction_basic(self):
        H = Hamiltonian({"ZI": 0.5, "IZ": -0.3})
        assert H.n_qubits == 2
        assert H.n_terms == 2

    def test_construction_single_qubit(self):
        H = Hamiltonian({"Z": 1.0})
        assert H.n_qubits == 1
        assert H.n_terms == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            Hamiltonian({})

    def test_invalid_pauli_raises(self):
        with pytest.raises(ValueError, match="Invalid Pauli"):
            Hamiltonian({"AB": 1.0})

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Hamiltonian({"Z": 1.0, "ZZ": 0.5})

    def test_zero_terms_removed(self):
        H = Hamiltonian({"Z": 1.0, "X": 0.0})
        assert H.n_terms == 1

    def test_case_insensitive(self):
        H = Hamiltonian({"zi": 0.5, "iz": -0.3})
        assert "ZI" in H.terms
        assert "IZ" in H.terms

    def test_expectation_z_on_zero_state(self):
        """⟨0|Z|0⟩ = 1"""
        H = Hamiltonian({"Z": 1.0})
        sv = np.array([1, 0], dtype=complex)
        assert np.isclose(H.expectation(sv), 1.0)

    def test_expectation_z_on_one_state(self):
        """⟨1|Z|1⟩ = -1"""
        H = Hamiltonian({"Z": 1.0})
        sv = np.array([0, 1], dtype=complex)
        assert np.isclose(H.expectation(sv), -1.0)

    def test_expectation_x_on_plus_state(self):
        """⟨+|X|+⟩ = 1"""
        H = Hamiltonian({"X": 1.0})
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert np.isclose(H.expectation(sv), 1.0)

    def test_expectation_x_on_minus_state(self):
        """⟨-|X|-⟩ = -1"""
        H = Hamiltonian({"X": 1.0})
        sv = np.array([1, -1], dtype=complex) / np.sqrt(2)
        assert np.isclose(H.expectation(sv), -1.0)

    def test_expectation_identity(self):
        """⟨ψ|I|ψ⟩ = 1 for any normalized state"""
        H = Hamiltonian({"I": 3.14})
        sv = np.array([1, 0], dtype=complex)
        assert np.isclose(H.expectation(sv), 3.14)

    def test_expectation_bell_state_zz(self):
        """⟨Φ⁺|ZZ|Φ⁺⟩ = 1 (Bell state is ZZ eigenstate)"""
        H = Hamiltonian({"ZZ": 1.0})
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.isclose(H.expectation(sv), 1.0)

    def test_expectation_bell_state_xx(self):
        """⟨Φ⁺|XX|Φ⁺⟩ = 1"""
        H = Hamiltonian({"XX": 1.0})
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.isclose(H.expectation(sv), 1.0)

    def test_expectation_matches_matrix(self):
        """Expectation via Pauli application matches full matrix method."""
        H = Hamiltonian({"ZI": 0.5, "IZ": -0.3, "XX": 0.2})
        sv = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        # Via full matrix
        mat = H.matrix()
        expected = np.real(sv.conj() @ mat @ sv)
        assert np.isclose(H.expectation(sv), expected, atol=1e-12)

    def test_expectation_random_state(self):
        """Expectation on random state matches matrix multiplication."""
        np.random.seed(42)
        H = Hamiltonian({"ZZ": -1.0, "XI": 0.5, "IX": 0.5, "YY": 0.3})
        sv = np.random.randn(4) + 1j * np.random.randn(4)
        sv /= np.linalg.norm(sv)
        mat = H.matrix()
        expected = np.real(sv.conj() @ mat @ sv)
        assert np.isclose(H.expectation(sv), expected, atol=1e-12)

    def test_matrix_hermitian(self):
        """Hamiltonian matrix should be Hermitian."""
        H = Hamiltonian({"ZI": 0.5, "IX": -0.3, "YZ": 0.2})
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T)

    def test_matrix_dimensions(self):
        H = Hamiltonian({"ZZZ": 1.0})
        assert H.matrix().shape == (8, 8)

    def test_ground_state_energy_z(self):
        """Ground state of Z is -1 (|1⟩ state)."""
        H = Hamiltonian({"Z": 1.0})
        assert np.isclose(H.ground_state_energy(), -1.0)

    def test_ground_state_vector(self):
        """Ground state of -Z is |0⟩."""
        H = Hamiltonian({"Z": -1.0})
        energy, state = H.ground_state()
        assert np.isclose(energy, -1.0)
        assert np.isclose(abs(state[0]), 1.0, atol=1e-10)

    def test_addition(self):
        H1 = Hamiltonian({"ZI": 1.0})
        H2 = Hamiltonian({"IZ": 0.5})
        H3 = H1 + H2
        assert H3.n_terms == 2
        assert np.isclose(H3.terms["ZI"], 1.0)
        assert np.isclose(H3.terms["IZ"], 0.5)

    def test_addition_overlapping(self):
        H1 = Hamiltonian({"ZZ": 1.0, "XX": 0.5})
        H2 = Hamiltonian({"ZZ": 0.3, "YY": 0.2})
        H3 = H1 + H2
        assert np.isclose(H3.terms["ZZ"], 1.3)

    def test_scalar_multiply(self):
        H = Hamiltonian({"Z": 2.0})
        H2 = 0.5 * H
        assert np.isclose(H2.terms["Z"], 1.0)

    def test_repr(self):
        H = Hamiltonian({"Z": 1.0})
        assert "Hamiltonian" in repr(H)

    def test_str(self):
        H = Hamiltonian({"ZI": 0.5, "IZ": -0.3})
        s = str(H)
        assert "2 qubits" in s


class TestStandardHamiltonians:
    """Tests for pre-built Hamiltonian constructors."""

    def test_ising_2_qubits(self):
        H = transverse_field_ising(2, J=1.0, h=0.5)
        assert H.n_qubits == 2
        assert "ZZ" in H.terms
        assert "XI" in H.terms or "IX" in H.terms

    def test_ising_3_qubits(self):
        H = transverse_field_ising(3)
        assert H.n_qubits == 3
        # Should have ZZ on (0,1) and (1,2), plus X on each qubit
        assert H.n_terms == 5  # 2 ZZ + 3 X

    def test_ising_hermitian(self):
        H = transverse_field_ising(4)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T)

    def test_heisenberg_2_qubits(self):
        H = heisenberg_xyz(2)
        assert H.n_qubits == 2
        assert "XX" in H.terms
        assert "YY" in H.terms
        assert "ZZ" in H.terms

    def test_heisenberg_hermitian(self):
        H = heisenberg_xyz(3)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T)

    def test_h2_default(self):
        H = molecular_hydrogen()
        assert H.n_qubits == 2
        assert H.n_terms == 5

    def test_h2_ground_state(self):
        """H₂ ground state energy at 0.74 Å should be negative."""
        H = molecular_hydrogen(0.74)
        E0 = H.ground_state_energy()
        # STO-3G energy varies by coefficient set; just verify it's reasonable
        assert -1.5 < E0 < -0.5

    def test_h2_invalid_bond_length(self):
        with pytest.raises(ValueError):
            molecular_hydrogen(0.42)


class TestPauliApplication:
    """Tests for efficient Pauli string application."""

    def test_identity_application(self):
        sv = np.array([0.6, 0.8], dtype=complex)
        result = _apply_pauli_string(sv, "I", 1)
        assert np.allclose(result, sv)

    def test_x_application(self):
        sv = np.array([1, 0], dtype=complex)
        result = _apply_pauli_string(sv, "X", 1)
        assert np.allclose(result, [0, 1])

    def test_z_application(self):
        sv = np.array([1, 0], dtype=complex)
        result = _apply_pauli_string(sv, "Z", 1)
        assert np.allclose(result, [1, 0])

    def test_zz_on_bell(self):
        """ZZ|Φ⁺⟩ = |Φ⁺⟩ (Bell state is +1 eigenstate of ZZ)."""
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        result = _apply_pauli_string(sv, "ZZ", 2)
        assert np.allclose(result, sv)

    def test_matches_matrix(self):
        """Pauli string application matches full matrix multiplication."""
        np.random.seed(123)
        sv = np.random.randn(8) + 1j * np.random.randn(8)
        sv /= np.linalg.norm(sv)
        result_fast = _apply_pauli_string(sv, "XYZ", 3)
        mat = _pauli_string_matrix("XYZ")
        result_full = mat @ sv
        assert np.allclose(result_fast, result_full, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════
# GRADIENT TESTS
# ═══════════════════════════════════════════════════════════════════

class TestParameterShiftGradient:
    """Tests for the parameter-shift rule."""

    def test_ry_z_analytic(self):
        """
        ∂/∂θ ⟨0|Ry†(θ) Z Ry(θ)|0⟩ = -sin(θ)

        This is the canonical test case for gradient correctness.
        """
        H = Hamiltonian({"Z": 1.0})
        for theta in [0.0, 0.3, np.pi / 4, np.pi / 2, np.pi, 2.5]:
            def ansatz(params):
                return MiniCircuit(1).ry(params[0], 0)
            grad = _param_shift_gradient(ansatz, H, np.array([theta]))
            expected = -np.sin(theta)
            assert np.isclose(grad[0], expected, atol=1e-6), \
                f"θ={theta}: got {grad[0]}, expected {expected}"

    def test_rx_z_analytic(self):
        """∂/∂θ ⟨0|Rx†(θ) Z Rx(θ)|0⟩ = -sin(θ) (same as Ry due to Z)"""
        H = Hamiltonian({"Z": 1.0})
        theta = np.pi / 3
        def ansatz(params):
            return MiniCircuit(1).rx(params[0], 0)
        # ⟨0|Rx†(θ) Z Rx(θ)|0⟩ = cos(θ), so derivative = -sin(θ)
        grad = _param_shift_gradient(ansatz, H, np.array([theta]))
        assert np.isclose(grad[0], -np.sin(theta), atol=1e-6)

    def test_zero_gradient_at_extremum(self):
        """Gradient should be zero at θ = 0 and θ = π for Ry-Z."""
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)
        grad_0 = _param_shift_gradient(ansatz, H, np.array([0.0]))
        grad_pi = _param_shift_gradient(ansatz, H, np.array([np.pi]))
        assert np.isclose(grad_0[0], 0.0, atol=1e-10)
        assert np.isclose(grad_pi[0], 0.0, atol=1e-10)

    def test_multi_parameter(self):
        """Gradient of 2-parameter circuit."""
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0).rz(params[1], 0)

        params = np.array([np.pi / 4, np.pi / 3])
        grad = _param_shift_gradient(ansatz, H, params)

        # Compare with finite difference
        grad_fd = _finite_diff_gradient(ansatz, H, params)
        assert np.allclose(grad, grad_fd, atol=1e-5)

    def test_two_qubit_circuit(self):
        """Gradient of a 2-qubit entangling circuit."""
        H = Hamiltonian({"ZI": 0.5, "IZ": 0.5})
        def ansatz(params):
            return MiniCircuit(2).ry(params[0], 0).ry(params[1], 1).cx(0, 1)

        params = np.array([0.5, 0.8])
        grad_ps = _param_shift_gradient(ansatz, H, params)
        grad_fd = _finite_diff_gradient(ansatz, H, params)
        assert np.allclose(grad_ps, grad_fd, atol=1e-5)


class TestFiniteDifferenceGradient:
    """Tests for finite-difference gradient."""

    def test_ry_z_analytic(self):
        H = Hamiltonian({"Z": 1.0})
        theta = np.pi / 3
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)
        grad = _finite_diff_gradient(ansatz, H, np.array([theta]))
        assert np.isclose(grad[0], -np.sin(theta), atol=1e-5)

    def test_multi_param(self):
        H = Hamiltonian({"ZI": 1.0, "IX": 0.5})
        def ansatz(params):
            return MiniCircuit(2).ry(params[0], 0).rx(params[1], 1)
        params = np.array([0.7, 1.2])
        grad = _finite_diff_gradient(ansatz, H, params)
        assert grad.shape == (2,)
        # Should be finite and non-zero for these parameters
        assert not np.allclose(grad, 0.0)


class TestAdjointGradient:
    """Tests for adjoint differentiation."""

    def test_ry_z_analytic(self):
        """Adjoint gradient matches analytic result."""
        H = Hamiltonian({"Z": 1.0})
        theta = np.pi / 4

        def _build(params):
            qc = MiniCircuit(1)
            instr = _Instruction("ry",  [0],
                np.array([[np.cos(params[0]/2), -np.sin(params[0]/2)],
                          [np.sin(params[0]/2),  np.cos(params[0]/2)]], dtype=complex),
                param_idx=0)
            qc._instructions = [instr]
            return qc

        grad = _adjoint_gradient(_build, H, np.array([theta]))
        assert np.isclose(grad[0], -np.sin(theta), atol=1e-6)

    def test_matches_param_shift(self):
        """Adjoint gradient matches parameter-shift gradient."""
        H = Hamiltonian({"ZI": 0.5, "IX": 0.3})
        params = np.array([0.7, 1.1])

        def _build(params):
            qc = MiniCircuit(2)
            c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
            c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
            qc._instructions = [
                _Instruction("ry", [0],
                    np.array([[c0, -s0], [s0, c0]], dtype=complex), param_idx=0),
                _Instruction("ry", [1],
                    np.array([[c1, -s1], [s1, c1]], dtype=complex), param_idx=1),
            ]
            return qc

        grad_adj = _adjoint_gradient(_build, H, params)
        # Use standard ansatz for param_shift (no param_idx needed)
        def ansatz(params):
            return MiniCircuit(2).ry(params[0], 0).ry(params[1], 1)
        grad_ps = _param_shift_gradient(ansatz, H, params)
        assert np.allclose(grad_adj, grad_ps, atol=1e-6)


class TestCrossMethodAgreement:
    """All three gradient methods should agree."""

    def test_single_param_all_methods(self):
        """All methods agree on single-parameter Ry-Z gradient."""
        H = Hamiltonian({"Z": 1.0})
        theta = 1.23

        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)

        def ansatz_adj(params):
            c, s = np.cos(params[0]/2), np.sin(params[0]/2)
            qc = MiniCircuit(1)
            qc._instructions = [
                _Instruction("ry", [0],
                    np.array([[c, -s], [s, c]], dtype=complex), param_idx=0)
            ]
            return qc

        grad_ps = _param_shift_gradient(ansatz, H, np.array([theta]))
        grad_fd = _finite_diff_gradient(ansatz, H, np.array([theta]))
        grad_adj = _adjoint_gradient(ansatz_adj, H, np.array([theta]))

        assert np.allclose(grad_ps, grad_fd, atol=1e-5)
        assert np.allclose(grad_ps, grad_adj, atol=1e-5)

    def test_multi_param_all_methods(self):
        """All methods agree on multi-parameter circuit."""
        H = Hamiltonian({"ZI": 0.5, "IZ": -0.3, "XX": 0.2})
        params = np.array([0.5, 1.0, 0.3])

        def ansatz(params):
            return (MiniCircuit(2)
                    .ry(params[0], 0)
                    .ry(params[1], 1)
                    .cx(0, 1)
                    .ry(params[2], 0))

        def ansatz_adj(params):
            qc = MiniCircuit(2)
            c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
            c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
            c2, s2 = np.cos(params[2]/2), np.sin(params[2]/2)
            cx_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
            qc._instructions = [
                _Instruction("ry", [0],
                    np.array([[c0, -s0], [s0, c0]], dtype=complex), param_idx=0),
                _Instruction("ry", [1],
                    np.array([[c1, -s1], [s1, c1]], dtype=complex), param_idx=1),
                _Instruction("cx", [0, 1], cx_mat, param_idx=None),
                _Instruction("ry", [0],
                    np.array([[c2, -s2], [s2, c2]], dtype=complex), param_idx=2),
            ]
            return qc

        grad_ps = _param_shift_gradient(ansatz, H, params)
        grad_fd = _finite_diff_gradient(ansatz, H, params)
        grad_adj = _adjoint_gradient(ansatz_adj, H, params)

        assert np.allclose(grad_ps, grad_fd, atol=1e-4)
        assert np.allclose(grad_ps, grad_adj, atol=1e-4)


class TestExpectationFunction:
    """Tests for the expectation() public API."""

    def test_basic(self):
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)
        val = expectation(ansatz, H, np.array([0.0]))
        assert np.isclose(val, 1.0)  # ⟨0|Z|0⟩ = 1

    def test_pi_rotation(self):
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)
        val = expectation(ansatz, H, np.array([np.pi]))
        assert np.isclose(val, -1.0)  # ⟨1|Z|1⟩ = -1


class TestExpectationAndGradient:
    """Tests for the combined expectation_and_gradient() API."""

    def test_basic(self):
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)
        val, grad = expectation_and_gradient(
            ansatz, H, np.array([np.pi / 3]), method="param_shift"
        )
        assert np.isclose(val, np.cos(np.pi / 3))
        assert np.isclose(grad[0], -np.sin(np.pi / 3), atol=1e-6)

    def test_finite_diff_method(self):
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)
        val, grad = expectation_and_gradient(
            ansatz, H, np.array([0.5]), method="finite_diff"
        )
        assert np.isclose(val, np.cos(0.5))
        assert np.isclose(grad[0], -np.sin(0.5), atol=1e-5)


class TestVQEOptimization:
    """Integration test: VQE optimization using gradients."""

    def test_h2_vqe_convergence(self):
        """
        VQE on H₂ should converge to near the exact ground state energy.

        This is the key integration test: optimizer + gradients + Hamiltonian
        all working together.
        """
        from scipy.optimize import minimize

        H = molecular_hydrogen(0.74)
        exact_E0 = H.ground_state_energy()

        def ansatz(params):
            return (MiniCircuit(2)
                    .ry(params[0], 0)
                    .ry(params[1], 1)
                    .cx(0, 1)
                    .ry(params[2], 0)
                    .ry(params[3], 1))

        # Use parameter-shift (most reliable, no param_idx needed)
        def cost(params):
            return expectation(ansatz, H, params)

        def cost_grad(params):
            return _param_shift_gradient(ansatz, H, params)

        np.random.seed(42)
        x0 = np.random.randn(4) * 0.1

        result = minimize(cost, x0, jac=cost_grad, method="L-BFGS-B",
                          options={"maxiter": 200})

        # Should get within 0.05 Ha of exact (finite ansatz expressibility)
        assert result.fun < exact_E0 + 0.05, \
            f"VQE energy {result.fun:.4f} too far from exact {exact_E0:.4f}"

    def test_single_qubit_optimization(self):
        """Optimize Ry angle to minimize ⟨Z⟩ → should find θ = π."""
        from scipy.optimize import minimize

        H = Hamiltonian({"Z": 1.0})

        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)

        def cost_and_grad(params):
            val = expectation(ansatz, H, params)
            grad = _param_shift_gradient(ansatz, H, params)
            return val, grad

        result = minimize(cost_and_grad, x0=[0.1], jac=True, method="L-BFGS-B")
        # Minimum of cos(θ) is at θ = π, giving -1
        assert np.isclose(result.fun, -1.0, atol=1e-5)

    def test_ising_vqe(self):
        """VQE on 2-qubit Ising model converges."""
        from scipy.optimize import minimize

        H = transverse_field_ising(2, J=1.0, h=0.5)
        exact_E0 = H.ground_state_energy()

        def ansatz(params):
            return (MiniCircuit(2)
                    .ry(params[0], 0)
                    .ry(params[1], 1)
                    .cx(0, 1)
                    .ry(params[2], 0)
                    .ry(params[3], 1)
                    .cx(1, 0)
                    .ry(params[4], 0)
                    .ry(params[5], 1))

        np.random.seed(7)
        x0 = np.random.randn(6) * 0.1

        result = minimize(
            lambda p: expectation(ansatz, H, p),
            x0, jac=lambda p: _param_shift_gradient(ansatz, H, p),
            method="L-BFGS-B", options={"maxiter": 300}
        )

        assert result.fun < exact_E0 + 0.1


class TestEdgeCases:
    """Edge cases and corner cases."""

    def test_zero_params(self):
        """Circuit with no parameters should return zero gradient."""
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1).h(0)
        grad = _param_shift_gradient(ansatz, H, np.array([]))
        assert len(grad) == 0

    def test_identity_hamiltonian(self):
        """Gradient of ⟨ψ|I|ψ⟩ = 1 is always 0."""
        H = Hamiltonian({"I": 1.0})
        def ansatz(params):
            return MiniCircuit(1).ry(params[0], 0)
        grad = _param_shift_gradient(ansatz, H, np.array([0.5]))
        assert np.isclose(grad[0], 0.0, atol=1e-10)

    def test_three_qubit(self):
        """Gradient on 3-qubit circuit."""
        H = Hamiltonian({"ZII": 1.0, "IZI": 1.0, "IIZ": 1.0})
        def ansatz(params):
            return (MiniCircuit(3)
                    .ry(params[0], 0)
                    .ry(params[1], 1)
                    .ry(params[2], 2))
        params = np.array([0.3, 0.6, 0.9])
        grad = _param_shift_gradient(ansatz, H, params)
        grad_fd = _finite_diff_gradient(ansatz, H, params)
        assert np.allclose(grad, grad_fd, atol=1e-5)

    def test_gradient_shape(self):
        """Gradient should have same shape as params."""
        H = Hamiltonian({"ZI": 1.0})
        def ansatz(params):
            return MiniCircuit(2).ry(params[0], 0).rx(params[1], 1).rz(params[2], 0)
        params = np.array([0.1, 0.2, 0.3])
        for method_fn in [_param_shift_gradient, _finite_diff_gradient]:
            grad = method_fn(ansatz, H, params)
            assert grad.shape == params.shape

    def test_invalid_method_raises(self):
        H = Hamiltonian({"Z": 1.0})
        def ansatz(params):
            return MiniCircuit(1)
        with pytest.raises(ValueError, match="Unknown method"):
            gradient(ansatz, H, np.array([0.1]), method="invalid")

    def test_dimension_mismatch_raises(self):
        H = Hamiltonian({"ZZ": 1.0})  # 2-qubit
        sv = np.array([1, 0], dtype=complex)  # 1-qubit
        with pytest.raises(ValueError, match="dimension"):
            H.expectation(sv)


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
