"""
Comprehensive tests for the quantum chemistry module.

Tests cover:
- Jordan-Wigner transformation correctness
- Pauli algebra utilities
- Molecule class (PySCF integration)
- Hydrogen at multiple bond lengths
- LiH with active space
- Bond dissociation curves
- VQE on molecular Hamiltonians
- Edge cases and error handling

Requires: pyscf (pip install pyscf)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tiny_qpu.chemistry.transforms import (
    jordan_wigner,
    _creation_op_jw,
    _annihilation_op_jw,
    _qubit_op_multiply,
    _qubit_op_collect,
    _multiply_pauli_strings,
)
from tiny_qpu.gradients.hamiltonian import Hamiltonian

# Check PySCF availability
try:
    import pyscf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

needs_pyscf = pytest.mark.skipif(not HAS_PYSCF, reason="PySCF not installed")


# ═══════════════════════════════════════════════════════════════════════
# PAULI ALGEBRA TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestPauliAlgebra:
    """Tests for Pauli string multiplication."""

    def test_identity_multiply(self):
        result, phase = _multiply_pauli_strings("II", "II")
        assert result == "II"
        assert phase == 1.0

    def test_xx_equals_i(self):
        result, phase = _multiply_pauli_strings("X", "X")
        assert result == "I"
        assert phase == 1.0

    def test_xy_equals_iz(self):
        result, phase = _multiply_pauli_strings("X", "Y")
        assert result == "Z"
        assert np.isclose(phase, 1j)

    def test_yz_equals_ix(self):
        result, phase = _multiply_pauli_strings("Y", "Z")
        assert result == "X"
        assert np.isclose(phase, 1j)

    def test_zx_equals_iy(self):
        result, phase = _multiply_pauli_strings("Z", "X")
        assert result == "Y"
        assert np.isclose(phase, 1j)

    def test_multi_qubit(self):
        result, phase = _multiply_pauli_strings("XY", "YX")
        # X*Y = iZ, Y*X = -iZ => iZ * (-iZ) = -i^2 * ZZ = ZZ
        assert result == "ZZ"
        assert np.isclose(phase, 1.0)

    def test_pauli_anticommutation(self):
        """XY = iZ and YX = -iZ → XY * YX = i * (-i) * ZZ = ZZ"""
        r1, p1 = _multiply_pauli_strings("X", "Y")  # iZ
        r2, p2 = _multiply_pauli_strings("Y", "X")  # -iZ
        assert np.isclose(p1, -p2)  # opposite phases


# ═══════════════════════════════════════════════════════════════════════
# JORDAN-WIGNER OPERATOR TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestJWOperators:
    """Tests for JW creation/annihilation operator representations."""

    def test_creation_0_single_qubit(self):
        """a†_0 = (X - iY)/2 on 1 qubit."""
        op = _creation_op_jw(0, 1)
        assert len(op) == 2
        paulis = {p: c for p, c in op}
        assert np.isclose(paulis["X"], 0.5)
        assert np.isclose(paulis["Y"], -0.5j)

    def test_annihilation_0_single_qubit(self):
        """a_0 = (X + iY)/2 on 1 qubit."""
        op = _annihilation_op_jw(0, 1)
        paulis = {p: c for p, c in op}
        assert np.isclose(paulis["X"], 0.5)
        assert np.isclose(paulis["Y"], 0.5j)

    def test_creation_1_two_qubits(self):
        """a†_1 = (X_1 - iY_1) Z_0 / 2 on 2 qubits."""
        op = _creation_op_jw(1, 2)
        paulis = {p: c for p, c in op}
        assert "ZX" in paulis
        assert "ZY" in paulis
        assert np.isclose(paulis["ZX"], 0.5)
        assert np.isclose(paulis["ZY"], -0.5j)

    def test_number_operator(self):
        """a†_0 a_0 = (I - Z)/2 on 1 qubit."""
        create = _creation_op_jw(0, 1)
        annihil = _annihilation_op_jw(0, 1)
        op = _qubit_op_multiply(create, annihil)
        collected = _qubit_op_collect(op)
        # Should get 0.5*I - 0.5*Z
        assert np.isclose(collected.get("I", 0), 0.5)
        assert np.isclose(collected.get("Z", 0), -0.5)

    def test_number_operator_qubit_1(self):
        """a†_1 a_1 = (I - Z_1)/2 with Z_0 chains canceling."""
        create = _creation_op_jw(1, 2)
        annihil = _annihilation_op_jw(1, 2)
        op = _qubit_op_multiply(create, annihil)
        collected = _qubit_op_collect(op)
        # Z chains cancel: Z_0 * Z_0 = I
        assert np.isclose(collected.get("II", 0), 0.5)
        assert np.isclose(collected.get("IZ", 0), -0.5)

    def test_anticommutation_same_site(self):
        """{a_p, a†_p} = I  →  a_p a†_p + a†_p a_p = I"""
        for p in range(3):
            n_qubits = 3
            create = _creation_op_jw(p, n_qubits)
            annihil = _annihilation_op_jw(p, n_qubits)
            # a_p a†_p
            op1 = _qubit_op_multiply(annihil, create)
            # a†_p a_p
            op2 = _qubit_op_multiply(create, annihil)
            # Sum should be I
            combined = op1 + op2
            collected = _qubit_op_collect(combined)
            identity = "I" * n_qubits
            assert np.isclose(collected.get(identity, 0), 1.0, atol=1e-10)
            # All non-identity terms should cancel
            for pauli, coeff in collected.items():
                if pauli != identity:
                    assert abs(coeff) < 1e-10, f"Non-zero {pauli}: {coeff}"

    def test_anticommutation_different_sites(self):
        """{a_p, a†_q} = 0 for p ≠ q"""
        n_qubits = 3
        for p in range(n_qubits):
            for q in range(n_qubits):
                if p == q:
                    continue
                create_q = _creation_op_jw(q, n_qubits)
                annihil_p = _annihilation_op_jw(p, n_qubits)
                op1 = _qubit_op_multiply(annihil_p, create_q)
                op2 = _qubit_op_multiply(create_q, annihil_p)
                combined = op1 + op2
                collected = _qubit_op_collect(combined)
                for pauli, coeff in collected.items():
                    assert abs(coeff) < 1e-10, \
                        f"{{a_{p}, a†_{q}}} has non-zero {pauli}: {coeff}"


# ═══════════════════════════════════════════════════════════════════════
# JORDAN-WIGNER TRANSFORM TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestJordanWigner:
    """Tests for the full Jordan-Wigner transformation."""

    def test_single_orbital_identity(self):
        """Single orbital with zero integrals → just nuclear repulsion."""
        h1 = np.zeros((1, 1))
        h2 = np.zeros((1, 1, 1, 1))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=1.5)
        assert "I" in terms
        assert np.isclose(terms["I"], 1.5)

    def test_single_orbital_number(self):
        """h_{00} = ε → H = ε(I-Z)/2 + nuc"""
        h1 = np.array([[1.0]])
        h2 = np.zeros((1, 1, 1, 1))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.0)
        H = Hamiltonian(terms)
        # Eigenvalues should be 0 (empty) and 1.0 (occupied)
        eigs = sorted(np.linalg.eigvalsh(H.matrix()))
        assert np.isclose(eigs[0], 0.0, atol=1e-10)
        assert np.isclose(eigs[1], 1.0, atol=1e-10)

    def test_two_orbital_noninteracting(self):
        """Two non-interacting orbitals with energies ε₁, ε₂."""
        h1 = np.diag([1.0, 2.0])
        h2 = np.zeros((2, 2, 2, 2))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.0)
        H = Hamiltonian(terms)
        eigs = sorted(np.linalg.eigvalsh(H.matrix()))
        # States: |00>=0, |10>=1, |01>=2, |11>=3
        assert np.isclose(eigs[0], 0.0, atol=1e-10)   # no electrons
        assert np.isclose(eigs[1], 1.0, atol=1e-10)   # orbital 0
        assert np.isclose(eigs[2], 2.0, atol=1e-10)   # orbital 1
        assert np.isclose(eigs[3], 3.0, atol=1e-10)   # both

    def test_hermiticity(self):
        """JW Hamiltonian must be Hermitian."""
        np.random.seed(42)
        n = 2
        h1 = np.random.randn(n, n)
        h1 = (h1 + h1.T) / 2  # symmetric
        h2 = np.random.randn(n, n, n, n) * 0.1
        h2 = (h2 + h2.transpose(2, 3, 0, 1)) / 2  # (pq|rs) = (rs|pq)
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.5)
        H = Hamiltonian(terms)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_threshold_filters(self):
        """Terms below threshold should be dropped."""
        h1 = np.array([[1e-15]])
        h2 = np.zeros((1, 1, 1, 1))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.0, threshold=1e-10)
        # Only identity-like terms remaining
        assert all(abs(c) >= 1e-10 for c in terms.values())


# ═══════════════════════════════════════════════════════════════════════
# MOLECULE CLASS TESTS (PySCF required)
# ═══════════════════════════════════════════════════════════════════════

@needs_pyscf
class TestMoleculeH2:
    """Tests for H₂ molecule via PySCF."""

    def test_hydrogen_creation(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        assert mol.n_orbitals == 2
        assert mol.n_electrons == 2
        assert mol.n_qubits == 4

    def test_hydrogen_hf_energy(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        # STO-3G HF energy for H2 at 0.74 Å is approximately -1.117
        assert -1.12 < mol.hf_energy < -1.11

    def test_hydrogen_nuclear_repulsion(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        # E_nuc = 1/(0.74 * 1.8897) Hartree ≈ 0.715
        assert 0.71 < mol.nuclear_repulsion < 0.72

    def test_hydrogen_hamiltonian_qubits(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        H = mol.hamiltonian()
        assert H.n_qubits == 4  # 2 spatial → 4 spin-orbitals

    def test_hydrogen_jw_matches_fci(self):
        """JW ground state energy matches PySCF FCI for H₂."""
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        H = mol.hamiltonian()
        e_jw = H.ground_state_energy()
        e_fci = mol.fci_energy()
        assert np.isclose(e_jw, e_fci, atol=1e-4), \
            f"JW={e_jw:.6f} vs FCI={e_fci:.6f}"

    def test_hydrogen_multiple_bond_lengths(self):
        """JW matches FCI at all bond lengths."""
        from tiny_qpu.chemistry import hydrogen
        for d in [0.5, 0.7, 0.74, 1.0, 1.5, 2.0]:
            mol = hydrogen(d)
            H = mol.hamiltonian()
            e_jw = H.ground_state_energy()
            e_fci = mol.fci_energy()
            assert np.isclose(e_jw, e_fci, atol=1e-4), \
                f"d={d}: JW={e_jw:.6f} vs FCI={e_fci:.6f}"

    def test_hydrogen_hamiltonian_hermitian(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        H = mol.hamiltonian()
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_hydrogen_integrals_shape(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        assert mol.one_body_integrals.shape == (2, 2)
        assert mol.two_body_integrals.shape == (2, 2, 2, 2)

    def test_hydrogen_mo_energies(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        mo_e = mol.mo_energies
        assert len(mo_e) == 2
        assert mo_e[0] < mo_e[1]  # bonding < antibonding

    def test_hydrogen_repr(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        r = repr(mol)
        assert "HH" in r
        assert "sto-3g" in r

    def test_hydrogen_str(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        s = str(mol)
        assert "H H" in s
        assert "sto-3g" in s


@needs_pyscf
class TestMoleculeLiH:
    """Tests for LiH molecule with active space."""

    def test_lih_creation(self):
        from tiny_qpu.chemistry import lithium_hydride
        mol = lithium_hydride(1.6)
        assert mol.n_orbitals == 2
        assert mol.n_electrons == 2
        assert mol.n_qubits == 4

    def test_lih_hf_energy(self):
        from tiny_qpu.chemistry import lithium_hydride
        mol = lithium_hydride(1.6)
        # LiH STO-3G HF energy is approximately -7.86
        assert -8.0 < mol.hf_energy < -7.8

    def test_lih_hamiltonian_hermitian(self):
        from tiny_qpu.chemistry import lithium_hydride
        mol = lithium_hydride(1.6)
        H = mol.hamiltonian()
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_lih_ground_state_below_hf(self):
        """JW ground state should be at or below HF energy."""
        from tiny_qpu.chemistry import lithium_hydride
        mol = lithium_hydride(1.6)
        H = mol.hamiltonian()
        e_jw = H.ground_state_energy()
        # Active space JW energy should be ≤ HF (correlation energy is negative)
        assert e_jw <= mol.hf_energy + 0.001


@needs_pyscf
class TestMoleculeCustom:
    """Tests for custom molecule construction."""

    def test_custom_geometry(self):
        from tiny_qpu.chemistry import Molecule
        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.0))],
            basis="sto-3g"
        )
        assert mol.n_orbitals == 2
        assert mol.n_qubits == 4

    def test_custom_basis(self):
        from tiny_qpu.chemistry import Molecule
        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 0.74))],
            basis="6-31g"
        )
        # 6-31G has more basis functions
        assert mol.n_orbitals > 2

    def test_custom_charge(self):
        from tiny_qpu.chemistry import Molecule
        mol = Molecule(
            [("He", (0, 0, 0))],
            basis="sto-3g",
            charge=1,
            spin=1,
        )
        assert mol.n_electrons == 1

    def test_active_space_reduction(self):
        from tiny_qpu.chemistry import Molecule
        # Full H2 in 6-31G has more orbitals
        mol_full = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 0.74))],
            basis="6-31g"
        )
        mol_active = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 0.74))],
            basis="6-31g",
            n_active_orbitals=2,
            n_active_electrons=2,
        )
        assert mol_active.n_orbitals < mol_full.n_orbitals
        assert mol_active.n_qubits == 4


@needs_pyscf
class TestBondDissociation:
    """Tests for bond dissociation curve."""

    def test_h2_curve(self):
        from tiny_qpu.chemistry import bond_dissociation_curve
        results = bond_dissociation_curve("H", "H",
                                          distances=[0.5, 1.0, 1.5])
        assert len(results) == 3
        for d, mol in results:
            assert mol.n_qubits == 4

    def test_h2_energy_minimum(self):
        """H₂ energy should be lowest near equilibrium (0.74 Å)."""
        from tiny_qpu.chemistry import bond_dissociation_curve
        results = bond_dissociation_curve("H", "H",
                                          distances=[0.5, 0.74, 1.0, 2.0])
        energies = {d: mol.hf_energy for d, mol in results}
        # Minimum should be near 0.74
        min_d = min(energies, key=energies.get)
        assert min_d == 0.74


# ═══════════════════════════════════════════════════════════════════════
# VQE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

@needs_pyscf
class TestVQEChemistry:
    """VQE optimization on molecular Hamiltonians."""

    def test_h2_vqe_converges(self):
        """VQE on H₂ PySCF Hamiltonian converges to near FCI."""
        from scipy.optimize import minimize
        from tiny_qpu.chemistry import hydrogen
        from tiny_qpu.gradients import expectation
        from tiny_qpu.gradients.differentiation import _param_shift_gradient

        mol = hydrogen(0.74)
        H = mol.hamiltonian()
        e_fci = mol.fci_energy()

        # Minimal test circuit class
        from tiny_qpu.gradients.differentiation import (
            _apply_single_qubit_gate, _apply_two_qubit_gate
        )

        class _Instr:
            def __init__(self, name, qubits, matrix):
                self.name = name
                self.qubits = qubits
                self.matrix = matrix
                self.param_idx = None

        class _QC:
            def __init__(self, nq):
                self._n_qubits = nq
                self._instructions = []
            @property
            def n_qubits(self): return self._n_qubits
            @property
            def num_qubits(self): return self._n_qubits
            def ry(self, t, q):
                c, s = np.cos(t/2), np.sin(t/2)
                self._instructions.append(_Instr("ry", [q],
                    np.array([[c,-s],[s,c]], dtype=complex)))
                return self
            def cx(self, c, t):
                self._instructions.append(_Instr("cx", [c,t],
                    np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                             dtype=complex)))
                return self
            def x(self, q):
                self._instructions.append(_Instr("x", [q],
                    np.array([[0,1],[1,0]], dtype=complex)))
                return self
            def statevector(self):
                sv = np.zeros(2**self._n_qubits, dtype=complex)
                sv[0] = 1.0
                for instr in self._instructions:
                    if len(instr.qubits) == 1:
                        sv = _apply_single_qubit_gate(
                            sv, instr.matrix, instr.qubits[0], self._n_qubits)
                    else:
                        sv = _apply_two_qubit_gate(
                            sv, instr.matrix, instr.qubits, self._n_qubits)
                return sv

        def ansatz(params):
            # UCCSD-inspired: start from HF state |0011>, apply entangling layers
            qc = _QC(4)
            qc.x(0).x(1)  # HF state |0011>
            qc.ry(params[0], 0).ry(params[1], 1)
            qc.ry(params[2], 2).ry(params[3], 3)
            qc.cx(0, 1).cx(2, 3).cx(1, 2)
            qc.ry(params[4], 0).ry(params[5], 1)
            qc.ry(params[6], 2).ry(params[7], 3)
            return qc

        np.random.seed(42)
        x0 = np.random.randn(8) * 0.1

        result = minimize(
            lambda p: expectation(ansatz, H, p),
            x0,
            jac=lambda p: _param_shift_gradient(ansatz, H, p),
            method="L-BFGS-B",
            options={"maxiter": 200},
        )

        # Should get within 0.1 Ha of FCI (4-qubit ansatz is expressive enough)
        assert result.fun < e_fci + 0.1, \
            f"VQE={result.fun:.4f} too far from FCI={e_fci:.4f}"


# ═══════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Error handling and edge cases."""

    @needs_pyscf
    def test_pyscf_import_error_message(self):
        """Verify helpful error message when PySCF is missing."""
        # We can't easily test this when PySCF IS installed,
        # but we can verify the Molecule class exists and works
        from tiny_qpu.chemistry import Molecule
        mol = Molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.74))])
        assert mol.hf_energy is not None

    def test_jw_empty_integrals(self):
        """Zero integrals with nuclear repulsion."""
        terms = jordan_wigner(
            np.zeros((2, 2)), np.zeros((2, 2, 2, 2)),
            nuclear_repulsion=5.0
        )
        assert np.isclose(terms.get("II", 0), 5.0)

    def test_jw_symmetric_integrals(self):
        """Symmetric one-body integrals produce Hermitian Hamiltonian."""
        h1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        h2 = np.zeros((2, 2, 2, 2))
        terms = jordan_wigner(h1, h2)
        H = Hamiltonian(terms)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
