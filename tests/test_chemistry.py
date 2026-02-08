"""
Comprehensive tests for the quantum chemistry module.

Tests cover:
- Jordan-Wigner transformation correctness
- Pauli algebra utilities
- Fermionic anticommutation relations
- Molecular Hamiltonian construction (using pre-computed integrals)
- Hydrogen at multiple bond lengths (JW vs FCI benchmark)
- LiH with active space
- VQE convergence on molecular Hamiltonians
- Edge cases and error handling

All molecular integrals are pre-computed from PySCF and embedded as
fixtures, so these tests run on any platform without PySCF installed.
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


# ═══════════════════════════════════════════════════════════════════════
# PRE-COMPUTED MOLECULAR FIXTURES (from PySCF, STO-3G basis)
# ═══════════════════════════════════════════════════════════════════════

def _spatial_to_spin(h1, eri):
    """Convert spatial-orbital integrals to spin-orbital integrals."""
    n_spatial = h1.shape[0]
    n_spin = 2 * n_spatial
    h1_spin = np.zeros((n_spin, n_spin))
    h2_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
    for p in range(n_spatial):
        for q in range(n_spatial):
            h1_spin[2*p, 2*q] = h1[p, q]
            h1_spin[2*p+1, 2*q+1] = h1[p, q]
            for r in range(n_spatial):
                for s in range(n_spatial):
                    val = eri[p, q, r, s]
                    h2_spin[2*p, 2*q, 2*r, 2*s] = val
                    h2_spin[2*p+1, 2*q+1, 2*r+1, 2*s+1] = val
                    h2_spin[2*p, 2*q, 2*r+1, 2*s+1] = val
                    h2_spin[2*p+1, 2*q+1, 2*r, 2*s] = val
    return h1_spin, h2_spin


# H₂ at equilibrium (0.74 Å), STO-3G
_H2_074 = {
    "h1": np.array([[-1.2533097866, 0.0], [0.0, -0.4750688488]]),
    "eri": np.array([
        [[[0.6747559268, 0.0], [0.0, 0.6637114014]],
         [[0.0, 0.181210462], [0.181210462, 0.0]]],
        [[[0.0, 0.181210462], [0.181210462, 0.0]],
         [[0.6637114014, 0.0], [0.0, 0.6976515045]]]
    ]),
    "nuc": 0.7151043390810812,
    "hf_energy": -1.1167593073964255,
    "fci_energy": -1.1372838344885023,
    "mo_energies": [-0.5784, 0.6710],  # approximate
}

# H₂ at various bond lengths
_H2_BONDS = {
    0.5: {
        "h1": np.array([[-1.4105283677, 0.0], [0.0, -0.2569357824]]),
        "eri": np.array([
            [[[0.7197060391, 0.0], [0.0, 0.7072398415]],
             [[0.0, 0.1688702277], [0.1688702277, 0.0]]],
            [[[0.0, 0.1688702277], [0.1688702277, 0.0]],
             [[0.7072398415, 0.0], [0.0, 0.7448393704]]]
        ]),
        "nuc": 1.05835442184,
        "fci_energy": -1.0551597944706257,
    },
    1.0: {
        "h1": np.array([[-1.1108441799, 0.0], [0.0, -0.5891210037]]),
        "eri": np.array([
            [[[0.6264024995, 0.0], [0.0, 0.6217067631]],
             [[0.0, 0.1967905835], [0.1967905835, 0.0]]],
            [[[0.0, 0.1967905835], [0.1967905835, 0.0]],
             [[0.6217067631, 0.0], [0.0, 0.6530707469]]]
        ]),
        "nuc": 0.52917721092,
        "fci_energy": -1.1011503302326187,
    },
    1.5: {
        "h1": np.array([[-0.9081808725, 0.0], [0.0, -0.6653369358]]),
        "eri": np.array([
            [[[0.5527033831, 0.0], [0.0, 0.5596841556]],
             [[0.0, 0.2295359361], [0.2295359361, 0.0]]],
            [[[0.0, 0.2295359361], [0.2295359361, 0.0]],
             [[0.5596841556, 0.0], [0.0, 0.5834207612]]]
        ]),
        "nuc": 0.35278480728,
        "fci_energy": -0.9981493534714101,
    },
    2.0: {
        "h1": np.array([[-0.7789220361, 0.0], [0.0, -0.6702666718]]),
        "eri": np.array([
            [[[0.5094628124, 0.0], [0.0, 0.5192012581]],
             [[0.0, 0.2591384749], [0.2591384749, 0.0]]],
            [[[0.0, 0.2591384749], [0.2591384749, 0.0]],
             [[0.5192012581, 0.0], [0.0, 0.5346641195]]]
        ]),
        "nuc": 0.26458860546,
        "fci_energy": -0.9486411121761855,
    },
}

# LiH at 1.6 Å, CASCI(2,2) active space
_LIH_16 = {
    "h1": np.array([[-0.7725817191, 0.048579605], [0.048579605, -0.3559395443]]),
    "eri": np.array([
        [[[0.4873109667, -0.0485795923], [-0.0485795923, 0.2236100448]],
         [[-0.0485795923, 0.0130639836], [0.0130639836, 0.0074841721]]],
        [[[-0.0485795923, 0.0130639836], [0.0130639836, 0.0074841721]],
         [[0.2236100448, 0.0074841721], [0.0074841721, 0.3378822715]]]
    ]),
    "nuc": -6.804012298302059,
    "hf_energy": -7.8618647698086574,
    "fci_energy": -7.882324378883502,
    "casci_energy": -7.862128833438598,
}


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
        assert result == "ZZ"
        assert np.isclose(phase, 1.0)

    def test_pauli_anticommutation(self):
        """XY = iZ and YX = -iZ → opposite phases."""
        _, p1 = _multiply_pauli_strings("X", "Y")
        _, p2 = _multiply_pauli_strings("Y", "X")
        assert np.isclose(p1, -p2)


# ═══════════════════════════════════════════════════════════════════════
# JORDAN-WIGNER OPERATOR TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestJWOperators:
    """Tests for JW creation/annihilation operator representations."""

    def test_creation_0_single_qubit(self):
        """a†_0 = (X - iY)/2 on 1 qubit."""
        op = _creation_op_jw(0, 1)
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
        """a†_1 includes Z chain on qubit 0."""
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
        assert np.isclose(collected.get("I", 0), 0.5)
        assert np.isclose(collected.get("Z", 0), -0.5)

    def test_number_operator_qubit_1(self):
        """a†_1 a_1 = (II - IZ)/2 — Z chains cancel."""
        create = _creation_op_jw(1, 2)
        annihil = _annihilation_op_jw(1, 2)
        op = _qubit_op_multiply(create, annihil)
        collected = _qubit_op_collect(op)
        assert np.isclose(collected.get("II", 0), 0.5)
        assert np.isclose(collected.get("IZ", 0), -0.5)

    def test_anticommutation_same_site(self):
        """{a_p, a†_p} = I for all p."""
        for p in range(3):
            n_qubits = 3
            create = _creation_op_jw(p, n_qubits)
            annihil = _annihilation_op_jw(p, n_qubits)
            op1 = _qubit_op_multiply(annihil, create)
            op2 = _qubit_op_multiply(create, annihil)
            collected = _qubit_op_collect(op1 + op2)
            identity = "I" * n_qubits
            assert np.isclose(collected.get(identity, 0), 1.0, atol=1e-10)
            for pauli, coeff in collected.items():
                if pauli != identity:
                    assert abs(coeff) < 1e-10

    def test_anticommutation_different_sites(self):
        """{a_p, a†_q} = 0 for p ≠ q."""
        n_qubits = 3
        for p in range(n_qubits):
            for q in range(n_qubits):
                if p == q:
                    continue
                create_q = _creation_op_jw(q, n_qubits)
                annihil_p = _annihilation_op_jw(p, n_qubits)
                op1 = _qubit_op_multiply(annihil_p, create_q)
                op2 = _qubit_op_multiply(create_q, annihil_p)
                collected = _qubit_op_collect(op1 + op2)
                for pauli, coeff in collected.items():
                    assert abs(coeff) < 1e-10, \
                        f"{{a_{p}, a†_{q}}} non-zero {pauli}: {coeff}"


# ═══════════════════════════════════════════════════════════════════════
# JORDAN-WIGNER TRANSFORM TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestJordanWigner:
    """Tests for the full Jordan-Wigner transformation."""

    def test_single_orbital_identity(self):
        """Zero integrals → just nuclear repulsion."""
        h1 = np.zeros((1, 1))
        h2 = np.zeros((1, 1, 1, 1))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=1.5)
        assert "I" in terms
        assert np.isclose(terms["I"], 1.5)

    def test_single_orbital_number(self):
        """h_{00} = ε → eigenvalues 0 and ε."""
        h1 = np.array([[1.0]])
        h2 = np.zeros((1, 1, 1, 1))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.0)
        H = Hamiltonian(terms)
        eigs = sorted(np.linalg.eigvalsh(H.matrix()))
        assert np.isclose(eigs[0], 0.0, atol=1e-10)
        assert np.isclose(eigs[1], 1.0, atol=1e-10)

    def test_two_orbital_noninteracting(self):
        """Two non-interacting orbitals: energies 0, ε₁, ε₂, ε₁+ε₂."""
        h1 = np.diag([1.0, 2.0])
        h2 = np.zeros((2, 2, 2, 2))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.0)
        H = Hamiltonian(terms)
        eigs = sorted(np.linalg.eigvalsh(H.matrix()))
        assert np.isclose(eigs[0], 0.0, atol=1e-10)
        assert np.isclose(eigs[1], 1.0, atol=1e-10)
        assert np.isclose(eigs[2], 2.0, atol=1e-10)
        assert np.isclose(eigs[3], 3.0, atol=1e-10)

    def test_hermiticity(self):
        """JW Hamiltonian must be Hermitian."""
        np.random.seed(42)
        n = 2
        h1 = np.random.randn(n, n)
        h1 = (h1 + h1.T) / 2
        h2 = np.random.randn(n, n, n, n) * 0.1
        h2 = (h2 + h2.transpose(2, 3, 0, 1)) / 2
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.5)
        H = Hamiltonian(terms)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_threshold_filters(self):
        """Terms below threshold should be dropped."""
        h1 = np.array([[1e-15]])
        h2 = np.zeros((1, 1, 1, 1))
        terms = jordan_wigner(h1, h2, nuclear_repulsion=0.0, threshold=1e-10)
        assert all(abs(c) >= 1e-10 for c in terms.values())


# ═══════════════════════════════════════════════════════════════════════
# MOLECULAR HAMILTONIAN TESTS (pre-computed fixtures)
# ═══════════════════════════════════════════════════════════════════════

class TestH2Molecule:
    """H₂ molecule: JW Hamiltonian vs exact FCI energy."""

    def test_h2_hamiltonian_qubits(self):
        """H₂ in STO-3G: 2 spatial → 4 spin-orbitals → 4 qubits."""
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_H2_074["nuc"])
        H = Hamiltonian(terms)
        assert H.n_qubits == 4

    def test_h2_hermitian(self):
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_H2_074["nuc"])
        H = Hamiltonian(terms)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_h2_jw_matches_fci(self):
        """JW ground state matches FCI for H₂ at 0.74 Å."""
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_H2_074["nuc"])
        H = Hamiltonian(terms)
        e_jw = H.ground_state_energy()
        assert np.isclose(e_jw, _H2_074["fci_energy"], atol=1e-6), \
            f"JW={e_jw:.6f} vs FCI={_H2_074['fci_energy']:.6f}"

    def test_h2_ground_below_hf(self):
        """Exact ground state must be ≤ HF energy (correlation is negative)."""
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_H2_074["nuc"])
        H = Hamiltonian(terms)
        assert H.ground_state_energy() <= _H2_074["hf_energy"] + 1e-6

    @pytest.mark.parametrize("distance", [0.5, 1.0, 1.5, 2.0])
    def test_h2_bond_lengths(self, distance):
        """JW matches FCI at multiple bond lengths."""
        data = _H2_BONDS[distance]
        h1_spin, h2_spin = _spatial_to_spin(data["h1"], data["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=data["nuc"])
        H = Hamiltonian(terms)
        e_jw = H.ground_state_energy()
        assert np.isclose(e_jw, data["fci_energy"], atol=1e-6), \
            f"d={distance}: JW={e_jw:.6f} vs FCI={data['fci_energy']:.6f}"

    def test_h2_energy_minimum_near_equilibrium(self):
        """Energy minimum should be at 0.74 Å (equilibrium)."""
        energies = {}
        for d, data in _H2_BONDS.items():
            h1_spin, h2_spin = _spatial_to_spin(data["h1"], data["eri"])
            terms = jordan_wigner(h1_spin, h2_spin,
                                  nuclear_repulsion=data["nuc"])
            H = Hamiltonian(terms)
            energies[d] = H.ground_state_energy()
        # Also add equilibrium
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_H2_074["nuc"])
        H = Hamiltonian(terms)
        energies[0.74] = H.ground_state_energy()
        # Minimum near 0.74 (should be 0.74 or 1.0 — both near minimum)
        min_d = min(energies, key=energies.get)
        assert min_d in [0.74, 1.0], f"Minimum at {min_d}, expected near 0.74"

    def test_h2_dissociation_limit(self):
        """Energy at large distance > energy at equilibrium."""
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms_eq = jordan_wigner(h1_spin, h2_spin,
                                 nuclear_repulsion=_H2_074["nuc"])
        h1_spin, h2_spin = _spatial_to_spin(
            _H2_BONDS[2.0]["h1"], _H2_BONDS[2.0]["eri"])
        terms_far = jordan_wigner(h1_spin, h2_spin,
                                  nuclear_repulsion=_H2_BONDS[2.0]["nuc"])
        e_eq = Hamiltonian(terms_eq).ground_state_energy()
        e_far = Hamiltonian(terms_far).ground_state_energy()
        assert e_far > e_eq


class TestLiHMolecule:
    """LiH molecule with CASCI(2,2) active space."""

    def test_lih_hamiltonian_qubits(self):
        """LiH active space: 2 orbitals → 4 qubits."""
        h1_spin, h2_spin = _spatial_to_spin(_LIH_16["h1"], _LIH_16["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_LIH_16["nuc"])
        H = Hamiltonian(terms)
        assert H.n_qubits == 4

    def test_lih_hermitian(self):
        h1_spin, h2_spin = _spatial_to_spin(_LIH_16["h1"], _LIH_16["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_LIH_16["nuc"])
        H = Hamiltonian(terms)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_lih_jw_matches_casci(self):
        """JW ground state matches CASCI energy for active space."""
        h1_spin, h2_spin = _spatial_to_spin(_LIH_16["h1"], _LIH_16["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_LIH_16["nuc"])
        H = Hamiltonian(terms)
        e_jw = H.ground_state_energy()
        # Should match CASCI(2,2) energy
        assert np.isclose(e_jw, _LIH_16["casci_energy"], atol=1e-4), \
            f"JW={e_jw:.6f} vs CASCI={_LIH_16['casci_energy']:.6f}"


# ═══════════════════════════════════════════════════════════════════════
# MOLECULE CLASS TESTS (structural, no PySCF needed for import)
# ═══════════════════════════════════════════════════════════════════════

class TestMoleculeClass:
    """Tests for the Molecule class interface."""

    def test_imports(self):
        from tiny_qpu.chemistry import (
            Molecule, hydrogen, lithium_hydride, water,
            bond_dissociation_curve, jordan_wigner,
        )

    def test_jordan_wigner_reexport(self):
        from tiny_qpu.chemistry import jordan_wigner as jw
        assert callable(jw)


# ═══════════════════════════════════════════════════════════════════════
# VQE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestVQEChemistry:
    """VQE optimization on molecular Hamiltonians."""

    def test_h2_vqe_converges(self):
        """VQE on H₂ Hamiltonian converges to near FCI."""
        from scipy.optimize import minimize
        from tiny_qpu.gradients import expectation
        from tiny_qpu.gradients.differentiation import (
            _param_shift_gradient,
            _apply_single_qubit_gate,
            _apply_two_qubit_gate,
        )

        # Build Hamiltonian from fixture
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_H2_074["nuc"])
        H = Hamiltonian(terms)
        e_fci = _H2_074["fci_energy"]

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

        assert result.fun < e_fci + 0.1, \
            f"VQE={result.fun:.4f} too far from FCI={e_fci:.4f}"


# ═══════════════════════════════════════════════════════════════════════
# EDGE CASES AND ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Error handling and edge cases."""

    def test_jw_empty_integrals(self):
        terms = jordan_wigner(
            np.zeros((2, 2)), np.zeros((2, 2, 2, 2)),
            nuclear_repulsion=5.0
        )
        assert np.isclose(terms.get("II", 0), 5.0)

    def test_jw_symmetric_integrals(self):
        h1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        h2 = np.zeros((2, 2, 2, 2))
        terms = jordan_wigner(h1, h2)
        H = Hamiltonian(terms)
        mat = H.matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_real_coefficients(self):
        """All JW coefficients should be real (imaginary parts cancel)."""
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        terms = jordan_wigner(h1_spin, h2_spin,
                              nuclear_repulsion=_H2_074["nuc"])
        for pauli, coeff in terms.items():
            assert isinstance(coeff, float), f"{pauli}: {coeff} not float"

    def test_spatial_to_spin_dimensions(self):
        """Spin-orbital arrays have correct dimensions."""
        h1_spin, h2_spin = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        assert h1_spin.shape == (4, 4)
        assert h2_spin.shape == (4, 4, 4, 4)

    def test_spatial_to_spin_symmetry(self):
        """Spin-orbital h1 should preserve hermiticity."""
        h1_spin, _ = _spatial_to_spin(_H2_074["h1"], _H2_074["eri"])
        assert np.allclose(h1_spin, h1_spin.T)


# ═══════════════════════════════════════════════════════════════════════
# OPTIONAL: LIVE PYSCF TESTS (only when PySCF is available)
# ═══════════════════════════════════════════════════════════════════════

try:
    import pyscf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

needs_pyscf = pytest.mark.skipif(not HAS_PYSCF, reason="PySCF not installed")


@needs_pyscf
class TestLivePySCF:
    """Live PySCF tests — only run when PySCF is installed."""

    def test_hydrogen_end_to_end(self):
        from tiny_qpu.chemistry import hydrogen
        mol = hydrogen(0.74)
        H = mol.hamiltonian()
        e_jw = H.ground_state_energy()
        e_fci = mol.fci_energy()
        assert np.isclose(e_jw, e_fci, atol=1e-4)

    def test_lithium_hydride_end_to_end(self):
        from tiny_qpu.chemistry import lithium_hydride
        mol = lithium_hydride(1.6)
        H = mol.hamiltonian()
        e_jw = H.ground_state_energy()
        assert e_jw < mol.hf_energy + 0.001

    def test_bond_dissociation_curve(self):
        from tiny_qpu.chemistry import bond_dissociation_curve
        results = bond_dissociation_curve("H", "H", distances=[0.5, 1.0])
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
