"""
Molecular electronic structure interface for quantum simulation.

Provides the Molecule class which uses PySCF (optional dependency) to
compute molecular integrals and builds qubit Hamiltonians via the
Jordan-Wigner transformation.

Pipeline:
    Molecule geometry → PySCF HF/integrals → Jordan-Wigner → Hamiltonian

Example:
    >>> from tiny_qpu.chemistry import Molecule
    >>> mol = Molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.74))], basis="sto-3g")
    >>> H = mol.hamiltonian()
    >>> print(f"Nuclear repulsion: {mol.nuclear_repulsion:.4f} Ha")
    >>> print(f"HF energy: {mol.hf_energy:.4f} Ha")
    >>> print(f"Qubit Hamiltonian: {H.n_qubits} qubits, {H.n_terms} terms")
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict

from .transforms import jordan_wigner


class Molecule:
    """
    Molecular electronic structure calculator.

    Wraps PySCF to perform Hartree-Fock calculations and extract
    molecular integrals, then uses Jordan-Wigner transformation
    to produce qubit Hamiltonians for VQE.

    Parameters
    ----------
    geometry : list of (str, tuple)
        Atomic symbols and Cartesian coordinates in Angstroms.
        Example: [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
    basis : str
        Gaussian basis set name. Default: "sto-3g".
    charge : int
        Net molecular charge. Default: 0.
    spin : int
        Number of unpaired electrons (2S). Default: 0.
    n_active_orbitals : int or None
        Number of active orbitals for active space reduction.
        None means use all orbitals.
    n_active_electrons : int or None
        Number of active electrons. None means use all electrons.

    Attributes
    ----------
    nuclear_repulsion : float
        Nuclear repulsion energy in Hartrees.
    hf_energy : float
        Hartree-Fock total energy in Hartrees.
    n_orbitals : int
        Number of molecular orbitals (or active orbitals).
    n_electrons : int
        Number of electrons (or active electrons).
    n_qubits : int
        Number of qubits needed (= n_orbitals for minimal mapping).
    """

    def __init__(self, geometry: List[Tuple[str, Tuple[float, float, float]]],
                 basis: str = "sto-3g",
                 charge: int = 0,
                 spin: int = 0,
                 n_active_orbitals: Optional[int] = None,
                 n_active_electrons: Optional[int] = None):

        self._geometry = geometry
        self._basis = basis
        self._charge = charge
        self._spin = spin
        self._n_active_orbitals = n_active_orbitals
        self._n_active_electrons = n_active_electrons

        # Computed quantities (populated by _run_pyscf)
        self._nuclear_repulsion: Optional[float] = None
        self._hf_energy: Optional[float] = None
        self._one_body: Optional[np.ndarray] = None
        self._two_body: Optional[np.ndarray] = None
        self._n_orbitals: Optional[int] = None
        self._n_electrons: Optional[int] = None
        self._mo_coeff: Optional[np.ndarray] = None
        self._mo_energies: Optional[np.ndarray] = None

        # Run PySCF immediately
        self._run_pyscf()

    def _run_pyscf(self) -> None:
        """Execute PySCF Hartree-Fock and extract integrals."""
        try:
            from pyscf import gto, scf, ao2mo
        except ImportError:
            raise ImportError(
                "PySCF is required for molecular Hamiltonian generation.\n"
                "Install it with: pip install pyscf\n"
                "Note: PySCF supports Linux and macOS. On Windows, use WSL."
            )

        # Build molecule
        atom_str = "; ".join(
            f"{sym} {x} {y} {z}"
            for sym, (x, y, z) in self._geometry
        )
        mol = gto.M(
            atom=atom_str,
            basis=self._basis,
            charge=self._charge,
            spin=self._spin,
            unit="Angstrom",
            verbose=0,
        )

        # Run RHF
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        self._hf_energy = float(mf.kernel())
        self._nuclear_repulsion = float(mol.energy_nuc())
        self._mo_coeff = mf.mo_coeff
        self._mo_energies = mf.mo_energy

        n_orbitals_total = mol.nao_nr()
        n_electrons_total = mol.nelectron

        # Active space selection
        if self._n_active_orbitals is not None:
            n_active = self._n_active_orbitals
            n_active_e = self._n_active_electrons or n_electrons_total
        else:
            n_active = n_orbitals_total
            n_active_e = n_electrons_total

        self._n_orbitals = n_active
        self._n_electrons = n_active_e

        # Get integrals in MO basis
        # One-electron integrals: h_pq = <p|h|q>
        h1 = self._mo_coeff.T @ mf.get_hcore() @ self._mo_coeff

        # Two-electron integrals: (pq|rs) in chemist notation
        eri_4d = ao2mo.full(mol, self._mo_coeff, compact=False)
        eri_4d = eri_4d.reshape(n_orbitals_total, n_orbitals_total,
                                n_orbitals_total, n_orbitals_total)

        # Active space reduction
        if self._n_active_orbitals is not None:
            n_frozen = (n_electrons_total - n_active_e) // 2
            active_range = slice(n_frozen, n_frozen + n_active)

            # Frozen core contribution
            frozen_energy = 0.0
            for i in range(n_frozen):
                frozen_energy += 2 * h1[i, i]
                for j in range(n_frozen):
                    frozen_energy += 2 * eri_4d[i, i, j, j] - eri_4d[i, j, j, i]

            self._nuclear_repulsion += frozen_energy

            # Effective one-body integrals in active space
            h1_active = h1[active_range, active_range].copy()
            for i in range(n_active):
                for j in range(n_active):
                    ai, aj = n_frozen + i, n_frozen + j
                    for k in range(n_frozen):
                        h1_active[i, j] += (
                            2 * eri_4d[ai, aj, k, k] - eri_4d[ai, k, k, aj]
                        )

            self._one_body = h1_active
            self._two_body = eri_4d[active_range, active_range,
                                     active_range, active_range]
        else:
            self._one_body = h1
            self._two_body = eri_4d

    @property
    def nuclear_repulsion(self) -> float:
        """Nuclear repulsion energy (Hartrees)."""
        return self._nuclear_repulsion

    @property
    def hf_energy(self) -> float:
        """Hartree-Fock total energy (Hartrees)."""
        return self._hf_energy

    @property
    def n_orbitals(self) -> int:
        """Number of (active) molecular orbitals."""
        return self._n_orbitals

    @property
    def n_electrons(self) -> int:
        """Number of (active) electrons."""
        return self._n_electrons

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for simulation (2 × n_orbitals for spin)."""
        return 2 * self._n_orbitals

    @property
    def one_body_integrals(self) -> np.ndarray:
        """One-electron integrals h_{pq} in MO basis."""
        return self._one_body.copy()

    @property
    def two_body_integrals(self) -> np.ndarray:
        """Two-electron integrals (pq|rs) in MO basis, chemist notation."""
        return self._two_body.copy()

    @property
    def mo_energies(self) -> np.ndarray:
        """Molecular orbital energies from HF."""
        return self._mo_energies.copy()

    def hamiltonian(self, threshold: float = 1e-10):
        """
        Build qubit Hamiltonian via Jordan-Wigner transformation.

        Converts spatial molecular orbital integrals to spin-orbital
        integrals, then applies Jordan-Wigner mapping.

        Returns
        -------
        Hamiltonian
            Pauli-string Hamiltonian ready for VQE.
        """
        from tiny_qpu.gradients import Hamiltonian

        # Convert spatial orbitals to spin-orbital integrals
        n_spatial = self._n_orbitals
        n_spin = 2 * n_spatial
        h1_spin = np.zeros((n_spin, n_spin))
        h2_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))

        for p in range(n_spatial):
            for q in range(n_spatial):
                # alpha-alpha and beta-beta
                h1_spin[2 * p, 2 * q] = self._one_body[p, q]
                h1_spin[2 * p + 1, 2 * q + 1] = self._one_body[p, q]
                for r in range(n_spatial):
                    for s in range(n_spatial):
                        val = self._two_body[p, q, r, s]
                        # All spin combinations: αα-αα, ββ-ββ, αα-ββ, ββ-αα
                        h2_spin[2*p, 2*q, 2*r, 2*s] = val
                        h2_spin[2*p+1, 2*q+1, 2*r+1, 2*s+1] = val
                        h2_spin[2*p, 2*q, 2*r+1, 2*s+1] = val
                        h2_spin[2*p+1, 2*q+1, 2*r, 2*s] = val

        terms = jordan_wigner(
            h1_spin, h2_spin,
            nuclear_repulsion=self._nuclear_repulsion,
            threshold=threshold,
        )

        if not terms:
            terms = {"I" * n_spin: self._nuclear_repulsion}

        return Hamiltonian(terms)

    def fci_energy(self) -> float:
        """
        Compute exact Full CI energy using PySCF.

        This is the exact ground state energy within the basis set,
        useful as a benchmark for VQE accuracy.

        Returns
        -------
        float
            FCI energy in Hartrees.
        """
        try:
            from pyscf import gto, scf, fci
        except ImportError:
            raise ImportError("PySCF required for FCI calculation")

        atom_str = "; ".join(
            f"{sym} {x} {y} {z}"
            for sym, (x, y, z) in self._geometry
        )
        mol = gto.M(
            atom=atom_str,
            basis=self._basis,
            charge=self._charge,
            spin=self._spin,
            unit="Angstrom",
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.kernel()

        cisolver = fci.FCI(mf)
        e_fci, _ = cisolver.kernel()
        return float(e_fci)

    def __repr__(self) -> str:
        atoms = [sym for sym, _ in self._geometry]
        formula = "".join(atoms)
        return (f"Molecule({formula}, basis={self._basis}, "
                f"qubits={self.n_qubits}, "
                f"E_HF={self.hf_energy:.6f} Ha)")

    def __str__(self) -> str:
        lines = [f"Molecule: {' '.join(sym for sym, _ in self._geometry)}"]
        lines.append(f"  Basis: {self._basis}")
        lines.append(f"  Charge: {self._charge}, Spin: {self._spin}")
        lines.append(f"  Orbitals: {self._n_orbitals}, Electrons: {self._n_electrons}")
        lines.append(f"  Qubits required: {self.n_qubits}")
        lines.append(f"  Nuclear repulsion: {self._nuclear_repulsion:.6f} Ha")
        lines.append(f"  HF energy: {self._hf_energy:.6f} Ha")
        return "\n".join(lines)


# ─── Convenience constructors ────────────────────────────────────────────

def hydrogen(bond_length: float = 0.74, basis: str = "sto-3g") -> Molecule:
    """
    H₂ molecule at given bond length.

    Parameters
    ----------
    bond_length : float
        H-H distance in Angstroms. Default: 0.74 (equilibrium).
    basis : str
        Basis set. Default: "sto-3g".
    """
    return Molecule(
        [("H", (0, 0, 0)), ("H", (0, 0, bond_length))],
        basis=basis,
    )


def lithium_hydride(bond_length: float = 1.6, basis: str = "sto-3g",
                    n_active_orbitals: int = 2,
                    n_active_electrons: int = 2) -> Molecule:
    """
    LiH molecule with active space reduction.

    Parameters
    ----------
    bond_length : float
        Li-H distance in Angstroms. Default: 1.6 (near equilibrium).
    basis : str
        Basis set. Default: "sto-3g".
    n_active_orbitals : int
        Active orbitals. Default: 2 (minimal for 2-qubit simulation).
    n_active_electrons : int
        Active electrons. Default: 2.
    """
    return Molecule(
        [("Li", (0, 0, 0)), ("H", (0, 0, bond_length))],
        basis=basis,
        n_active_orbitals=n_active_orbitals,
        n_active_electrons=n_active_electrons,
    )


def water(basis: str = "sto-3g",
          n_active_orbitals: Optional[int] = 4,
          n_active_electrons: Optional[int] = 4) -> Molecule:
    """
    H₂O molecule at experimental geometry.

    Parameters
    ----------
    basis : str
        Basis set. Default: "sto-3g".
    n_active_orbitals : int or None
        Active orbitals. Default: 4 (manageable qubit count).
    n_active_electrons : int or None
        Active electrons. Default: 4.
    """
    # Experimental geometry (Angstroms)
    return Molecule(
        [("O", (0.0, 0.0, 0.1173)),
         ("H", (0.0, 0.7572, -0.4692)),
         ("H", (0.0, -0.7572, -0.4692))],
        basis=basis,
        n_active_orbitals=n_active_orbitals,
        n_active_electrons=n_active_electrons,
    )


def bond_dissociation_curve(atom1: str, atom2: str,
                            distances: Optional[List[float]] = None,
                            basis: str = "sto-3g",
                            **kwargs) -> List[Tuple[float, 'Molecule']]:
    """
    Compute molecular data at multiple bond lengths.

    Parameters
    ----------
    atom1, atom2 : str
        Atomic symbols.
    distances : list of float or None
        Bond lengths in Angstroms. Default: [0.5, 0.6, ..., 2.5].
    basis : str
        Basis set.

    Returns
    -------
    list of (distance, Molecule)
        Molecule objects at each bond length.
    """
    if distances is None:
        distances = [0.5 + 0.1 * i for i in range(21)]

    results = []
    for d in distances:
        mol = Molecule(
            [(atom1, (0, 0, 0)), (atom2, (0, 0, d))],
            basis=basis,
            **kwargs,
        )
        results.append((d, mol))
    return results
