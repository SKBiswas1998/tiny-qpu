"""
Quantum Chemistry Benchmark Framework
========================================
Run VQE benchmarks across molecules, ansatze, optimizers, and noise levels.
Export publication-ready results.

Usage:
    from tiny_qpu.benchmark import ChemistryBenchmark
    
    bench = ChemistryBenchmark()
    results = bench.run_all()
    bench.export_csv("benchmark_results.csv")
    bench.summary()
"""
import numpy as np
import time
import json
import csv
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from .molecules import MoleculeLibrary, MolecularData, PauliTerm
from ..core import Circuit
from ..core import gates
from ..core.statevector import StateVector


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    molecule: str
    bond_length: float
    n_qubits: int
    ansatz: str
    optimizer: str
    depth: int
    noise_model: str
    noise_param: float
    
    # Results
    vqe_energy: float
    exact_energy: float
    error_ha: float         # Absolute error in Hartree
    error_mha: float        # Error in milli-Hartree
    error_kcal: float       # Error in kcal/mol
    chemical_accuracy: bool # < 1.6 mHa (1 kcal/mol)
    
    # Performance
    iterations: int
    time_seconds: float
    n_function_evals: int
    
    # Noise analysis
    fidelity: float = 1.0   # State fidelity vs ideal
    
    def __str__(self):
        acc = "✓" if self.chemical_accuracy else "✗"
        return (f"{self.molecule:5s} R={self.bond_length:.3f}  "
                f"E={self.vqe_energy:.6f}  err={self.error_mha:.2f} mHa  "
                f"[{acc}]  {self.time_seconds:.2f}s  noise={self.noise_model}")


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    @property
    def chemical_accuracy_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.chemical_accuracy for r in self.results) / len(self.results)
    
    def filter(self, **kwargs) -> 'BenchmarkSuite':
        """Filter results by field values."""
        filtered = []
        for r in self.results:
            match = all(getattr(r, k) == v for k, v in kwargs.items())
            if match:
                filtered.append(r)
        suite = BenchmarkSuite(results=filtered)
        return suite
    
    def summary(self) -> str:
        lines = [
            "=" * 78,
            "  Quantum Chemistry Benchmark Results",
            "=" * 78,
            f"  Total runs: {len(self.results)}",
            f"  Chemical accuracy rate: {self.chemical_accuracy_rate:.1%}",
            "",
            f"  {'Molecule':8s} {'R(A)':>6s} {'Ansatz':>10s} {'Noise':>10s} "
            f"{'VQE(Ha)':>12s} {'Error(mHa)':>10s} {'Acc':>4s} {'Time':>6s}",
            "-" * 78,
        ]
        
        for r in self.results:
            acc = "✓" if r.chemical_accuracy else "✗"
            lines.append(
                f"  {r.molecule:8s} {r.bond_length:>6.3f} {r.ansatz:>10s} "
                f"{r.noise_model:>10s} {r.vqe_energy:>12.6f} "
                f"{r.error_mha:>10.2f} {acc:>4s} {r.time_seconds:>5.2f}s"
            )
        
        lines.append("=" * 78)
        return '\n'.join(lines)


# =============================================================================
# VQE Engine (generalized from apps/vqe.py)
# =============================================================================

class VQEBenchmark:
    """
    VQE solver for benchmark molecular Hamiltonians.
    Supports multiple ansatze and noise-aware optimization.
    """
    
    def __init__(self, mol: MolecularData, ansatz: str = 'hardware_efficient',
                 depth: int = 2, optimizer: str = 'COBYLA'):
        self.mol = mol
        self.ansatz_type = ansatz
        self.depth = depth
        self.optimizer = optimizer
        self._eval_count = 0
    
    def _build_ansatz_params(self) -> int:
        """Return number of parameters for the ansatz."""
        n = self.mol.n_qubits
        if self.ansatz_type == 'hardware_efficient':
            return n * 2 * self.depth  # Ry + Rz per qubit per layer
        elif self.ansatz_type == 'ry_linear':
            return n * self.depth
        elif self.ansatz_type == 'uccsd_inspired':
            return n * 3 * self.depth
        return n * 2 * self.depth
    
    def _build_state(self, params: np.ndarray) -> np.ndarray:
        """Build trial state from ansatz parameters."""
        n = self.mol.n_qubits
        state = StateVector(n)
        
        if self.ansatz_type == 'hardware_efficient':
            idx = 0
            for layer in range(self.depth):
                for q in range(n):
                    state.apply_single_gate(gates.Ry(params[idx]), q)
                    idx += 1
                for q in range(n):
                    state.apply_single_gate(gates.Rz(params[idx]), q)
                    idx += 1
                # Entangling layer
                for q in range(n - 1):
                    state.apply_two_qubit_gate(gates.CNOT, q, q + 1)
        
        elif self.ansatz_type == 'ry_linear':
            idx = 0
            for layer in range(self.depth):
                for q in range(n):
                    state.apply_single_gate(gates.Ry(params[idx]), q)
                    idx += 1
                for q in range(n - 1):
                    state.apply_two_qubit_gate(gates.CNOT, q, q + 1)
        
        elif self.ansatz_type == 'uccsd_inspired':
            idx = 0
            for layer in range(self.depth):
                for q in range(n):
                    state.apply_single_gate(gates.Rx(params[idx]), q)
                    idx += 1
                for q in range(n):
                    state.apply_single_gate(gates.Ry(params[idx]), q)
                    idx += 1
                for q in range(n):
                    state.apply_single_gate(gates.Rz(params[idx]), q)
                    idx += 1
                for q in range(n - 1):
                    state.apply_two_qubit_gate(gates.CNOT, q, q + 1)
                if n > 2:
                    state.apply_two_qubit_gate(gates.CNOT, n-1, 0)
        
        return state._data
    
    def _compute_energy(self, params: np.ndarray) -> float:
        """Compute <psi|H|psi> for given parameters."""
        self._eval_count += 1
        state = self._build_state(params)
        H = self.mol.to_matrix()
        energy = np.real(state.conj() @ H @ state)
        return float(energy)
    
    def _compute_energy_noisy(self, params: np.ndarray, 
                               noise_channel, noise_param: float) -> float:
        """Compute energy with noise using density matrix."""
        self._eval_count += 1
        from ..noise import DensityMatrix
        
        n = self.mol.n_qubits
        dm = DensityMatrix(n)
        
        if self.ansatz_type == 'hardware_efficient':
            idx = 0
            for layer in range(self.depth):
                for q in range(n):
                    dm.apply_gate(gates.Ry(params[idx]), [q])
                    if noise_channel:
                        dm.apply_channel(noise_channel, [q])
                    idx += 1
                for q in range(n):
                    dm.apply_gate(gates.Rz(params[idx]), [q])
                    if noise_channel:
                        dm.apply_channel(noise_channel, [q])
                    idx += 1
                for q in range(n - 1):
                    dm.apply_gate(gates.CNOT, [q, q + 1])
                    if noise_channel:
                        for qq in [q, q+1]:
                            dm.apply_channel(noise_channel, [qq])
        else:
            # Fallback: compute state, convert to density matrix
            state = self._build_state(params)
            dm = DensityMatrix.from_statevector(state)
        
        H = self.mol.to_matrix()
        energy = np.real(np.trace(dm.matrix @ H))
        return float(energy)
    
    def run(self, maxiter: int = 200, seed: Optional[int] = None,
            noise_channel=None, noise_param: float = 0.0) -> dict:
        """Run VQE optimization."""
        from scipy.optimize import minimize
        
        if seed is not None:
            np.random.seed(seed)
        
        n_params = self._build_ansatz_params()
        self._eval_count = 0
        
        if noise_channel is not None and noise_param > 0:
            obj_func = lambda p: self._compute_energy_noisy(p, noise_channel, noise_param)
        else:
            obj_func = self._compute_energy
        
        # Multiple restarts to avoid local minima
        n_restarts = 5 if noise_channel is None else 2
        best_result = None
        best_energy = float('inf')
        
        start = time.time()
        for restart in range(n_restarts):
            # Smart initialization: small angles near zero
            x0 = np.random.uniform(-0.5, 0.5, n_params)
            
            try:
                result = minimize(obj_func, x0, method=self.optimizer,
                                 options={'maxiter': maxiter})
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_result = result
            except Exception:
                continue
        
        if best_result is None:
            best_result = type('obj', (object,), {'fun': 0, 'x': np.zeros(n_params), 'nit': 0, 'success': False})()
        
        result = best_result
        elapsed = time.time() - start
        
        # Compute ideal state fidelity against Pauli ground state
        ideal_state = self._build_state(result.x)
        pauli_energy, pauli_ground = self.mol.exact_diag()
        fidelity = float(np.abs(np.dot(ideal_state.conj(), pauli_ground)) ** 2)
        
        return {
            'energy': result.fun,
            'exact_energy': pauli_energy,
            'params': result.x,
            'iterations': getattr(result, 'nit', maxiter),
            'n_evals': self._eval_count,
            'time': elapsed,
            'fidelity': fidelity,
            'converged': result.success if hasattr(result, 'success') else True,
        }


# =============================================================================
# Main Benchmark Runner
# =============================================================================

class ChemistryBenchmark:
    """
    Run comprehensive VQE benchmarks across molecules and conditions.
    
    Usage:
        bench = ChemistryBenchmark()
        
        # Quick benchmark
        suite = bench.run_quick()
        
        # Full benchmark
        suite = bench.run_full()
        
        # Custom
        suite = bench.run(
            molecules=['H2', 'HeH+'],
            bond_lengths={'H2': [0.5, 0.735, 1.0, 1.5]},
            ansatze=['hardware_efficient', 'ry_linear'],
            noise_levels=[0.0, 0.01, 0.05],
        )
        
        # Export
        bench.export_csv(suite, "results.csv")
        bench.export_json(suite, "results.json")
    """
    
    def __init__(self, seed: int = 42, maxiter: int = 200):
        self.seed = seed
        self.maxiter = maxiter
    
    def run(self, molecules: Optional[List[str]] = None,
            bond_lengths: Optional[Dict[str, List[float]]] = None,
            ansatze: Optional[List[str]] = None,
            depths: Optional[List[int]] = None,
            noise_levels: Optional[List[float]] = None,
            verbose: bool = True) -> BenchmarkSuite:
        """Run benchmark with custom configuration."""
        
        if molecules is None:
            molecules = ['H2', 'HeH+']
        if ansatze is None:
            ansatze = ['hardware_efficient']
        if depths is None:
            depths = [2]
        if noise_levels is None:
            noise_levels = [0.0]
        
        default_bonds = {
            'H2': [0.735],
            'HeH+': [0.90],
            'LiH': [1.546],
            'H4': [1.00],
        }
        if bond_lengths is None:
            bond_lengths = {m: default_bonds.get(m, [None]) for m in molecules}
        
        suite = BenchmarkSuite(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        total = (len(molecules) * sum(len(bond_lengths.get(m, [None])) for m in molecules) 
                 * len(ansatze) * len(depths) * len(noise_levels))
        
        if verbose:
            print(f"Running {total} benchmark configurations...")
            print(f"{'#':>3s} {'Molecule':>8s} {'R':>6s} {'Ansatz':>12s} "
                  f"{'d':>2s} {'Noise':>6s} {'Energy':>12s} {'Err(mHa)':>9s} "
                  f"{'Acc':>4s} {'Time':>6s}")
            print("-" * 78)
        
        run_idx = 0
        for mol_name in molecules:
            for R in bond_lengths.get(mol_name, [None]):
                mol = MoleculeLibrary.get(mol_name, R)
                
                for ansatz in ansatze:
                    for depth in depths:
                        for noise_p in noise_levels:
                            run_idx += 1
                            
                            # Setup noise
                            noise_ch = None
                            noise_label = "clean"
                            if noise_p > 0:
                                from ..noise import depolarizing
                                noise_ch = depolarizing(noise_p)
                                noise_label = f"depol_{noise_p}"
                            
                            # Run VQE
                            vqe = VQEBenchmark(mol, ansatz, depth, 'COBYLA')
                            result = vqe.run(
                                maxiter=self.maxiter,
                                seed=self.seed,
                                noise_channel=noise_ch,
                                noise_param=noise_p,
                            )
                            
                            error = abs(result['energy'] - result['exact_energy'])
                            error_mha = error * 1000
                            error_kcal = error * 627.509  # Ha to kcal/mol
                            
                            br = BenchmarkResult(
                                molecule=mol_name,
                                bond_length=mol.bond_length,
                                n_qubits=mol.n_qubits,
                                ansatz=ansatz,
                                optimizer='COBYLA',
                                depth=depth,
                                noise_model=noise_label,
                                noise_param=noise_p,
                                vqe_energy=result['energy'],
                                exact_energy=result['exact_energy'],
                                error_ha=error,
                                error_mha=error_mha,
                                error_kcal=error_kcal,
                                chemical_accuracy=error_kcal < 1.0,
                                iterations=result['iterations'],
                                time_seconds=result['time'],
                                n_function_evals=result['n_evals'],
                                fidelity=result['fidelity'],
                            )
                            
                            suite.add(br)
                            
                            if verbose:
                                acc = "✓" if br.chemical_accuracy else "✗"
                                print(f"{run_idx:>3d} {mol_name:>8s} "
                                      f"{mol.bond_length:>6.3f} {ansatz:>12s} "
                                      f"{depth:>2d} {noise_p:>6.3f} "
                                      f"{result['energy']:>12.6f} "
                                      f"{error_mha:>9.2f} {acc:>4s} "
                                      f"{result['time']:>5.2f}s")
        
        if verbose:
            print(f"\nChemical accuracy rate: {suite.chemical_accuracy_rate:.1%}")
        
        return suite
    
    def run_quick(self, verbose: bool = True) -> BenchmarkSuite:
        """Quick benchmark: H2 and HeH+ at equilibrium, clean only."""
        return self.run(
            molecules=['H2', 'HeH+'],
            ansatze=['hardware_efficient'],
            depths=[2],
            noise_levels=[0.0],
            verbose=verbose,
        )
    
    def run_full(self, verbose: bool = True) -> BenchmarkSuite:
        """Full benchmark: all molecules, multiple conditions."""
        return self.run(
            molecules=['H2', 'HeH+', 'LiH', 'H4'],
            bond_lengths={
                'H2': [0.5, 0.735, 1.0, 1.5, 2.0],
                'HeH+': [0.75, 0.90, 1.0, 1.5],
                'LiH': [1.0, 1.546, 2.0],
                'H4': [0.75, 1.0, 1.5],
            },
            ansatze=['hardware_efficient', 'ry_linear', 'uccsd_inspired'],
            depths=[2, 3],
            noise_levels=[0.0, 0.005, 0.01, 0.05],
            verbose=verbose,
        )
    
    def run_noise_sweep(self, molecule: str = 'H2',
                        bond_length: float = 0.735,
                        noise_levels: Optional[List[float]] = None,
                        verbose: bool = True) -> BenchmarkSuite:
        """Sweep noise levels for a single molecule."""
        if noise_levels is None:
            noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
        
        return self.run(
            molecules=[molecule],
            bond_lengths={molecule: [bond_length]},
            ansatze=['hardware_efficient'],
            depths=[2],
            noise_levels=noise_levels,
            verbose=verbose,
        )
    
    @staticmethod
    def export_csv(suite: BenchmarkSuite, filepath: str):
        """Export results to CSV for papers/analysis."""
        fields = [
            'molecule', 'bond_length', 'n_qubits', 'ansatz', 'optimizer',
            'depth', 'noise_model', 'noise_param', 'vqe_energy', 'exact_energy',
            'error_ha', 'error_mha', 'error_kcal', 'chemical_accuracy',
            'iterations', 'time_seconds', 'n_function_evals', 'fidelity'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in suite.results:
                writer.writerow(asdict(r))
        
        print(f"Exported {len(suite.results)} results to {filepath}")
    
    @staticmethod
    def export_json(suite: BenchmarkSuite, filepath: str):
        """Export results to JSON."""
        data = {
            'timestamp': suite.timestamp,
            'n_results': len(suite.results),
            'chemical_accuracy_rate': suite.chemical_accuracy_rate,
            'results': [asdict(r) for r in suite.results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Exported {len(suite.results)} results to {filepath}")

