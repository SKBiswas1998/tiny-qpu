"""
Quantum Benchmark Visualizations
===================================
Publication-quality plots for molecular benchmarks.

Usage:
    from tiny_qpu.benchmark.visualize import BenchmarkPlotter

    plotter = BenchmarkPlotter()
    plotter.potential_energy_surfaces()
    plotter.noise_analysis('H2')
    plotter.ansatz_comparison()
    plotter.full_report()
"""
import numpy as np
import os
from typing import List, Optional, Dict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class BenchmarkPlotter:
    """Generate publication-quality benchmark plots."""

    # Professional color scheme
    COLORS = {
        'H2': '#2196F3',       # Blue
        'HeH+': '#FF5722',     # Deep Orange
        'LiH': '#4CAF50',      # Green
        'H4': '#9C27B0',       # Purple
        'clean': '#2196F3',
        'noisy': '#FF5722',
        'hw_eff': '#2196F3',
        'ry_lin': '#4CAF50',
        'uccsd': '#FF9800',
    }

    STYLE = {
        'figure.facecolor': '#0D1117',
        'axes.facecolor': '#161B22',
        'axes.edgecolor': '#30363D',
        'axes.labelcolor': '#C9D1D9',
        'text.color': '#C9D1D9',
        'xtick.color': '#8B949E',
        'ytick.color': '#8B949E',
        'grid.color': '#21262D',
        'grid.alpha': 0.8,
        'font.family': 'monospace',
        'font.size': 11,
    }

    def __init__(self, output_dir: str = '.', dark_mode: bool = True):
        if not HAS_MPL:
            raise ImportError("matplotlib required: pip install matplotlib")
        self.output_dir = output_dir
        self.dark_mode = dark_mode
        if dark_mode:
            plt.rcParams.update(self.STYLE)

    def potential_energy_surfaces(self, molecules: Optional[List[str]] = None,
                                   save: bool = True) -> str:
        """Plot potential energy surfaces for all molecules."""
        from .molecules import MoleculeLibrary

        if molecules is None:
            molecules = ['H2', 'HeH+', 'LiH', 'H4']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Potential Energy Surfaces — tiny-qpu',
                     fontsize=16, fontweight='bold', y=0.98)

        for idx, mol_name in enumerate(molecules):
            ax = axes[idx // 2][idx % 2]
            surface = MoleculeLibrary.potential_energy_surface(mol_name, n_points=25)
            Rs = [r for r, _ in surface]
            Es = [e for _, e in surface]

            color = self.COLORS.get(mol_name, '#2196F3')

            ax.plot(Rs, Es, '-', color=color, linewidth=2.5, label='Exact (FCI)')
            ax.fill_between(Rs, Es, min(Es) - 0.05, alpha=0.1, color=color)

            # Mark equilibrium
            min_idx = np.argmin(Es)
            ax.plot(Rs[min_idx], Es[min_idx], 'o', color='#FFD700',
                    markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate(f'  R={Rs[min_idx]:.2f} Å\n  E={Es[min_idx]:.4f} Ha',
                       (Rs[min_idx], Es[min_idx]),
                       fontsize=9, color='#FFD700')

            mol = MoleculeLibrary.get(mol_name)
            ax.set_title(f'{mol.formula}  ({mol.n_qubits} qubits, {mol.n_terms} Pauli terms)',
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Bond Length (Å)')
            ax.set_ylabel('Energy (Hartree)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'pes_curves.png')
        if save:
            fig.savefig(path, dpi=180, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            plt.close()
            print(f"Saved: {path}")
        return path

    def noise_analysis(self, molecule: str = 'H2',
                        noise_levels: Optional[List[float]] = None,
                        save: bool = True) -> str:
        """Plot noise degradation curve for a molecule."""
        from . import ChemistryBenchmark

        if noise_levels is None:
            noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

        bench = ChemistryBenchmark(seed=42, maxiter=200)
        suite = bench.run_noise_sweep(molecule, noise_levels=noise_levels, verbose=False)

        errors = [r.error_mha for r in suite.results]
        energies = [r.vqe_energy for r in suite.results]
        exact = suite.results[0].exact_energy

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(f'Noise Analysis — {molecule}', fontsize=16, fontweight='bold')

        color = self.COLORS.get(molecule, '#2196F3')

        # Error vs noise
        ax1.plot(noise_levels, errors, 'o-', color=color, linewidth=2.5,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax1.axhline(y=1.594, color='#FFD700', linestyle='--', alpha=0.7,
                    label='Chemical accuracy (1 kcal/mol)')
        ax1.fill_between(noise_levels, 0, 1.594, alpha=0.08, color='#FFD700')
        ax1.set_xlabel('Depolarizing Noise (p)')
        ax1.set_ylabel('Error (milli-Hartree)')
        ax1.set_title('Error vs Noise Level')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Energy vs noise
        ax2.plot(noise_levels, energies, 's-', color='#FF5722', linewidth=2.5,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax2.axhline(y=exact, color='#4CAF50', linestyle='--', alpha=0.7,
                    label=f'Exact = {exact:.4f} Ha')
        ax2.set_xlabel('Depolarizing Noise (p)')
        ax2.set_ylabel('VQE Energy (Hartree)')
        ax2.set_title('Energy Degradation')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f'noise_{molecule.replace("+","p")}.png')
        if save:
            fig.savefig(path, dpi=180, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            plt.close()
            print(f"Saved: {path}")
        return path

    def ansatz_comparison(self, molecule: str = 'H2',
                           save: bool = True) -> str:
        """Compare ansatz performance."""
        from . import ChemistryBenchmark

        bench = ChemistryBenchmark(seed=42, maxiter=300)
        suite = bench.run(
            molecules=[molecule],
            ansatze=['hardware_efficient', 'ry_linear', 'uccsd_inspired'],
            depths=[1, 2, 3, 4],
            noise_levels=[0.0],
            verbose=False,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(f'Ansatz Comparison — {molecule}', fontsize=16, fontweight='bold')

        ansatz_map = {
            'hardware_efficient': ('HW-Efficient', '#2196F3', 'o'),
            'ry_linear': ('Ry-Linear', '#4CAF50', 's'),
            'uccsd_inspired': ('UCCSD-Inspired', '#FF9800', 'D'),
        }

        for ansatz in ['hardware_efficient', 'ry_linear', 'uccsd_inspired']:
            results = [r for r in suite.results if r.ansatz == ansatz]
            if not results:
                continue
            depths = [r.depth for r in results]
            errors = [r.error_mha for r in results]
            times = [r.time_seconds for r in results]
            label, color, marker = ansatz_map[ansatz]

            ax1.plot(depths, errors, f'{marker}-', color=color, linewidth=2,
                    markersize=8, label=label, markeredgecolor='white', markeredgewidth=1.5)
            ax2.plot(depths, times, f'{marker}-', color=color, linewidth=2,
                    markersize=8, label=label, markeredgecolor='white', markeredgewidth=1.5)

        ax1.axhline(y=1.594, color='#FFD700', linestyle='--', alpha=0.7,
                    label='Chemical accuracy')
        ax1.set_xlabel('Ansatz Depth')
        ax1.set_ylabel('Error (milli-Hartree)')
        ax1.set_title('Accuracy vs Depth')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        ax2.set_xlabel('Ansatz Depth')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Runtime vs Depth')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f'ansatz_{molecule.replace("+","p")}.png')
        if save:
            fig.savefig(path, dpi=180, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            plt.close()
            print(f"Saved: {path}")
        return path

    def molecule_overview(self, save: bool = True) -> str:
        """Bar chart comparing all molecules."""
        from .molecules import MoleculeLibrary
        from . import ChemistryBenchmark

        bench = ChemistryBenchmark(seed=42, maxiter=300)
        suite = bench.run(
            molecules=['H2', 'HeH+', 'LiH', 'H4'],
            ansatze=['hardware_efficient'],
            depths=[2],
            noise_levels=[0.0],
            verbose=False,
        )

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Molecular Benchmark Overview', fontsize=16, fontweight='bold')

        names = [r.molecule for r in suite.results]
        errors = [r.error_mha for r in suite.results]
        times = [r.time_seconds for r in suite.results]
        qubits = [r.n_qubits for r in suite.results]
        colors = [self.COLORS.get(n, '#2196F3') for n in names]

        # Error bars
        bars1 = ax1.bar(names, errors, color=colors, edgecolor='white', linewidth=0.5)
        ax1.axhline(y=1.594, color='#FFD700', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Error (mHa)')
        ax1.set_title('Accuracy')

        # Time bars
        bars2 = ax2.bar(names, times, color=colors, edgecolor='white', linewidth=0.5)
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Runtime')

        # Qubits
        bars3 = ax3.bar(names, qubits, color=colors, edgecolor='white', linewidth=0.5)
        ax3.set_ylabel('Qubits')
        ax3.set_title('Problem Size')

        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'molecule_overview.png')
        if save:
            fig.savefig(path, dpi=180, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            plt.close()
            print(f"Saved: {path}")
        return path

    def full_report(self, save: bool = True) -> List[str]:
        """Generate all plots."""
        paths = []
        print("Generating benchmark visualizations...")
        paths.append(self.potential_energy_surfaces(save=save))
        paths.append(self.noise_analysis('H2', save=save))
        paths.append(self.ansatz_comparison('H2', save=save))
        paths.append(self.molecule_overview(save=save))
        print(f"\nGenerated {len(paths)} plots")
        return paths
