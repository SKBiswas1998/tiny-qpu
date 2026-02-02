"""
Animated Visualizations (Matplotlib)
========================================
Animated GIFs for quantum concepts — no ffmpeg or LaTeX required.

Usage:
    from tiny_qpu.benchmark.animate import QuantumAnimator
    
    anim = QuantumAnimator(output_dir='.')
    anim.vqe_optimization()
    anim.bloch_sphere_gates()
    anim.pes_scan()
    anim.noise_degradation()
"""
import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import FancyArrowPatch, Circle
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import art3d
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class QuantumAnimator:
    """Generate animated GIFs for quantum computing concepts."""

    DARK_BG = '#0D1117'
    DARK_AXES = '#161B22'

    def __init__(self, output_dir: str = '.', fps: int = 20):
        if not HAS_MPL:
            raise ImportError("matplotlib required")
        self.output_dir = output_dir
        self.fps = fps
        plt.rcParams.update({
            'figure.facecolor': self.DARK_BG,
            'axes.facecolor': self.DARK_AXES,
            'axes.edgecolor': '#30363D',
            'axes.labelcolor': '#C9D1D9',
            'text.color': '#C9D1D9',
            'xtick.color': '#8B949E',
            'ytick.color': '#8B949E',
            'font.family': 'monospace',
        })

    def bloch_sphere_gates(self, save: bool = True) -> str:
        """Animate qubit state evolution on the Bloch sphere."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.DARK_BG)

        # Draw sphere wireframe
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Gate sequence: H, X, Z, T, Y applied to |0>
        # Bloch coordinates: (theta, phi) -> (sin(t)cos(p), sin(t)sin(p), cos(t))
        gates = [
            ('|0⟩', 0, 0),
            ('H|0⟩', np.pi/2, 0),
            ('XH|0⟩', np.pi/2, np.pi),
            ('ZXH|0⟩', np.pi/2, 0),
            ('T·ZXH|0⟩', np.pi/2, np.pi/4),
            ('Y·T·ZXH|0⟩', np.pi/2, np.pi + np.pi/4),
        ]

        # Interpolate between states
        n_interp = 30  # frames per gate
        all_states = []
        all_labels = []

        for i in range(len(gates) - 1):
            _, t1, p1 = gates[i]
            label, t2, p2 = gates[i + 1]
            for j in range(n_interp):
                frac = j / n_interp
                t = t1 + (t2 - t1) * frac
                p = p1 + (p2 - p1) * frac
                all_states.append((t, p))
                all_labels.append(label if j > n_interp // 2 else gates[i][0])

        # Add pause at end
        for _ in range(20):
            all_states.append(all_states[-1])
            all_labels.append(all_labels[-1])

        arrow_line = [None]
        title_text = [None]
        trail_x, trail_y, trail_z = [], [], []
        trail_line = [None]

        def init():
            ax.clear()
            ax.set_facecolor(self.DARK_BG)
            return []

        def update(frame):
            ax.clear()
            ax.set_facecolor(self.DARK_BG)

            # Sphere
            ax.plot_surface(x, y, z, alpha=0.04, color='#2196F3')
            ax.plot_wireframe(x, y, z, alpha=0.08, color='#2196F3',
                            linewidth=0.3, rstride=4, cstride=4)

            # Axes
            ax.plot([-1.3, 1.3], [0, 0], [0, 0], color='#444', linewidth=0.5)
            ax.plot([0, 0], [-1.3, 1.3], [0, 0], color='#444', linewidth=0.5)
            ax.plot([0, 0], [0, 0], [-1.3, 1.3], color='#444', linewidth=0.5)

            # Pole labels
            ax.text(0, 0, 1.4, '|0⟩', color='#4CAF50', fontsize=14, ha='center')
            ax.text(0, 0, -1.4, '|1⟩', color='#FF5722', fontsize=14, ha='center')
            ax.text(1.4, 0, 0, 'X', color='#666', fontsize=10, ha='center')
            ax.text(0, 1.4, 0, 'Y', color='#666', fontsize=10, ha='center')

            # State vector
            theta, phi = all_states[frame]
            sx = np.sin(theta) * np.cos(phi)
            sy = np.sin(theta) * np.sin(phi)
            sz = np.cos(theta)

            trail_x.append(sx)
            trail_y.append(sy)
            trail_z.append(sz)

            # Trail
            if len(trail_x) > 1:
                ax.plot(trail_x, trail_y, trail_z,
                       color='#FFD700', alpha=0.4, linewidth=1.5)

            # Arrow
            ax.quiver(0, 0, 0, sx, sy, sz,
                     color='#FFD700', arrow_length_ratio=0.12, linewidth=3)

            # Dot at tip
            ax.scatter([sx], [sy], [sz], color='#FFD700', s=80, zorder=10)

            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            ax.set_title(f'Bloch Sphere — {all_labels[frame]}',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_axis_off()
            ax.view_init(elev=25, azim=frame * 0.5 - 45)

            return []

        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=len(all_states), interval=1000 // self.fps, blit=False,
        )

        path = os.path.join(self.output_dir, 'bloch_sphere.gif')
        if save:
            anim.save(path, writer='pillow', fps=self.fps, dpi=100)
            plt.close()
            print(f"Saved: {path} ({os.path.getsize(path) // 1024} KB)")
        return path

    def vqe_optimization(self, save: bool = True) -> str:
        """Animate VQE optimizer finding the ground state."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Simple energy landscape with one clear global min
        theta_range = np.linspace(-3, 3, 300)

        def E(x):
            return -0.85 * np.exp(-x**2) + 0.15 * np.cos(3*x) * np.exp(-0.3*x**2)

        landscape = [E(t) for t in theta_range]
        exact_min = min(landscape)

        # Smooth trajectory from x=2.8 down to x=0
        xs = [2.8, 2.4, 2.0, 1.6, 1.2, 0.9, 0.65, 0.45, 0.3, 0.18,
              0.10, 0.05, 0.02, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ys = [E(x) for x in xs]

        n_frames = len(xs)

        def update(frame):
            ax1.clear()
            ax2.clear()

            idx = min(frame, n_frames - 1)

            # Left: energy landscape + optimizer dot
            ax1.plot(theta_range, landscape, color='#2196F3', linewidth=2.5)
            ax1.axhline(y=exact_min, color='#4CAF50', linestyle='--', alpha=0.6,
                       label=f'Exact: {exact_min:.3f} Ha')
            ax1.fill_between(theta_range, exact_min - 0.03, exact_min + 0.03,
                           alpha=0.1, color='#4CAF50')

            # Trail
            if idx > 0:
                ax1.plot(xs[:idx+1], ys[:idx+1], 'o-', color='#FFD700',
                        markersize=3, linewidth=1, alpha=0.5)
            # Current position
            ax1.plot(xs[idx], ys[idx], 'o', color='#FFD700', markersize=14,
                    markeredgecolor='white', markeredgewidth=2, zorder=10)

            ax1.set_xlabel(chr(952) + ' (parameter)')
            ax1.set_ylabel('E(' + chr(952) + ') (Hartree)')
            ax1.set_title('VQE Energy Landscape', fontweight='bold')
            ax1.legend(fontsize=10, loc='upper left')
            ax1.grid(True, alpha=0.2)
            ax1.set_xlim(-3.2, 3.2)
            ax1.set_ylim(-1.0, 0.3)

            # Right: convergence curve
            history = ys[:idx+1]
            ax2.plot(range(len(history)), history, '-', color='#FF5722', linewidth=2.5)
            ax2.plot(len(history)-1, history[-1], 'o', color='#FF5722', markersize=8,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax2.axhline(y=exact_min, color='#4CAF50', linestyle='--', alpha=0.6,
                       label='Exact')
            ax2.fill_between([0, n_frames], exact_min - 0.03, exact_min + 0.03,
                           alpha=0.1, color='#4CAF50')
            ax2.set_xlabel('VQE Iteration')
            ax2.set_ylabel('Energy (Hartree)')
            ax2.set_title(f'Convergence (iter {idx})', fontweight='bold')
            ax2.set_xlim(0, n_frames)
            ax2.set_ylim(-1.0, 0.3)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.2)

            error = abs(ys[idx] - exact_min) * 1000
            status = 'CONVERGED!' if error < 5 else f'Error: {error:.1f} mHa'
            color = '#4CAF50' if error < 5 else '#C9D1D9'
            fig.suptitle(f'VQE Optimization ' + chr(8212) + f' {status}',
                        fontsize=14, fontweight='bold', color=color)
            plt.tight_layout()
            return []

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=400, blit=False,
        )

        path = os.path.join(self.output_dir, 'vqe_optimization.gif')
        if save:
            anim.save(path, writer='pillow', fps=4, dpi=120)
            plt.close()
            print(f"Saved: {path} ({os.path.getsize(path) // 1024} KB)")
        return path

    def pes_scan(self, molecule: str = 'H2', save: bool = True) -> str:
        """Animate potential energy surface scan."""
        import sys
        sys.path.insert(0, 'src')
        from tiny_qpu.benchmark.molecules import MoleculeLibrary

        surface = MoleculeLibrary.potential_energy_surface(molecule, n_points=30)
        Rs = [r for r, _ in surface]
        Es = [e for _, e in surface]

        fig, ax = plt.subplots(figsize=(10, 6))

        mol = MoleculeLibrary.get(molecule)

        def update(frame):
            ax.clear()
            n = frame + 1

            # Plot up to current point
            ax.plot(Rs[:n], Es[:n], 'o-', color='#2196F3', linewidth=2.5,
                   markersize=6, markeredgecolor='white', markeredgewidth=1)

            # Fill under curve
            if n > 1:
                ax.fill_between(Rs[:n], Es[:n], min(Es) - 0.05,
                              alpha=0.1, color='#2196F3')

            # Current point highlight
            ax.plot(Rs[n-1], Es[n-1], 'o', color='#FFD700', markersize=14,
                   markeredgecolor='white', markeredgewidth=2, zorder=10)

            # Minimum so far
            min_e = min(Es[:n])
            min_r = Rs[Es[:n].index(min_e)]
            if n > 3:
                ax.annotate(f'E_min = {min_e:.4f} Ha\nR = {min_r:.2f} Å',
                           (min_r, min_e), fontsize=11, color='#FFD700',
                           xytext=(20, 20), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='#FFD700'))

            ax.set_xlim(Rs[0] - 0.1, Rs[-1] + 0.1)
            ax.set_ylim(min(Es) - 0.1, max(Es) + 0.1)
            ax.set_xlabel('Bond Length (Å)', fontsize=12)
            ax.set_ylabel('Energy (Hartree)', fontsize=12)
            ax.set_title(f'{mol.formula} Potential Energy Surface  '
                        f'({n}/{len(Rs)} points)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.2)

            return []

        total_frames = len(Rs) + 15  # extra frames at the end

        def update_with_pause(frame):
            return update(min(frame, len(Rs) - 1))

        anim = animation.FuncAnimation(
            fig, update_with_pause, frames=total_frames,
            interval=1000 // self.fps, blit=False,
        )

        path = os.path.join(self.output_dir, f'pes_{molecule.replace("+","p")}.gif')
        if save:
            anim.save(path, writer='pillow', fps=self.fps, dpi=120)
            plt.close()
            print(f"Saved: {path} ({os.path.getsize(path) // 1024} KB)")
        return path

    def noise_degradation(self, save: bool = True) -> str:
        """Animate noise increasing and destroying quantum state."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        noise_levels = np.linspace(0, 0.25, 50)

        # Pre-computed: error grows ~linearly with noise for small p
        # Bell state: fidelity = 1 - (8/5)p for depolarizing
        errors = [p * 1700 for p in noise_levels]  # approx mHa
        fidelities = [max(0, 1 - 1.6 * p) for p in noise_levels]

        def update(frame):
            ax1.clear()
            ax2.clear()
            n = frame + 1

            # Error plot
            ax1.plot(noise_levels[:n], errors[:n], 'o-', color='#FF5722',
                    linewidth=2.5, markersize=4)
            ax1.axhline(y=1.594, color='#FFD700', linestyle='--', alpha=0.7,
                       label='Chemical accuracy')
            ax1.fill_between([0, 0.25], 0, 1.594, alpha=0.05, color='#FFD700')
            ax1.set_xlim(0, 0.25)
            ax1.set_ylim(0, 450)
            ax1.set_xlabel('Depolarizing Noise (p)')
            ax1.set_ylabel('Error (milli-Hartree)')
            ax1.set_title('Error vs Noise', fontweight='bold')
            ax1.grid(True, alpha=0.2)
            ax1.legend(fontsize=10)

            # Fidelity plot
            ax2.plot(noise_levels[:n], fidelities[:n], 's-', color='#4CAF50',
                    linewidth=2.5, markersize=4)
            ax2.axhline(y=0.99, color='#FFD700', linestyle='--', alpha=0.7,
                       label='99% fidelity')
            ax2.set_xlim(0, 0.25)
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel('Depolarizing Noise (p)')
            ax2.set_ylabel('State Fidelity')
            ax2.set_title('Fidelity vs Noise', fontweight='bold')
            ax2.grid(True, alpha=0.2)
            ax2.legend(fontsize=10)

            p = noise_levels[min(frame, len(noise_levels)-1)]
            fig.suptitle(f'Noise Degradation — p = {p:.3f}',
                        fontsize=14, fontweight='bold')

            plt.tight_layout()
            return []

        anim = animation.FuncAnimation(
            fig, update, frames=len(noise_levels) + 10,
            interval=1000 // self.fps, blit=False,
        )

        path = os.path.join(self.output_dir, 'noise_degradation.gif')
        if save:
            anim.save(path, writer='pillow', fps=self.fps, dpi=120)
            plt.close()
            print(f"Saved: {path} ({os.path.getsize(path) // 1024} KB)")
        return path

    def all_animations(self) -> list:
        """Generate all animations."""
        paths = []
        print("Generating quantum animations...")
        paths.append(self.bloch_sphere_gates())
        paths.append(self.vqe_optimization())
        paths.append(self.pes_scan('H2'))
        paths.append(self.noise_degradation())
        print(f"\nGenerated {len(paths)} animations")
        return paths
