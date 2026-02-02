import sys
sys.path.insert(0, 'src')
from tiny_qpu.benchmark.animate import QuantumAnimator
from tiny_qpu.benchmark.visualize import BenchmarkPlotter

# Animated GIFs
anim = QuantumAnimator(output_dir='output', fps=15)
print("=" * 50)
print(" Generating Animated GIFs")
print("=" * 50)

print("\n1/4 Bloch Sphere (qubit gate rotations)...")
anim.bloch_sphere_gates()

print("\n2/4 VQE Optimization (finding ground state)...")
anim.vqe_optimization()

print("\n3/4 H2 Potential Energy Surface scan...")
anim.pes_scan('H2')

print("\n4/4 Noise Degradation...")
anim.noise_degradation()

# Static plots too
print("\n" + "=" * 50)
print(" Generating Static Plots")
print("=" * 50)

plotter = BenchmarkPlotter(output_dir='output', dark_mode=True)

print("\n5 PES curves (all molecules)...")
plotter.potential_energy_surfaces()

print("\n6 Noise analysis (H2)...")
plotter.noise_analysis('H2', noise_levels=[0.0, 0.01, 0.05, 0.10])

print("\n7 Molecule overview...")
plotter.molecule_overview()

print("\n" + "=" * 50)
print(" ALL DONE!")
print("=" * 50)

import os
for f in sorted(os.listdir('output')):
    size = os.path.getsize(os.path.join('output', f)) // 1024
    print(f"  {f:30s} {size:>6d} KB")
