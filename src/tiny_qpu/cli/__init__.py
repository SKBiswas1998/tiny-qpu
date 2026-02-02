"""
Command-line interface for tiny-qpu.

Usage:
    tiny-qpu qrng --bytes 32
    tiny-qpu maxcut --random 6
    tiny-qpu bb84 --demo
    tiny-qpu vqe --molecule h2
    tiny-qpu run bell
"""
import argparse
import sys
import time


def cmd_run(args):
    """Run a quantum program."""
    from ..core import Circuit
    
    if args.file == 'bell':
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = qc.run(shots=args.shots)
        print("\nBell State Results:")
        for state, count in sorted(result.counts.items()):
            pct = 100 * count / args.shots
            bar = '█' * int(pct / 2)
            print(f"  |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}")
    elif args.file == 'ghz':
        qc = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
        result = qc.run(shots=args.shots)
        print("\nGHZ State Results:")
        for state, count in sorted(result.counts.items()):
            pct = 100 * count / args.shots
            bar = '█' * int(pct / 2)
            print(f"  |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print(f"Unknown program: {args.file}")
        print("Available: bell, ghz")


def cmd_qrng(args):
    """Generate quantum random numbers."""
    from ..apps import QRNG
    
    qrng = QRNG()
    
    if args.bits:
        bits = qrng.random_bits(args.bits)
        print(''.join(str(b) for b in bits))
    elif args.bytes:
        data = qrng.random_bytes(args.bytes)
        if args.hex:
            print(data.hex())
        else:
            sys.stdout.buffer.write(data)
    elif args.int:
        low, high = args.int
        print(qrng.random_int(low, high))
    elif args.float:
        print(qrng.random_float())
    elif args.uuid:
        print(qrng.random_uuid4())
    else:
        print(qrng.random_bytes(32).hex())


def cmd_maxcut(args):
    """Solve MaxCut using QAOA."""
    from ..apps import QAOA
    
    print(f"Solving MaxCut with QAOA (p={args.p})...")
    
    if args.random:
        qaoa = QAOA.random_graph(args.random, edge_prob=0.5, p=args.p)
        print(f"Random graph: {qaoa.num_qubits} nodes, {len(qaoa.edges)} edges")
    elif args.edges:
        edges = []
        for e in args.edges.split(','):
            i, j = map(int, e.split('-'))
            edges.append((i, j))
        qaoa = QAOA(edges, p=args.p)
    else:
        edges = [(0, 1), (1, 2), (2, 0)]
        qaoa = QAOA(edges, p=args.p)
        print("Using triangle graph")
    
    print(f"Edges: {qaoa.edges}")
    
    start = time.time()
    result = qaoa.optimize(shots=args.shots)
    elapsed = time.time() - start
    
    print(f"\nResult:")
    print(f"  Best partition: {result.bitstring}")
    print(f"  Cut value: {result.cut_value()}")
    print(f"  Time: {elapsed:.2f}s")
    
    set_0 = [i for i, b in enumerate(reversed(result.bitstring)) if b == '0']
    set_1 = [i for i, b in enumerate(reversed(result.bitstring)) if b == '1']
    print(f"  Set A: {set_0}")
    print(f"  Set B: {set_1}")


def cmd_bb84(args):
    """Run BB84 quantum key distribution."""
    from ..apps import BB84
    
    if args.demo:
        BB84.demo(key_length=args.key_length)
    else:
        bb84 = BB84(key_length=args.key_length)
        result = bb84.run(with_eavesdropper=args.eve)
        
        print(f"\nKey: {result.key.hex()}")
        print(f"Error rate: {result.error_rate:.2%}")
        
        if result.eavesdropper_detected:
            print("⚠️  WARNING: Possible eavesdropper detected!")


def cmd_vqe(args):
    """Run VQE molecular simulation."""
    from ..apps import VQE, MolecularHamiltonian
    
    print(f"VQE Molecular Simulation")
    print("=" * 50)
    
    if args.molecule == 'h2':
        print(f"Molecule: H₂ (Hydrogen)")
        print(f"Bond length: {args.bond_length} Å")
        
        h2 = MolecularHamiltonian.H2(args.bond_length)
        vqe = VQE(h2, depth=args.depth)
        
        print(f"Qubits: {h2.num_qubits}")
        print(f"Parameters: {vqe.num_params}")
        print(f"\nOptimizing...")
        
        start = time.time()
        result = vqe.run(maxiter=args.maxiter, seed=args.seed)
        elapsed = time.time() - start
        
        exact = h2.exact_ground_state()
        error = abs(result.energy - exact)
        
        print(f"\nResults:")
        print(f"  VQE Energy:   {result.energy:.6f} Ha")
        print(f"  Exact Energy: {exact:.6f} Ha")
        print(f"  Error:        {error:.6f} Ha")
        print(f"  Iterations:   {result.num_iterations}")
        print(f"  Time:         {elapsed:.2f}s")
        
        if error < 0.001:
            print("  ✓ Chemical accuracy achieved!")
    
    elif args.scan:
        print("Potential Energy Surface Scan")
        print("-" * 50)
        
        h2 = MolecularHamiltonian.H2(0.735)
        vqe = VQE(h2, depth=args.depth)
        
        bond_lengths = [0.5, 0.6, 0.735, 0.9, 1.0, 1.2, 1.5, 2.0]
        energies = vqe.potential_energy_surface(bond_lengths, maxiter=100)
        
        min_r = min(energies, key=energies.get)
        print(f"\nEquilibrium bond length: {min_r:.2f} Å")
    else:
        print("Available molecules: h2")
        print("Use --scan for potential energy surface")


def cmd_info(args):
    """Show tiny-qpu information."""
    from .. import __version__
    
    print(f"""
tiny-qpu v{__version__}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A minimal, fast quantum computing library.

Applications:
  • QRNG   - Quantum random number generation
  • QAOA   - MaxCut graph optimization  
  • BB84   - Quantum key distribution
  • VQE    - Molecular ground state simulation

Usage:
  tiny-qpu qrng --bytes 32 --hex
  tiny-qpu maxcut --random 6
  tiny-qpu bb84 --demo
  tiny-qpu vqe --molecule h2
  tiny-qpu run bell

GitHub: https://github.com/SKBiswas1998/tiny-qpu
""")



def cmd_benchmark(args):
    """Run molecular chemistry benchmarks."""
    from ..benchmark import ChemistryBenchmark
    from ..benchmark.molecules import MoleculeLibrary

    if args.list:
        print("\nAvailable molecules:")
        print("-" * 60)
        for name, info in MoleculeLibrary.list_molecules().items():
            print(f"  {name:6s}  {info['qubits']}q  {info['n_terms']:2d} terms  "
                  f"E={info['exact_energy']:.6f} Ha")
            print(f"         R={info['bond_range']} A  {info['description']}")
        return

    bench = ChemistryBenchmark(seed=args.seed or 42, maxiter=args.maxiter)

    if args.noise_sweep:
        mol = (args.molecule or 'H2').upper()
        mol = 'HeH+' if mol == 'HEH+' else 'LiH' if mol == 'LIH' else mol
        R = args.bond_length or 0.735
        suite = bench.run_noise_sweep(mol, bond_length=R)
    elif args.full:
        suite = bench.run_full()
    elif args.quick:
        suite = bench.run_quick()
    else:
        molecules = [(args.molecule or 'H2').upper()]
        molecules = ['HeH+' if m == 'HEH+' else 'LiH' if m == 'LIH' else m for m in molecules]
        suite = bench.run(molecules=molecules, depths=[args.depth])

    print("\n" + suite.summary())

    if args.export:
        ChemistryBenchmark.export_csv(suite, args.export)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='tiny-qpu',
        description='A minimal quantum computing toolkit'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run quantum program')
    run_parser.add_argument('file', help='Program: bell, ghz')
    run_parser.add_argument('--shots', type=int, default=1024)
    run_parser.set_defaults(func=cmd_run)
    
    # QRNG command
    qrng_parser = subparsers.add_parser('qrng', help='Quantum random numbers')
    qrng_parser.add_argument('--bits', type=int)
    qrng_parser.add_argument('--bytes', type=int)
    qrng_parser.add_argument('--hex', action='store_true')
    qrng_parser.add_argument('--int', type=int, nargs=2, metavar=('LOW', 'HIGH'))
    qrng_parser.add_argument('--float', action='store_true')
    qrng_parser.add_argument('--uuid', action='store_true')
    qrng_parser.set_defaults(func=cmd_qrng)
    
    # MaxCut command
    maxcut_parser = subparsers.add_parser('maxcut', help='Solve MaxCut with QAOA')
    maxcut_parser.add_argument('--edges', help='Edges as "0-1,1-2,2-3"')
    maxcut_parser.add_argument('--random', type=int, metavar='N')
    maxcut_parser.add_argument('--p', type=int, default=1)
    maxcut_parser.add_argument('--shots', type=int, default=512)
    maxcut_parser.set_defaults(func=cmd_maxcut)
    
    # BB84 command
    bb84_parser = subparsers.add_parser('bb84', help='BB84 key distribution')
    bb84_parser.add_argument('--key-length', type=int, default=128)
    bb84_parser.add_argument('--eve', action='store_true')
    bb84_parser.add_argument('--demo', action='store_true')
    bb84_parser.set_defaults(func=cmd_bb84)
    
    # VQE command
    vqe_parser = subparsers.add_parser('vqe', help='VQE molecular simulation')
    vqe_parser.add_argument('--molecule', default='h2', help='Molecule: h2')
    vqe_parser.add_argument('--bond-length', type=float, default=0.735)
    vqe_parser.add_argument('--depth', type=int, default=3, help='Ansatz depth')
    vqe_parser.add_argument('--maxiter', type=int, default=200)
    vqe_parser.add_argument('--seed', type=int, default=None)
    vqe_parser.add_argument('--scan', action='store_true', help='PES scan')
    vqe_parser.set_defaults(func=cmd_vqe)
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Chemistry benchmark suite')
    bench_parser.add_argument('--molecule', '-m', default=None, help='Molecule: H2, HeH+, LiH, H4')
    bench_parser.add_argument('--bond-length', '-r', type=float, default=None)
    bench_parser.add_argument('--depth', '-d', type=int, default=2, help='Ansatz depth')
    bench_parser.add_argument('--maxiter', type=int, default=300)
    bench_parser.add_argument('--seed', type=int, default=42)
    bench_parser.add_argument('--quick', action='store_true', help='Quick: H2+HeH+ clean')
    bench_parser.add_argument('--full', action='store_true', help='Full: all molecules+noise')
    bench_parser.add_argument('--noise-sweep', action='store_true', help='Noise level sweep')
    bench_parser.add_argument('--list', action='store_true', help='List available molecules')
    bench_parser.add_argument('--export', type=str, default=None, help='Export CSV path')
    bench_parser.set_defaults(func=cmd_benchmark)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show tiny-qpu info')
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
