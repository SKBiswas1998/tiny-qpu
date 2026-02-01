"""
Command-line interface for tiny-qpu.

Usage:
    tiny-qpu run program.qasm --shots 1000
    tiny-qpu qrng --bytes 32
    tiny-qpu maxcut --nodes 5 --edges "0-1,1-2,2-3,3-0"
    tiny-qpu bb84 --key-length 128
"""
import argparse
import sys
import time


def cmd_run(args):
    """Run a quantum assembly program."""
    from ..core import Circuit
    
    # For now, use the old QPU for .qasm files
    # TODO: Add QASM parser to Circuit
    print(f"Running {args.file} with {args.shots} shots...")
    
    # Simple demo circuit if no file
    if args.file == 'bell':
        qc = Circuit(2).h(0).cx(0, 1).measure_all()
        result = qc.run(shots=args.shots)
        print("\nBell State Results:")
        for state, count in sorted(result.counts.items()):
            pct = 100 * count / args.shots
            bar = '█' * int(pct / 2)
            print(f"  |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print(f"QASM file execution not yet implemented. Try: tiny-qpu run bell")


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
        # Default: 32 random bytes as hex
        print(qrng.random_bytes(32).hex())


def cmd_maxcut(args):
    """Solve MaxCut using QAOA."""
    from ..apps import QAOA
    
    print(f"Solving MaxCut with QAOA (p={args.p}, shots={args.shots})...")
    
    if args.random:
        qaoa = QAOA.random_graph(args.random, edge_prob=0.5, p=args.p)
        print(f"Random graph: {qaoa.num_qubits} nodes, {len(qaoa.edges)} edges")
    elif args.edges:
        # Parse edges like "0-1,1-2,2-3"
        edges = []
        for e in args.edges.split(','):
            i, j = map(int, e.split('-'))
            edges.append((i, j))
        qaoa = QAOA(edges, p=args.p)
    else:
        # Default: triangle
        edges = [(0, 1), (1, 2), (2, 0)]
        qaoa = QAOA(edges, p=args.p)
        print("Using default triangle graph")
    
    print(f"Edges: {qaoa.edges}")
    
    start = time.time()
    result = qaoa.optimize(shots=args.shots)
    elapsed = time.time() - start
    
    print(f"\nResult:")
    print(f"  Best partition: {result.bitstring}")
    print(f"  Cut value: {result.cut_value()}")
    print(f"  Time: {elapsed:.2f}s")
    
    # Show partition
    set_0 = [i for i, b in enumerate(reversed(result.bitstring)) if b == '0']
    set_1 = [i for i, b in enumerate(reversed(result.bitstring)) if b == '1']
    print(f"  Set A: {set_0}")
    print(f"  Set B: {set_1}")


def cmd_bb84(args):
    """Run BB84 quantum key distribution."""
    from ..apps import BB84
    
    print(f"Running BB84 QKD (key length: {args.key_length} bits)")
    
    if args.demo:
        BB84.demo(key_length=args.key_length)
    else:
        bb84 = BB84(key_length=args.key_length)
        result = bb84.run(with_eavesdropper=args.eve)
        
        print(f"\nKey: {result.key.hex()}")
        print(f"Error rate: {result.error_rate:.2%}")
        
        if result.eavesdropper_detected:
            print("⚠️  WARNING: Possible eavesdropper detected!")


def cmd_info(args):
    """Show tiny-qpu information."""
    from .. import __version__
    
    print(f"""
tiny-qpu v{__version__}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A minimal, fast quantum computing library.

Features:
  • Fast imports (<500ms vs Qiskit's 5+ seconds)
  • Fluent API: Circuit(2).h(0).cx(0,1).measure_all()
  • Educational mode: see state after each gate
  • Practical applications: QRNG, QAOA, BB84

Applications:
  • QRNG  - Quantum random number generation
  • QAOA  - Graph optimization (MaxCut)
  • BB84  - Quantum key distribution

Usage:
  tiny-qpu qrng --bytes 32
  tiny-qpu maxcut --random 6
  tiny-qpu bb84 --demo
  tiny-qpu run bell

GitHub: https://github.com/SKBiswas1998/tiny-qpu
""")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='tiny-qpu',
        description='A minimal quantum computing toolkit'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run quantum program')
    run_parser.add_argument('file', help='QASM file or "bell" for demo')
    run_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    run_parser.set_defaults(func=cmd_run)
    
    # QRNG command
    qrng_parser = subparsers.add_parser('qrng', help='Quantum random numbers')
    qrng_parser.add_argument('--bits', type=int, help='Generate N random bits')
    qrng_parser.add_argument('--bytes', type=int, help='Generate N random bytes')
    qrng_parser.add_argument('--hex', action='store_true', help='Output as hex')
    qrng_parser.add_argument('--int', type=int, nargs=2, metavar=('LOW', 'HIGH'),
                            help='Random int in [LOW, HIGH)')
    qrng_parser.add_argument('--float', action='store_true', help='Random float [0,1)')
    qrng_parser.add_argument('--uuid', action='store_true', help='Generate UUID4')
    qrng_parser.set_defaults(func=cmd_qrng)
    
    # MaxCut command
    maxcut_parser = subparsers.add_parser('maxcut', help='Solve MaxCut with QAOA')
    maxcut_parser.add_argument('--edges', help='Edges as "0-1,1-2,2-3"')
    maxcut_parser.add_argument('--random', type=int, metavar='N', help='Random N-node graph')
    maxcut_parser.add_argument('--p', type=int, default=1, help='QAOA layers')
    maxcut_parser.add_argument('--shots', type=int, default=512, help='Shots per evaluation')
    maxcut_parser.set_defaults(func=cmd_maxcut)
    
    # BB84 command
    bb84_parser = subparsers.add_parser('bb84', help='BB84 key distribution')
    bb84_parser.add_argument('--key-length', type=int, default=128, help='Key length in bits')
    bb84_parser.add_argument('--eve', action='store_true', help='Simulate eavesdropper')
    bb84_parser.add_argument('--demo', action='store_true', help='Run comparison demo')
    bb84_parser.set_defaults(func=cmd_bb84)
    
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
