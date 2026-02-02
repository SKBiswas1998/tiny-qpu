"""
Shor's Algorithm for Integer Factorization
=============================================
Uses QPE to find period of modular exponentiation.

Usage:
    from tiny_qpu.algorithms import shor_factor, demo_factoring
    
    result = shor_factor(15)
    print(f"15 = {result.factors[0]} x {result.factors[1]}")
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from math import gcd, log2, ceil
from fractions import Fraction
from ..core import Circuit
from ..core import gates
from ..core.statevector import StateVector


@dataclass
class ShorResult:
    """Result of Shor's factorization."""
    N: int
    factors: Tuple[int, int]
    a: int
    period: int
    phase: float
    attempts: int
    method: str
    
    @property
    def success(self) -> bool:
        p, q = self.factors
        return p * q == self.N and p > 1 and q > 1
    
    def __str__(self) -> str:
        if self.success:
            return (f"Shor: {self.N} = {self.factors[0]} x {self.factors[1]} "
                    f"(a={self.a}, r={self.period}, attempts={self.attempts})")
        return f"Shor: Failed to factor {self.N}"


def _modular_exp_unitary(a: int, power: int, N: int, n: int) -> np.ndarray:
    """Build unitary for |y> -> |a^power * y mod N>."""
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=np.complex128)
    a_pow = pow(a, power, N)
    for y in range(dim):
        if y < N:
            U[(y * a_pow) % N, y] = 1.0
        else:
            U[y, y] = 1.0
    return U


def _qpe_for_period(a: int, N: int, n_counting: int = 8,
                    shots: int = 2048, seed: Optional[int] = None) -> Optional[int]:
    """Use Quantum Phase Estimation to find period of a^x mod N."""
    n_target = max(ceil(log2(N + 1)), 2)
    n_total = n_counting + n_target
    
    if seed is not None:
        np.random.seed(seed)
    
    state = StateVector(n_total)
    
    # Prepare |1> on target register (eigenstate of modular mult)
    state.apply_single_gate(gates.X, n_counting)
    
    # Hadamard on counting register
    for i in range(n_counting):
        state.apply_single_gate(gates.H, i)
    
    # Controlled modular exponentiation
    for k in range(n_counting):
        power = 2 ** k
        U = _modular_exp_unitary(a, power, N, n_target)
        
        # Apply controlled-U: when qubit k = |1>, apply U to target
        sv = state._data.copy()
        new_sv = sv.copy()
        
        for idx in range(2 ** n_total):
            control_bit = (idx >> (n_total - 1 - k)) & 1
            if control_bit == 1:
                # Extract target bits
                target_val = 0
                for t in range(n_target):
                    target_val |= ((idx >> (n_total - 1 - (n_counting + t))) & 1) << (n_target - 1 - t)
                
                new_sv[idx] = 0
                
                # Apply U
                for j_t in range(2 ** n_target):
                    if abs(U[target_val, j_t]) > 1e-15:
                        # Build source index
                        source_idx = idx
                        for t in range(n_target):
                            bit_pos = n_total - 1 - (n_counting + t)
                            old_bit = (source_idx >> bit_pos) & 1
                            new_bit = (j_t >> (n_target - 1 - t)) & 1
                            if old_bit != new_bit:
                                source_idx ^= (1 << bit_pos)
                        
                        new_sv[idx] += U[target_val, j_t] * sv[source_idx]
        
        state._data = new_sv
    
    # Inverse QFT on counting register
    for i in range(n_counting // 2):
        state.apply_two_qubit_gate(gates.SWAP, i, n_counting - 1 - i)
    
    for i in range(n_counting - 1, -1, -1):
        for j in range(n_counting - 1, i, -1):
            angle = -np.pi / (2 ** (j - i))
            state.apply_two_qubit_gate(gates.CP(angle), j, i)
        state.apply_single_gate(gates.H, i)
    
    # Measure counting register
    probs = np.abs(state._data) ** 2
    counting_probs = np.zeros(2 ** n_counting)
    for idx in range(2 ** n_total):
        counting_idx = idx >> n_target
        counting_probs[counting_idx] += probs[idx]
    
    counting_probs /= counting_probs.sum()
    
    # Sample multiple times and try each
    samples = np.random.choice(2 ** n_counting, size=shots, p=counting_probs)
    unique_samples = set(samples)
    
    # Try to extract period from each measured phase
    for measured_int in sorted(unique_samples, key=lambda x: -counting_probs[x]):
        if measured_int == 0:
            continue
        
        phase = measured_int / (2 ** n_counting)
        
        # Continued fractions to find r
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        
        if r > 0 and pow(a, r, N) == 1:
            return r
        
        # Try small multiples
        for mult in range(2, 6):
            if pow(a, r * mult, N) == 1:
                return r * mult
    
    return None


def _find_period_classical(a: int, N: int) -> int:
    """Find period classically (for verification)."""
    r = 1
    current = a % N
    while current != 1 and r <= N:
        current = (current * a) % N
        r += 1
    return r if r <= N else None


def shor_factor(N: int, max_attempts: int = 20,
                n_counting: int = 8, shots: int = 2048,
                seed: Optional[int] = None,
                use_quantum: bool = True) -> ShorResult:
    """
    Factor integer N using Shor's algorithm.
    
    Args:
        N: Integer to factor
        max_attempts: Max random base attempts
        n_counting: QPE precision qubits
        shots: QPE measurement shots
        seed: Random seed
        use_quantum: Use quantum period finding
    
    Returns:
        ShorResult with factors
    """
    if seed is not None:
        np.random.seed(seed)
    
    if N < 4:
        return ShorResult(N, (1, N), 0, 0, 0, 0, 'trivial')
    if N % 2 == 0:
        return ShorResult(N, (2, N // 2), 0, 0, 0, 0, 'even')
    
    # Check perfect power
    for b in range(2, int(np.log2(N)) + 2):
        a_root = round(N ** (1.0/b))
        for candidate in [a_root - 1, a_root, a_root + 1]:
            if candidate > 1 and candidate ** b == N:
                return ShorResult(N, (candidate, N // candidate), 0, 0, 0, 0, 'perfect_power')
    
    for attempt in range(1, max_attempts + 1):
        a = np.random.randint(2, N)
        
        d = gcd(a, N)
        if d > 1:
            return ShorResult(N, (d, N // d), a, 0, 0, attempt, 'gcd_shortcut')
        
        if use_quantum:
            r = _qpe_for_period(a, N, n_counting, shots)
        else:
            r = _find_period_classical(a, N)
        
        if r is None or r == 0 or r % 2 != 0:
            continue
        
        x = pow(a, r // 2, N)
        if x == N - 1:
            continue
        
        f1 = gcd(x + 1, N)
        f2 = gcd(x - 1, N)
        
        for f in [f1, f2]:
            if 1 < f < N:
                return ShorResult(N, (f, N // f), a, r, 0, attempt,
                                'quantum' if use_quantum else 'classical')
    
    return ShorResult(N, (1, N), 0, 0, 0, max_attempts, 'failed')


def factor(N: int, **kwargs) -> Tuple[int, int]:
    """Simple interface: returns (p, q) such that N = p * q."""
    return shor_factor(N, **kwargs).factors


def demo_factoring():
    """Demonstrate Shor's algorithm."""
    print("Shor's Algorithm - Integer Factorization")
    print("=" * 55)
    
    for N in [15, 21, 33, 35, 55, 77, 91]:
        result = shor_factor(N, seed=42)
        if result.success:
            p, q = result.factors
            print(f"  {N:3d} = {p} x {q}  "
                  f"(a={result.a}, r={result.period}, method={result.method})")
        else:
            print(f"  {N:3d}: Failed")
