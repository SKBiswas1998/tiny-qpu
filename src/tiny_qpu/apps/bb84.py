"""
BB84 Quantum Key Distribution Protocol.

BB84 is the first quantum cryptography protocol (Bennett & Brassard, 1984).
It allows two parties (Alice and Bob) to establish a shared secret key
with security guaranteed by quantum mechanics.

Usage:
    from tiny_qpu.apps import BB84
    
    # Simulate key distribution
    bb84 = BB84(key_length=256)
    result = bb84.run()
    
    print(f"Shared key: {result.key.hex()}")
    print(f"Detected eavesdropping: {result.eavesdropper_detected}")
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from ..core import Circuit


@dataclass
class BB84Result:
    """Result of BB84 protocol execution."""
    key: bytes                      # Final shared secret key
    key_bits: List[int]             # Key as list of bits
    raw_key_length: int             # Length before error correction
    error_rate: float               # Estimated error rate
    eavesdropper_detected: bool     # True if QBER > threshold
    
    # Detailed statistics
    alice_bits: List[int]           # Alice's original bits
    alice_bases: List[int]          # Alice's bases (0=Z, 1=X)
    bob_bases: List[int]            # Bob's bases
    bob_results: List[int]          # Bob's measurement results
    matching_bases: List[int]       # Indices where bases matched
    
    def __repr__(self) -> str:
        return (f"BB84Result(key_length={len(self.key_bits)}, "
                f"error_rate={self.error_rate:.2%}, "
                f"eavesdropper={self.eavesdropper_detected})")


class BB84:
    """
    BB84 Quantum Key Distribution Protocol.
    
    The protocol works as follows:
    1. Alice generates random bits and random bases (Z or X)
    2. Alice encodes each bit in the chosen basis and sends to Bob
       - Z basis: |0⟩ or |1⟩
       - X basis: |+⟩ or |-⟩
    3. Bob randomly chooses a basis to measure each qubit
    4. Alice and Bob publicly compare bases (not bit values!)
    5. They keep only bits where bases matched (sifting)
    6. They compare a subset to estimate error rate (QBER)
    7. If QBER < threshold, remaining bits form the key
    
    Security: If Eve intercepts and measures, she disturbs the state,
    causing errors that Alice and Bob can detect.
    
    Args:
        key_length: Desired final key length in bits
        error_threshold: Maximum QBER before aborting (default: 11%)
    """
    
    def __init__(self, key_length: int = 128, error_threshold: float = 0.11):
        self.key_length = key_length
        self.error_threshold = error_threshold
        
        # We need ~4x raw bits to get desired key length after sifting
        # (50% basis match, plus some for error estimation)
        self.raw_length = key_length * 4
    
    def _alice_prepare(self, bit: int, basis: int) -> Circuit:
        """
        Alice prepares a qubit encoding her bit in her chosen basis.
        
        Z basis (basis=0): |0⟩ for bit=0, |1⟩ for bit=1
        X basis (basis=1): |+⟩ for bit=0, |-⟩ for bit=1
        """
        qc = Circuit(1)
        
        if bit == 1:
            qc.x(0)  # Flip to |1⟩
        
        if basis == 1:
            qc.h(0)  # Change to X basis: |0⟩→|+⟩, |1⟩→|-⟩
        
        return qc
    
    def _bob_measure(self, qc: Circuit, basis: int) -> int:
        """
        Bob measures in his chosen basis.
        """
        if basis == 1:
            qc.h(0)  # Change from X to Z basis before measuring
        
        qc.measure(0, 0)
        result = qc.run(shots=1)
        bitstring = list(result.counts.keys())[0]
        return int(bitstring)
    
    def _eve_intercept(self, qc: Circuit) -> Tuple[Circuit, int]:
        """
        Eve intercepts and measures (optional eavesdropper simulation).
        
        Eve randomly chooses a basis and measures, then re-prepares.
        This disturbs the state ~25% of the time when bases don't match.
        """
        eve_basis = np.random.randint(0, 2)
        
        # Eve measures
        if eve_basis == 1:
            qc.h(0)
        qc.measure(0, 0)
        result = qc.run(shots=1)
        eve_bit = int(list(result.counts.keys())[0])
        
        # Eve re-prepares (can't perfectly clone!)
        new_qc = self._alice_prepare(eve_bit, eve_basis)
        return new_qc, eve_bit
    
    def run(self, with_eavesdropper: bool = False, 
            seed: Optional[int] = None) -> BB84Result:
        """
        Execute the BB84 protocol.
        
        Args:
            with_eavesdropper: If True, simulate Eve intercepting
            seed: Random seed for reproducibility
            
        Returns:
            BB84Result with shared key and statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Step 1: Alice generates random bits and bases
        alice_bits = [np.random.randint(0, 2) for _ in range(self.raw_length)]
        alice_bases = [np.random.randint(0, 2) for _ in range(self.raw_length)]
        
        # Step 2: Bob chooses random measurement bases
        bob_bases = [np.random.randint(0, 2) for _ in range(self.raw_length)]
        
        # Step 3: Quantum transmission and measurement
        bob_results = []
        for i in range(self.raw_length):
            # Alice prepares
            qc = self._alice_prepare(alice_bits[i], alice_bases[i])
            
            # Optional: Eve intercepts
            if with_eavesdropper:
                qc, _ = self._eve_intercept(qc)
            
            # Bob measures
            result = self._bob_measure(qc, bob_bases[i])
            bob_results.append(result)
        
        # Step 4: Sifting - keep only matching bases
        matching_indices = [i for i in range(self.raw_length) 
                          if alice_bases[i] == bob_bases[i]]
        
        sifted_alice = [alice_bits[i] for i in matching_indices]
        sifted_bob = [bob_results[i] for i in matching_indices]
        
        # Step 5: Error estimation (use ~10% of sifted bits)
        n_check = max(1, len(sifted_alice) // 10)
        check_indices = np.random.choice(len(sifted_alice), n_check, replace=False)
        
        errors = sum(1 for i in check_indices 
                    if sifted_alice[i] != sifted_bob[i])
        error_rate = errors / n_check if n_check > 0 else 0.0
        
        # Step 6: Remove checked bits from key
        remaining_indices = [i for i in range(len(sifted_alice)) 
                           if i not in check_indices]
        key_bits = [sifted_alice[i] for i in remaining_indices]
        
        # Truncate to desired length
        key_bits = key_bits[:self.key_length]
        
        # Step 7: Check if eavesdropper detected
        eavesdropper_detected = error_rate > self.error_threshold
        
        # Convert bits to bytes
        key_bytes = self._bits_to_bytes(key_bits)
        
        return BB84Result(
            key=key_bytes,
            key_bits=key_bits,
            raw_key_length=len(sifted_alice),
            error_rate=error_rate,
            eavesdropper_detected=eavesdropper_detected,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_results=bob_results,
            matching_bases=matching_indices
        )
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes."""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits = bits + [0]
        
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        
        return bytes(result)
    
    @staticmethod
    def demo(key_length: int = 32, verbose: bool = True) -> Tuple[BB84Result, BB84Result]:
        """
        Run a demonstration comparing secure vs intercepted communication.
        
        Returns:
            Tuple of (secure_result, intercepted_result)
        """
        if verbose:
            print("=" * 50)
            print("BB84 Quantum Key Distribution Demo")
            print("=" * 50)
        
        bb84 = BB84(key_length=key_length)
        
        # Secure channel (no eavesdropper)
        if verbose:
            print("\n--- Secure Channel (No Eavesdropper) ---")
        secure = bb84.run(with_eavesdropper=False, seed=42)
        if verbose:
            print(f"Key generated: {secure.key.hex()[:32]}...")
            print(f"Error rate: {secure.error_rate:.2%}")
            print(f"Eavesdropper detected: {secure.eavesdropper_detected}")
        
        # Intercepted channel
        if verbose:
            print("\n--- Intercepted Channel (Eve Present) ---")
        intercepted = bb84.run(with_eavesdropper=True, seed=42)
        if verbose:
            print(f"Key generated: {intercepted.key.hex()[:32]}...")
            print(f"Error rate: {intercepted.error_rate:.2%}")
            print(f"Eavesdropper detected: {intercepted.eavesdropper_detected}")
            
            if intercepted.eavesdropper_detected:
                print("\n⚠️  High error rate detected - communication compromised!")
                print("    Alice and Bob should abort and try a different channel.")
        
        return secure, intercepted
