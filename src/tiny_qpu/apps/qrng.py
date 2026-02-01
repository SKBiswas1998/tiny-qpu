"""
Quantum Random Number Generator (QRNG)

Generates true random numbers using quantum superposition.
Unlike classical PRNGs, quantum randomness is fundamentally unpredictable.

Usage:
    from tiny_qpu.apps import QRNG
    
    qrng = QRNG()
    
    # Get random bits
    bits = qrng.random_bits(128)  # 128 random bits
    
    # Get random integer
    n = qrng.random_int(0, 100)   # Random int in [0, 100)
    
    # Get random float
    f = qrng.random_float()       # Random float in [0, 1)
    
    # Get random bytes (for cryptography)
    key = qrng.random_bytes(32)   # 256-bit key
"""
import numpy as np
from typing import List, Optional
from ..core import Circuit


class QRNG:
    """
    Quantum Random Number Generator.
    
    Uses quantum superposition to generate true random numbers.
    Each qubit in |+⟩ state produces one truly random bit when measured.
    
    Attributes:
        num_qubits: Number of qubits used per batch (default: 8)
        
    Example:
        >>> qrng = QRNG()
        >>> print(qrng.random_int(1, 100))  # Random number 1-99
        42
        >>> print(qrng.random_bytes(16).hex())  # Random 128-bit key
        'a3f2b8c9d4e5f6a7b8c9d0e1f2a3b4c5'
    """
    
    def __init__(self, num_qubits: int = 8, seed: Optional[int] = None):
        """
        Initialize QRNG.
        
        Args:
            num_qubits: Qubits per batch (more = fewer circuit executions)
            seed: Random seed (for testing only - defeats quantum randomness!)
        """
        self.num_qubits = num_qubits
        self._seed = seed
        self._buffer: List[int] = []
    
    def _generate_batch(self) -> List[int]:
        """Generate a batch of random bits using quantum circuit."""
        qc = Circuit(self.num_qubits)
        
        # Put all qubits in superposition
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Measure all
        qc.measure_all()
        
        # Run once
        result = qc.run(shots=1, seed=self._seed)
        bitstring = list(result.counts.keys())[0]
        
        # Convert to list of ints
        return [int(b) for b in bitstring]
    
    def _ensure_buffer(self, n: int) -> None:
        """Ensure buffer has at least n bits."""
        while len(self._buffer) < n:
            self._buffer.extend(self._generate_batch())
    
    def _consume_bits(self, n: int) -> List[int]:
        """Consume n bits from buffer."""
        self._ensure_buffer(n)
        bits = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return bits
    
    def random_bit(self) -> int:
        """Generate a single random bit (0 or 1)."""
        return self._consume_bits(1)[0]
    
    def random_bits(self, n: int) -> List[int]:
        """Generate n random bits."""
        return self._consume_bits(n)
    
    def random_bitstring(self, n: int) -> str:
        """Generate random bitstring of length n."""
        return ''.join(str(b) for b in self.random_bits(n))
    
    def random_bytes(self, n: int) -> bytes:
        """
        Generate n random bytes.
        
        Suitable for cryptographic keys, IVs, nonces, etc.
        """
        bits = self.random_bits(n * 8)
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)
    
    def random_int(self, low: int, high: int) -> int:
        """
        Generate random integer in range [low, high).
        
        Uses rejection sampling for uniform distribution.
        """
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        
        range_size = high - low
        bits_needed = (range_size - 1).bit_length()
        
        # Rejection sampling for uniform distribution
        while True:
            bits = self.random_bits(bits_needed)
            value = 0
            for b in bits:
                value = (value << 1) | b
            
            if value < range_size:
                return low + value
    
    def random_float(self) -> float:
        """Generate random float in [0, 1) with 53-bit precision."""
        # IEEE 754 double has 53-bit mantissa
        bits = self.random_bits(53)
        value = 0
        for b in bits:
            value = (value << 1) | b
        return value / (2**53)
    
    def random_uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate random float in [low, high)."""
        return low + self.random_float() * (high - low)
    
    def random_choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        if not seq:
            raise ValueError("Cannot choose from empty sequence")
        idx = self.random_int(0, len(seq))
        return seq[idx]
    
    def random_shuffle(self, seq: list) -> list:
        """Return a shuffled copy of the sequence (Fisher-Yates)."""
        result = list(seq)
        for i in range(len(result) - 1, 0, -1):
            j = self.random_int(0, i + 1)
            result[i], result[j] = result[j], result[i]
        return result
    
    def random_sample(self, population, k: int) -> list:
        """Choose k unique random elements from population."""
        if k > len(population):
            raise ValueError("Sample size larger than population")
        
        pool = list(population)
        result = []
        for _ in range(k):
            idx = self.random_int(0, len(pool))
            result.append(pool.pop(idx))
        return result
    
    def random_uuid4(self) -> str:
        """Generate a random UUID4."""
        b = self.random_bytes(16)
        # Set version (4) and variant (RFC 4122)
        b = bytearray(b)
        b[6] = (b[6] & 0x0f) | 0x40  # Version 4
        b[8] = (b[8] & 0x3f) | 0x80  # Variant RFC 4122
        
        hex_str = b.hex()
        return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


def random_bits(n: int) -> List[int]:
    """Convenience function: generate n random bits."""
    return QRNG().random_bits(n)


def random_bytes(n: int) -> bytes:
    """Convenience function: generate n random bytes."""
    return QRNG().random_bytes(n)


def random_int(low: int, high: int) -> int:
    """Convenience function: generate random int in [low, high)."""
    return QRNG().random_int(low, high)


def random_float() -> float:
    """Convenience function: generate random float in [0, 1)."""
    return QRNG().random_float()
