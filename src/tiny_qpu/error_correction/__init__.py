"""
Quantum Error Correction Codes
================================
Implements fundamental QEC codes that protect quantum information.

Codes:
- BitFlipCode [[3,1,1]]: Corrects single X errors
- PhaseFlipCode [[3,1,1]]: Corrects single Z errors  
- ShorCode [[9,1,3]]: Corrects arbitrary single-qubit errors
- SteaneCode [[7,1,3]]: CSS code, corrects arbitrary single-qubit errors

Usage:
    from tiny_qpu.error_correction import BitFlipCode, ShorCode
    from tiny_qpu.noise import bit_flip
    
    code = BitFlipCode()
    result = code.demonstrate(error_rate=0.1, shots=1000)
    print(f"Without QEC: {result['uncorrected_error']:.1%}")
    print(f"With QEC:    {result['corrected_error']:.1%}")
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..core import Circuit


@dataclass
class QECResult:
    """Result of error correction demonstration."""
    logical_error_rate: float       # Error rate WITH correction
    physical_error_rate: float      # Error rate WITHOUT correction
    shots: int
    syndrome_counts: Dict[str, int] # How often each syndrome appeared
    code_name: str
    
    @property
    def improvement(self) -> float:
        if self.physical_error_rate > 0:
            return self.physical_error_rate / max(self.logical_error_rate, 1e-10)
        return 0.0
    
    def __str__(self) -> str:
        lines = [
            f"{self.code_name} Error Correction Results:",
            f"  Physical error rate: {self.physical_error_rate:.4f}",
            f"  Logical error rate:  {self.logical_error_rate:.4f}",
            f"  Improvement factor:  {self.improvement:.1f}x",
            f"  Shots: {self.shots}",
        ]
        return '\n'.join(lines)


class BitFlipCode:
    """
    [[3,1,1]] Bit Flip Repetition Code
    
    Encodes 1 logical qubit into 3 physical qubits.
    Corrects any single X (bit flip) error.
    
    Encoding: |0_L> = |000>, |1_L> = |111>
    Syndromes:
        00 -> no error
        01 -> error on qubit 2
        10 -> error on qubit 0  
        11 -> error on qubit 1
    """
    
    def __init__(self):
        self.name = "Bit Flip [[3,1,1]]"
        self.n_physical = 3
        self.n_logical = 1
    
    def encode(self) -> Circuit:
        """Create encoding circuit: |psi>|00> -> |psi_L>"""
        qc = Circuit(5)  # 3 data + 2 ancilla (syndrome)
        # Encode: |psi>|00> using CNOT fan-out
        qc.cx(0, 1)  # Copy qubit 0 to qubit 1
        qc.cx(0, 2)  # Copy qubit 0 to qubit 2
        return qc
    
    def syndrome_measure(self, qc: Circuit) -> Circuit:
        """Add syndrome measurement gates."""
        # Syndrome bit 0: parity of qubits 0,1 -> ancilla 3
        qc.cx(0, 3)
        qc.cx(1, 3)
        # Syndrome bit 1: parity of qubits 1,2 -> ancilla 4
        qc.cx(1, 4)
        qc.cx(2, 4)
        return qc
    
    def correct(self, qc: Circuit) -> Circuit:
        """Add correction based on syndrome."""
        # If syndrome = 10 (ancilla3=1, ancilla4=0): flip qubit 0
        # If syndrome = 11 (ancilla3=1, ancilla4=1): flip qubit 1
        # If syndrome = 01 (ancilla3=0, ancilla4=1): flip qubit 2
        # Using Toffoli gates for conditional correction
        
        # Flip qubit 0 if syndrome = 10
        qc.x(4)           # Negate ancilla4
        qc.ccx(3, 4, 0)   # If both ancillas -> flip q0
        qc.x(4)           # Undo negation
        
        # Flip qubit 1 if syndrome = 11
        qc.ccx(3, 4, 1)
        
        # Flip qubit 2 if syndrome = 01
        qc.x(3)           # Negate ancilla3
        qc.ccx(3, 4, 2)   # If both ancillas -> flip q2
        qc.x(3)           # Undo negation
        
        return qc
    
    def demonstrate(self, error_rate: float = 0.1, 
                    shots: int = 1000, seed: Optional[int] = None) -> QECResult:
        """
        Demonstrate error correction vs uncorrected.
        
        Applies random bit flip errors and compares corrected vs uncorrected.
        """
        if seed is not None:
            np.random.seed(seed)
        
        uncorrected_errors = 0
        corrected_errors = 0
        syndrome_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
        
        for _ in range(shots):
            # --- Uncorrected path ---
            # Single qubit, apply error
            if np.random.random() < error_rate:
                uncorrected_errors += 1
            
            # --- Corrected path ---
            # Encode into 3 qubits
            errors = [np.random.random() < error_rate for _ in range(3)]
            
            # Count bit flips
            num_errors = sum(errors)
            
            # Syndrome detection
            s0 = errors[0] ^ errors[1]  # Parity of q0, q1
            s1 = errors[1] ^ errors[2]  # Parity of q1, q2
            syndrome = f"{int(s0)}{int(s1)}"
            syndrome_counts[syndrome] = syndrome_counts.get(syndrome, 0) + 1
            
            # Correction: majority vote
            if num_errors <= 1:
                # Can correct: 0 or 1 error
                corrected_errors += 0
            else:
                # 2+ errors: correction fails
                corrected_errors += 1
        
        return QECResult(
            logical_error_rate=corrected_errors / shots,
            physical_error_rate=uncorrected_errors / shots,
            shots=shots,
            syndrome_counts=syndrome_counts,
            code_name=self.name
        )
    
    def demonstrate_circuit(self, error_qubit: Optional[int] = None,
                           shots: int = 1000, seed: Optional[int] = None):
        """
        Full circuit-based demonstration.
        
        Args:
            error_qubit: Which qubit to apply error to (0,1,2), None for no error
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Build full QEC circuit
        qc = Circuit(5)  # 3 data + 2 syndrome ancilla
        
        # Encode
        qc.cx(0, 1)
        qc.cx(0, 2)
        
        # Apply error
        if error_qubit is not None:
            qc.x(error_qubit)
        
        # Syndrome measurement
        self.syndrome_measure(qc)
        
        # Correction
        self.correct(qc)
        
        # Measure data qubit
        qc.measure_all()
        result = qc.run(shots=shots, seed=seed)
        
        return result


class PhaseFlipCode:
    """
    [[3,1,1]] Phase Flip Code
    
    Corrects single Z (phase flip) errors.
    Uses Hadamard basis to convert phase flips to bit flips.
    
    Encoding: |0_L> = |+++>, |1_L> = |---> 
    """
    
    def __init__(self):
        self.name = "Phase Flip [[3,1,1]]"
        self.n_physical = 3
        self.n_logical = 1
    
    def demonstrate(self, error_rate: float = 0.1,
                    shots: int = 1000, seed: Optional[int] = None) -> QECResult:
        if seed is not None:
            np.random.seed(seed)
        
        uncorrected_errors = 0
        corrected_errors = 0
        syndrome_counts = {}
        
        for _ in range(shots):
            if np.random.random() < error_rate:
                uncorrected_errors += 1
            
            errors = [np.random.random() < error_rate for _ in range(3)]
            num_errors = sum(errors)
            
            s0 = errors[0] ^ errors[1]
            s1 = errors[1] ^ errors[2]
            syndrome = f"{int(s0)}{int(s1)}"
            syndrome_counts[syndrome] = syndrome_counts.get(syndrome, 0) + 1
            
            if num_errors >= 2:
                corrected_errors += 1
        
        return QECResult(
            logical_error_rate=corrected_errors / shots,
            physical_error_rate=uncorrected_errors / shots,
            shots=shots,
            syndrome_counts=syndrome_counts,
            code_name=self.name
        )


class ShorCode:
    """
    [[9,1,3]] Shor Code
    
    The first quantum error correction code (Shor 1995).
    Encodes 1 logical qubit into 9 physical qubits.
    Corrects ANY single-qubit error (X, Y, or Z).
    
    Structure:
        Phase flip code on 3 blocks, each using bit flip code:
        |0_L> = (|000> + |111>)(|000> + |111>)(|000> + |111>) / 2sqrt(2)
        |1_L> = (|000> - |111>)(|000> - |111>)(|000> - |111>) / 2sqrt(2)
    """
    
    def __init__(self):
        self.name = "Shor [[9,1,3]]"
        self.n_physical = 9
        self.n_logical = 1
    
    def demonstrate(self, error_rate: float = 0.1,
                    shots: int = 1000, seed: Optional[int] = None) -> QECResult:
        """
        Demonstrate Shor code correction.
        
        Simulates arbitrary single-qubit errors (X, Y, Z with equal probability).
        """
        if seed is not None:
            np.random.seed(seed)
        
        uncorrected_errors = 0
        corrected_errors = 0
        syndrome_counts = {}
        
        for _ in range(shots):
            # Uncorrected
            if np.random.random() < error_rate:
                uncorrected_errors += 1
            
            # Corrected path: 9 qubits, any single error correctable
            # Each qubit can get X, Y, or Z error independently
            errors = []
            for _ in range(9):
                if np.random.random() < error_rate:
                    error_type = np.random.choice(['X', 'Y', 'Z'])
                    errors.append(error_type)
                else:
                    errors.append(None)
            
            num_errors = sum(1 for e in errors if e is not None)
            
            # Shor code can correct 1 arbitrary error
            # Fails if 2+ errors on different blocks or same type
            if num_errors == 0:
                pass
            elif num_errors == 1:
                pass  # Always correctable
            else:
                # Approximate: fails with high probability for 2+ errors
                corrected_errors += 1
            
            syndrome = f"{num_errors}"
            syndrome_counts[syndrome] = syndrome_counts.get(syndrome, 0) + 1
        
        return QECResult(
            logical_error_rate=corrected_errors / shots,
            physical_error_rate=uncorrected_errors / shots,
            shots=shots,
            syndrome_counts=syndrome_counts,
            code_name=self.name
        )
    
    def encode_circuit(self) -> Circuit:
        """
        Create the full 9-qubit Shor encoding circuit.
        
        Returns circuit that maps |psi>|0>^8 -> |psi_L>
        """
        qc = Circuit(9)
        
        # Phase flip encoding (across 3 blocks)
        qc.cx(0, 3)
        qc.cx(0, 6)
        
        # Bit flip encoding within each block
        # Block 1: qubits 0,1,2
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        
        # Block 2: qubits 3,4,5
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        
        # Block 3: qubits 6,7,8
        qc.h(6)
        qc.cx(6, 7)
        qc.cx(6, 8)
        
        return qc


class SteaneCode:
    """
    [[7,1,3]] Steane Code
    
    CSS (Calderbank-Shor-Steane) code based on classical [7,4,3] Hamming code.
    Encodes 1 logical qubit into 7 physical qubits.
    Corrects any single-qubit error.
    More efficient than Shor code (7 vs 9 qubits).
    
    |0_L> = |0000000> + |1010101> + |0110011> + |1100110>
            + |0001111> + |1011010> + |0111100> + |1101001>  (normalized)
    |1_L> = X_L |0_L> (apply X to all qubits)
    """
    
    def __init__(self):
        self.name = "Steane [[7,1,3]]"
        self.n_physical = 7
        self.n_logical = 1
        
        # Parity check matrix for [7,4,3] Hamming code
        self.H_matrix = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ])
    
    def demonstrate(self, error_rate: float = 0.1,
                    shots: int = 1000, seed: Optional[int] = None) -> QECResult:
        """Demonstrate Steane code correction."""
        if seed is not None:
            np.random.seed(seed)
        
        uncorrected_errors = 0
        corrected_errors = 0
        syndrome_counts = {}
        
        for _ in range(shots):
            if np.random.random() < error_rate:
                uncorrected_errors += 1
            
            # Apply errors to 7 qubits
            errors = [np.random.random() < error_rate for _ in range(7)]
            num_errors = sum(errors)
            
            # Compute syndrome using parity check
            error_vec = np.array(errors, dtype=int)
            syndrome_vec = (self.H_matrix @ error_vec) % 2
            syndrome = ''.join(str(int(s)) for s in syndrome_vec)
            syndrome_counts[syndrome] = syndrome_counts.get(syndrome, 0) + 1
            
            # Steane code corrects any single error
            if num_errors == 0:
                pass
            elif num_errors == 1:
                pass  # Syndrome uniquely identifies error location
            else:
                # 2+ errors: may miscorrect
                corrected_errors += 1
        
        return QECResult(
            logical_error_rate=corrected_errors / shots,
            physical_error_rate=uncorrected_errors / shots,
            shots=shots,
            syndrome_counts=syndrome_counts,
            code_name=self.name
        )
    
    def encode_circuit(self) -> Circuit:
        """Create Steane code encoding circuit."""
        qc = Circuit(7)
        
        # Prepare |0_L> from |0000000>
        qc.h(0)
        qc.h(1)
        qc.h(2)
        
        # Apply CNOT pattern based on Hamming code
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 6)
        qc.cx(1, 3)
        qc.cx(1, 5)
        qc.cx(1, 6)
        qc.cx(2, 4)
        qc.cx(2, 5)
        qc.cx(2, 6)
        
        return qc


def compare_codes(error_rates: Optional[List[float]] = None,
                  shots: int = 10000, seed: int = 42) -> Dict:
    """
    Compare all error correction codes across error rates.
    
    Returns:
        Dictionary with results for each code at each error rate.
    """
    if error_rates is None:
        error_rates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    
    codes = [BitFlipCode(), PhaseFlipCode(), ShorCode(), SteaneCode()]
    results = {}
    
    print(f"{'Error Rate':>10} | {'No QEC':>8} | {'BitFlip':>8} | {'PhaseFlip':>8} | {'Shor':>8} | {'Steane':>8}")
    print("-" * 70)
    
    for p in error_rates:
        row = {'physical': p}
        
        for code in codes:
            result = code.demonstrate(error_rate=p, shots=shots, seed=seed)
            row[code.name] = result.logical_error_rate
        
        results[p] = row
        
        print(f"{p:>10.2f} | {p:>8.4f} | {row[codes[0].name]:>8.4f} | {row[codes[1].name]:>8.4f} | {row[codes[2].name]:>8.4f} | {row[codes[3].name]:>8.4f}")
    
    return results
