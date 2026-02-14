"""
Quantum error mitigation for NISQ circuits.

Two complementary mitigation techniques:

1. **Zero Noise Extrapolation (ZNE)**: Estimates the zero-noise
   expectation value by running circuits at multiple noise levels
   and extrapolating to zero noise.

2. **Measurement Error Mitigation**: Corrects systematic readout
   errors using calibration matrices.

Quick start:
    >>> from tiny_qpu.mitigation import zne_mitigate, LinearExtrapolator
    >>> result = zne_mitigate(executor, circuit, scale_factors=[1, 3, 5])

    >>> from tiny_qpu.mitigation import MeasurementMitigator
    >>> mit = MeasurementMitigator(n_qubits=2)
    >>> mit.calibrate_from_noise(readout_error=0.05)
    >>> corrected = mit.apply(noisy_counts)
"""

from .zne import (
    Extrapolator,
    LinearExtrapolator,
    RichardsonExtrapolator,
    PolynomialExtrapolator,
    ExponentialExtrapolator,
    fold_global,
    fold_gates_at_random,
    zne_mitigate,
    simulate_zne,
)
from .measurement import (
    MeasurementMitigator,
    TensoredMitigator,
    simulate_readout_noise,
)

__all__ = [
    # ZNE
    "Extrapolator",
    "LinearExtrapolator",
    "RichardsonExtrapolator",
    "PolynomialExtrapolator",
    "ExponentialExtrapolator",
    "fold_global",
    "fold_gates_at_random",
    "zne_mitigate",
    "simulate_zne",
    # Measurement
    "MeasurementMitigator",
    "TensoredMitigator",
    "simulate_readout_noise",
]
