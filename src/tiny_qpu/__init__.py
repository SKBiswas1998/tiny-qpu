"""
tiny-qpu: A lightweight quantum processing unit simulator.

Designed for composability, speed, and research utility.
Provides efficient statevector simulation, parameterized circuits,
and a pure-Python OpenQASM 2.0 parser.

Example
-------
>>> from tiny_qpu import Circuit, gates, StatevectorBackend
>>> qc = Circuit(2)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> backend = StatevectorBackend()
>>> result = backend.run(qc)
>>> print(result.probabilities())
{0: 0.5, 3: 0.5}
"""

__version__ = "0.2.0"

from tiny_qpu.circuit import Circuit, Parameter
from tiny_qpu.backends.statevector import StatevectorBackend
from tiny_qpu.backends.density_matrix import DensityMatrixBackend

__all__ = [
    "Circuit",
    "Parameter",
    "StatevectorBackend",
    "DensityMatrixBackend",
    "__version__",
]
