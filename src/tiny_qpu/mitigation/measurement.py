"""
Measurement error mitigation for quantum circuits.

Corrects systematic readout (measurement) errors using a calibration
matrix approach:

1. **Calibration**: Prepare each computational basis state |00...0⟩,
   |00...1⟩, ..., |11...1⟩ and measure to build a confusion matrix M
   where M[i,j] = P(measure i | prepared j).

2. **Mitigation**: Given noisy measurement distribution p_noisy,
   solve M · p_ideal = p_noisy for p_ideal.

Methods:
- **Matrix inversion**: p_ideal = M⁻¹ · p_noisy  (fast, can give
  negative probabilities)
- **Least-squares**: minimize ||M · p - p_noisy||² subject to
  p ≥ 0 and Σp = 1  (physical, always valid)
- **Iterative Bayesian**: update p via Bayes' rule until convergence
  (stable, naturally positive)

For large systems (>5 qubits), full calibration is exponential.
We provide **tensored mitigation** that calibrates each qubit
independently and applies corrections via tensor product structure.

Reference:
    Bravyi, Sheldon, Smolin, Gambetta, "Mitigating measurement errors
    in multiqubit experiments", PRA 103, 042605 (2021).

Usage:
    >>> from tiny_qpu.mitigation import MeasurementMitigator
    >>> mit = MeasurementMitigator(n_qubits=2)
    >>> mit.calibrate(executor)
    >>> corrected = mit.apply(noisy_counts)
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union


class MeasurementMitigator:
    """
    Measurement error mitigation via calibration matrix.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    method : str
        Correction method: 'inverse' (fast), 'least_squares' (physical),
        or 'bayesian' (iterative). Default: 'least_squares'.
    """

    def __init__(self, n_qubits: int, method: str = "least_squares"):
        if n_qubits < 1 or n_qubits > 12:
            raise ValueError(f"n_qubits must be 1-12, got {n_qubits}")
        if method not in ("inverse", "least_squares", "bayesian"):
            raise ValueError(f"Unknown method '{method}'. Use: inverse, least_squares, bayesian")

        self._n_qubits = n_qubits
        self._method = method
        self._dim = 2 ** n_qubits
        self._cal_matrix = None  # shape (dim, dim)
        self._cal_matrix_inv = None

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def calibration_matrix(self) -> Optional[np.ndarray]:
        """Return the calibration (confusion) matrix, or None if not calibrated."""
        return self._cal_matrix.copy() if self._cal_matrix is not None else None

    @property
    def is_calibrated(self) -> bool:
        return self._cal_matrix is not None

    def calibrate(
        self,
        executor: Callable[[str, int], Dict[str, int]],
        shots: int = 8192,
    ) -> np.ndarray:
        """
        Run calibration circuits to build the confusion matrix.

        Parameters
        ----------
        executor : callable
            Function(state_label: str, shots: int) → dict[str, int].
            Prepares the given basis state (e.g., '010') and measures,
            returning a dictionary of bitstring counts.
        shots : int
            Number of measurement shots per calibration circuit.

        Returns
        -------
        np.ndarray
            The calibration matrix M of shape (2^n, 2^n).
        """
        n = self._n_qubits
        dim = self._dim
        cal = np.zeros((dim, dim), dtype=float)

        for j in range(dim):
            # Prepare basis state j
            state_label = format(j, f"0{n}b")
            counts = executor(state_label, shots)
            total = sum(counts.values())

            for bitstring, count in counts.items():
                i = int(bitstring, 2)
                if i < dim:
                    cal[i, j] = count / total

        self._cal_matrix = cal
        self._cal_matrix_inv = None  # Reset cached inverse
        return cal

    def calibrate_from_noise(
        self,
        readout_error: Union[float, List[float]] = 0.01,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Build calibration matrix from a noise model (no executor needed).

        Useful for testing and demonstration.

        Parameters
        ----------
        readout_error : float or list of float
            Probability of bit flip on readout. If a single float, the
            same error is applied to all qubits. If a list, per-qubit
            error rates.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            The calibration matrix.
        """
        n = self._n_qubits
        rng = np.random.default_rng(seed)

        if isinstance(readout_error, (int, float)):
            errors = [float(readout_error)] * n
        else:
            errors = [float(e) for e in readout_error]
            if len(errors) != n:
                raise ValueError(f"Expected {n} error rates, got {len(errors)}")

        # Build single-qubit confusion matrices
        qubit_matrices = []
        for p in errors:
            m = np.array([[1 - p, p], [p, 1 - p]])
            qubit_matrices.append(m)

        # Tensor product to get full confusion matrix
        cal = qubit_matrices[0]
        for m in qubit_matrices[1:]:
            cal = np.kron(cal, m)

        self._cal_matrix = cal
        self._cal_matrix_inv = None
        return cal

    def apply(
        self,
        counts: Dict[str, int],
        method: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Apply measurement error mitigation to noisy counts.

        Parameters
        ----------
        counts : dict
            Noisy measurement counts {bitstring: count}.
        method : str, optional
            Override the correction method for this call.

        Returns
        -------
        dict
            Corrected probability distribution {bitstring: probability}.
        """
        if self._cal_matrix is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        m = method or self._method
        n = self._n_qubits
        dim = self._dim

        # Convert counts to probability vector
        total = sum(counts.values())
        p_noisy = np.zeros(dim)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            if idx < dim:
                p_noisy[idx] = count / total

        # Apply correction
        if m == "inverse":
            p_corrected = self._apply_inverse(p_noisy)
        elif m == "least_squares":
            p_corrected = self._apply_least_squares(p_noisy)
        elif m == "bayesian":
            p_corrected = self._apply_bayesian(p_noisy)
        else:
            raise ValueError(f"Unknown method '{m}'")

        # Convert back to dict
        result = {}
        for i in range(dim):
            if abs(p_corrected[i]) > 1e-10:
                bitstring = format(i, f"0{n}b")
                result[bitstring] = float(p_corrected[i])

        return result

    def _apply_inverse(self, p_noisy: np.ndarray) -> np.ndarray:
        """Direct matrix inversion: p = M⁻¹ · p_noisy."""
        if self._cal_matrix_inv is None:
            self._cal_matrix_inv = np.linalg.inv(self._cal_matrix)
        return self._cal_matrix_inv @ p_noisy

    def _apply_least_squares(self, p_noisy: np.ndarray) -> np.ndarray:
        """
        Constrained least-squares: min ||M·p - p_noisy||²
        subject to p ≥ 0 and Σp = 1.

        Uses iterative projected gradient descent.
        """
        M = self._cal_matrix
        dim = len(p_noisy)

        # Initialize with inverse (or uniform if inverse fails)
        try:
            p = self._apply_inverse(p_noisy)
        except np.linalg.LinAlgError:
            p = np.ones(dim) / dim

        # Projected gradient descent
        lr = 0.5
        for iteration in range(200):
            # Gradient: 2 M^T (M p - p_noisy)
            residual = M @ p - p_noisy
            grad = 2 * M.T @ residual

            # Update
            p_new = p - lr * grad

            # Project onto simplex: p ≥ 0 and Σp = 1
            p_new = _project_simplex(p_new)

            # Check convergence
            if np.linalg.norm(p_new - p) < 1e-10:
                break
            p = p_new

        return p

    def _apply_bayesian(
        self, p_noisy: np.ndarray, max_iter: int = 100
    ) -> np.ndarray:
        """
        Iterative Bayesian unfolding (IBU).

        Updates: p_j^{k+1} = p_j^k · Σ_i [M_{ij} · p_noisy_i / (M · p^k)_i]
        """
        M = self._cal_matrix
        dim = len(p_noisy)

        # Initialize with uniform
        p = np.ones(dim) / dim

        for iteration in range(max_iter):
            # Predicted noisy distribution
            p_pred = M @ p
            p_pred = np.maximum(p_pred, 1e-15)  # avoid division by zero

            # Bayesian update
            ratio = p_noisy / p_pred
            correction = M.T @ ratio
            p_new = p * correction

            # Normalize
            total = np.sum(p_new)
            if total > 0:
                p_new /= total

            # Check convergence
            if np.linalg.norm(p_new - p) < 1e-10:
                break
            p = p_new

        return p

    def assignment_fidelity(self) -> float:
        """
        Compute the average assignment fidelity.

        F_avg = (1/2^n) Σ_i M[i,i]

        A perfect readout has F_avg = 1.0.
        """
        if self._cal_matrix is None:
            raise RuntimeError("Not calibrated.")
        return float(np.trace(self._cal_matrix) / self._dim)

    def worst_fidelity(self) -> float:
        """Return the minimum diagonal element of the calibration matrix."""
        if self._cal_matrix is None:
            raise RuntimeError("Not calibrated.")
        return float(np.min(np.diag(self._cal_matrix)))

    def __repr__(self) -> str:
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"MeasurementMitigator({self._n_qubits}q, {self._method}, {status})"


class TensoredMitigator:
    """
    Tensored (per-qubit) measurement error mitigation.

    Scales linearly with qubit count instead of exponentially,
    by assuming qubit readout errors are independent.

    Calibration requires only 2n circuits instead of 2^n.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (can handle larger systems than MeasurementMitigator).
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("n_qubits must be ≥ 1")
        self._n_qubits = n_qubits
        self._qubit_matrices = None  # List of 2×2 calibration matrices

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def is_calibrated(self) -> bool:
        return self._qubit_matrices is not None

    def calibrate(
        self,
        executor: Callable[[str, int], Dict[str, int]],
        shots: int = 8192,
    ) -> List[np.ndarray]:
        """
        Calibrate each qubit independently.

        Requires only 2 circuits (all-0 and all-1) instead of 2^n.

        Parameters
        ----------
        executor : callable
            Function(state_label: str, shots: int) → dict[str, int].
        shots : int
            Shots per calibration circuit.

        Returns
        -------
        list of np.ndarray
            Per-qubit 2×2 calibration matrices.
        """
        n = self._n_qubits

        # Prepare |00...0⟩ and |11...1⟩
        state_0 = "0" * n
        state_1 = "1" * n
        counts_0 = executor(state_0, shots)
        counts_1 = executor(state_1, shots)
        total_0 = sum(counts_0.values())
        total_1 = sum(counts_1.values())

        self._qubit_matrices = []
        for qubit_idx in range(n):
            # P(measure 0 on qubit i | prepared 0)
            p00 = 0.0
            for bs, count in counts_0.items():
                if len(bs) == n and bs[qubit_idx] == "0":
                    p00 += count / total_0
            p10 = 1 - p00  # P(measure 1 | prepared 0)

            # P(measure 1 on qubit i | prepared 1)
            p11 = 0.0
            for bs, count in counts_1.items():
                if len(bs) == n and bs[qubit_idx] == "1":
                    p11 += count / total_1
            p01 = 1 - p11  # P(measure 0 | prepared 1)

            m = np.array([[p00, p01], [p10, p11]])
            self._qubit_matrices.append(m)

        return list(self._qubit_matrices)

    def calibrate_from_noise(
        self,
        readout_error: Union[float, List[float]] = 0.01,
    ) -> List[np.ndarray]:
        """Build per-qubit calibration matrices from a noise model."""
        n = self._n_qubits
        if isinstance(readout_error, (int, float)):
            errors = [float(readout_error)] * n
        else:
            errors = list(readout_error)

        self._qubit_matrices = []
        for p in errors:
            m = np.array([[1 - p, p], [p, 1 - p]])
            self._qubit_matrices.append(m)
        return list(self._qubit_matrices)

    def apply(self, counts: Dict[str, int]) -> Dict[str, float]:
        """
        Apply tensored correction to noisy counts.

        Corrects each qubit's marginal distribution independently,
        then combines via tensor product structure.

        Parameters
        ----------
        counts : dict
            Noisy measurement counts {bitstring: count}.

        Returns
        -------
        dict
            Corrected probability distribution.
        """
        if not self.is_calibrated:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        n = self._n_qubits
        dim = 2 ** n
        total = sum(counts.values())

        # Build probability vector
        p_noisy = np.zeros(dim)
        for bs, count in counts.items():
            idx = int(bs, 2)
            if idx < dim:
                p_noisy[idx] = count / total

        # Apply correction via tensor product of per-qubit inverses
        p_corrected = p_noisy.copy()
        for qubit_idx in range(n):
            m_inv = np.linalg.inv(self._qubit_matrices[qubit_idx])
            p_corrected = _apply_single_qubit_correction(
                p_corrected, m_inv, qubit_idx, n
            )

        # Project to physical (non-negative, normalized)
        p_corrected = _project_simplex(p_corrected)

        # Convert back to dict
        result = {}
        for i in range(dim):
            if p_corrected[i] > 1e-10:
                result[format(i, f"0{n}b")] = float(p_corrected[i])
        return result

    def qubit_fidelities(self) -> List[float]:
        """Return per-qubit assignment fidelities."""
        if not self.is_calibrated:
            raise RuntimeError("Not calibrated.")
        return [float((m[0, 0] + m[1, 1]) / 2) for m in self._qubit_matrices]

    def __repr__(self) -> str:
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"TensoredMitigator({self._n_qubits}q, {status})"


# ─── Utility functions ───────────────────────────────────────────────


def _apply_single_qubit_correction(
    probs: np.ndarray,
    m_inv: np.ndarray,
    qubit_idx: int,
    n_qubits: int,
) -> np.ndarray:
    """
    Apply a single-qubit correction matrix to a probability vector.

    Reshapes the probability vector into a tensor of shape (2, 2, ..., 2),
    contracts the correction matrix on the qubit_idx axis, then reshapes back.
    """
    shape = [2] * n_qubits
    tensor = probs.reshape(shape)
    tensor = np.tensordot(m_inv, tensor, axes=([1], [qubit_idx]))
    tensor = np.moveaxis(tensor, 0, qubit_idx)
    return tensor.reshape(-1)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector onto the probability simplex (p ≥ 0, Σp = 1).

    Uses the efficient O(n log n) algorithm from:
    Duchi et al., "Efficient Projections onto the ℓ₁-Ball" (2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0)


def simulate_readout_noise(
    ideal_counts: Dict[str, int],
    readout_error: Union[float, List[float]] = 0.01,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Apply simulated readout noise to ideal measurement counts.

    Parameters
    ----------
    ideal_counts : dict
        Ideal (noiseless) measurement counts.
    readout_error : float or list
        Bit-flip probability on readout.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Noisy measurement counts.
    """
    rng = np.random.default_rng(seed)
    n_qubits = len(next(iter(ideal_counts)))

    if isinstance(readout_error, (int, float)):
        errors = [float(readout_error)] * n_qubits
    else:
        errors = list(readout_error)

    noisy_counts = {}
    for bitstring, count in ideal_counts.items():
        for _ in range(count):
            noisy_bits = list(bitstring)
            for i, bit in enumerate(noisy_bits):
                if rng.random() < errors[i]:
                    noisy_bits[i] = "1" if bit == "0" else "0"
            noisy_bs = "".join(noisy_bits)
            noisy_counts[noisy_bs] = noisy_counts.get(noisy_bs, 0) + 1

    return noisy_counts
