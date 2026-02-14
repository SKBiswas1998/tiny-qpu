"""
Zero Noise Extrapolation (ZNE) for quantum error mitigation.

ZNE estimates the zero-noise expectation value by:
1. Running a circuit at multiple noise levels (noise scaling)
2. Fitting a model to the (noise, result) data
3. Extrapolating to the zero-noise limit

Noise scaling methods:
- **Unitary folding** (circuit-level): G → G G† G (increases depth 3×)
  - Global folding: fold entire circuit uniformly
  - Per-gate folding: fold individual gates (finer control)
- **Noise factor scaling**: directly scale noise model parameters

Extrapolation methods:
- **Linear**: y = a + b·λ  (simplest, works for small noise)
- **Richardson**: polynomial through all points (exact if noise model
  is polynomial, but can oscillate for many points)
- **Exponential**: y = a + b·exp(c·λ)  (good for depolarizing noise)
- **Polynomial**: y = a₀ + a₁·λ + ... + aₖ·λᵏ  (flexible, user-chosen degree)

Reference:
    Temme, Bravyi, Gambetta, "Error Mitigation for Short-Depth Quantum
    Circuits", PRL 119, 180509 (2017).
    Li, Benjamin, "Efficient Variational Quantum Simulator Incorporating
    Active Error Minimization", PRX 7, 021050 (2017).

Usage:
    >>> from tiny_qpu.mitigation import zne_mitigate, LinearExtrapolator
    >>> mitigated = zne_mitigate(
    ...     executor=run_circuit,
    ...     circuit=qc,
    ...     scale_factors=[1, 2, 3],
    ...     extrapolator=LinearExtrapolator(),
    ... )
"""

import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple, Union
from abc import ABC, abstractmethod


# ─── Extrapolation models ────────────────────────────────────────────


class Extrapolator(ABC):
    """Base class for ZNE extrapolation models."""

    @abstractmethod
    def fit_and_extrapolate(
        self,
        scale_factors: np.ndarray,
        expectation_values: np.ndarray,
    ) -> float:
        """
        Fit model to (scale_factors, expectation_values) and return
        the extrapolated value at scale_factor = 0.

        Parameters
        ----------
        scale_factors : array of shape (n,)
            Noise amplification factors (≥ 1).
        expectation_values : array of shape (n,)
            Measured expectation values at each noise level.

        Returns
        -------
        float
            Estimated zero-noise expectation value.
        """
        pass

    def fit(
        self,
        scale_factors: np.ndarray,
        expectation_values: np.ndarray,
    ) -> dict:
        """
        Fit model and return parameters for inspection.

        Returns
        -------
        dict
            Model parameters and extrapolated value.
        """
        extrapolated = self.fit_and_extrapolate(scale_factors, expectation_values)
        return {"zero_noise_value": extrapolated}


class LinearExtrapolator(Extrapolator):
    """
    Linear extrapolation: y = a + b·λ

    Requires ≥ 2 data points. Uses least-squares fit.
    Best for small noise levels where the noise dependence is approximately
    linear. This is the simplest and most robust extrapolation method.
    """

    def fit_and_extrapolate(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        if len(sf) < 2:
            raise ValueError("Linear extrapolation requires ≥ 2 data points")
        # Least-squares fit: y = a + b*x
        coeffs = np.polyfit(sf, ev, deg=1)  # [b, a]
        return float(np.polyval(coeffs, 0.0))

    def fit(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        coeffs = np.polyfit(sf, ev, deg=1)
        return {
            "zero_noise_value": float(np.polyval(coeffs, 0.0)),
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "model": "linear",
        }


class RichardsonExtrapolator(Extrapolator):
    """
    Richardson extrapolation: polynomial of degree n-1 through n points.

    This is the unique polynomial interpolant through all data points,
    evaluated at λ=0. Equivalent to the classical Richardson extrapolation
    used in numerical analysis.

    Exact if the noise model is a polynomial of degree ≤ n-1.
    Can oscillate (Runge's phenomenon) for many noisy data points.
    """

    def fit_and_extrapolate(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        n = len(sf)
        if n < 2:
            raise ValueError("Richardson extrapolation requires ≥ 2 data points")
        # Fit polynomial of degree n-1 through all points
        coeffs = np.polyfit(sf, ev, deg=n - 1)
        return float(np.polyval(coeffs, 0.0))

    def fit(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        n = len(sf)
        coeffs = np.polyfit(sf, ev, deg=n - 1)
        return {
            "zero_noise_value": float(np.polyval(coeffs, 0.0)),
            "degree": n - 1,
            "coefficients": coeffs.tolist(),
            "model": "richardson",
        }


class PolynomialExtrapolator(Extrapolator):
    """
    Polynomial extrapolation: y = a₀ + a₁·λ + ... + aₖ·λᵏ

    User specifies the polynomial degree. Uses least-squares fit,
    so can handle more data points than the degree.

    Parameters
    ----------
    degree : int
        Polynomial degree (default: 2). Requires ≥ degree+1 data points.
    """

    def __init__(self, degree: int = 2):
        if degree < 1:
            raise ValueError("Polynomial degree must be ≥ 1")
        self._degree = degree

    def fit_and_extrapolate(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        if len(sf) < self._degree + 1:
            raise ValueError(
                f"Polynomial degree {self._degree} requires "
                f"≥ {self._degree + 1} data points, got {len(sf)}"
            )
        coeffs = np.polyfit(sf, ev, deg=self._degree)
        return float(np.polyval(coeffs, 0.0))

    def fit(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        coeffs = np.polyfit(sf, ev, deg=self._degree)
        return {
            "zero_noise_value": float(np.polyval(coeffs, 0.0)),
            "degree": self._degree,
            "coefficients": coeffs.tolist(),
            "model": "polynomial",
        }


class ExponentialExtrapolator(Extrapolator):
    """
    Exponential extrapolation: y = a + b·exp(c·λ)

    Good model for depolarizing noise, where expectation values decay
    exponentially with circuit depth/noise.

    Uses a two-step fit:
    1. Estimate asymptote a from the trend of the data
    2. Fit log(y - a) = log(b) + c·λ linearly

    Falls back to linear if the exponential fit fails.

    Requires ≥ 3 data points.
    """

    def fit_and_extrapolate(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        if len(sf) < 3:
            raise ValueError("Exponential extrapolation requires ≥ 3 data points")

        try:
            params = self._fit_exponential(sf, ev)
            a, b, c = params
            return float(a + b)  # exp(c*0) = 1, so y(0) = a + b
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            # Fallback to linear
            coeffs = np.polyfit(sf, ev, deg=1)
            return float(np.polyval(coeffs, 0.0))

    def _fit_exponential(self, sf, ev):
        """Fit y = a + b*exp(c*x) using iterative linearization."""
        # Estimate asymptote: for large noise, y → a
        # Use the last two points to estimate the trend
        sorted_idx = np.argsort(sf)
        sf_sorted = sf[sorted_idx]
        ev_sorted = ev[sorted_idx]

        # Try multiple asymptote guesses
        best_residual = np.inf
        best_params = None

        for a_guess in [ev_sorted[-1], 0.0, ev_sorted[-1] * 1.2]:
            shifted = ev_sorted - a_guess
            # Need all shifted values to have the same sign for log
            if np.all(shifted > 1e-10):
                log_shifted = np.log(shifted)
                fit = np.polyfit(sf_sorted, log_shifted, deg=1)
                c, log_b = fit[0], fit[1]
                b = np.exp(log_b)
                # Compute residual
                predicted = a_guess + b * np.exp(c * sf_sorted)
                residual = np.sum((ev_sorted - predicted) ** 2)
                if residual < best_residual:
                    best_residual = residual
                    best_params = (a_guess, b, c)
            elif np.all(shifted < -1e-10):
                log_shifted = np.log(-shifted)
                fit = np.polyfit(sf_sorted, log_shifted, deg=1)
                c, log_b = fit[0], fit[1]
                b = -np.exp(log_b)
                predicted = a_guess + b * np.exp(c * sf_sorted)
                residual = np.sum((ev_sorted - predicted) ** 2)
                if residual < best_residual:
                    best_residual = residual
                    best_params = (a_guess, b, c)

        if best_params is None:
            raise RuntimeError("Exponential fit failed")
        return best_params

    def fit(self, scale_factors, expectation_values):
        sf = np.asarray(scale_factors, dtype=float)
        ev = np.asarray(expectation_values, dtype=float)
        result = {"model": "exponential"}
        try:
            a, b, c = self._fit_exponential(sf, ev)
            result.update({
                "zero_noise_value": float(a + b),
                "asymptote": float(a),
                "amplitude": float(b),
                "rate": float(c),
            })
        except (RuntimeError, ValueError):
            coeffs = np.polyfit(sf, ev, deg=1)
            result.update({
                "zero_noise_value": float(np.polyval(coeffs, 0.0)),
                "fallback": "linear",
            })
        return result


# ─── Noise scaling: Unitary folding ─────────────────────────────────


def fold_global(circuit_ops: List[dict], scale_factor: int) -> List[dict]:
    """
    Global unitary folding: G → G (G† G)^n

    Increases circuit depth by factor (2n + 1) where scale_factor = 2n + 1.
    The scale factor must be an odd positive integer.

    Parameters
    ----------
    circuit_ops : list of dict
        Circuit operations, each with keys: 'gate', 'qubits', 'params'.
    scale_factor : int
        Noise amplification factor (must be odd ≥ 1).

    Returns
    -------
    list of dict
        Folded circuit operations.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(f"Global fold scale_factor must be odd ≥ 1, got {scale_factor}")

    if scale_factor == 1:
        return list(circuit_ops)

    n_folds = (scale_factor - 1) // 2
    folded = list(circuit_ops)
    for _ in range(n_folds):
        # Append G† (reversed circuit with inverted gates)
        folded.extend(_invert_ops(circuit_ops))
        # Append G again
        folded.extend(circuit_ops)
    return folded


def fold_gates_at_random(
    circuit_ops: List[dict],
    scale_factor: float,
    rng: Optional[np.random.Generator] = None,
) -> List[dict]:
    """
    Random per-gate folding to achieve a non-integer scale factor.

    Each gate g is replaced by g g† g with probability p, where p
    is chosen so that the expected depth increase matches the scale factor.

    Parameters
    ----------
    circuit_ops : list of dict
        Circuit operations.
    scale_factor : float
        Target noise amplification (≥ 1.0).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    list of dict
        Folded circuit operations.
    """
    if scale_factor < 1.0:
        raise ValueError(f"Scale factor must be ≥ 1.0, got {scale_factor}")

    if rng is None:
        rng = np.random.default_rng()

    n_ops = len(circuit_ops)
    if n_ops == 0:
        return []

    # Number of full folds for all gates
    n_full = int((scale_factor - 1) // 2)
    # Remaining fractional part: fold this fraction of gates one more time
    remainder = (scale_factor - 1) / 2 - n_full
    n_extra = int(round(remainder * n_ops))

    # Select which gates get the extra fold
    extra_indices = set(rng.choice(n_ops, size=min(n_extra, n_ops), replace=False))

    folded = []
    for i, op in enumerate(circuit_ops):
        folded.append(op)
        n = n_full + (1 if i in extra_indices else 0)
        inv = _invert_op(op)
        for _ in range(n):
            folded.append(inv)
            folded.append(op)

    return folded


def _invert_ops(ops: List[dict]) -> List[dict]:
    """Reverse a list of operations and invert each gate."""
    return [_invert_op(op) for op in reversed(ops)]


def _invert_op(op: dict) -> dict:
    """
    Invert a single gate operation.

    For rotation gates (rx, ry, rz, etc.): negate the angle.
    For self-inverse gates (h, x, y, z, cx, cz, swap): return unchanged.
    For others: negate all parameters (safe for unitary rotations).
    """
    inv = dict(op)  # shallow copy
    gate = op.get("gate", "").lower()

    # Self-inverse gates
    self_inverse = {"h", "x", "y", "z", "cx", "cnot", "cz", "swap",
                    "ccx", "toffoli", "cswap", "fredkin", "i", "id"}
    if gate in self_inverse:
        return inv

    # Phase gates: negate angle
    rotation_gates = {"rx", "ry", "rz", "u1", "p", "rxx", "ryy", "rzz",
                      "crx", "cry", "crz", "cp", "u3"}
    if gate in rotation_gates and "params" in op:
        inv["params"] = [-p for p in op["params"]]
        return inv

    # S, T gates
    if gate == "s":
        inv["gate"] = "sdg"
        return inv
    if gate == "sdg":
        inv["gate"] = "s"
        return inv
    if gate == "t":
        inv["gate"] = "tdg"
        return inv
    if gate == "tdg":
        inv["gate"] = "t"
        return inv

    # Default: negate params (works for most rotation-like gates)
    if "params" in op:
        inv["params"] = [-p for p in op["params"]]
    return inv


# ─── Main ZNE function ──────────────────────────────────────────────


def zne_mitigate(
    executor: Callable,
    circuit,
    scale_factors: Sequence[Union[int, float]] = (1, 3, 5),
    extrapolator: Optional[Extrapolator] = None,
    num_shots: int = 1024,
    seed: Optional[int] = None,
) -> dict:
    """
    Perform Zero Noise Extrapolation on a quantum circuit.

    This is the main entry point for ZNE error mitigation.

    Parameters
    ----------
    executor : callable
        Function(circuit, noise_factor, num_shots) → float.
        Executes the circuit at the given noise level and returns
        an expectation value. The noise_factor parameter controls
        how much the noise is amplified (1.0 = normal, 2.0 = double, etc.).
    circuit : object
        Quantum circuit to mitigate. Passed directly to executor.
    scale_factors : sequence of float
        Noise amplification factors to sample. Default: (1, 3, 5).
        Must be sorted in ascending order with first element ≥ 1.
    extrapolator : Extrapolator, optional
        Extrapolation model. Default: LinearExtrapolator().
    num_shots : int
        Number of measurement shots per evaluation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results containing:
        - 'mitigated_value': Zero-noise extrapolated expectation value
        - 'unmitigated_value': Raw value at scale_factor=1
        - 'expectation_values': List of (scale_factor, value) pairs
        - 'improvement': Difference between mitigated and unmitigated
        - 'extrapolation': Fit details from the extrapolator

    Example
    -------
    >>> def run(circuit, noise_factor, shots):
    ...     # Run circuit with noise_factor × base noise
    ...     return expectation_value
    >>> result = zne_mitigate(run, my_circuit, scale_factors=[1, 2, 3])
    >>> print(f"Mitigated: {result['mitigated_value']:.6f}")
    """
    sf = np.asarray(scale_factors, dtype=float)
    if len(sf) < 2:
        raise ValueError("Need ≥ 2 scale factors for extrapolation")
    if sf[0] < 1.0:
        raise ValueError("Minimum scale factor must be ≥ 1.0")

    if extrapolator is None:
        extrapolator = LinearExtrapolator()

    # Execute circuit at each noise level
    rng = np.random.default_rng(seed)
    expectation_values = []
    for factor in sf:
        val = executor(circuit, float(factor), num_shots)
        expectation_values.append(float(val))

    ev = np.asarray(expectation_values)

    # Extrapolate to zero noise
    fit_result = extrapolator.fit(sf, ev)
    mitigated = fit_result["zero_noise_value"]

    return {
        "mitigated_value": mitigated,
        "unmitigated_value": expectation_values[0],
        "expectation_values": list(zip(sf.tolist(), expectation_values)),
        "improvement": mitigated - expectation_values[0],
        "extrapolation": fit_result,
    }


# ─── Convenience: simulate ZNE with a noise model ───────────────────


def simulate_zne(
    ideal_value: float,
    noise_rate: float = 0.01,
    circuit_depth: int = 10,
    scale_factors: Sequence[Union[int, float]] = (1, 3, 5),
    extrapolator: Optional[Extrapolator] = None,
    noise_model: str = "depolarizing",
    seed: Optional[int] = None,
) -> dict:
    """
    Demonstrate ZNE with a simulated noise model (no real circuit needed).

    Useful for education and testing. Generates synthetic noisy data
    from an ideal expectation value and applies ZNE.

    Parameters
    ----------
    ideal_value : float
        True (noiseless) expectation value.
    noise_rate : float
        Base noise rate per gate (depolarizing: p, damping: γ).
    circuit_depth : int
        Number of noisy gates in the circuit.
    scale_factors : sequence of float
        Noise amplification factors.
    extrapolator : Extrapolator, optional
        Extrapolation model.
    noise_model : str
        Type of noise: 'depolarizing', 'amplitude_damping', or 'linear'.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Same as zne_mitigate, plus 'ideal_value' and 'noise_model'.
    """
    rng = np.random.default_rng(seed)

    def noisy_executor(circuit, noise_factor, shots):
        effective_noise = noise_rate * noise_factor
        if noise_model == "depolarizing":
            # Depolarizing: ⟨O⟩_noisy = ⟨O⟩_ideal × (1 - p)^d
            decay = (1 - effective_noise) ** circuit_depth
            noisy = ideal_value * decay
        elif noise_model == "amplitude_damping":
            # Amplitude damping: exponential decay
            decay = np.exp(-effective_noise * circuit_depth)
            noisy = ideal_value * decay
        elif noise_model == "linear":
            # Simple linear noise model
            noisy = ideal_value * (1 - effective_noise * circuit_depth)
        else:
            raise ValueError(f"Unknown noise model: {noise_model}")

        # Add shot noise
        shot_noise = rng.normal(0, abs(ideal_value) * 0.01 / np.sqrt(shots))
        return noisy + shot_noise

    result = zne_mitigate(
        executor=noisy_executor,
        circuit=None,
        scale_factors=scale_factors,
        extrapolator=extrapolator,
        num_shots=10000,  # High shots to minimize shot noise in demo
        seed=seed,
    )
    result["ideal_value"] = ideal_value
    result["noise_model"] = noise_model
    result["error_unmitigated"] = abs(result["unmitigated_value"] - ideal_value)
    result["error_mitigated"] = abs(result["mitigated_value"] - ideal_value)
    return result
