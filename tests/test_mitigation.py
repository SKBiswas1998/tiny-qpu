"""
Tests for quantum error mitigation module.

Tests cover:
- ZNE extrapolation models (linear, Richardson, polynomial, exponential)
- ZNE with simulated noise (depolarizing, amplitude damping)
- Unitary folding (global, per-gate)
- Measurement error mitigation (full calibration matrix)
- Tensored measurement mitigation (per-qubit)
- Readout noise simulation
- Edge cases and error handling
"""

import numpy as np
import pytest
from tiny_qpu.mitigation import (
    # ZNE
    LinearExtrapolator,
    RichardsonExtrapolator,
    PolynomialExtrapolator,
    ExponentialExtrapolator,
    fold_global,
    fold_gates_at_random,
    zne_mitigate,
    simulate_zne,
    # Measurement
    MeasurementMitigator,
    TensoredMitigator,
    simulate_readout_noise,
)


# ═══════════════════════════════════════════════════════════════════
# ZNE Extrapolation Models
# ═══════════════════════════════════════════════════════════════════


class TestLinearExtrapolator:
    """Linear: y = a + b·λ, extrapolate to λ=0."""

    def test_exact_linear_data(self):
        """Perfectly linear data → exact recovery."""
        ext = LinearExtrapolator()
        sf = np.array([1.0, 2.0, 3.0])
        # y = 1.0 - 0.2*x → y(0) = 1.0
        ev = 1.0 - 0.2 * sf
        result = ext.fit_and_extrapolate(sf, ev)
        assert abs(result - 1.0) < 1e-10

    def test_two_points(self):
        """Minimum data: 2 points."""
        ext = LinearExtrapolator()
        result = ext.fit_and_extrapolate([1.0, 3.0], [0.8, 0.4])
        # slope = (0.4 - 0.8) / (3 - 1) = -0.2, intercept = 1.0
        assert abs(result - 1.0) < 1e-10

    def test_noisy_data(self):
        """Noisy data → reasonable estimate."""
        ext = LinearExtrapolator()
        sf = np.array([1, 2, 3, 4, 5], dtype=float)
        ev = 1.0 - 0.1 * sf + np.array([0.01, -0.02, 0.015, -0.005, 0.01])
        result = ext.fit_and_extrapolate(sf, ev)
        assert abs(result - 1.0) < 0.1  # Within 10%

    def test_fit_returns_metadata(self):
        ext = LinearExtrapolator()
        info = ext.fit([1.0, 2.0, 3.0], [0.8, 0.6, 0.4])
        assert "slope" in info
        assert "intercept" in info
        assert info["model"] == "linear"
        assert abs(info["zero_noise_value"] - 1.0) < 1e-10

    def test_too_few_points_raises(self):
        ext = LinearExtrapolator()
        with pytest.raises(ValueError, match="≥ 2"):
            ext.fit_and_extrapolate([1.0], [0.5])


class TestRichardsonExtrapolator:
    """Richardson: polynomial of degree n-1 through n points."""

    def test_quadratic_data(self):
        """3 points from y = 1 - 0.1x - 0.05x² → y(0) = 1.0"""
        ext = RichardsonExtrapolator()
        sf = np.array([1.0, 2.0, 3.0])
        ev = 1.0 - 0.1 * sf - 0.05 * sf ** 2
        result = ext.fit_and_extrapolate(sf, ev)
        assert abs(result - 1.0) < 1e-8

    def test_two_points_same_as_linear(self):
        ext = RichardsonExtrapolator()
        result = ext.fit_and_extrapolate([1.0, 3.0], [0.8, 0.4])
        assert abs(result - 1.0) < 1e-10

    def test_fit_returns_degree(self):
        ext = RichardsonExtrapolator()
        info = ext.fit([1.0, 2.0, 3.0], [0.85, 0.6, 0.3])
        assert info["degree"] == 2
        assert info["model"] == "richardson"


class TestPolynomialExtrapolator:
    """Polynomial: user-specified degree with least-squares."""

    def test_degree_2(self):
        ext = PolynomialExtrapolator(degree=2)
        sf = np.array([1.0, 2.0, 3.0, 4.0])
        ev = 1.0 - 0.1 * sf - 0.02 * sf ** 2
        result = ext.fit_and_extrapolate(sf, ev)
        assert abs(result - 1.0) < 1e-8

    def test_degree_1_matches_linear(self):
        poly = PolynomialExtrapolator(degree=1)
        lin = LinearExtrapolator()
        sf = np.array([1.0, 2.0, 3.0])
        ev = np.array([0.8, 0.6, 0.4])
        assert abs(poly.fit_and_extrapolate(sf, ev) -
                   lin.fit_and_extrapolate(sf, ev)) < 1e-10

    def test_too_few_points_raises(self):
        ext = PolynomialExtrapolator(degree=3)
        with pytest.raises(ValueError, match="≥ 4"):
            ext.fit_and_extrapolate([1, 2, 3], [0.8, 0.6, 0.4])

    def test_invalid_degree_raises(self):
        with pytest.raises(ValueError, match="≥ 1"):
            PolynomialExtrapolator(degree=0)


class TestExponentialExtrapolator:
    """Exponential: y = a + b·exp(c·λ)."""

    def test_exponential_decay(self):
        """Data from y = 0.0 + 1.0 * exp(-0.3*x) → y(0) = 1.0"""
        ext = ExponentialExtrapolator()
        sf = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ev = np.exp(-0.3 * sf)
        result = ext.fit_and_extrapolate(sf, ev)
        assert abs(result - 1.0) < 0.15  # Reasonable for exponential

    def test_with_asymptote(self):
        """Data from y = 0.5 + 0.5 * exp(-0.2*x)."""
        ext = ExponentialExtrapolator()
        sf = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ev = 0.5 + 0.5 * np.exp(-0.2 * sf)
        result = ext.fit_and_extrapolate(sf, ev)
        assert abs(result - 1.0) < 0.2  # y(0) = 0.5 + 0.5 = 1.0

    def test_fallback_to_linear(self):
        """Pathological data that can't fit exponential → falls back to linear."""
        ext = ExponentialExtrapolator()
        sf = np.array([1.0, 2.0, 3.0])
        ev = np.array([0.0, 0.0, 0.0])  # Can't fit exponential
        result = ext.fit_and_extrapolate(sf, ev)
        assert abs(result) < 0.1  # Linear extrapolation of constant 0

    def test_too_few_points_raises(self):
        ext = ExponentialExtrapolator()
        with pytest.raises(ValueError, match="≥ 3"):
            ext.fit_and_extrapolate([1, 2], [0.8, 0.6])


# ═══════════════════════════════════════════════════════════════════
# Unitary Folding
# ═══════════════════════════════════════════════════════════════════


class TestUnitaryFolding:
    """Test circuit folding for noise amplification."""

    @pytest.fixture
    def simple_circuit(self):
        return [
            {"gate": "h", "qubits": [0]},
            {"gate": "cx", "qubits": [0, 1]},
            {"gate": "rz", "qubits": [1], "params": [0.5]},
        ]

    def test_fold_1_identity(self, simple_circuit):
        """Scale factor 1 → unchanged circuit."""
        folded = fold_global(simple_circuit, 1)
        assert len(folded) == 3

    def test_fold_3_triples_depth(self, simple_circuit):
        """Scale factor 3 → G G† G (3× depth)."""
        folded = fold_global(simple_circuit, 3)
        assert len(folded) == 9  # 3 + 3 + 3

    def test_fold_5(self, simple_circuit):
        """Scale factor 5 → G G† G G† G (5× depth)."""
        folded = fold_global(simple_circuit, 5)
        assert len(folded) == 15

    def test_fold_even_raises(self, simple_circuit):
        with pytest.raises(ValueError, match="odd"):
            fold_global(simple_circuit, 2)

    def test_fold_zero_raises(self, simple_circuit):
        with pytest.raises(ValueError):
            fold_global(simple_circuit, 0)

    def test_inverse_rotation(self):
        """Rotation gate inversion negates angle."""
        ops = [{"gate": "rz", "qubits": [0], "params": [1.5]}]
        folded = fold_global(ops, 3)
        # Original, then inverse (negated), then original
        assert folded[0]["params"] == [1.5]
        assert folded[1]["params"] == [-1.5]
        assert folded[2]["params"] == [1.5]

    def test_inverse_self_inverse(self):
        """Self-inverse gates (H, X, CX) stay unchanged."""
        ops = [{"gate": "h", "qubits": [0]}]
        folded = fold_global(ops, 3)
        assert all(op["gate"] == "h" for op in folded)

    def test_inverse_s_gate(self):
        """S gate inverts to Sdg."""
        ops = [{"gate": "s", "qubits": [0]}]
        folded = fold_global(ops, 3)
        assert folded[1]["gate"] == "sdg"

    def test_random_folding_average_scale(self):
        """Random folding achieves approximate target scale factor."""
        ops = [{"gate": "rx", "qubits": [0], "params": [0.5]}] * 20
        rng = np.random.default_rng(42)
        folded = fold_gates_at_random(ops, 2.5, rng=rng)
        actual_scale = len(folded) / len(ops)
        assert abs(actual_scale - 2.5) < 0.5

    def test_random_folding_scale_1(self):
        """Scale factor 1.0 → unchanged."""
        ops = [{"gate": "h", "qubits": [0]}] * 5
        folded = fold_gates_at_random(ops, 1.0)
        assert len(folded) == 5

    def test_random_folding_invalid_raises(self):
        with pytest.raises(ValueError, match="≥ 1.0"):
            fold_gates_at_random([], 0.5)


# ═══════════════════════════════════════════════════════════════════
# ZNE Integration
# ═══════════════════════════════════════════════════════════════════


class TestZNEMitigate:
    """End-to-end ZNE tests."""

    def test_basic_zne(self):
        """ZNE recovers ideal value from linearly noisy executor."""
        ideal = -1.137
        noise_rate = 0.05

        def executor(circuit, noise_factor, shots):
            return ideal * (1 - noise_rate * noise_factor)

        result = zne_mitigate(
            executor=executor,
            circuit=None,
            scale_factors=[1, 2, 3],
            extrapolator=LinearExtrapolator(),
        )
        assert abs(result["mitigated_value"] - ideal) < 0.01
        assert result["unmitigated_value"] != ideal
        assert "improvement" in result

    def test_zne_with_richardson(self):
        """Richardson extrapolation with quadratic noise."""
        ideal = 0.75

        def executor(circuit, noise_factor, shots):
            return ideal * (1 - 0.02 * noise_factor - 0.005 * noise_factor ** 2)

        result = zne_mitigate(
            executor=executor,
            circuit=None,
            scale_factors=[1, 2, 3],
            extrapolator=RichardsonExtrapolator(),
        )
        assert abs(result["mitigated_value"] - ideal) < 0.01

    def test_zne_too_few_factors_raises(self):
        with pytest.raises(ValueError, match="≥ 2"):
            zne_mitigate(lambda c, n, s: 0.5, None, scale_factors=[1])

    def test_zne_invalid_min_factor_raises(self):
        with pytest.raises(ValueError, match="≥ 1.0"):
            zne_mitigate(lambda c, n, s: 0.5, None, scale_factors=[0.5, 1, 2])


class TestSimulateZNE:
    """Test the simulation/demonstration convenience function."""

    def test_depolarizing_noise(self):
        result = simulate_zne(
            ideal_value=-1.0,
            noise_rate=0.02,
            circuit_depth=10,
            scale_factors=[1, 3, 5],
            seed=42,
        )
        assert result["error_mitigated"] < result["error_unmitigated"]
        assert result["ideal_value"] == -1.0
        assert result["noise_model"] == "depolarizing"

    def test_amplitude_damping_noise(self):
        result = simulate_zne(
            ideal_value=-0.8,
            noise_rate=0.01,
            circuit_depth=15,
            noise_model="amplitude_damping",
            seed=123,
        )
        assert result["error_mitigated"] < result["error_unmitigated"] + 0.05

    def test_linear_noise_model(self):
        result = simulate_zne(
            ideal_value=0.5,
            noise_rate=0.03,
            circuit_depth=5,
            noise_model="linear",
            seed=99,
        )
        assert abs(result["mitigated_value"] - 0.5) < 0.1

    def test_exponential_extrapolator(self):
        result = simulate_zne(
            ideal_value=-1.0,
            noise_rate=0.02,
            circuit_depth=10,
            scale_factors=[1, 2, 3, 4, 5],
            extrapolator=ExponentialExtrapolator(),
            seed=42,
        )
        assert result["error_mitigated"] < 0.2

    def test_unknown_noise_model_raises(self):
        with pytest.raises(ValueError, match="Unknown noise model"):
            simulate_zne(ideal_value=1.0, noise_model="unknown", seed=1)


# ═══════════════════════════════════════════════════════════════════
# Measurement Error Mitigation
# ═══════════════════════════════════════════════════════════════════


class TestMeasurementMitigator:
    """Full calibration matrix approach."""

    def test_calibrate_from_noise(self):
        mit = MeasurementMitigator(n_qubits=2)
        cal = mit.calibrate_from_noise(readout_error=0.05)
        assert cal.shape == (4, 4)
        assert mit.is_calibrated
        # Diagonal elements should be large (high fidelity)
        assert all(cal[i, i] > 0.8 for i in range(4))
        # Columns should sum to 1
        assert np.allclose(cal.sum(axis=0), 1.0)

    def test_calibrate_per_qubit_errors(self):
        mit = MeasurementMitigator(n_qubits=2)
        cal = mit.calibrate_from_noise(readout_error=[0.01, 0.05])
        assert cal.shape == (4, 4)
        # Different per-qubit errors → different off-diagonal patterns
        # |01⟩↔|11⟩ flip (qubit 0) has rate 0.01
        # |00⟩↔|01⟩ flip (qubit 1) has rate 0.05
        assert not np.isclose(cal[1, 0], cal[2, 0])  # different qubit flip rates

    def test_apply_inverse(self):
        mit = MeasurementMitigator(n_qubits=2, method="inverse")
        mit.calibrate_from_noise(readout_error=0.05)
        # Perfect |00⟩ state with readout noise
        noisy = {"00": 900, "01": 30, "10": 30, "11": 40}
        corrected = mit.apply(noisy)
        # |00⟩ should dominate more after correction
        assert corrected.get("00", 0) > 0.85

    def test_apply_least_squares(self):
        mit = MeasurementMitigator(n_qubits=2, method="least_squares")
        mit.calibrate_from_noise(readout_error=0.05)
        noisy = {"00": 480, "11": 470, "01": 25, "10": 25}
        corrected = mit.apply(noisy)
        # Bell state: should be ~50/50 on |00⟩ and |11⟩
        assert corrected.get("00", 0) > 0.45
        assert corrected.get("11", 0) > 0.45
        # All probabilities should be non-negative
        assert all(v >= -1e-10 for v in corrected.values())

    def test_apply_bayesian(self):
        mit = MeasurementMitigator(n_qubits=2, method="bayesian")
        mit.calibrate_from_noise(readout_error=0.05)
        noisy = {"00": 900, "01": 30, "10": 30, "11": 40}
        corrected = mit.apply(noisy)
        assert corrected.get("00", 0) > 0.85
        # Bayesian always gives non-negative
        assert all(v >= 0 for v in corrected.values())

    def test_assignment_fidelity(self):
        mit = MeasurementMitigator(n_qubits=2)
        mit.calibrate_from_noise(readout_error=0.05)
        f = mit.assignment_fidelity()
        assert 0.9 < f < 1.0

    def test_worst_fidelity(self):
        mit = MeasurementMitigator(n_qubits=2)
        mit.calibrate_from_noise(readout_error=0.05)
        f = mit.worst_fidelity()
        assert f > 0.85

    def test_not_calibrated_raises(self):
        mit = MeasurementMitigator(n_qubits=2)
        with pytest.raises(RuntimeError, match="Not calibrated"):
            mit.apply({"00": 100})

    def test_invalid_qubits_raises(self):
        with pytest.raises(ValueError):
            MeasurementMitigator(n_qubits=0)
        with pytest.raises(ValueError):
            MeasurementMitigator(n_qubits=15)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            MeasurementMitigator(n_qubits=2, method="magic")

    def test_calibrate_with_executor(self):
        """Test calibration with a simulated executor."""
        rng = np.random.default_rng(42)

        def executor(state_label, shots):
            """Simulate noisy preparation and measurement."""
            n = len(state_label)
            counts = {}
            for _ in range(shots):
                bits = list(state_label)
                for i in range(n):
                    if rng.random() < 0.03:
                        bits[i] = "1" if bits[i] == "0" else "0"
                bs = "".join(bits)
                counts[bs] = counts.get(bs, 0) + 1
            return counts

        mit = MeasurementMitigator(n_qubits=2)
        cal = mit.calibrate(executor, shots=10000)
        assert cal.shape == (4, 4)
        assert mit.assignment_fidelity() > 0.9

    def test_correction_improves_accuracy(self):
        """Verify mitigation actually improves results."""
        mit = MeasurementMitigator(n_qubits=2, method="least_squares")
        mit.calibrate_from_noise(readout_error=0.1)

        # Ideal: pure |00⟩ state
        ideal_probs = {"00": 1.0, "01": 0.0, "10": 0.0, "11": 0.0}

        # Simulate noisy measurement
        noisy = simulate_readout_noise(
            {"00": 1000}, readout_error=0.1, seed=42
        )
        total = sum(noisy.values())
        noisy_probs = {k: v / total for k, v in noisy.items()}

        # Apply correction
        corrected = mit.apply(noisy)

        # Corrected should be closer to ideal
        noisy_error = abs(noisy_probs.get("00", 0) - 1.0)
        corrected_error = abs(corrected.get("00", 0) - 1.0)
        assert corrected_error < noisy_error

    def test_method_override(self):
        """Can override method per-call."""
        mit = MeasurementMitigator(n_qubits=1, method="inverse")
        mit.calibrate_from_noise(readout_error=0.05)
        counts = {"0": 900, "1": 100}
        result_inv = mit.apply(counts, method="inverse")
        result_ls = mit.apply(counts, method="least_squares")
        # Both should give similar results
        assert abs(result_inv.get("0", 0) - result_ls.get("0", 0)) < 0.1

    def test_repr(self):
        mit = MeasurementMitigator(n_qubits=3)
        assert "3q" in repr(mit)
        assert "not calibrated" in repr(mit)
        mit.calibrate_from_noise(0.01)
        assert "calibrated" in repr(mit)


class TestTensoredMitigator:
    """Per-qubit tensored mitigation."""

    def test_calibrate_from_noise(self):
        mit = TensoredMitigator(n_qubits=3)
        matrices = mit.calibrate_from_noise(readout_error=0.05)
        assert len(matrices) == 3
        assert all(m.shape == (2, 2) for m in matrices)
        assert mit.is_calibrated

    def test_per_qubit_errors(self):
        mit = TensoredMitigator(n_qubits=3)
        mit.calibrate_from_noise(readout_error=[0.01, 0.05, 0.1])
        fids = mit.qubit_fidelities()
        assert fids[0] > fids[1] > fids[2]

    def test_apply_correction(self):
        mit = TensoredMitigator(n_qubits=2)
        mit.calibrate_from_noise(readout_error=0.1)

        noisy = {"00": 800, "01": 60, "10": 60, "11": 80}
        corrected = mit.apply(noisy)
        assert corrected.get("00", 0) > 0.75
        assert all(v >= -1e-10 for v in corrected.values())

    def test_not_calibrated_raises(self):
        mit = TensoredMitigator(n_qubits=2)
        with pytest.raises(RuntimeError, match="Not calibrated"):
            mit.apply({"00": 100})

    def test_repr(self):
        mit = TensoredMitigator(n_qubits=4)
        assert "4q" in repr(mit)

    def test_tensored_matches_full_for_independent_noise(self):
        """For independent noise, tensored ≈ full calibration."""
        n = 2
        error = 0.05

        full = MeasurementMitigator(n, method="least_squares")
        full.calibrate_from_noise(error)

        tens = TensoredMitigator(n)
        tens.calibrate_from_noise(error)

        noisy = {"00": 850, "01": 50, "10": 50, "11": 50}
        result_full = full.apply(noisy)
        result_tens = tens.apply(noisy)

        # Should agree closely for independent noise
        for bs in ["00", "01", "10", "11"]:
            assert abs(result_full.get(bs, 0) - result_tens.get(bs, 0)) < 0.1


# ═══════════════════════════════════════════════════════════════════
# Readout Noise Simulation
# ═══════════════════════════════════════════════════════════════════


class TestReadoutNoiseSimulation:
    """Test the noise simulation utility."""

    def test_no_noise(self):
        """Zero noise → identical counts."""
        counts = {"00": 500, "11": 500}
        noisy = simulate_readout_noise(counts, readout_error=0.0, seed=42)
        assert noisy == counts

    def test_noise_adds_bitflips(self):
        """Non-zero noise → some bit flips."""
        counts = {"00": 10000}
        noisy = simulate_readout_noise(counts, readout_error=0.1, seed=42)
        # Should have some non-|00⟩ counts
        assert len(noisy) > 1
        assert noisy.get("00", 0) < 10000

    def test_total_counts_preserved(self):
        """Total count should be preserved."""
        counts = {"00": 500, "11": 500}
        noisy = simulate_readout_noise(counts, readout_error=0.05, seed=42)
        assert sum(noisy.values()) == 1000

    def test_per_qubit_noise(self):
        """Per-qubit noise rates."""
        counts = {"00": 10000}
        noisy = simulate_readout_noise(counts, readout_error=[0.0, 0.5], seed=42)
        # Qubit 0 has no noise, qubit 1 has 50% flip
        # So we expect roughly 50% "00" and 50% "01"
        assert noisy.get("00", 0) > 4000
        assert noisy.get("01", 0) > 4000
        # Very few "10" or "11" (qubit 0 has no noise)
        assert noisy.get("10", 0) < 100
        assert noisy.get("11", 0) < 100

    def test_reproducibility(self):
        counts = {"01": 1000}
        a = simulate_readout_noise(counts, 0.05, seed=42)
        b = simulate_readout_noise(counts, 0.05, seed=42)
        assert a == b


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_qubit_mitigation(self):
        mit = MeasurementMitigator(n_qubits=1, method="least_squares")
        mit.calibrate_from_noise(readout_error=0.1)
        corrected = mit.apply({"0": 850, "1": 150})
        assert corrected.get("0", 0) > 0.85

    def test_zero_noise_zne(self):
        """No noise → mitigated = unmitigated."""
        def executor(c, nf, s):
            return -1.0

        result = zne_mitigate(executor, None, [1, 2, 3])
        assert abs(result["mitigated_value"] - (-1.0)) < 1e-10
        assert abs(result["improvement"]) < 1e-10

    def test_perfect_calibration(self):
        """Identity calibration matrix → no correction."""
        mit = MeasurementMitigator(n_qubits=2, method="inverse")
        mit.calibrate_from_noise(readout_error=0.0)
        counts = {"00": 500, "11": 500}
        corrected = mit.apply(counts)
        assert abs(corrected.get("00", 0) - 0.5) < 1e-10
        assert abs(corrected.get("11", 0) - 0.5) < 1e-10

    def test_large_scale_factors(self):
        """ZNE with large scale factors still works."""
        def executor(c, nf, s):
            return 1.0 * (1 - 0.01 * nf)

        result = zne_mitigate(executor, None, [1, 5, 9, 13])
        assert abs(result["mitigated_value"] - 1.0) < 0.02

    def test_negative_expectation(self):
        """ZNE works with negative expectation values."""
        def executor(c, nf, s):
            return -0.5 * (1 - 0.05 * nf)

        result = zne_mitigate(executor, None, [1, 2, 3])
        assert abs(result["mitigated_value"] - (-0.5)) < 0.05

    def test_empty_circuit_folding(self):
        """Empty circuit → empty folded circuit."""
        assert fold_global([], 1) == []
        assert fold_gates_at_random([], 1.0) == []
