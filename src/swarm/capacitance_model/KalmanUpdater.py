"""
Simple Kalman Filter for capacitance estimation.

For a static state with Gaussian measurements:
- High confidence (low variance) predictions get high weight
- Low confidence (high variance) predictions are rejected entirely

For variable capacitance (time-varying), set process_noise > 0.
"""

import numpy as np
from typing import Tuple, List


class KalmanCapacitanceUpdater:
    """
    Minimal Kalman filter for nearest-neighbor capacitance estimation.

    Features:
    - Variance gating: rejects updates with variance > threshold
    - Process noise: set > 0 for time-varying capacitance
    - Simple state: one scalar per capacitance element
    """

    def __init__(
        self,
        n_dots: int,
        prior_mean: float = 0.0,
        prior_variance: float = 0.5,
        variance_threshold: float = 0.05,
        process_noise: float = 0.0,
    ):
        """
        Args:
            n_dots: Number of quantum dots
            prior_mean: Initial capacitance estimate (0 for symmetric training)
            prior_variance: Initial uncertainty (large = uninformative prior)
            variance_threshold: Reject updates with variance > this value
            process_noise: Expected variance growth between updates (0 = static)
        """
        self.n_dots = n_dots
        self.variance_threshold = variance_threshold
        self.process_noise = process_noise
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

        # State: (n_dots, n_dots) matrices for means and variances
        self.means = np.zeros((n_dots, n_dots))
        self.variances = np.zeros((n_dots, n_dots))

        # Initialize nearest-neighbor pairs
        for i in range(n_dots - 1):
            self.means[i, i + 1] = prior_mean
            self.means[i + 1, i] = prior_mean
            self.variances[i, i + 1] = prior_variance
            self.variances[i + 1, i] = prior_variance

        # Stats for debugging
        self.total_accepted = 0
        self.total_rejected = 0

    def update(self, i: int, j: int, delta: float, measurement_variance: float) -> bool:
        """
        Kalman update for a single capacitance element.

        Args:
            i, j: Matrix indices (gate i, dot j)
            delta: ML model prediction (residual from current state)
            measurement_variance: exp(log_var) from ML model

        Returns:
            bool: True if update was applied, False if rejected
        """
        # Variance gating: reject unreliable predictions
        if measurement_variance > self.variance_threshold:
            self.total_rejected += 1
            return False

        # Prediction step: add process noise (for time-varying capacitance)
        P = self.variances[i, j] + self.process_noise
        x = self.means[i, j]
        R = measurement_variance

        # Kalman gain
        K = P / (P + R)

        # Update step
        self.means[i, j] = x + K * delta
        self.variances[i, j] = (1 - K) * P

        self.total_accepted += 1
        return True

    def update_from_scan(
        self, left_dot: int, ml_outputs: List[Tuple[float, float]]
    ) -> Tuple[int, int]:
        """
        Process ML outputs for a dot pair scan.

        Args:
            left_dot: Left dot index (scan is for dots left_dot and left_dot+1)
            ml_outputs: [(delta_RL, log_var_RL), (delta_LR, log_var_LR)]

        Returns:
            (accepted, rejected): Count of accepted and rejected updates
        """
        if len(ml_outputs) != 2:
            raise ValueError(f"Expected 2 outputs, got {len(ml_outputs)}")

        i = left_dot
        accepted = 0
        rejected = 0

        # RL coupling: gate i+1 to dot i
        delta_RL, log_var_RL = ml_outputs[0]
        var_RL = np.exp(log_var_RL)
        if self.update(i + 1, i, delta_RL, var_RL):
            accepted += 1
        else:
            rejected += 1

        # LR coupling: gate i to dot i+1
        delta_LR, log_var_LR = ml_outputs[1]
        var_LR = np.exp(log_var_LR)
        if self.update(i, i + 1, delta_LR, var_LR):
            accepted += 1
        else:
            rejected += 1

        return accepted, rejected

    def get_capacitance_stats(self, i: int, j: int) -> Tuple[float, float]:
        """Get current estimate and variance for element (i, j)."""
        return self.means[i, j], self.variances[i, j]

    def get_full_matrix(self) -> np.ndarray:
        """Return full Cgd matrix with diagonal set to 1.0."""
        cgd = self.means.copy()
        for i in range(self.n_dots):
            cgd[i, i] = 1.0
        return cgd

    def reset(self, prior_mean: float = None, prior_variance: float = None):
        """Reset all estimates to prior."""
        if prior_mean is None:
            prior_mean = self.prior_mean
        if prior_variance is None:
            prior_variance = self.prior_variance

        for i in range(self.n_dots - 1):
            self.means[i, i + 1] = prior_mean
            self.means[i + 1, i] = prior_mean
            self.variances[i, i + 1] = prior_variance
            self.variances[i + 1, i] = prior_variance

        self.total_accepted = 0
        self.total_rejected = 0

    def get_stats(self) -> dict:
        """Return debugging statistics."""
        total = self.total_accepted + self.total_rejected
        return {
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": self.total_accepted / total if total > 0 else 0.0,
        }
