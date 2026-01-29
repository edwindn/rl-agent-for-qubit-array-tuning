"""
Direct updater for capacitance estimation.

Behaves like the Kalman updater interface-wise, but does NOT perform a Bayesian update.
It ignores prior estimates and directly writes the predicted values into the matrix.
"""

import numpy as np
from typing import Tuple, List


class DirectCapacitanceUpdater:
    """
    Direct updater for capacitance estimation with NN and NNN support.

    Features:
    - Variance gating: rejects updates with variance > threshold
    - Simple state: one scalar per capacitance element (set directly to predictions)
    - Supports both nearest-neighbor (NN) and next-nearest-neighbor (NNN) couplings
    - Symmetric matrix: updates automatically mirrored to both (i,j) and (j,i)
    - Hard bounds: capacitance means clamped to mean_bounds, log_var clamped before exp()
    """

    def __init__(
        self,
        n_dots: int,
        prior_mean: float = 0.0,
        prior_variance: float = 0.5,
        variance_threshold: float = 0.05,
        process_noise: float = 0.0,
        include_nnn: bool = True,
        mean_bounds: Tuple[float, float] = (-1.0, 1.0),
        log_var_bounds: Tuple[float, float] = (-6.0, 2.0),
        prior_mean_nnn: float = None,
    ):
        """
        Args:
            n_dots: Number of quantum dots
            prior_mean: Initial capacitance estimate for NN couplings
            prior_variance: Initial uncertainty (large = uninformative prior)
            variance_threshold: Reject updates with variance > this value
            process_noise: Included for API parity with Kalman updater (unused here)
            include_nnn: Whether to track next-nearest-neighbor couplings
            mean_bounds: (min, max) hard bounds for capacitance values
            log_var_bounds: (min, max) bounds for log variance before exp()
                            Default (-6, 2) gives variance in ~[0.002, 7.4]
            prior_mean_nnn: Initial capacitance estimate for NNN couplings
                            (defaults to prior_mean if not specified)
        """
        self.n_dots = n_dots
        self.variance_threshold = variance_threshold
        self.process_noise = process_noise
        self.prior_mean = prior_mean
        self.prior_mean_nnn = prior_mean_nnn if prior_mean_nnn is not None else prior_mean
        self.prior_variance = prior_variance
        self.include_nnn = include_nnn
        self.mean_bounds = mean_bounds
        self.log_var_bounds = log_var_bounds

        # State: (n_dots, n_dots) matrices for means and variances
        self.means = np.zeros((n_dots, n_dots))
        self.variances = np.zeros((n_dots, n_dots))

        # Initialize nearest-neighbor pairs (NN)
        for i in range(n_dots - 1):
            self.means[i, i + 1] = prior_mean
            self.means[i + 1, i] = prior_mean  # symmetric
            self.variances[i, i + 1] = prior_variance
            self.variances[i + 1, i] = prior_variance

        # Initialize next-nearest-neighbor pairs (NNN) if enabled
        if include_nnn:
            for i in range(n_dots - 2):
                self.means[i, i + 2] = self.prior_mean_nnn
                self.means[i + 2, i] = self.prior_mean_nnn  # symmetric
                self.variances[i, i + 2] = prior_variance
                self.variances[i + 2, i] = prior_variance

        # Stats for debugging
        self.total_accepted = 0
        self.total_rejected = 0

    def _clamp_and_exp_log_var(self, log_var: float) -> float:
        """Clamp log variance to bounds and convert to variance."""
        clamped = np.clip(log_var, self.log_var_bounds[0], self.log_var_bounds[1])
        return np.exp(clamped)

    def update(self, i: int, j: int, delta: float, measurement_variance: float) -> bool:
        """
        Direct update for a single capacitance element.
        Automatically enforces symmetry by normalizing to upper triangular
        and mirroring to lower triangular.

        Args:
            i, j: Matrix indices (will be normalized to upper triangular)
            delta: ML model prediction (directly written to the matrix)
            measurement_variance: exp(log_var) from ML model

        Returns:
            bool: True if update was applied, False if rejected
        """
        # Normalize to upper triangular (canonical form: row < col)
        row, col = min(i, j), max(i, j)

        # Variance gating: reject unreliable predictions
        if measurement_variance > self.variance_threshold:
            self.total_rejected += 1
            return False

        # Direct update: ignore prior state, write prediction directly
        new_mean = delta
        new_var = measurement_variance

        # Clamp mean to physical bounds
        new_mean = np.clip(new_mean, self.mean_bounds[0], self.mean_bounds[1])

        # Update upper triangular and mirror to lower (enforce symmetry)
        self.means[row, col] = new_mean
        self.means[col, row] = new_mean
        self.variances[row, col] = new_var
        self.variances[col, row] = new_var

        self.total_accepted += 1
        return True

    def update_from_scan(
        self, left_dot: int, ml_outputs: List[Tuple[float, float]]
    ) -> Tuple[int, int]:
        """
        Process ML outputs for a dot pair scan.

        Args:
            left_dot: Left dot index (scan is for dots left_dot and left_dot+1)
            ml_outputs: For NNN mode (3 outputs):
                [(delta_NN, log_var_NN), (delta_NNN_right, log_var_NNN_right), (delta_NNN_left, log_var_NNN_left)]
                For legacy NN mode (2 outputs):
                [(delta_RL, log_var_RL), (delta_LR, log_var_LR)]

        Returns:
            (accepted, rejected): Count of accepted and rejected updates
        """
        i = left_dot
        accepted = 0
        rejected = 0

        if self.include_nnn and len(ml_outputs) == 3:
            # NNN mode: 3 outputs [NN, NNN_right, NNN_left]
            # update() enforces symmetry automatically, so we only call once per coupling

            # NN coupling: Cgd[i, i+1] (symmetric with Cgd[i+1, i])
            delta_NN, log_var_NN = ml_outputs[0]
            var_NN = self._clamp_and_exp_log_var(log_var_NN)
            if self.update(i, i + 1, delta_NN, var_NN):
                accepted += 1
            else:
                rejected += 1

            # NNN_right: Cgd[i, i+2] (symmetric with Cgd[i+2, i])
            if i + 2 < self.n_dots:
                delta_NNN_right, log_var_NNN_right = ml_outputs[1]
                var_NNN_right = self._clamp_and_exp_log_var(log_var_NNN_right)
                if self.update(i, i + 2, delta_NNN_right, var_NNN_right):
                    accepted += 1
                else:
                    rejected += 1

            # NNN_left: Cgd[i+1, i-1] (symmetric with Cgd[i-1, i+1])
            if i - 1 >= 0:
                delta_NNN_left, log_var_NNN_left = ml_outputs[2]
                var_NNN_left = self._clamp_and_exp_log_var(log_var_NNN_left)
                if self.update(i + 1, i - 1, delta_NNN_left, var_NNN_left):
                    accepted += 1
                else:
                    rejected += 1

        elif len(ml_outputs) == 2:
            # Legacy NN mode: 2 outputs [RL, LR]
            # Both predictions target the same symmetric coupling Cgd[i, i+1].
            # update() normalizes both to (i, i+1), so last accepted update wins.

            # RL coupling: Cgd[i+1, i] -> normalizes to Cgd[i, i+1]
            delta_RL, log_var_RL = ml_outputs[0]
            var_RL = self._clamp_and_exp_log_var(log_var_RL)
            if self.update(i + 1, i, delta_RL, var_RL):
                accepted += 1
            else:
                rejected += 1

            # LR coupling: Cgd[i, i+1] (already canonical)
            delta_LR, log_var_LR = ml_outputs[1]
            var_LR = self._clamp_and_exp_log_var(log_var_LR)
            if self.update(i, i + 1, delta_LR, var_LR):
                accepted += 1
            else:
                rejected += 1

        else:
            raise ValueError(f"Expected 2 or 3 outputs, got {len(ml_outputs)}")

        return accepted, rejected

    def get_capacitance_stats(self, i: int, j: int) -> Tuple[float, float]:
        """Get current estimate and variance for element (i, j).

        Note: Matrix is symmetric, so (i,j) and (j,i) return the same values.
        """
        return self.means[i, j], self.variances[i, j]

    def get_full_matrix(self) -> np.ndarray:
        """Return full Cgd matrix with diagonal set to 1.0."""
        cgd = self.means.copy()
        for i in range(self.n_dots):
            cgd[i, i] = 1.0
        return cgd

    def reset(self, prior_mean: float = None, prior_variance: float = None, prior_mean_nnn: float = None):
        """Reset all estimates to prior."""
        if prior_mean is None:
            prior_mean = self.prior_mean
        if prior_mean_nnn is None:
            prior_mean_nnn = self.prior_mean_nnn
        if prior_variance is None:
            prior_variance = self.prior_variance

        # Reset NN positions
        for i in range(self.n_dots - 1):
            self.means[i, i + 1] = prior_mean
            self.means[i + 1, i] = prior_mean
            self.variances[i, i + 1] = prior_variance
            self.variances[i + 1, i] = prior_variance

        # Reset NNN positions if enabled
        if self.include_nnn:
            for i in range(self.n_dots - 2):
                self.means[i, i + 2] = prior_mean_nnn
                self.means[i + 2, i] = prior_mean_nnn
                self.variances[i, i + 2] = prior_variance
                self.variances[i + 2, i] = prior_variance

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
