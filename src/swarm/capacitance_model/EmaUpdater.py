import numpy as np
from typing import Dict, Tuple, List, Union, Callable


class EmaCapacitancePredictor:
    """
    Exponential Moving Average (EMA) predictor for capacitance values in a quantum dot array.

    This class maintains an N×N capacitance matrix where each element represents
    the capacitance between a gate and a dot. Unlike the Bayesian approach, this
    directly updates the matrix values with ML model outputs, and stores the
    exact ML prediction variances at each step.

    Attributes:
        n_dots (int): Number of quantum dots in the array
        means (np.ndarray): N×N matrix of capacitance estimates
        variances (np.ndarray): N×N matrix storing the ML model prediction variances
        prior_config (Dict or Callable): Configuration for initial values
    """

    def __init__(self, n_dots: int, nn: bool, prior_config: Union[Dict[Tuple[int, int], Tuple[float, float]], Callable], ema_length: int | None = None):
        """
        Initialize the EMA capacitance predictor with initial values.

        Args:
            n_dots (int): Number of quantum dots in the array
            nn: whether we are in the nearest-neighbour regime or not
            prior_config (Dict or Callable): Either a dictionary mapping (i,j) pairs to
                (prior_mean, prior_variance) tuples, or a callable that takes (i,j) and
                returns (prior_mean, prior_variance). Only the mean is used for initialization.
            ema_length (int | None): If specified, use exponential moving average with this lengthscale.
                If None, use the last value directly without EMA.

        Example:
            # Function-based prior (distance-dependent)
            def distance_prior(i, j):
                if i == j:
                    return (1.0, 0.01)  # Self-capacitance (diagonal)
                elif abs(i-j) == 1:
                    return (0.40, 0.2)  # Nearest neighbors
                else:
                    return (0.2, 0.1)   # Distant pairs
            predictor = EmaCapacitancePredictor(5, True, distance_prior, ema_length=10)
        """
        if ema_length is not None:
            raise NotImplementedError("EMA updates not yet done")
            # NOTE will need to handle priors being initialised to the mean and thus initial updates will be unstable,
            # use variance updates with an initially very high variance, or low confidence

        self.n_dots = n_dots
        self.nearest_neighbour = nn
        self.prior_config = prior_config
        self.ema_length = ema_length

        # TODO prior config not being used for now
        print("WARNING: EMA prior not being used, prior means set to mean value of distribution (0.3)")

        if not nn:
            raise NotImplementedError("EMA capacitance update only supports nearest-neighbour coupling for now.")

        # Initialize matrices for capacitance estimates and ML prediction variances
        # indexed as (i_gate, i_dot)
        self.means = np.zeros((n_dots, n_dots))
        self.variances = np.zeros((n_dots, n_dots))

        # Initialize matrices with default values
        self._initialize_matrices()

    def _initialize_matrices(self):
        """Initialize the capacitance matrices with default values."""
        if self.nearest_neighbour:
            # Set nearest neighbor couplings to 0.3 in the means matrix
            for i in range(self.n_dots - 1):
                self.means[i, i+1] = 0.3
                self.means[i+1, i] = 0.3

    def direct_update(self, i: int, j: int, ml_estimate: float, ml_variance: float):
        """
        Directly update a capacitance element with ML model output.
        Uses exponential moving average if ema_length is specified, otherwise uses last value.
        Automatically enforces symmetry by normalizing to upper triangular and mirroring.

        Args:
            i (int): gate index
            j (int): dot index
            ml_estimate (float): ML model's capacitance estimate
            ml_variance (float): ML model's prediction variance

        Raises:
            ValueError: If indices are invalid
        """
        # Validate inputs
        if not (0 <= i < self.n_dots and 0 <= j < self.n_dots):
            raise ValueError(f"Invalid indices: ({i}, {j}). Must be in range [0, {self.n_dots})")

        # Normalize to upper triangular (canonical form: row < col)
        row, col = min(i, j), max(i, j)

        # In nearest neighbour mode, only allow updates to nearest neighbor pairs
        if self.nearest_neighbour and abs(row - col) != 1:
            raise ValueError(f"In nearest neighbour mode, can only update adjacent pairs. Got (gate {i}, dot {j})")

        # Update the mean using EMA if ema_length is specified, otherwise use last value
        if self.ema_length is not None:
            # Exponential moving average: new_mean = alpha * new_value + (1 - alpha) * old_mean
            # where alpha = 1 / ema_length, scaled by confidence (inverse variance)
            # Lower variance -> higher confidence -> stronger update
            base_alpha = 1.0 / self.ema_length
            confidence_scale = 1.0 / (0.5 + ml_variance)
            alpha = base_alpha * confidence_scale
            # Clip alpha to [0, 1] to maintain stability
            alpha = np.clip(alpha, 0.0, 1.0)
            new_mean = alpha * ml_estimate + (1 - alpha) * self.means[row, col]
        else:
            # Just use the last value
            new_mean = ml_estimate

        # Update upper triangular and mirror to lower (enforce symmetry)
        self.means[row, col] = new_mean
        self.means[col, row] = new_mean
        self.variances[row, col] = ml_variance
        self.variances[col, row] = ml_variance


    def update_from_scan(self, left_dot: int, ml_outputs: List[Tuple[float, float]]):
        """
        Process ML model output for a dot pair scan and update relevant capacitances.

        In NN mode, both predictions (RL and LR) target the same symmetric coupling
        Cgd[i, i+1]. direct_update() normalizes both to the canonical position, so
        the second update overwrites the first (last value wins for non-EMA mode).

        Args:
            left_dot: left dot in the scan - we assume pair (i, i+1)
            ml_outputs (List[Tuple[float, float]]): List of tuples containing
                (capacitance_estimate, log_variance) for each coupling between gate i and dot j.

        Example:
            predictor.update_from_scan(
                left_dot=2,
                ml_outputs=[(0.23, -2.3), (0.18, -1.9)]
            )
        """
        expected_len = 2 if self.nearest_neighbour else 3
        if len(ml_outputs) != expected_len:
            raise ValueError(f"ml_outputs must contain {expected_len} measurements, but got {len(ml_outputs)}")

        i = left_dot

        # RL coupling: cgd[i+1, i] -> normalizes to cgd[i, i+1]
        estimate_RL, log_var_RL = ml_outputs[0]
        variance_RL = np.exp(log_var_RL)
        self.direct_update(i+1, i, estimate_RL, variance_RL)

        # LR coupling: cgd[i, i+1] (already canonical)
        estimate_LR, log_var_LR = ml_outputs[1]
        variance_LR = np.exp(log_var_LR)
        self.direct_update(i, i+1, estimate_LR, variance_LR)


    def get_capacitance_stats(self, i: int, j: int) -> Tuple[float, float]:
        """
        Return current estimate for a specific capacitance.

        Args:
            i (int): gate index
            j (int): dot index

        Returns:
            Tuple[float, float]: (mean, variance) of the estimate.

        Raises:
            ValueError: If indices are invalid
        """
        if not (0 <= i < self.n_dots and 0 <= j < self.n_dots):
            raise ValueError(f"Invalid indices: ({i}, {j}). Must be in range [0, {self.n_dots})")

        return self.means[i, j], self.variances[i, j]


    def get_full_matrix(self, return_variance: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return the full matrix of current estimates.

        Args:
            return_variance (bool): If True, return both mean and variance matrices

        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]:
                If return_variance=False: matrix of means
                If return_variance=True: (means_matrix, variances_matrix)
        """
        means = self.means.copy()
        variances = self.variances.copy()

        # Set diagonal to 1.0
        for i in range(self.n_dots):
            means[i, i] = 1.0
            variances[i, i] = 0.0001  # arbitrary small number for compatibility

        if return_variance:
            return means, variances
        else:
            return means



# Example usage and testing
if __name__ == "__main__":
    # Example: Function-based prior configuration
    n_dots = 5

    def distance_prior(i: int, j: int) -> tuple:
        if i == j:
            return (1, 0.01)  # Self-capacitance (diagonal)
        elif abs(i - j) == 1:
            return (0.40, 0.2)  # Nearest neighbors
        elif abs(i - j) == 2:
            return (0.2, 0.1)   # Distant pairs
        else:
            return (0.0, 0.1)   # Very distant pairs

    predictor = EmaCapacitancePredictor(n_dots, nn=True, prior_config=distance_prior)

    # Simulating ML model outputs: [RL, LR] for scan of dots (1,2)
    ml_estimates = [0.23, 0.18]  # Capacitance predictions
    log_variances = [-2.3, -1.9]  # Log variance (ignored for EMA)
    predictor.update_from_scan(left_dot=1, ml_outputs=[(ml_estimates[0], log_variances[0]), (ml_estimates[1], log_variances[1])])

    # Get specific capacitance stats
    mean, var = predictor.get_capacitance_stats(1, 2)
    print(f"C(1,2): mean={mean:.4f}, variance={var:.6f}")

    # Get full matrix
    full_matrix = predictor.get_full_matrix()
    print(f"Full capacitance matrix:\n{full_matrix}")
