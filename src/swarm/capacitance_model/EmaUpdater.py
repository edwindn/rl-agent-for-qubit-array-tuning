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

        # Initialize matrices for capacitance estimates and ML prediction variances
        # indexed as (i_gate, i_dot)
        self.means = np.zeros((n_dots, n_dots))
        self.variances = np.zeros((n_dots, n_dots))

        # Initialize matrices with default values
        self._initialize_matrices()

    def _initialize_matrices(self):
        """Initialize the capacitance matrices with default values."""
        # NNN mode: initialize both NN and NNN couplings
        for i in range(self.n_dots - 1):
            # Nearest neighbor couplings
            self.means[i, i+1] = 0.3
            self.means[i+1, i] = 0.3
        for i in range(self.n_dots - 2):
            # Next-nearest neighbor couplings
            self.means[i, i+2] = 0.15
            self.means[i+2, i] = 0.15

    def direct_update(self, dot_pair: Tuple[int, int], ml_estimate: float):
        """
        Directly update a capacitance element with ML model output.
        Sets the capacitance value directly to the ML prediction.
        Automatically enforces symmetry.

        Args:
            dot_pair (Tuple[int, int]): Tuple of (i, j) dot indices
            ml_estimate (float): ML model's capacitance estimate

        Raises:
            ValueError: If indices are invalid
        """
        i, j = dot_pair

        # Handle edge cases where indices are out of bounds
        if i < 0 or j >= self.n_dots:
            if i == -1 or j == self.n_dots:
                # Edge case at boundaries - skip update
                return
            else:
                raise ValueError(f"Invalid dot indices ({i}, {j}) for matrix of size {self.n_dots}")

        # Validate dot pair separation (NNN mode: allow distance 1 and 2)
        if abs(i - j) not in [1, 2]:
            raise ValueError(f"Can only update adjacent or next-nearest pairs. Got dot pair ({i}, {j})")

        # Set the value directly (symmetric update)
        self.means[i, j] = ml_estimate
        self.means[j, i] = ml_estimate


    def update_from_scan(self, left_dot: int, ml_outputs: List[Tuple[float, float]]):
        """
        Process ML model output for a dot pair scan and update relevant capacitances.

        NNN mode: 3 outputs [NN, NNN_right, NNN_left] for different couplings

        Args:
            left_dot: left dot in the scan - we assume pair (i, i+1)
            ml_outputs (List[Tuple[float, float]]): List of 3 tuples containing
                (capacitance_estimate, log_variance) for each coupling.

        Example:
            predictor.update_from_scan(
                left_dot=2,
                ml_outputs=[(0.23, -2.3), (0.18, -1.9), (0.15, -2.1)]
            )
        """
        if len(ml_outputs) != 3:
            raise ValueError(f"ml_outputs must contain 3 measurements, but got {len(ml_outputs)}")

        i = left_dot

        # NNN mode: [NN, NNN_right, NNN_left]
        # NN: Cgd[i, i+1]
        estimate_nn, log_var_nn = ml_outputs[0]

        # NNN_right: Cgd[i, i+2]
        estimate_nnn_right, log_var_nnn_right = ml_outputs[1]

        # NNN_left: Cgd[i+1, i-1]
        estimate_nnn_left, log_var_nnn_left = ml_outputs[2]

        # Define dot pairs following the same convention as KrigingUpdater
        dot_pairs = [(i, i+1), (i, i+2), (i+1, i-1)]
        estimates = [estimate_nn, estimate_nnn_right, estimate_nnn_left]

        # Update each coupling directly
        for dot_pair, estimate in zip(dot_pairs, estimates):
            self.direct_update(dot_pair, estimate)


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

    predictor = EmaCapacitancePredictor(n_dots, nn=False, prior_config=distance_prior)

    # Simulating ML model outputs: [NN, NNN_right, NNN_left] for scan of dots (1,2)
    ml_estimates = [0.23, 0.18, 0.15]  # Capacitance predictions
    log_variances = [-2.3, -1.9, -2.1]  # Log variance
    predictor.update_from_scan(left_dot=1, ml_outputs=[(ml_estimates[0], log_variances[0]), (ml_estimates[1], log_variances[1]), (ml_estimates[2], log_variances[2])])

    # Get specific capacitance stats
    mean, var = predictor.get_capacitance_stats(1, 2)
    print(f"C(1,2): mean={mean:.4f}, variance={var:.6f}")

    # Get full matrix
    full_matrix = predictor.get_full_matrix()
    print(f"Full capacitance matrix:\n{full_matrix}")
