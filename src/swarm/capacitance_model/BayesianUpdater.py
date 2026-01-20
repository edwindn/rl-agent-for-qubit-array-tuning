import numpy as np
from typing import Dict, Tuple, List, Union, Callable
import warnings


class BayesianCapacitancePredictor:
    """
    Bayesian parameter estimation for fixed capacitance values in a quantum dot array.
    
    This class maintains a symmetric N×N capacitance matrix where each element represents
    the capacitance between a gate and a dot. Each matrix element stores both the mean
    and variance of a Gaussian posterior distribution, which is updated using conjugate
    Bayesian inference as new measurements arrive from ML model predictions.
    
    The class provides direct integration with trained neural networks that output
    log variance for uncertainty estimation. Use update_from_ml_model() for direct
    integration with your trained capacitance prediction model.
    
    Attributes:
        n_dots (int): Number of quantum dots in the array
        means (np.ndarray): N×N matrix of posterior means
        variances (np.ndarray): N×N matrix of posterior variances
        prior_config (Dict or Callable): Configuration for prior distributions
    """
    
    def __init__(self, n_dots: int, nn: bool, prior_config: Union[Dict[Tuple[int, int], Tuple[float, float]], Callable]):
        """
        Initialize the capacitance predictor with prior distributions.
        
        Args:
            n_dots (int): Number of quantum dots in the array
            nn: whether we are in the nearest-neighbour regime or not
            prior_config (Dict or Callable): Either a dictionary mapping (i,j) pairs to 
                (prior_mean, prior_variance) tuples, or a callable that takes (i,j) and 
                returns (prior_mean, prior_variance)
        
        Example:
            # Dictionary-based prior
            priors = {(i,j): (0.25, 0.1) for i in range(5) for j in range(5)}
            predictor = BayesianCapacitancePredictor(5, priors)
            
            # Function-based prior (distance-dependent)
            def distance_prior(i, j):
                if i == j:
                    return (0.5, 0.05)  # Self-capacitance
                elif abs(i-j) == 1:
                    return (0.25, 0.1)  # Nearest neighbors
                else:
                    return (0.1, 0.2)   # Distant pairs
            predictor = BayesianCapacitancePredictor(5, distance_prior)
        """
        self.n_dots = n_dots
        self.nearest_neighbour = nn
        self.prior_config = prior_config

        if not nn:
            raise NotImplementedError("Capacitance Bayesian update only supports nearest-neighbour coupling for now.")
        
        # Initialize symmetric matrices for means and variances
        # indexed as (i_gate, i_dot)
        self.means = np.zeros((n_dots, n_dots))
        self.variances = np.zeros((n_dots, n_dots))
        
        # Set up prior distributions
        self._initialize_priors()
        
        # Validation
        self._validate_initialization()
    
    def _initialize_priors(self):
        """Initialize the prior distributions for all matrix elements."""
        if self.nearest_neighbour:
            # Only initialize nearest neighbor pairs: [i,i+1] and [i+1,i]
            # note we do not require symmetry, but this is quicker to initialise and
            # symmetry makes sense with no further priors
            for i in range(self.n_dots - 1):
                if callable(self.prior_config):
                    prior_mean, prior_var = self.prior_config(i, i+1)
                else:
                    prior_mean, prior_var = self.prior_config.get((i, i+1), (0.25, 0.1))

                self.means[i, i+1] = prior_mean
                self.means[i+1, i] = prior_mean
                self.variances[i, i+1] = prior_var
                self.variances[i+1, i] = prior_var
        else:
            # Initialize all matrix elements
            for i in range(self.n_dots):
                for j in range(self.n_dots):
                    if callable(self.prior_config):
                        prior_mean, prior_var = self.prior_config(i, j)
                    else:
                        prior_mean, prior_var = self.prior_config.get((i, j), (0.25, 0.1))

                    self.means[i, j] = prior_mean
                    self.variances[i, j] = prior_var
    
    def _validate_initialization(self):
        """Validate the initialization of the matrices."""
        # TODO no longer needed
        # # Check symmetry
        # if not np.allclose(self.means, self.means.T):
        #     raise ValueError("Mean matrix is not symmetric")
        # if not np.allclose(self.variances, self.variances.T):
        #     raise ValueError("Variance matrix is not symmetric")

        # Check positive variances for initialized elements only
        if self.nearest_neighbour:
            # Only check nearest neighbor pairs
            for i in range(self.n_dots - 1):
                if self.variances[i, i+1] <= 0 or self.variances[i+1, i] <= 0:
                    raise ValueError("All variances must be positive")
        else:
            # Check all elements
            if np.any(self.variances <= 0):
                raise ValueError("All variances must be positive")
    
    def bayesian_update(self, i: int, j: int, ml_estimate: float, ml_variance: float):
        """
        Perform conjugate Bayesian update for a single capacitance element.

        Uses the conjugate prior property of Gaussian distributions to update
        the posterior mean and variance given a new measurement.

        Args:
            i (int): gate index
            j (int): dot index
            ml_estimate (float): ML model's capacitance estimate
            ml_variance (float): Variance/uncertainty of the ML estimate

        Raises:
            ValueError: If indices are invalid or variances are non-positive
        """
        # Validate inputs
        if not (0 <= i < self.n_dots and 0 <= j < self.n_dots):
            raise ValueError(f"Invalid indices: ({i}, {j}). Must be in range [0, {self.n_dots})")
        if ml_variance <= 0:
            raise ValueError("ML variance must be positive")

        # In nearest neighbour mode, only allow updates to nearest neighbor pairs
        if self.nearest_neighbour and abs(i - j) != 1:
            raise ValueError(f"In nearest neighbour mode, can only update adjacent pairs. Got (gate {i}, dot {j})")

        # Get current posterior for element (i,j)
        current_mean = self.means[i, j]
        current_var = self.variances[i, j]

        # Conjugate update formulas
        precision_prior = 1 / current_var
        precision_ml = 1 / ml_variance
        precision_post = precision_prior + precision_ml

        new_mean = (current_mean * precision_prior + ml_estimate * precision_ml) / precision_post
        new_var = 1 / precision_post

        # Sanity check: posterior variance should not increase
        if new_var > current_var:
            warnings.warn(f"Posterior variance increased for element ({i},{j}). "
                         f"Old: {current_var:.6f}, New: {new_var:.6f}")

        self.means[i, j] = new_mean
        self.variances[i, j] = new_var

        # TODO this assumes we update Cdd. In the refactor we update Cgd. to be removed
        # # Update both (i,j) and (j,i) to maintain symmetry
        # self.means[i, j] = new_mean
        # self.means[j, i] = new_mean
        # self.variances[i, j] = new_var
        # self.variances[j, i] = new_var

    def update_from_scan(self, left_dot: int, ml_outputs: List[Tuple[float, float]]):
        """
        Process ML model output for a dot pair scan and update relevant capacitances.
        
        Args:
            dot_pair (Tuple[int, int]): The measured dot pair (i, j) # not yet implemented
            left_dot: left dot in the scan - we assume pair (i, i+1)
            ml_outputs (List[Tuple[float, float]]): List of tuples containing
                (capacitance_estimate, log_variance) for each coupling between gate i and dot j
        
        Example:
            predictor.update_from_scan(
                dot_pair=(2, 3),
                ml_outputs=[(0.23, -2.3), (0.18, -1.9), (0.31, -2.5)]
            )
        """

        # TODO this code currently only works with nearest neighbour prediction.

        expected_len = 2 if self.nearest_neighbour else 3
        if len(ml_outputs) != expected_len:
            raise ValueError(f"ml_outputs must contain {expected_len} measurements, but got {len(ml_outputs)}")
        
        i = left_dot

        estimate_RL, log_var_RL = ml_outputs[0]
        variance_RL = np.exp(log_var_RL)
        self.bayesian_update(i+1, i, estimate_RL, variance_RL)

        estimate_LR, log_var_LR = ml_outputs[1]
        variance_LR = np.exp(log_var_LR)
        self.bayesian_update(i, i+1, estimate_LR, variance_LR)
        
        # i, j = dot_pair
        
        # # First measurement: direct capacitance C_ij
        # estimate_ij, log_var_ij = ml_outputs[0]
        # variance_ij = np.exp(log_var_ij)
        # self.bayesian_update(i, j, estimate_ij, variance_ij)
        
        # # Second and third measurements: neighboring capacitances
        # # We need to determine which neighboring dots these correspond to

        # for idx, (estimate, log_variance) in enumerate(ml_outputs[1:], 1):
        #     variance = np.exp(log_variance)
            
        #     # Find appropriate neighboring dot pairs
        #     # This is a heuristic - you may need to modify based on your specific setup
        #     if idx == 1:  # C_ik - find neighbor of i
        #         k = self._find_neighbor(i, exclude=[j])
        #         if k is not None:
        #             self.bayesian_update(i, k, estimate, variance)
        #     elif idx == 2:  # C_jk - find neighbor of j
        #         k = self._find_neighbor(j, exclude=[i])
        #         if k is not None:
        #             self.bayesian_update(j, k, estimate, variance)
    
    def _find_neighbor(self, dot_idx: int, exclude: List[int] = None) -> Union[int, None]:
        """
        Find a neighboring dot for the given dot index.
        
        This is a simplified neighbor-finding heuristic. In practice, you might
        have a more sophisticated graph structure or physical layout.
        
        Args:
            dot_idx (int): Index of the dot to find a neighbor for
            exclude (List[int]): List of dot indices to exclude from consideration
        
        Returns:
            int or None: Index of a neighboring dot, or None if none found
        """
        raise NotImplementedError
        return

        if exclude is None:
            exclude = []
        
        # Simple linear arrangement assumption - neighbors are ±1 index
        candidates = [dot_idx - 1, dot_idx + 1]
        
        for neighbor in candidates:
            if (0 <= neighbor < self.n_dots and 
                neighbor not in exclude and 
                neighbor != dot_idx):
                return neighbor
        
        return None
    
    def get_capacitance_stats(self, i: int, j: int) -> Tuple[float, float]:
        """
        Return current mean and variance estimates for a specific capacitance.
        
        Args:
            i (int): gate index
            j (int): dot index
        
        Returns:
            Tuple[float, float]: (mean, variance) of the posterior distribution
        
        Raises:
            ValueError: If indices are invalid
        """
        if not (0 <= i < self.n_dots and 0 <= j < self.n_dots):
            raise ValueError(f"Invalid indices: ({i}, {j}). Must be in range [0, {self.n_dots})")
        
        return self.means[i, j], self.variances[i, j]
    
    def get_full_matrix(self, return_variance: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return the full symmetric matrix of current estimates.
        
        Args:
            return_variance (bool): If True, return both mean and variance matrices
        
        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: 
                If return_variance=False: matrix of means
                If return_variance=True: (means_matrix, variances_matrix)
        """
        means = self.means.copy()
        variances = self.variances.copy()

        for i in range(self.n_dots):
            means[i, i] = 1.0
            variances[i, i] = 0.0001 #arbitrary small number

        if return_variance:
            return means, variances
        else:
            return means
    
    def get_confidence_interval(self, i: int, j: int, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for a specific capacitance estimate.
        
        Args:
            i (int): gate index
            j (int): dot index
            confidence_level (float): Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound) of confidence interval
        """
        mean, variance = self.get_capacitance_stats(i, j)
        std = np.sqrt(variance)
        
        # For Gaussian distribution
        from scipy.stats import norm
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha/2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return lower, upper
    
    def reset_element(self, i: int, j: int):
        """
        Reset a specific matrix element to its prior distribution.
        
        Args:
            i (int): gate index
            j (int): dot index
        """
        if callable(self.prior_config):
            prior_mean, prior_var = self.prior_config(i, j)
        else:
            prior_mean, prior_var = self.prior_config.get((i, j), (0.25, 0.1))
        
        self.means[i, j] = prior_mean
        self.variances[i, j] = prior_var


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Dictionary-based prior configuration
    n_dots = 5
    prior_dict = {}
    
    for i in range(n_dots):
        for j in range(n_dots):
            if i == j:
                prior_dict[(i, j)] = (1, 0.01)  # Self-capacitance
            elif abs(i - j) == 1:
                prior_dict[(i, j)] = (0.40, 0.2)  # Nearest neighbors
            elif abs(i - j) == 2:
                prior_dict[(i, j)] = (0.2, 0.1)   # Distant pairs
            else:
                prior_dict[(i, j)] = (0., 0.1)   # Distant pairs
    
    predictor = BayesianCapacitancePredictor(n_dots, prior_dict)
    
    # Simulating ML model outputs: [Cgd[1,2], Cgd[1,3], Cgd[0,2]]
    ml_estimates = [0.23, 0.18, 0.31]  # Capacitance predictions
    log_variances = [-2.3, -1.9, -2.5]  # Log variance from uncertainty head
    predictor.update_from_scan((1,2), [(ml_estimates[0], log_variances[0]), (ml_estimates[1], log_variances[1]), (ml_estimates[2], log_variances[2])])
    
    # Get specific capacitance stats
    mean, var = predictor.get_capacitance_stats(1, 2)
    print(f"C(1,2): mean={mean:.4f}, variance={var:.6f}")
    
    # Get confidence interval
    lower, upper = predictor.get_confidence_interval(1, 2)
    print(f"95% CI for C(1,2): [{lower:.4f}, {upper:.4f}]")
    
    # Get full matrix
    full_matrix = predictor.get_full_matrix()
    print(f"Full capacitance matrix:\n{full_matrix}")
