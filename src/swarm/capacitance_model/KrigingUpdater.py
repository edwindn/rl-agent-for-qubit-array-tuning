from typing import Dict, Tuple, List, Union, Callable
import numpy as np
import math
import warnings
from functools import partial

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from swarm.capacitance_model.capacitance_utils import get_channel_targets
get_channel_targets = partial(get_channel_targets, has_sensor=False)


class InterpolatedCapacitancePredictor:
    """
    Spatially aware parameter estimation for varying capacitance values in a quantum dot array.
    
    This class maintains a history of capacitance estimates and their uncertainties
    at the points that we have sampled so far. It then calculates an optimal prior at
    any new location based on an interpolation between previously seen values.

    The update at the new point is then done using regular Bayesian update
    with the model's prediction.
    
    The class provides direct integration with trained neural networks that output
    log variance for uncertainty estimation. Use update_from_ml_model() for direct
    integration with your trained capacitance prediction model.

    Note the voltages we use for our mapping should be the actual physical voltages, not virtualised ones
    
    Attributes:
        n_dots (int): Number of quantum dots in the array
        means (np.ndarray): list of (coordinates, NxN matrix) pairs of posterior means
        variances (np.ndarray): list of (coordinates, NxN matrix) pairs of posterior errors
        prior_config (Dict or Callable): Configuration for prior distributions
        history: Dict mapping scan index to a list of (coordinates, means, variances) previous points
    """

    def __init__(
        self,
        n_dots: int,
        nn: int,
        prior_config: Union[Dict[Tuple[int, int], Tuple[float, float]], Callable],
        length_scale: float = 0.5, # has dimensions of voltage, hyperparameter
        noise_level: float = 1e-4,
        max_points_to_consider: int = 20,
    ):
        raise NotImplementedError("Refactor requires modifications to support Cgd prediction.")
        self.n_dots = n_dots
        self.max_points_to_consider = max_points_to_consider
        self.prior_config = prior_config

        
        # In the absence of any data, we will use these priors as the best guess we have
        # This will also be updated as the running best estimate for the capacitances
        self.means = np.zeros((n_dots, n_dots))
        self.variances = np.zeros((n_dots, n_dots))
        self._initialize_priors()
        self._validate_initialization()

        self.default_prior_means = self.means
        self.default_prior_vars = self.variances


        # note - dot pairs are 0-indexed
        dots = list(range(n_dots))
        
        self.history = {k: [] for k in dots[:-1]} # we use the leftmost dot as the scan index (same convention as get_channel_targets)
        # each entry in history will be a tuple of (coordinates, means, variances) for that dot pair

        # initialise the RBF kernel
        kernel = RBF(length_scale=length_scale)
        self.gaussian_process = partial(GaussianProcessRegressor, kernel=kernel, normalize_y=True, optimizer=None)


    def _initialize_priors(self):
        """Initialize the prior distributions for all matrix elements."""
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
        # Check symmetry
        if not np.allclose(self.means, self.means.T):
            raise ValueError("Mean matrix is not symmetric")
        if not np.allclose(self.variances, self.variances.T):
            raise ValueError("Variance matrix is not symmetric")
        
        # Check positive variances
        if np.any(self.variances <= 0):
            raise ValueError("All variances must be positive")

        
    def _compute_kriging_prior(self, scan_idx: int, new_coords: Tuple[float, float], max_points: int = 20,
                               signal_variance: float = 1.0, jitter: float = 1e-6) -> Tuple[List, List]:
        """Computes the prior mean and variance at a new data point based on past history"""
        if scan_idx not in self.history:
            raise ValueError(f"Invalid scan index {scan_idx}")

        history = self.history[scan_idx]

        prior_means = get_channel_targets(scan_idx, self.default_prior_means, self.n_dots)
        prior_vars = get_channel_targets(scan_idx, self.default_prior_vars, self.n_dots)

        if not history:
            # return the correcty indexed default prior means and vars
            return prior_means, prior_vars

        x_star = np.asarray(new_coords, dtype=float).reshape(1, 2) # query entry point

        coords = []
        obs_means = [[], [], []]
        obs_vars = [[], [], []]

        for entry in history:
            coord, means, vars_ = entry
            coords.append(tuple(coord))
            for k in range(3):
                obs_means[k].append(float(means[k]))
                obs_vars[k].append(float(vars_[k]))

        X = np.asarray(coords, dtype=float) # (n, 2)

        if X.shape[0] > max_points:
            # choose the nearest max_points data points
            norm = np.sum((X - x_star)**2, axis=1)
            idxs = np.argsort(norm)[:max_points]
            X = X[idxs, :]
            for k in range(3):
                obs_means[k] = [obs_means[k][i] for i in idxs]
                obs_vars[k] = [obs_vars[k][i] for i in idxs]


        # perform kriging update for each of the 3 capacitance values
        for k in range(3):
            y = np.asarray(obs_means[k])
            noise = np.asarray(obs_vars[k])

            alpha = noise + jitter
            gp = self.gaussian_process(alpha=alpha)
            gp.fit(X, y)
            mu_star, std_star = gp.predict(x_star, return_std=True)

            prior_means[k] = float(mu_star[0])
            prior_vars[k] = float(std_star[0]**2)

        return prior_means, prior_vars

    
    def _update_matrices(self, dot_pair: Tuple[int, int], new_mean: float, new_var: float):
        """Updates the model's mean and variances matrices based on the new estimates"""
        i, j = dot_pair

        assert j > i, "dot_pair must be ordered (i < j)"
        assert j - i in [1, 2], "dot_pair must be adjacent or next-adjacent dots"

        if i < 0 or j >= self.n_dots:
            if i == -1 or j == self.n_dots:
                # handles dot pairs at the two ends
                return
            else:
                raise ValueError(f"Invalid dot indices ({i}, {j}) for matrix of size {self.n_dots}")

        # symmetric update
        self.means[i, j] = new_mean
        self.means[j, i] = new_mean
        self.variances[i, j] = new_var
        self.variances[j, i] = new_var
    

    def bayesian_update(self, mean: float, var: float, prior_mean: float, prior_var: float):
        """
        Perform conjugate Bayesian update for a single capacitance element.
        
        Uses the conjugate prior property of Gaussian distributions to update
        the posterior mean and variance given a new measurement.
        """
        var += 1e-8
        prior_var += 1e-8

        # Validate inputs
        assert var > 0, "Measurement variance must be positive"
        assert prior_var > 0, "Prior variance must be positive"
        
        # Conjugate update formulas
        precision_prior = 1 / prior_var
        precision_ml = 1 / var
        precision_post = precision_prior + precision_ml
        
        new_mean = (prior_mean * precision_prior + mean * precision_ml) / precision_post
        new_var = 1 / precision_post
        
        # Sanity check: posterior variance should not increase
        if new_var > prior_var:
            warnings.warn(f"Posterior variance increased for element ({i},{j}). "
                         f"Old: {prior_var:.6f}, New: {new_var:.6f}")

        return new_mean, new_var


    def update_from_scan(self, dot_pair: Tuple[int, int], voltages: Tuple[float, float], ml_outputs: List[Tuple[float, float]]):
        """
        Process ML model output for a dot pair scan and update relevant capacitances.
        
        Args:
            dot_pair (Tuple[int, int]): The measured dot pair (i, j)
            ml_outputs (List[Tuple[float, float]]): List of 3 tuples containing
                (capacitance_estimate, log_variance) for:
                - C_ij: Capacitance between the measured dots
                - C_ik: Capacitance between dot i and neighboring dot k (k = j+1)
                - C_jk: Capacitance between dot j and neighboring dot k (k = i-1)
        
        Example:
            predictor.update_from_scan(
                dot_pair=(2, 3),
                ml_outputs=[(0.23, -2.3), (0.18, -1.9), (0.31, -2.5)]
            )
        """
        if len(ml_outputs) != 3:
            raise ValueError("ml_outputs must contain exactly 3 measurements")
        
        i, j = dot_pair

        dot_pairs = [(i, j), (i, j+1), (i-1, j)]

        scan_idx = i
        assert scan_idx in self.history, f"Invalid scan index {scan_idx}"

        prior_means, prior_vars = self._compute_kriging_prior(scan_idx, voltages, max_points=self.max_points_to_consider)

        predicted_means, predicted_logvars = zip(*ml_outputs)
        predicted_vars = [math.exp(lv) for lv in predicted_logvars]

        new_means = []
        new_vars = []

        for idx, (mean, var, prior_mean, prior_var, dot_pair) in enumerate(zip(predicted_means, predicted_vars, prior_means, prior_vars, dot_pairs)):
            new_mean, new_var = self.bayesian_update(mean, var, prior_mean, prior_var)
            self._update_matrices(dot_pair, new_mean, new_var) # handles the edge cases internally

            new_means.append(new_mean)
            new_vars.append(new_var)

        assert idx == 2, "Expected exactly 3 updates, something went wrong"

        self.history[scan_idx].append((voltages, new_means, new_vars))


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
            i (int): First dot index
            j (int): Second dot index
        
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
        if return_variance:
            return self.means.copy(), self.variances.copy()
        else:
            return self.means.copy()
    
    def get_confidence_interval(self, i: int, j: int, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for a specific capacitance estimate.
        
        Args:
            i (int): First dot index
            j (int): Second dot index
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
            i (int): First dot index
            j (int): Second dot index
        """
        if callable(self.prior_config):
            prior_mean, prior_var = self.prior_config(i, j)
        else:
            prior_mean, prior_var = self.prior_config.get((i, j), (0.25, 0.1))
        
        self.means[i, j] = prior_mean
        self.means[j, i] = prior_mean
        self.variances[i, j] = prior_var
        self.variances[j, i] = prior_var
    
    def get_matrix_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of the current capacitance matrix.
        
        Returns:
            Dict[str, float]: Dictionary containing summary statistics
        """
        # Only consider upper triangular (excluding diagonal) to avoid double counting
        upper_tri_indices = np.triu_indices(self.n_dots, k=1)
        off_diagonal_means = self.means[upper_tri_indices]
        off_diagonal_vars = self.variances[upper_tri_indices]
        
        # Diagonal elements (self-capacitance)
        diagonal_means = np.diag(self.means)
        diagonal_vars = np.diag(self.variances)
        
        return {
            'off_diagonal_mean_avg': np.mean(off_diagonal_means),
            'off_diagonal_mean_std': np.std(off_diagonal_means),
            'off_diagonal_var_avg': np.mean(off_diagonal_vars),
            'diagonal_mean_avg': np.mean(diagonal_means),
            'diagonal_var_avg': np.mean(diagonal_vars),
            'total_uncertainty': np.sum(off_diagonal_vars) + np.sum(diagonal_vars)
        }
