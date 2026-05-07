"""
Convergence tracking for benchmark runners.

Provides a common interface for all benchmark runners to track and store
plunger and barrier distances consistently. Loads config values and prints
them clearly to avoid silent failures.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from env_init import get_voltage_ranges_from_config, ENV_CONFIG_PATH


@dataclass
class ConvergenceTracker:
    """
    Tracks convergence metrics throughout optimization.

    Used by all benchmark runners to ensure consistent data storage.
    Stores plunger and barrier distances separately for flexible analysis.
    Also stores scan numbers for each record to enable consistent x-axis plotting.
    """

    num_plungers: int
    num_barriers: int
    plunger_range: float  # from config - max possible distance per plunger
    barrier_range: float  # from config - max possible distance per barrier

    # History of total distances at each step, with corresponding scan numbers
    scan_numbers: List[int] = field(default_factory=list)  # scan number for each record
    plunger_distance_history: List[float] = field(default_factory=list)
    barrier_distance_history: List[float] = field(default_factory=list)

    @classmethod
    def from_env(cls, env) -> "ConvergenceTracker":
        """
        Create tracker from environment, loading config values.

        Args:
            env: QuantumDeviceEnv instance

        Returns:
            Initialized ConvergenceTracker

        Raises:
            FileNotFoundError: If env_config.yaml not found
            KeyError: If required config keys missing
        """
        plunger_range, barrier_range = cls._load_voltage_ranges()

        num_plungers = env.num_plunger_voltages
        num_barriers = env.num_barrier_voltages
        max_dist = plunger_range * num_plungers + barrier_range * num_barriers

        print(f"ConvergenceTracker initialized:")
        print(f"  Plungers: {num_plungers} x {plunger_range}V range")
        print(f"  Barriers: {num_barriers} x {barrier_range}V range")
        print(f"  Max possible distance: {max_dist:.1f}V")

        return cls(
            num_plungers=num_plungers,
            num_barriers=num_barriers,
            plunger_range=plunger_range,
            barrier_range=barrier_range,
        )

    @staticmethod
    def _load_voltage_ranges() -> tuple:
        """
        Load voltage ranges from centralized env config.

        Returns:
            (plunger_range, barrier_range) - midpoint of min/max config values

        Raises:
            FileNotFoundError: If config file not found
        """
        print(f"Loading voltage ranges from: {ENV_CONFIG_PATH}")
        plunger_range, barrier_range = get_voltage_ranges_from_config()
        print(f"  Plunger range: {plunger_range}V")
        print(f"  Barrier range: {barrier_range}V")
        return plunger_range, barrier_range

    def record(self, plunger_dists: np.ndarray, barrier_dists: np.ndarray, scan_number: int):
        """
        Record distances from a single evaluation step.

        Args:
            plunger_dists: Array of absolute distances for each plunger
            barrier_dists: Array of absolute distances for each barrier
            scan_number: The scan number (x-axis value for plotting)
        """
        self.scan_numbers.append(scan_number)
        self.plunger_distance_history.append(float(np.sum(plunger_dists)))
        self.barrier_distance_history.append(float(np.sum(barrier_dists)))

    def reset(self):
        """Clear history for a new trial."""
        self.scan_numbers = []
        self.plunger_distance_history = []
        self.barrier_distance_history = []

    @property
    def max_possible_distance(self) -> float:
        """Maximum possible total distance given config ranges."""
        return (self.plunger_range * self.num_plungers) + (self.barrier_range * self.num_barriers)

    def get_total_distance_history(self) -> List[float]:
        """Get total distance (plunger + barrier) at each step."""
        return [
            p + b
            for p, b in zip(self.plunger_distance_history, self.barrier_distance_history)
        ]

    def get_normalized_history(self) -> List[float]:
        """
        Get normalized convergence score history.

        Returns:
            List of scores where 0 = max distance (worst), 1 = converged
        """
        max_dist = self.max_possible_distance
        return [
            1.0 - (p + b) / max_dist
            for p, b in zip(self.plunger_distance_history, self.barrier_distance_history)
        ]

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "num_plungers": self.num_plungers,
            "num_barriers": self.num_barriers,
            "plunger_range": self.plunger_range,
            "barrier_range": self.barrier_range,
            "scan_numbers": self.scan_numbers,
            "plunger_distance_history": self.plunger_distance_history,
            "barrier_distance_history": self.barrier_distance_history,
        }

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.plunger_distance_history)


if __name__ == "__main__":
    # Quick test
    print("Testing ConvergenceTracker config loading...")
    plunger_range, barrier_range = ConvergenceTracker._load_voltage_ranges()
    print(f"\nLoaded successfully:")
    print(f"  plunger_range = {plunger_range}")
    print(f"  barrier_range = {barrier_range}")
