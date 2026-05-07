"""
Shared environment initialization for benchmarks.

Provides a consistent interface for creating quantum device environments
across all benchmark methods (Nelder-Mead, Bayesian, RL, etc.)
"""

import sys
from pathlib import Path
import numpy as np
import yaml

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv

# =============================================================================
# Centralized config path - change this to use a different env config
# =============================================================================
ENV_CONFIG_PATH = src_dir / "swarm" / "environment" / "env_config.yaml"


def load_env_config() -> dict:
    """
    Load the environment config from the centralized path.

    Returns:
        Config dictionary

    Raises:
        FileNotFoundError: If config file not found
    """
    if not ENV_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Environment config not found: {ENV_CONFIG_PATH}")

    with open(ENV_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_voltage_ranges_from_config() -> tuple:
    """
    Get plunger and barrier voltage ranges from config.

    Returns:
        (plunger_range, barrier_range) - midpoint of min/max config values
    """
    config = load_env_config()

    plunger_cfg = config["simulator"]["full_plunger_range_width"]
    barrier_cfg = config["simulator"]["full_barrier_range_width"]

    plunger_range = (plunger_cfg["min"] + plunger_cfg["max"]) / 2
    barrier_range = (barrier_cfg["min"] + barrier_cfg["max"]) / 2

    return plunger_range, barrier_range


def create_benchmark_env(
    num_dots: int = 2,
    use_barriers: bool = True,
    seed: int = None,
    capacitance_model: str = None,
) -> QuantumDeviceEnv:
    """
    Create a quantum device environment for benchmarking.

    Args:
        num_dots: Number of quantum dots in the array
        use_barriers: Whether to include barrier voltage optimization
        seed: Random seed for reproducibility
        capacitance_model: Capacitance update method ("perfect", "fake", None)

    Returns:
        QuantumDeviceEnv instance with ground truth accessible via device_state
    """
    # Create env using centralized config path
    env = QuantumDeviceEnv(
        training=True,
        config_path=str(ENV_CONFIG_PATH),
        num_dots=num_dots,
        use_barriers=use_barriers,
    )

    # Update capacitance model setting
    if capacitance_model is not None:
        env.capacitance_model = capacitance_model
    else:
        env.capacitance_model = None  # No updates, use initial

    # Reset with seed
    obs, info = env.reset(seed=int(seed) if seed is not None else None)

    return env


def get_ground_truth(env: QuantumDeviceEnv) -> tuple:
    """
    Extract ground truth voltages from environment.

    Args:
        env: QuantumDeviceEnv instance

    Returns:
        (plunger_gt, barrier_gt): Ground truth voltage arrays
    """
    plunger_gt = env.device_state["gate_ground_truth"]
    barrier_gt = env.device_state["barrier_ground_truth"]
    return plunger_gt, barrier_gt


def get_voltage_ranges(env: QuantumDeviceEnv, rng: np.random.Generator = None) -> dict:
    """
    Get voltage ranges from environment's internal bounds.

    Returns the env's internal voltage bounds to ensure benchmarks
    operate in the same voltage space as RL training.

    Args:
        env: QuantumDeviceEnv instance
        rng: Unused, kept for backward compatibility

    Returns:
        dict with plunger_min, plunger_max, barrier_min, barrier_max
    """
    return {
        "plunger_min": env.plunger_min,
        "plunger_max": env.plunger_max,
        "barrier_min": env.barrier_min,
        "barrier_max": env.barrier_max,
    }


def get_current_voltages(env: QuantumDeviceEnv) -> tuple:
    """
    Get current voltages from environment state.

    Args:
        env: QuantumDeviceEnv instance

    Returns:
        (plunger_v, barrier_v): Current voltage arrays
    """
    plunger_v = env.device_state["current_gate_voltages"]
    barrier_v = env.device_state["current_barrier_voltages"]
    return plunger_v, barrier_v


def random_initial_voltages(env: QuantumDeviceEnv, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate random initial voltages within the environment's voltage ranges.

    Args:
        env: QuantumDeviceEnv instance
        rng: Random number generator (optional)

    Returns:
        Concatenated array of [plunger_voltages, barrier_voltages]
    """
    if rng is None:
        rng = np.random.default_rng()

    ranges = get_voltage_ranges(env)

    plunger_v = rng.uniform(
        low=ranges["plunger_min"],
        high=ranges["plunger_max"],
        size=env.num_plunger_voltages
    )

    barrier_v = rng.uniform(
        low=ranges["barrier_min"],
        high=ranges["barrier_max"],
        size=env.num_barrier_voltages
    )

    return np.concatenate([plunger_v, barrier_v])


def split_voltages(voltages: np.ndarray, num_plungers: int) -> tuple:
    """
    Split concatenated voltage array into plunger and barrier components.

    Args:
        voltages: Concatenated [plunger, barrier] array
        num_plungers: Number of plunger voltages

    Returns:
        (plunger_v, barrier_v): Split voltage arrays
    """
    plunger_v = voltages[:num_plungers]
    barrier_v = voltages[num_plungers:]
    return plunger_v, barrier_v


if __name__ == "__main__":
    # Quick test
    env = create_benchmark_env(num_dots=2, use_barriers=True, seed=42)
    plunger_gt, barrier_gt = get_ground_truth(env)
    print(f"Plunger ground truth: {plunger_gt}")
    print(f"Barrier ground truth: {barrier_gt}")
    print(f"Voltage ranges: {get_voltage_ranges(env)}")
