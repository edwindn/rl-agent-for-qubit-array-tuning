import numpy as np

from typing import Tuple


def score_functions(plunger_voltages: np.ndarray, barrier_voltages: np.ndarray,
                    plunger_gt: np.ndarray, barrier_gt: np.ndarray) -> Tuple[float]:
    """
    Metric to minimise
    """

    plunger_score = ((plunger_voltages - plunger_gt)**2).sum()
    barrier_score = ((barrier_voltages - barrier_gt)**2).sum()

    return plunger_score, barrier_score


def normalize_voltages(voltages, v_min, v_max):
    """
    Normalize voltages from physical range to [-1, 1] for environment action space.

    Args:
        voltages: Physical voltage values
        v_min: Minimum voltage range
        v_max: Maximum voltage range

    Returns:
        Normalized voltages in [-1, 1]
    """
    normalized = (voltages - v_min) / (v_max - v_min)  # [0, 1]
    normalized = normalized * 2 - 1  # [-1, 1]
    return normalized