import numpy as np
from typing import List, Tuple


def get_targets_with_nnn(channel_idx: int, cgd_matrix: np.ndarray, num_dots: int, has_sensor: bool = True) -> np.ndarray:
    """
    Extract 3 capacitance values: 1 NN (symmetric) + 2 NNN

    For channel i (scanning dots i and i+1):
    - NN: Cgd[i, i+1] (= Cgd[i+1, i] by symmetry, we just use one)
    - NNN_right: Cgd[i, i+2] (0 if i+2 >= num_dots, edge case)
    - NNN_left: Cgd[i+1, i-1] (0 if i-1 < 0, edge case)

    Args:
        channel_idx: index of leftmost dot (0 <= idx <= num_dots - 2)
        cgd_matrix: CGD matrix of shape (num_dots, num_dots+1) or (num_dots, num_dots)
        num_dots: Number of dots in the system
        has_sensor: Whether cgd_matrix includes sensor column

    Returns:
        np.ndarray of shape (3,): [nn, nnn_right, nnn_left]
    """
    assert channel_idx in list(range(num_dots - 1)), f"Out-of-bounds channel index given for {num_dots} dots."

    if has_sensor:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] - 1 == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots+1})"
    else:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots})"

    i = channel_idx

    # NN coupling: Cgd[i, i+1] (gate i+1 affects dot i)
    nn = float(cgd_matrix[i, i + 1])

    # NNN_right: Cgd[i, i+2] (gate i+2 affects dot i)
    if i + 2 < num_dots:
        nnn_right = float(cgd_matrix[i, i + 2])
    else:
        nnn_right = 0.0  # Edge case: no gate i+2

    # NNN_left: Cgd[i+1, i-1] (gate i-1 affects dot i+1)
    if i - 1 >= 0:
        nnn_left = float(cgd_matrix[i + 1, i - 1])
    else:
        nnn_left = 0.0  # Edge case: no gate i-1

    return np.array([nn, nnn_right, nnn_left], dtype=np.float32)


def get_nearest_targets(channel_idx: int, cgd_matrix: np.ndarray, num_dots: int, has_sensor: bool  = True) -> np.ndarray:
    """
    Extract capacitance values for just nearest-neighbour couplings (two values)

    Args:
        channel_idx: index of leftmost dot (0 <= idx <= num_dots - 2)

    Returns:
        tuple of (c1, c2) where
        c1 = coupling of gate idx+1 to dot idx
        c2 = coupling of gate idx to dot idx+1
    """

    assert channel_idx in list(range(num_dots-1)), f"Out-of-bounds channel index given for {num_dots} dots."

    if has_sensor:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] - 1 == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots+1})"
    else:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots})"

    c1 = float(cgd_matrix[channel_idx, channel_idx + 1])
    c2 = float(cgd_matrix[channel_idx + 1, channel_idx])

    return np.array([c1, c2], dtype=np.float32)


def get_channel_targets(channel_idx: int, cgd_matrix: np.ndarray, num_dots: int, has_sensor: bool = True) -> np.ndarray:
    """
    Get the target CGD values for a specific channel.
    
    For each channel (1-indexed as specified):
    - Channel 0 (image 1): [0, Cgd[1,2], Cgd[1,3]] (Cgd[0,2] doesn't exist, so use 0)
    - Channel 1 (image 2): [Cgd[2,3], Cgd[2,4], Cgd[1,3]]
    - Channel 2 (image 3): [Cgd[3,4], Cgd[3,5], Cgd[2,4]]
    - etc.
    
    Args:
        channel_idx: Index of the leftmost dot in the pair being swept (0-indexed)
        cgd_matrix: CGD matrix of shape (num_dots, num_dots+1)
        num_dots: Number of dots in the system
        
    Returns:
        List of 3 target values for this channel
    """

    assert channel_idx in list(range(num_dots-1)), f"Out-of-bounds channel index given for {num_dots} dots."

    if has_sensor:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] - 1 == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots+1})"
    else:
        assert cgd_matrix.shape[0] == cgd_matrix.shape[1] == num_dots, f"CGD matrix must have shape ({num_dots}, {num_dots})"

    # Extract the pairs of dots to consider for the channel index (0-indexed)
    left_pair = (channel_idx-1, channel_idx+1) # out of bounds for channel 0
    middle_pair = (channel_idx, channel_idx+1)
    right_pair = (channel_idx, channel_idx+2) # out of bounds for last channel
    
    targets = []

    if left_pair[0] < 0:
        targets.append(0.0)
    else:
        targets.append(float(cgd_matrix[left_pair[0], left_pair[1]]))

    targets.append(float(cgd_matrix[middle_pair[0], middle_pair[1]]))

    if right_pair[1] > num_dots - 1:
        targets.append(0.0)
    else:
        targets.append(float(cgd_matrix[right_pair[0], right_pair[1]]))

    # reorder to match expected model output: l, m, r -> m, r, l
    targets = np.array(targets, dtype=np.float32)[[1, 2, 0]]
    
    return targets