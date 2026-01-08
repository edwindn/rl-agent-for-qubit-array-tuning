import numpy as np
from typing import List, Tuple


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