"""
Build per-agent observations from QuantumDeviceEnv state.

Replicates the observation building logic from multi_agent_wrapper.py
to construct observations that match what the trained policies expect.
"""

import numpy as np
from typing import Dict, List, Tuple


def build_agent_channel_map(num_dots: int) -> Tuple[Dict[str, List[int]], List[str], List[str]]:
    """
    Build channel assignment map for agents.

    Args:
        num_dots: Number of quantum dots

    Returns:
        Tuple of (channel_map, plunger_agent_ids, barrier_agent_ids)
    """
    num_plungers = num_dots
    num_barriers = num_dots - 1
    num_csds = num_dots - 1  # Number of charge stability diagrams

    channel_map = {}
    plunger_agent_ids = []
    barrier_agent_ids = []

    # Plunger agents: 2 channels each (adjacent CSDs)
    for i in range(num_plungers):
        agent_id = f"plunger_{i}"
        plunger_agent_ids.append(agent_id)

        if i == 0:
            # First plunger: duplicate first channel
            channel_map[agent_id] = [0, 0]
        elif i == num_plungers - 1:
            # Last plunger: duplicate last channel
            channel_map[agent_id] = [num_csds - 1, num_csds - 1]
        else:
            # Middle plungers: adjacent pair
            channel_map[agent_id] = [i - 1, i]

    # Barrier agents: 1 channel each
    for i in range(num_barriers):
        agent_id = f"barrier_{i}"
        barrier_agent_ids.append(agent_id)
        channel_map[agent_id] = [i]

    return channel_map, plunger_agent_ids, barrier_agent_ids


def build_observations(
    global_image: np.ndarray,
    plunger_voltages: np.ndarray,
    barrier_voltages: np.ndarray,
    plunger_min: np.ndarray,
    plunger_max: np.ndarray,
    barrier_min: np.ndarray,
    barrier_max: np.ndarray,
    channel_map: Dict[str, List[int]],
    plunger_agent_ids: List[str],
    barrier_agent_ids: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build per-agent observations from global environment state.

    Args:
        global_image: CSD image stack of shape (H, W, num_csds)
        plunger_voltages: Current plunger voltages (absolute values)
        barrier_voltages: Current barrier voltages (absolute values)
        plunger_min: Min plunger voltage bounds
        plunger_max: Max plunger voltage bounds
        barrier_min: Min barrier voltage bounds
        barrier_max: Max barrier voltage bounds
        channel_map: Agent to channel assignment
        plunger_agent_ids: List of plunger agent IDs
        barrier_agent_ids: List of barrier agent IDs

    Returns:
        Dictionary mapping agent_id to observation dict with 'image' and 'voltage'
    """
    observations = {}
    num_plungers = len(plunger_agent_ids)

    # Normalize voltages to [-1, 1]
    plunger_voltages_norm = 2 * (plunger_voltages - plunger_min) / (plunger_max - plunger_min) - 1
    barrier_voltages_norm = 2 * (barrier_voltages - barrier_min) / (barrier_max - barrier_min) - 1

    # Build plunger observations
    for agent_id in plunger_agent_ids:
        agent_idx = int(agent_id.split("_")[1])
        channels = channel_map[agent_id]

        img1 = global_image[:, :, channels[0]]
        img2 = global_image[:, :, channels[1]]

        # Apply transformations based on position (from multi_agent_wrapper.py)
        if agent_idx == 0:
            # First agent: no flipping
            agent_image = np.stack([img1, img2], axis=2)
        elif agent_idx == num_plungers - 1:
            # Final agent: flip both images (transpose)
            img1 = np.transpose(img1, (1, 0))
            img2 = np.transpose(img2, (1, 0))
            agent_image = np.stack([img1, img2], axis=2)
        else:
            # Middle agents: flip only second image
            img2 = np.transpose(img2, (1, 0))
            agent_image = np.stack([img1, img2], axis=2)

        observations[agent_id] = {
            "image": agent_image.astype(np.float32),
            "voltage": np.array([plunger_voltages_norm[agent_idx]], dtype=np.float32),
        }

    # Build barrier observations
    for agent_id in barrier_agent_ids:
        agent_idx = int(agent_id.split("_")[1])
        channels = channel_map[agent_id]

        # Barrier agents get 1 channel
        agent_image = global_image[:, :, channels[0]:channels[0] + 1]

        observations[agent_id] = {
            "image": agent_image.astype(np.float32),
            "voltage": np.array([barrier_voltages_norm[agent_idx]], dtype=np.float32),
        }

    return observations


def observations_to_torch(
    observations: Dict[str, Dict[str, np.ndarray]],
    device: str = "cuda",
) -> Dict[str, Dict[str, "torch.Tensor"]]:
    """
    Convert numpy observations to batched torch tensors.

    Args:
        observations: Dictionary of agent observations (numpy)
        device: Torch device

    Returns:
        Dictionary with torch tensors, batch dimension added
    """
    import torch

    torch_obs = {}
    for agent_id, obs in observations.items():
        torch_obs[agent_id] = {
            "image": torch.tensor(obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
            "voltage": torch.tensor(obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
        }
    return torch_obs


def actions_to_voltages(
    actions: Dict[str, np.ndarray],
    current_plunger_v: np.ndarray,
    current_barrier_v: np.ndarray,
    plunger_min: np.ndarray,
    plunger_max: np.ndarray,
    barrier_min: np.ndarray,
    barrier_max: np.ndarray,
    voltage_step_plunger: float,
    voltage_step_barrier: float,
    plunger_agent_ids: List[str],
    barrier_agent_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert policy actions to voltage updates.

    Actions are in [-1, 1] and represent direction/magnitude of voltage change.
    The actual voltage change is: action * voltage_step.

    Args:
        actions: Dictionary mapping agent_id to action value [-1, 1]
        current_plunger_v: Current plunger voltages
        current_barrier_v: Current barrier voltages
        plunger_min: Min plunger voltage bounds
        plunger_max: Max plunger voltage bounds
        barrier_min: Min barrier voltage bounds
        barrier_max: Max barrier voltage bounds
        voltage_step_plunger: Max voltage step per action
        voltage_step_barrier: Max voltage step per action
        plunger_agent_ids: List of plunger agent IDs
        barrier_agent_ids: List of barrier agent IDs

    Returns:
        Tuple of (new_plunger_voltages, new_barrier_voltages)
    """
    new_plunger_v = current_plunger_v.copy()
    new_barrier_v = current_barrier_v.copy()

    # Apply plunger actions
    for agent_id in plunger_agent_ids:
        if agent_id in actions:
            i = int(agent_id.split("_")[1])
            action = actions[agent_id]
            if hasattr(action, "__len__"):
                action = float(action[0])
            else:
                action = float(action)
            # Clip action to [-1, 1]
            action = np.clip(action, -1.0, 1.0)
            # Apply voltage step
            new_plunger_v[i] += action * voltage_step_plunger
            # Clip to bounds
            new_plunger_v[i] = np.clip(new_plunger_v[i], plunger_min[i], plunger_max[i])

    # Apply barrier actions
    for agent_id in barrier_agent_ids:
        if agent_id in actions:
            i = int(agent_id.split("_")[1])
            action = actions[agent_id]
            if hasattr(action, "__len__"):
                action = float(action[0])
            else:
                action = float(action)
            # Clip action to [-1, 1]
            action = np.clip(action, -1.0, 1.0)
            # Apply voltage step
            new_barrier_v[i] += action * voltage_step_barrier
            # Clip to bounds
            new_barrier_v[i] = np.clip(new_barrier_v[i], barrier_min[i], barrier_max[i])

    return new_plunger_v, new_barrier_v


if __name__ == "__main__":
    # Quick test
    num_dots = 2
    channel_map, plunger_ids, barrier_ids = build_agent_channel_map(num_dots)

    print(f"Num dots: {num_dots}")
    print(f"Channel map: {channel_map}")
    print(f"Plunger IDs: {plunger_ids}")
    print(f"Barrier IDs: {barrier_ids}")

    # Test with dummy data
    image = np.random.rand(128, 128, num_dots - 1).astype(np.float32)
    plunger_v = np.array([-50.0, -40.0])
    barrier_v = np.array([2.0])

    obs = build_observations(
        global_image=image,
        plunger_voltages=plunger_v,
        barrier_voltages=barrier_v,
        plunger_min=np.array([-100.0, -100.0]),
        plunger_max=np.array([0.0, 0.0]),
        barrier_min=np.array([0.0]),
        barrier_max=np.array([10.0]),
        channel_map=channel_map,
        plunger_agent_ids=plunger_ids,
        barrier_agent_ids=barrier_ids,
    )

    for agent_id, agent_obs in obs.items():
        print(f"{agent_id}: image={agent_obs['image'].shape}, voltage={agent_obs['voltage']}")
