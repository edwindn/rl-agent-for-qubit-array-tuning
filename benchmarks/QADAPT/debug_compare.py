"""
Debug: Compare benchmark inference with training inference to find discrepancy.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add paths
benchmarks_dir = Path(__file__).parent.parent
project_root = benchmarks_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(benchmarks_dir))
sys.path.insert(0, str(src_dir))

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
from policy_loader import load_policies, get_deterministic_action


def debug_single_step(seed=42):
    """Debug a single step to understand what's happening."""

    print("="*70)
    print(f"DEBUGGING SEED {seed}")
    print("="*70)

    # Create env
    env = MultiAgentEnvWrapper(training=True, return_voltage=True)

    # Load policies
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = load_policies("../../artifacts/rl_checkpoint_best:v3482", device=device)

    # Reset
    obs_dict, info = env.reset(seed=seed)

    # Get initial state
    plunger_v = env.base_env.device_state["current_gate_voltages"]
    plunger_gt = env.base_env.device_state["gate_ground_truth"]
    plunger_min = env.base_env.plunger_min
    plunger_max = env.base_env.plunger_max

    print(f"\nInitial state:")
    print(f"  Plunger voltages: {plunger_v}")
    print(f"  Plunger GT: {plunger_gt}")
    print(f"  Plunger min: {plunger_min}")
    print(f"  Plunger max: {plunger_max}")
    print(f"  Plunger range widths: {plunger_max - plunger_min}")

    # Check observations
    print(f"\nObservations for plunger_0:")
    obs = obs_dict["plunger_0"]
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Image min/max: {obs['image'].min():.4f} / {obs['image'].max():.4f}")
    print(f"  Voltage obs: {obs['voltage']}")

    # Calculate what voltage obs SHOULD be
    # From env: voltage is normalized to [-1, 1] based on plunger range
    expected_voltage_obs = 2 * (plunger_v[0] - plunger_min[0]) / (plunger_max[0] - plunger_min[0]) - 1
    print(f"  Expected voltage obs: {expected_voltage_obs:.4f}")
    print(f"  Actual voltage obs: {obs['voltage'][0]:.4f}")
    print(f"  Match: {np.isclose(expected_voltage_obs, obs['voltage'][0], atol=0.01)}")

    # Get policy action
    torch_obs = {
        "image": torch.tensor(obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
        "voltage": torch.tensor(obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    action = get_deterministic_action(policies["plunger_policy"], torch_obs)
    action_val = action.cpu().numpy().flatten()[0]

    print(f"\nPolicy action for plunger_0: {action_val:.4f}")

    # Calculate what voltage this action maps to
    # From env._rescale_gate_voltages: action in [-1,1] -> voltage in [min, max]
    # voltage = (action + 1) / 2 * (max - min) + min
    target_voltage = (action_val + 1) / 2 * (plunger_max[0] - plunger_min[0]) + plunger_min[0]
    print(f"  This maps to target voltage: {target_voltage:.2f}")
    print(f"  Ground truth is: {plunger_gt[0]:.2f}")
    print(f"  Current voltage is: {plunger_v[0]:.2f}")

    # What action SHOULD the policy output to reach GT?
    # target_voltage = GT, so action = 2 * (GT - min) / (max - min) - 1
    ideal_action = 2 * (plunger_gt[0] - plunger_min[0]) / (plunger_max[0] - plunger_min[0]) - 1
    print(f"\n  Ideal action to reach GT: {ideal_action:.4f}")
    print(f"  Actual action: {action_val:.4f}")
    print(f"  Action error: {abs(action_val - ideal_action):.4f}")

    # Is GT even reachable within the voltage range?
    gt_in_range = (plunger_gt[0] >= plunger_min[0]) and (plunger_gt[0] <= plunger_max[0])
    print(f"\n  GT in voltage range: {gt_in_range}")
    if not gt_in_range:
        print(f"  WARNING: Ground truth {plunger_gt[0]:.2f} is OUTSIDE range [{plunger_min[0]:.2f}, {plunger_max[0]:.2f}]!")

    # Step and see what happens
    actions = {}
    for agent_id, agent_obs in obs_dict.items():
        policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"
        torch_obs = {
            "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
            "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
        }
        act = get_deterministic_action(policies[policy_name], torch_obs)
        actions[agent_id] = act.cpu().numpy().flatten()

    obs_dict2, rewards, _, _, _ = env.step(actions)

    new_plunger_v = env.base_env.device_state["current_gate_voltages"]
    new_plunger_gt = env.base_env.device_state["gate_ground_truth"]

    print(f"\nAfter step:")
    print(f"  New plunger voltages: {new_plunger_v}")
    print(f"  New plunger GT: {new_plunger_gt}")
    print(f"  GT changed: {not np.allclose(plunger_gt, new_plunger_gt)}")

    old_dist = np.abs(plunger_v - plunger_gt)
    new_dist = np.abs(new_plunger_v - new_plunger_gt)
    print(f"\n  Old distances: {old_dist}")
    print(f"  New distances: {new_dist}")
    print(f"  Improved: {np.sum(new_dist) < np.sum(old_dist)}")

    return np.sum(new_dist) < np.sum(old_dist)


if __name__ == "__main__":
    # Test multiple seeds
    results = []
    for seed in [1, 2, 3, 42, 100, 200]:
        try:
            improved = debug_single_step(seed)
            results.append((seed, improved))
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            results.append((seed, None))
        print("\n")

    print("="*70)
    print("SUMMARY")
    print("="*70)
    for seed, improved in results:
        status = "IMPROVED" if improved else "WORSENED" if improved is False else "ERROR"
        print(f"  Seed {seed}: {status}")
