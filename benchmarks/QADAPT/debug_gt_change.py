"""
Debug: Check if GT changes are causing apparent failures.
"""

import sys
from pathlib import Path
import numpy as np
import torch

benchmarks_dir = Path(__file__).parent.parent
project_root = benchmarks_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(benchmarks_dir))
sys.path.insert(0, str(src_dir))

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
from policy_loader import load_policies, get_deterministic_action


def run_episode(seed=42, max_steps=10):
    env = MultiAgentEnvWrapper(training=True, return_voltage=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = load_policies("../../artifacts/rl_checkpoint_best:v3482", device=device)

    obs_dict, _ = env.reset(seed=seed)

    # Get INITIAL ground truth
    initial_gt = env.base_env.device_state["gate_ground_truth"].copy()
    initial_v = env.base_env.device_state["current_gate_voltages"].copy()

    print(f"Initial GT: {initial_gt}")
    print(f"Initial V:  {initial_v}")
    print(f"Initial dist from initial GT: {np.abs(initial_v - initial_gt)}")
    print()

    for step in range(max_steps):
        # Get actions
        actions = {}
        for agent_id, agent_obs in obs_dict.items():
            policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"
            torch_obs = {
                "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
                "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
            }
            action = get_deterministic_action(policies[policy_name], torch_obs)
            actions[agent_id] = action.cpu().numpy().flatten()

        obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)

        current_v = env.base_env.device_state["current_gate_voltages"]
        current_gt = env.base_env.device_state["gate_ground_truth"]

        # Distance from CURRENT gt (what policy should track)
        dist_current_gt = np.abs(current_v - current_gt)
        # Distance from INITIAL gt (fixed target)
        dist_initial_gt = np.abs(current_v - initial_gt)

        print(f"Step {step+1}:")
        print(f"  Current GT: {current_gt}")
        print(f"  GT change:  {current_gt - initial_gt}")
        print(f"  Current V:  {current_v}")
        print(f"  Dist from CURRENT GT: {dist_current_gt} (sum: {np.sum(dist_current_gt):.2f})")
        print(f"  Dist from INITIAL GT: {dist_initial_gt} (sum: {np.sum(dist_initial_gt):.2f})")
        print()

        if terminateds.get("__all__", False):
            break


if __name__ == "__main__":
    print("="*70)
    print("Testing how GT changes affect apparent convergence")
    print("="*70 + "\n")
    run_episode(seed=42, max_steps=5)
