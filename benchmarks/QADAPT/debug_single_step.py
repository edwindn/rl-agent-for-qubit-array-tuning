"""Debug script to trace observation and action flow."""

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
from swarm.environment.env import QuantumDeviceEnv
from policy_loader import load_policies, get_deterministic_action

# Create env directly using MultiAgentEnvWrapper (same as training)
print("Creating environment...")
env = MultiAgentEnvWrapper(training=True, return_voltage=True)

print(f"\nEnvironment config:")
print(f"  num_dots: {env.num_gates}")
print(f"  use_deltas: {env.base_env.use_deltas}")
print(f"  resolution: {env.base_env.resolution}")

# Load policies
print("\nLoading policies...")
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "../../artifacts/rl_checkpoint_best:v3482"
policies = load_policies(checkpoint_path, device=device)
print(f"  Loaded: {list(policies.keys())}")

# Reset environment
print("\nResetting environment...")
obs_dict, info = env.reset(seed=42)

# Print initial state
print(f"\nInitial device state:")
print(f"  Plunger voltages: {env.base_env.device_state['current_gate_voltages']}")
print(f"  Barrier voltages: {env.base_env.device_state['current_barrier_voltages']}")
print(f"  Plunger ground truth: {env.base_env.device_state['gate_ground_truth']}")
print(f"  Barrier ground truth: {env.base_env.device_state['barrier_ground_truth']}")
print(f"  Plunger range: {env.base_env.plunger_min} to {env.base_env.plunger_max}")
print(f"  Barrier range: {env.base_env.barrier_min} to {env.base_env.barrier_max}")

# Calculate initial distances
plunger_dists = env.base_env.device_state['current_gate_voltages'] - env.base_env.device_state['gate_ground_truth']
barrier_dists = env.base_env.device_state['current_barrier_voltages'] - env.base_env.device_state['barrier_ground_truth']
init_obj = float(np.sum(plunger_dists**2) + np.sum(barrier_dists**2))
print(f"\nInitial distances:")
print(f"  Plunger: {plunger_dists}")
print(f"  Barrier: {barrier_dists}")
print(f"  Objective: {init_obj:.2f}")

# Check observation shapes and values
print("\nObservation check (plunger_0):")
agent_obs = obs_dict["plunger_0"]
print(f"  Image shape: {agent_obs['image'].shape}")
print(f"  Image range: [{agent_obs['image'].min():.4f}, {agent_obs['image'].max():.4f}]")
print(f"  Voltage: {agent_obs['voltage']}")

# Get policy action
torch_obs = {
    "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
    "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
}
print(f"\nTorch observation shapes:")
print(f"  image: {torch_obs['image'].shape}")
print(f"  voltage: {torch_obs['voltage'].shape}")

action = get_deterministic_action(policies["plunger_policy"], torch_obs)
print(f"\nPolicy action for plunger_0: {action.cpu().numpy()}")

# Get all actions
print("\nGetting all agent actions...")
actions = {}
for agent_id, agent_obs in obs_dict.items():
    policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"
    torch_obs = {
        "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
        "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    action = get_deterministic_action(policies[policy_name], torch_obs)
    actions[agent_id] = action.cpu().numpy().flatten()
    print(f"  {agent_id}: action={actions[agent_id][0]:.4f}, obs_voltage={agent_obs['voltage'][0]:.4f}")

# Step environment
print("\nStepping environment...")
obs_dict2, rewards, terminateds, truncateds, infos = env.step(actions)

# Print new state
print(f"\nNew device state:")
print(f"  Plunger voltages: {env.base_env.device_state['current_gate_voltages']}")
print(f"  Barrier voltages: {env.base_env.device_state['current_barrier_voltages']}")

# Calculate new distances
plunger_dists2 = env.base_env.device_state['current_gate_voltages'] - env.base_env.device_state['gate_ground_truth']
barrier_dists2 = env.base_env.device_state['current_barrier_voltages'] - env.base_env.device_state['barrier_ground_truth']
new_obj = float(np.sum(plunger_dists2**2) + np.sum(barrier_dists2**2))
print(f"\nNew distances:")
print(f"  Plunger: {plunger_dists2}")
print(f"  Barrier: {barrier_dists2}")
print(f"  Objective: {new_obj:.2f} (change: {new_obj - init_obj:+.2f})")

# Print rewards
print(f"\nRewards:")
for agent_id, reward in rewards.items():
    print(f"  {agent_id}: {reward:.4f}")

print("\n" + "="*60)
print("ANALYSIS:")
if new_obj < init_obj:
    print(f"  Objective IMPROVED by {init_obj - new_obj:.2f}")
else:
    print(f"  Objective WORSENED by {new_obj - init_obj:.2f}")
    print("  This suggests policy is not working correctly!")

# Run 5 more steps to see the trend
print("\n" + "="*60)
print("RUNNING 5 MORE STEPS:")
obj_history = [init_obj, new_obj]

for step in range(5):
    # Get actions for all agents
    actions = {}
    for agent_id, agent_obs in obs_dict2.items():
        policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"
        torch_obs = {
            "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
            "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
        }
        action = get_deterministic_action(policies[policy_name], torch_obs)
        actions[agent_id] = action.cpu().numpy().flatten()

    # Step
    obs_dict2, rewards, terminateds, truncateds, infos = env.step(actions)

    # Calculate objective
    plunger_v = env.base_env.device_state['current_gate_voltages']
    barrier_v = env.base_env.device_state['current_barrier_voltages']
    plunger_gt = env.base_env.device_state['gate_ground_truth']
    barrier_gt = env.base_env.device_state['barrier_ground_truth']
    plunger_dists = plunger_v - plunger_gt
    barrier_dists = barrier_v - barrier_gt
    step_obj = float(np.sum(plunger_dists**2) + np.sum(barrier_dists**2))
    obj_history.append(step_obj)

    print(f"  Step {step+2}: obj={step_obj:.2f} (change: {step_obj - obj_history[-2]:+.2f})")
    print(f"    Plunger dists: {plunger_dists}")

print("\n" + "="*60)
print("OBJECTIVE TREND:", obj_history)
if all(obj_history[i] >= obj_history[i-1] for i in range(1, len(obj_history))):
    print("CRITICAL: Objective monotonically increases - policy is broken!")
elif all(obj_history[i] <= obj_history[i-1] for i in range(1, len(obj_history))):
    print("GOOD: Objective monotonically decreases - policy is working!")
else:
    print("MIXED: Objective fluctuates - needs investigation")
