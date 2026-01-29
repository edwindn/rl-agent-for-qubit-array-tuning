"""
Debug script to compare training inference (via RLlib evaluate) vs benchmark inference (direct forward_inference).

This script:
1. Creates identical environment states with the same seed
2. Gets observations from both approaches
3. Compares the resulting actions and voltages
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


def get_stochastic_action(policy, observation, num_samples=100):
    """
    Sample actions stochastically from the policy (like RLlib's evaluate with explore=True).
    Returns the mean of sampled actions to compare with deterministic.
    """
    with torch.no_grad():
        output = policy.forward_inference({"obs": observation})
        action_dist_inputs = output["action_dist_inputs"]
        action_dim = action_dist_inputs.shape[-1] // 2
        mean = action_dist_inputs[..., :action_dim]
        log_std = action_dist_inputs[..., action_dim:]
        std = torch.exp(log_std)

        # Sample multiple times and take mean to estimate expected action
        samples = []
        for _ in range(num_samples):
            noise = torch.randn_like(mean)
            sample = mean + std * noise
            samples.append(sample)

        sampled_mean = torch.stack(samples).mean(dim=0)

        return mean, std, sampled_mean


def main():
    print("=" * 70)
    print("COMPARING TRAINING INFERENCE VS BENCHMARK INFERENCE")
    print("=" * 70)

    # Create environment
    print("\nCreating environment...")
    env = MultiAgentEnvWrapper(training=True, return_voltage=True)

    print(f"  num_dots: {env.num_gates}")
    print(f"  use_deltas: {env.base_env.use_deltas}")

    # Load policies
    print("\nLoading policies...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "../../artifacts/rl_checkpoint_best:v3482"

    try:
        policies = load_policies(checkpoint_path, device=device)
        print(f"  Loaded: {list(policies.keys())}")
    except Exception as e:
        print(f"  Error loading policies: {e}")
        print("  Please provide a valid checkpoint path")
        return

    # Reset environment with fixed seed
    seed = 42
    print(f"\nResetting environment with seed={seed}...")
    obs_dict, info = env.reset(seed=seed)

    # Print initial state
    print(f"\nInitial device state:")
    print(f"  Plunger voltages: {env.base_env.device_state['current_gate_voltages']}")
    print(f"  Plunger ground truth: {env.base_env.device_state['gate_ground_truth']}")
    print(f"  Barrier voltages: {env.base_env.device_state['current_barrier_voltages']}")
    print(f"  Barrier ground truth: {env.base_env.device_state['barrier_ground_truth']}")

    init_plunger_dist = env.base_env.device_state['current_gate_voltages'] - env.base_env.device_state['gate_ground_truth']
    init_barrier_dist = env.base_env.device_state['current_barrier_voltages'] - env.base_env.device_state['barrier_ground_truth']
    init_obj = float(np.sum(init_plunger_dist**2) + np.sum(init_barrier_dist**2))
    print(f"\nInitial distances:")
    print(f"  Plunger: {init_plunger_dist}")
    print(f"  Barrier: {init_barrier_dist}")
    print(f"  Objective: {init_obj:.4f}")

    # Analyze policy outputs for each agent
    print("\n" + "=" * 70)
    print("POLICY OUTPUT ANALYSIS")
    print("=" * 70)

    for agent_id in sorted(obs_dict.keys()):
        agent_obs = obs_dict[agent_id]
        policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"

        # Convert to torch tensor
        torch_obs = {
            "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
            "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
        }

        # Get deterministic action (mean)
        det_action = get_deterministic_action(policies[policy_name], torch_obs)
        det_action_val = det_action.cpu().numpy().flatten()[0]

        # Get stochastic info
        mean, std, sampled_mean = get_stochastic_action(policies[policy_name], torch_obs)
        std_val = std.cpu().numpy().flatten()[0]
        sampled_mean_val = sampled_mean.cpu().numpy().flatten()[0]

        print(f"\n{agent_id}:")
        print(f"  Obs voltage (normalized): {agent_obs['voltage'][0]:.4f}")
        print(f"  Obs image range: [{agent_obs['image'].min():.4f}, {agent_obs['image'].max():.4f}]")
        print(f"  Policy mean (deterministic): {det_action_val:.4f}")
        print(f"  Policy std: {std_val:.4f}")
        print(f"  Sampled mean (100 samples): {sampled_mean_val:.4f}")
        print(f"  Difference (sampled - det): {sampled_mean_val - det_action_val:.6f}")

    # Run a few steps with deterministic actions
    print("\n" + "=" * 70)
    print("RUNNING STEPS WITH DETERMINISTIC ACTIONS")
    print("=" * 70)

    obj_history = [init_obj]

    for step in range(10):
        # Get deterministic actions
        actions = {}
        for agent_id, agent_obs in obs_dict.items():
            policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"
            torch_obs = {
                "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
                "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
            }
            action = get_deterministic_action(policies[policy_name], torch_obs)
            actions[agent_id] = action.cpu().numpy().flatten()

        # Step environment
        obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)

        # Calculate objective
        plunger_v = env.base_env.device_state['current_gate_voltages']
        barrier_v = env.base_env.device_state['current_barrier_voltages']
        plunger_gt = env.base_env.device_state['gate_ground_truth']
        barrier_gt = env.base_env.device_state['barrier_ground_truth']

        plunger_dist = plunger_v - plunger_gt
        barrier_dist = barrier_v - barrier_gt
        step_obj = float(np.sum(plunger_dist**2) + np.sum(barrier_dist**2))
        obj_history.append(step_obj)

        change = step_obj - obj_history[-2]
        status = "↓ BETTER" if change < 0 else "↑ WORSE"

        print(f"\nStep {step + 1}:")
        print(f"  Objective: {step_obj:.4f} ({status} by {abs(change):.4f})")
        print(f"  Plunger dist: {plunger_dist}")
        print(f"  Barrier dist: {barrier_dist}")

        # Check for convergence
        done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
        if done:
            print(f"  Episode ended (terminated={terminateds.get('__all__')}, truncated={truncateds.get('__all__')})")
            break

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nObjective history: {[f'{x:.2f}' for x in obj_history]}")
    print(f"Initial objective: {obj_history[0]:.4f}")
    print(f"Final objective: {obj_history[-1]:.4f}")
    print(f"Total change: {obj_history[-1] - obj_history[0]:+.4f}")

    if obj_history[-1] < obj_history[0]:
        print("\n✓ Policy is reducing objective (working correctly)")
    else:
        print("\n✗ Policy is NOT reducing objective (something is wrong!)")
        print("\nPossible issues:")
        print("  1. Checkpoint might be from early training (not converged)")
        print("  2. Observation format might differ from training")
        print("  3. Action interpretation might differ from training")


if __name__ == "__main__":
    main()
