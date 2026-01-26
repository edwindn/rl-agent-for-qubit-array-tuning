"""Debug: Compare PhysicalObjective optimal voltages vs env ground truth."""

import sys
from pathlib import Path
import numpy as np

# Add paths
benchmarks_dir = Path(__file__).parent.parent
project_root = benchmarks_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(benchmarks_dir))
sys.path.insert(0, str(src_dir))

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
from objective import PhysicalObjective, get_distances

# Create env
print("Creating environment...")
env = MultiAgentEnvWrapper(training=True, return_voltage=True)

# Reset
obs_dict, info = env.reset(seed=42)

# Get current voltages
plunger_v = env.base_env.device_state['current_gate_voltages']
barrier_v = env.base_env.device_state['current_barrier_voltages']
voltages = np.concatenate([plunger_v, barrier_v])

# Env's ground truth
env_plunger_gt = env.base_env.device_state['gate_ground_truth']
env_barrier_gt = env.base_env.device_state['barrier_ground_truth']

# PhysicalObjective's optimal voltages
obj = PhysicalObjective(env.base_env)
physics_plunger_opt, physics_barrier_opt = obj.get_optimal_voltages(voltages)

print("\nCurrent voltages:")
print(f"  Plunger: {plunger_v}")
print(f"  Barrier: {barrier_v}")

print("\nEnv ground truth (what policy was trained on):")
print(f"  Plunger: {env_plunger_gt}")
print(f"  Barrier: {env_barrier_gt}")

print("\nPhysicsObjective optimal (what benchmark uses):")
print(f"  Plunger: {physics_plunger_opt}")
print(f"  Barrier: {physics_barrier_opt}")

print("\nDifference (env GT - physics optimal):")
print(f"  Plunger diff: {env_plunger_gt - physics_plunger_opt}")
print(f"  Barrier diff: {env_barrier_gt - physics_barrier_opt}")

print("\nDistances from current to env GT:")
print(f"  Plunger: {plunger_v - env_plunger_gt}")
print(f"  Barrier: {barrier_v - env_barrier_gt}")

print("\nDistances from current to physics optimal:")
print(f"  Plunger: {plunger_v - physics_plunger_opt}")
print(f"  Barrier: {barrier_v - physics_barrier_opt}")

# Check if they're approximately the same
plunger_match = np.allclose(env_plunger_gt, physics_plunger_opt, atol=1.0)
barrier_match = np.allclose(env_barrier_gt, physics_barrier_opt, atol=1.0)

print("\n" + "="*60)
if plunger_match and barrier_match:
    print("MATCH: Env ground truth and physics objective agree (within 1V)")
else:
    print("MISMATCH: Env ground truth and physics objective DISAGREE!")
    print("This means the benchmark uses a different target than training!")
    print("\nPlunger max diff:", np.max(np.abs(env_plunger_gt - physics_plunger_opt)))
    print("Barrier max diff:", np.max(np.abs(env_barrier_gt - physics_barrier_opt)))
