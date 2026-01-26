"""
Load RLlib checkpoint policies for standalone inference.

Loads plunger_policy and barrier_policy from an RLlib checkpoint without
requiring a full Ray server. Uses RLModule.from_checkpoint() which works
with just the ray library installed.
"""

import sys
from pathlib import Path
from typing import Dict

import torch

# Add src directory to path for swarm imports
benchmarks_dir = Path(__file__).parent.parent
project_root = benchmarks_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


def load_policies(checkpoint_path: str, device: str = "cuda") -> Dict[str, "RLModule"]:
    """
    Load plunger_policy and barrier_policy from RLlib checkpoint.

    Args:
        checkpoint_path: Path to RLlib checkpoint directory (e.g., iteration_69)
        device: Device to load models on ("cuda" or "cpu")

    Returns:
        Dictionary with "plunger_policy" and "barrier_policy" RLModule instances

    Raises:
        FileNotFoundError: If checkpoint files are missing
        RuntimeError: If checkpoint loading fails
    """
    from ray.rllib.core.rl_module.rl_module import RLModule

    # Convert to absolute path (RLlib's from_checkpoint requires absolute paths)
    checkpoint_path = Path(checkpoint_path).resolve()
    rl_module_path = checkpoint_path / "learner_group" / "learner" / "rl_module"

    # Validate checkpoint structure
    required_policies = ["plunger_policy", "barrier_policy"]
    for policy_name in required_policies:
        policy_path = rl_module_path / policy_name
        if not policy_path.exists():
            raise FileNotFoundError(f"Missing {policy_name} in checkpoint: {policy_path}")
        if not (policy_path / "module_state.pkl").exists():
            raise FileNotFoundError(f"Missing module_state.pkl for {policy_name}")

    policies = {}
    for policy_name in required_policies:
        policy_path = rl_module_path / policy_name

        # Load the RLModule from checkpoint
        module = RLModule.from_checkpoint(str(policy_path))

        # Move to device and set to eval mode
        module = module.to(device)
        module.eval()

        policies[policy_name] = module

    return policies


def get_deterministic_action(policy: "RLModule", observation: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Get deterministic action from policy (using mean of Gaussian).

    Args:
        policy: RLModule instance
        observation: Dictionary with 'image' and 'voltage' tensors (batched)

    Returns:
        Action tensor of shape (batch_size, action_dim)
    """
    with torch.no_grad():
        # Forward pass through policy
        output = policy.forward_inference({"obs": observation})

        # action_dist_inputs contains [mean, log_std] concatenated
        # For deterministic action, use just the mean (first half)
        action_dist_inputs = output["action_dist_inputs"]
        action_dim = action_dist_inputs.shape[-1] // 2
        mean = action_dist_inputs[..., :action_dim]

        return mean


def validate_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that checkpoint has required structure.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if valid, raises exception otherwise
    """
    checkpoint_path = Path(checkpoint_path).resolve()

    required_files = [
        "learner_group/learner/rl_module/plunger_policy/module_state.pkl",
        "learner_group/learner/rl_module/barrier_policy/module_state.pkl",
    ]

    for f in required_files:
        if not (checkpoint_path / f).exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    return True


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/iteration_69")
    args = parser.parse_args()

    print(f"Loading policies from: {args.checkpoint}")
    validate_checkpoint(args.checkpoint)
    policies = load_policies(args.checkpoint)

    print(f"Loaded policies: {list(policies.keys())}")
    for name, policy in policies.items():
        print(f"  {name}: {type(policy)}")
