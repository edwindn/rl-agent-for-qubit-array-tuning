"""Common utility functions shared between training and inference."""
import glob
import re
from pathlib import Path

import torch


def parse_config_overrides(unknown_args):
    """Parse config override arguments in the format --key.subkey value or --key=value (allows dynamically overriding settings when calling train.py)"""
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            # Handle both --key=value and --key value formats
            if '=' in arg:
                # Format: --key=value
                key_value = arg[2:]  # Remove '--' prefix
                key, value = key_value.split('=', 1)
                i += 1
            elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # Format: --key value
                key = arg[2:]  # Remove '--' prefix
                value = unknown_args[i + 1]
                i += 2
            else:
                # Standalone flag or no value
                i += 1
                continue

            # Type conversion
            try:
                # Handle None as string
                if value.lower() == 'none':
                    value = None
                elif value.lower() in ('true', 'false'):
                    # Handle boolean
                    value = value.lower() == 'true'
                else:
                    # Try to convert to number (handles both int, float, and scientific notation)
                    try:
                        # First try float (handles scientific notation like 1e-05)
                        float_val = float(value)
                        # If it's a whole number, convert to int
                        if float_val.is_integer() and 'e' not in value.lower() and '.' not in value:
                            value = int(float_val)
                        else:
                            value = float_val
                    except ValueError:
                        # Keep as string if not a number
                        pass
            except (ValueError, AttributeError):
                pass  # Keep as string

            overrides[key] = value
        else:
            i += 1
    return overrides


def map_sweep_parameters(overrides):
    """Map sweep parameter names to config paths for wandb sweep compatibility."""
    # Mapping from sweep parameter names to config paths
    sweep_param_mapping = {
        # Core training parameters
        'minibatch_size': 'rl_config.training.minibatch_size',
        'num_epochs': 'rl_config.training.num_epochs',
        'lr': 'rl_config.training.lr',
        'gamma': 'rl_config.training.gamma',
        'lambda_': 'rl_config.training.lambda_',
        'clip_param': 'rl_config.training.clip_param',
        'entropy_coeff': 'rl_config.training.entropy_coeff',
        'vf_loss_coeff': 'rl_config.training.vf_loss_coeff',
        'kl_target': 'rl_config.training.kl_target',
        'grad_clip': 'rl_config.training.grad_clip',
        'grad_clip_by': 'rl_config.training.grad_clip_by',
        'train_batch_size': 'rl_config.training.train_batch_size',

        # Algorithm choice
        'algorithm': 'rl_config.algorithm',

        # Training control
        'num_iterations': 'defaults.num_iterations',
    }

    mapped_overrides = {}

    for key, value in overrides.items():
        if key in sweep_param_mapping:
            # Map sweep parameter to config path
            config_path = sweep_param_mapping[key]
            mapped_overrides[config_path] = value
            print(f"Mapped sweep parameter: {key} -> {config_path} = {value}")
        else:
            # Keep original key (might be a nested config path already)
            mapped_overrides[key] = value
            print(f"Direct config override: {key} = {value}")

    return mapped_overrides


def apply_config_overrides(config, overrides):
    """Apply config overrides using dot notation to nested dictionary."""
    # First map sweep parameters to config paths
    mapped_overrides = map_sweep_parameters(overrides)

    for key, value in mapped_overrides.items():
        keys = key.split('.')
        current = config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the final value
        current[keys[-1]] = value
        print(f"Config override applied: {key} = {value}")

    return config


def fix_optimizer_betas_after_checkpoint_load(algo):
    """Fix optimizer beta parameters that may have been saved as tensors.

    When checkpoints are saved with LR schedules, optimizer parameters like betas
    can be stored as tensors. This causes errors when loading with certain optimizer
    configurations. This function converts any tensor betas back to scalar values.

    Args:
        algo: The RLlib Algorithm instance after restore_from_path() has been called
    """
    def fix_betas(learner):
        for name, optimizer in learner._named_optimizers.items():
            for param_group in optimizer.param_groups:
                if 'betas' in param_group:
                    betas = param_group['betas']
                    param_group['betas'] = (
                        float(betas[0]) if torch.is_tensor(betas[0]) else betas[0],
                        float(betas[1]) if torch.is_tensor(betas[1]) else betas[1]
                    )

    algo.learner_group.foreach_learner(fix_betas)


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the given directory.

    Args:
        checkpoint_dir (str or Path): Directory containing checkpoint folders

    Returns:
        tuple: (checkpoint_path, iteration_number) or (None, None) if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None, None

    # Find all iteration directories
    iteration_pattern = checkpoint_dir / "iteration_*"
    iteration_dirs = glob.glob(str(iteration_pattern))

    if not iteration_dirs:
        return None, None

    # Extract iteration numbers and find the maximum
    max_iteration = 0
    latest_checkpoint = None

    for iteration_dir in iteration_dirs:
        # Extract iteration number from directory name
        match = re.search(r'iteration_(\d+)', iteration_dir)
        if match:
            iteration_num = int(match.group(1))
            if iteration_num > max_iteration:
                max_iteration = iteration_num
                latest_checkpoint = iteration_dir

    return latest_checkpoint, max_iteration


def clean_checkpoint_folder(checkpoint_dir):
    """Clean checkpoint folder at start of training, keeping yaml files."""
    import shutil

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    # Remove all iteration_* directories
    iteration_pattern = checkpoint_dir / "iteration_*"
    iteration_dirs = glob.glob(str(iteration_pattern))

    for iteration_dir in iteration_dirs:
        try:
            shutil.rmtree(iteration_dir)
        except Exception as e:
            print(f"Warning: Could not delete {iteration_dir}: {e}")


def delete_old_checkpoint_if_needed(checkpoint_dir):
    """Keep only the latest checkpoint, delete all older ones."""
    import shutil

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    # Find all iteration directories
    iteration_pattern = checkpoint_dir / "iteration_*"
    iteration_dirs = glob.glob(str(iteration_pattern))

    if len(iteration_dirs) <= 1:
        return  # Keep at least 1 checkpoint

    # Extract iteration numbers and sort
    iteration_info = []
    for iteration_dir in iteration_dirs:
        match = re.search(r'iteration_(\d+)', iteration_dir)
        if match:
            iteration_num = int(match.group(1))
            iteration_info.append((iteration_num, iteration_dir))

    # Sort by iteration number (oldest first)
    iteration_info.sort(key=lambda x: x[0])

    # Delete all but the latest checkpoint
    for iteration_num, iteration_dir in iteration_info[:-1]:
        try:
            shutil.rmtree(iteration_dir)
        except Exception as e:
            print(f"Warning: Could not delete old checkpoint {iteration_dir}: {e}")


def create_env_to_module_connector(env, spaces, device, use):
    """
    Creates module connector for action to memory handling.
    Note: do not modify the signature, ray expects arguments 0-2

    Args:
        env: The (vectorized) gym environment
        spaces: Dict with space info like {'__env__': ([obs_space, act_space]), '__env_single__': ([obs_space, act_space])}
        device: Torch device (can be None)
        use: Whether to use the custom connector or not
    """
    if use:
        from qadapt.voltage_model.prev_action_handling import CustomPrevActionHandling
        return [CustomPrevActionHandling()]
    else:
        # Return empty list - let Ray handle everything with defaults
        return []
