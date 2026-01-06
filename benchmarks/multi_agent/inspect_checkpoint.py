#!/usr/bin/env python3
"""
Inspect Ray RLlib checkpoint to print exact model architecture from saved weights.
This script analyzes the pickled weight files to determine the actual model structure.
"""
import os
import sys
import argparse
import pickle
import json
from pathlib import Path
from collections import OrderedDict

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Default weights directory
current_dir = Path(__file__).parent
WEIGHTS_DIR = current_dir / "weights" / "artifacts"


def find_available_checkpoints(weights_dir=WEIGHTS_DIR):
    """Find all available checkpoints in the weights directory."""
    if not weights_dir.exists():
        return []

    checkpoints = []
    for subdir in weights_dir.iterdir():
        if subdir.is_dir():
            if (subdir / "rllib_checkpoint.json").exists():
                checkpoints.append((subdir.name, subdir))

    return sorted(checkpoints)


def find_checkpoint(checkpoint_name=None):
    """Find a checkpoint by name or return the latest one."""
    available = find_available_checkpoints()

    if not available:
        raise ValueError(
            f"No checkpoints found in {WEIGHTS_DIR}\n"
            f"Please download model weights to this directory."
        )

    if checkpoint_name is None:
        checkpoint_name, checkpoint_path = available[-1]
        print(f"Auto-selected latest checkpoint: {checkpoint_name}")
        return checkpoint_path

    # Find specific checkpoint
    for name, path in available:
        if name == checkpoint_name:
            return path

    # Checkpoint not found
    available_names = [name for name, _ in available]
    raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found.\n"
        f"Available checkpoints: {', '.join(available_names)}"
    )


def extract_layer_info(param_name, param_shape):
    """
    Extract layer type and dimensions from parameter name and shape.

    Args:
        param_name: Parameter name (e.g., "encoder.0.weight")
        param_shape: Parameter shape tuple

    Returns:
        dict with layer_type, input_dim, output_dim, etc.
    """
    info = {
        'param_name': param_name,
        'shape': param_shape,
        'layer_type': None,
        'input_dim': None,
        'output_dim': None,
        'details': {}
    }

    # Determine layer type from shape
    if len(param_shape) == 2:
        # Linear layer: (out_features, in_features) for PyTorch
        info['layer_type'] = 'Linear'
        info['output_dim'] = param_shape[0]
        info['input_dim'] = param_shape[1]

    elif len(param_shape) == 4:
        # Conv2d: (out_channels, in_channels, kernel_height, kernel_width)
        info['layer_type'] = 'Conv2d'
        info['output_dim'] = param_shape[0]
        info['input_dim'] = param_shape[1]
        info['details']['kernel_size'] = (param_shape[2], param_shape[3])

    elif len(param_shape) == 1:
        # Bias, BatchNorm parameters, or LayerNorm parameters
        if 'bias' in param_name:
            info['layer_type'] = 'Bias'
            info['output_dim'] = param_shape[0]
        elif 'running_mean' in param_name or 'running_var' in param_name:
            info['layer_type'] = 'BatchNorm_stats'
            info['output_dim'] = param_shape[0]
        elif 'weight' in param_name and ('norm' in param_name.lower() or 'bn' in param_name.lower()):
            info['layer_type'] = 'Norm_weight'
            info['output_dim'] = param_shape[0]
        else:
            info['layer_type'] = 'Vector'
            info['output_dim'] = param_shape[0]

    elif len(param_shape) == 3:
        # Could be Conv1d or other 3D tensors
        info['layer_type'] = 'Conv1d_or_other_3D'
        info['output_dim'] = param_shape[0]
        info['input_dim'] = param_shape[1]
        info['details']['kernel_size'] = param_shape[2]

    else:
        info['layer_type'] = f'{len(param_shape)}D_tensor'

    return info


def organize_layers(state_dict):
    """
    Organize state dict parameters into a hierarchical layer structure.

    Args:
        state_dict: Dictionary of parameter names to tensors

    Returns:
        Organized dictionary of layers grouped by component
    """
    # Extract layer information
    all_params = {}
    for param_name, param_tensor in state_dict.items():
        if hasattr(param_tensor, 'shape'):
            shape = tuple(param_tensor.shape)
            all_params[param_name] = extract_layer_info(param_name, shape)

    # Group by base layer (remove .weight and .bias suffixes)
    layers = {}
    for param_name, param_info in all_params.items():
        # Extract base layer name
        base_name = param_name
        param_type = None

        if param_name.endswith('.weight'):
            base_name = param_name[:-7]  # Remove '.weight'
            param_type = 'weight'
        elif param_name.endswith('.bias'):
            base_name = param_name[:-5]  # Remove '.bias'
            param_type = 'bias'
        elif 'running_mean' in param_name or 'running_var' in param_name:
            base_name = param_name.rsplit('.', 1)[0]
            param_type = param_name.rsplit('.', 1)[1]

        if base_name not in layers:
            layers[base_name] = {}

        if param_type:
            layers[base_name][param_type] = param_info
        else:
            layers[base_name]['other'] = param_info

    return layers


def print_model_architecture(state_dict, policy_name, verbose=False):
    """
    Print organized model architecture from state dict.

    Args:
        state_dict: PyTorch state dictionary
        policy_name: Name of the policy
        verbose: If True, print all parameters

    Returns:
        dict: Architecture information for JSON serialization
    """
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE FOR: {policy_name.upper()}")
    print(f"{'='*80}")

    # Organize layers
    layers = organize_layers(state_dict)

    # Group layers by component (encoder, pi, vf, etc.)
    components = {}
    for layer_name in layers.keys():
        # Extract component name (first part before dot or number)
        parts = layer_name.split('.')

        # Try to identify component
        if len(parts) > 0:
            # Common component names
            component = parts[0]
        else:
            component = 'root'

        if component not in components:
            components[component] = []
        components[component].append(layer_name)

    # Build architecture info for JSON
    architecture_info = {
        'policy_name': policy_name,
        'components': {},
        'total_parameters': 0
    }

    # Print each component
    total_params = 0

    for component_name in sorted(components.keys()):
        print(f"\n[{component_name.upper()}]")

        layer_names = sorted(components[component_name])
        component_layers = []

        layer_num = 0
        for layer_name in layer_names:
            layer_params = layers[layer_name]

            # Skip if only bias (weight should be printed with it)
            if 'weight' not in layer_params and 'bias' in layer_params:
                continue

            # Get weight info
            if 'weight' in layer_params:
                weight_info = layer_params['weight']
                layer_type = weight_info['layer_type']

                # Print layer header
                print(f"\n  Layer {layer_num}: {layer_name}")

                layer_info = {
                    'layer_name': layer_name,
                    'layer_type': layer_type,
                    'weight_shape': list(weight_info['shape']),
                }

                # Print layer details based on type
                if layer_type == 'Linear':
                    in_dim = weight_info['input_dim']
                    out_dim = weight_info['output_dim']
                    print(f"    Type: {layer_type}({in_dim} → {out_dim})")
                    print(f"    Weight shape: {weight_info['shape']}")

                    layer_info['input_dim'] = in_dim
                    layer_info['output_dim'] = out_dim

                    # Count parameters
                    param_count = in_dim * out_dim

                elif layer_type == 'Conv2d':
                    in_ch = weight_info['input_dim']
                    out_ch = weight_info['output_dim']
                    kernel = weight_info['details'].get('kernel_size', 'unknown')
                    print(f"    Type: {layer_type}(in_channels={in_ch}, out_channels={out_ch}, kernel={kernel})")
                    print(f"    Weight shape: {weight_info['shape']}")

                    layer_info['in_channels'] = in_ch
                    layer_info['out_channels'] = out_ch
                    layer_info['kernel_size'] = kernel

                    # Count parameters
                    param_count = weight_info['shape'][0] * weight_info['shape'][1] * weight_info['shape'][2] * weight_info['shape'][3]

                elif layer_type == 'Conv1d_or_other_3D':
                    in_dim = weight_info['input_dim']
                    out_dim = weight_info['output_dim']
                    kernel = weight_info['details'].get('kernel_size', 'unknown')
                    print(f"    Type: {layer_type}(in={in_dim}, out={out_dim}, kernel={kernel})")
                    print(f"    Weight shape: {weight_info['shape']}")

                    layer_info['input_dim'] = in_dim
                    layer_info['output_dim'] = out_dim
                    layer_info['kernel_size'] = kernel

                    param_count = weight_info['shape'][0] * weight_info['shape'][1] * weight_info['shape'][2]

                else:
                    print(f"    Type: {layer_type}")
                    print(f"    Weight shape: {weight_info['shape']}")

                    # Count parameters
                    shape = weight_info['shape']
                    param_count = 1
                    for dim in shape:
                        param_count *= dim

                # Add bias parameters
                if 'bias' in layer_params:
                    bias_info = layer_params['bias']
                    print(f"    Bias shape: {bias_info['shape']}")
                    layer_info['bias_shape'] = list(bias_info['shape'])

                    bias_count = bias_info['shape'][0] if len(bias_info['shape']) > 0 else 0
                    param_count += bias_count

                print(f"    Parameters: {param_count:,}")
                layer_info['parameters'] = param_count
                total_params += param_count

                component_layers.append(layer_info)
                layer_num += 1

            elif 'other' in layer_params:
                # Handle other parameter types
                other_info = layer_params['other']
                print(f"\n  {layer_name}")
                print(f"    Type: {other_info['layer_type']}")
                print(f"    Shape: {other_info['shape']}")

                # Count parameters
                shape = other_info['shape']
                param_count = 1
                for dim in shape:
                    param_count *= dim
                print(f"    Parameters: {param_count:,}")
                total_params += param_count

                layer_info = {
                    'layer_name': layer_name,
                    'layer_type': other_info['layer_type'],
                    'shape': list(other_info['shape']),
                    'parameters': param_count
                }
                component_layers.append(layer_info)

        architecture_info['components'][component_name] = component_layers

    # Print verbose parameter listing if requested
    if verbose:
        print(f"\n{'─'*80}")
        print("ALL PARAMETERS (VERBOSE)")
        print(f"{'─'*80}")
        all_params = []
        for param_name in sorted(state_dict.keys()):
            param = state_dict[param_name]
            if hasattr(param, 'shape'):
                shape = tuple(param.shape)
                dtype = str(param.dtype) if hasattr(param, 'dtype') else 'unknown'
                print(f"  {param_name}")
                print(f"    Shape: {shape}, dtype: {dtype}")
                all_params.append({
                    'name': param_name,
                    'shape': list(shape),
                    'dtype': dtype
                })
        architecture_info['all_parameters'] = all_params

    print(f"\n{'─'*80}")
    print(f"Total parameters: {total_params:,}")
    print(f"{'='*80}\n")

    architecture_info['total_parameters'] = total_params
    return architecture_info


def inspect_checkpoint(checkpoint_path, verbose=False, save_json=False):
    """
    Inspect a Ray RLlib checkpoint and print model architecture from weights.

    Args:
        checkpoint_path: Path to checkpoint directory
        verbose: If True, print all parameters
        save_json: If True, save architecture info to JSON file

    Returns:
        dict: Complete checkpoint architecture information
    """
    checkpoint_path = Path(checkpoint_path)

    # Collect all architecture info
    checkpoint_info = {
        'checkpoint_name': checkpoint_path.name,
        'checkpoint_path': str(checkpoint_path),
        'policies': {}
    }

    print(f"\n{'='*80}")
    print(f"INSPECTING CHECKPOINT: {checkpoint_path.name}")
    print(f"{'='*80}\n")

    # Read RLlib checkpoint metadata
    checkpoint_json_path = checkpoint_path / "rllib_checkpoint.json"
    if not checkpoint_json_path.exists():
        raise FileNotFoundError(f"Not a valid RLlib checkpoint: {checkpoint_json_path} not found")

    with open(checkpoint_json_path, 'r') as f:
        checkpoint_metadata = json.load(f)

    print(f"Checkpoint type: {checkpoint_metadata.get('type', 'Unknown')}")
    print(f"RLlib version: {checkpoint_metadata.get('ray_version', 'Unknown')}")
    print(f"Checkpoint version: {checkpoint_metadata.get('checkpoint_version', 'Unknown')}")

    checkpoint_info['checkpoint_type'] = checkpoint_metadata.get('type', 'Unknown')
    checkpoint_info['rllib_version'] = checkpoint_metadata.get('ray_version', 'Unknown')
    checkpoint_info['checkpoint_version'] = checkpoint_metadata.get('checkpoint_version', 'Unknown')

    # Look for learner group checkpoints (RLlib 2.x structure with learner_group)
    learner_group_dir = checkpoint_path / "learner_group" / "learner" / "rl_module"
    if learner_group_dir.exists():
        print(f"\n{'─'*80}")
        print("LEARNER GROUP STATE DETECTED (RLlib 2.x)")
        print(f"{'─'*80}")

        # List policies/modules
        policy_dirs = [d for d in learner_group_dir.iterdir() if d.is_dir() and (d / "module_state.pkl").exists()]
        print(f"\nFound {len(policy_dirs)} policy/policies: {[d.name for d in policy_dirs]}")

        for policy_dir in sorted(policy_dirs):
            policy_name = policy_dir.name

            # Look for module_state.pkl
            model_file = policy_dir / "module_state.pkl"
            if model_file.exists():
                print(f"\nLoading weights from: {model_file.relative_to(checkpoint_path)}")
                try:
                    with open(model_file, 'rb') as f:
                        state_dict = pickle.load(f)

                    # Print architecture from state dict
                    arch_info = print_model_architecture(state_dict, policy_name, verbose=verbose)
                    checkpoint_info['policies'][policy_name] = arch_info

                except Exception as e:
                    print(f"Error loading {model_file}: {e}")
                    import traceback
                    traceback.print_exc()

    # Look for learner checkpoints (alternate RLlib 2.x structure)
    learner_dir = checkpoint_path / "learner"
    if learner_dir.exists() and not learner_group_dir.exists():
        print(f"\n{'─'*80}")
        print("LEARNER STATE DETECTED (RLlib 2.x)")
        print(f"{'─'*80}")

        # List learner workers
        learner_workers = list(learner_dir.glob("learner_worker_*"))
        print(f"\nFound {len(learner_workers)} learner worker(s)")

        for worker_dir in sorted(learner_workers):
            print(f"\nAnalyzing: {worker_dir.name}")

            # Look for module state
            module_state_dir = worker_dir / "module_state"
            if module_state_dir.exists():
                # List policies/modules
                policy_dirs = [d for d in module_state_dir.iterdir() if d.is_dir()]
                print(f"Found {len(policy_dirs)} policy/policies: {[d.name for d in policy_dirs]}")

                for policy_dir in sorted(policy_dirs):
                    policy_name = policy_dir.name

                    # Try .pt files first
                    model_file = policy_dir / "module_state.pt"
                    if model_file.exists():
                        print(f"\nLoading weights from: {model_file.relative_to(checkpoint_path)}")
                        try:
                            import torch
                            state_dict = torch.load(model_file, map_location='cpu')
                            arch_info = print_model_architecture(state_dict, policy_name, verbose=verbose)
                            checkpoint_info['policies'][policy_name] = arch_info
                        except Exception as e:
                            print(f"Error loading {model_file}: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        # Try .pkl files
                        model_file = policy_dir / "module_state.pkl"
                        if model_file.exists():
                            print(f"\nLoading weights from: {model_file.relative_to(checkpoint_path)}")
                            try:
                                with open(model_file, 'rb') as f:
                                    state_dict = pickle.load(f)
                                arch_info = print_model_architecture(state_dict, policy_name, verbose=verbose)
                                checkpoint_info['policies'][policy_name] = arch_info
                            except Exception as e:
                                print(f"Error loading {model_file}: {e}")
                                import traceback
                                traceback.print_exc()

    # Look for policies directory (older RLlib structure)
    policies_dir = checkpoint_path / "policies"
    if policies_dir.exists():
        print(f"\n{'─'*80}")
        print("POLICIES DIRECTORY DETECTED (RLlib 1.x)")
        print(f"{'─'*80}")

        policy_dirs = [d for d in policies_dir.iterdir() if d.is_dir()]
        print(f"\nFound {len(policy_dirs)} policy/policies: {[d.name for d in policy_dirs]}")

        for policy_dir in sorted(policy_dirs):
            policy_name = policy_dir.name

            # Look for model weights
            model_files = list(policy_dir.glob("*.pt")) + list(policy_dir.glob("*.pth"))

            if model_files:
                for model_file in sorted(model_files):
                    print(f"\nLoading weights from: {model_file.relative_to(checkpoint_path)}")
                    try:
                        import torch
                        state_dict = torch.load(model_file, map_location='cpu')

                        # Print architecture from state dict
                        policy_display_name = f"{policy_name}/{model_file.stem}" if len(model_files) > 1 else policy_name
                        arch_info = print_model_architecture(state_dict, policy_display_name, verbose=verbose)
                        checkpoint_info['policies'][policy_display_name] = arch_info

                    except Exception as e:
                        print(f"Error loading {model_file}: {e}")
                        import traceback
                        traceback.print_exc()

    # Save to JSON if requested
    if save_json:
        output_filename = f"{checkpoint_path.name}_weights_info.json"
        output_path = checkpoint_path.parent / output_filename

        with open(output_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

        print(f"\n{'─'*80}")
        print(f"Saved architecture info to: {output_path}")
        print(f"{'─'*80}\n")

    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE")
    print(f"{'='*80}\n")

    return checkpoint_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect Ray RLlib checkpoint and print model architecture from weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect latest checkpoint
  python inspect_checkpoint.py

  # Inspect specific checkpoint by name
  python inspect_checkpoint.py --checkpoint run_240

  # Inspect with verbose output (all parameters listed)
  python inspect_checkpoint.py --checkpoint run_240 --verbose

  # Inspect checkpoint by full path
  python inspect_checkpoint.py --checkpoint /path/to/checkpoint
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint name or path (default: latest checkpoint)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print all parameters with shapes and dtypes"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save architecture info to JSON file ({checkpoint_name}_weights_info.json)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Find checkpoint
        if args.checkpoint is None:
            checkpoint_path = find_checkpoint(None)
        elif Path(args.checkpoint).exists():
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = find_checkpoint(args.checkpoint)

        # Inspect checkpoint
        inspect_checkpoint(checkpoint_path, verbose=args.verbose, save_json=args.save)

        return 0

    except KeyboardInterrupt:
        print("\nInspection interrupted by user")
        return 1

    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
