"""
Collect per-step episode data from a trained RL policy for calibration analysis.

Uses algo.evaluate() (same as inference.py / eval_runs/main.py) so that actions
go through the full RLlib pipeline (connectors, action distribution, etc.).
Per-step data is captured via a DataCollectingEnv wrapper that saves to disk.

Sign convention: ML model outputs raw values v. We store model_values = -v
(qarray sign convention). absolute_prediction = current_estimate + model_values.

Saves to .npy for use with test_variances.py.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import yaml
import tempfile
from pathlib import Path
from functools import partial

SRC_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from swarm.capacitance_model.capacitance_utils import get_nearest_targets, get_targets_with_nnn

DEFAULT_WANDB_ARTIFACT = "rl_agents_for_tuning/RLModel/rl_checkpoint_best:v3482"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "checkpoints"
CAP_MODEL_CHECKPOINT = "swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "data" / "episode_data.npy"


# ---------------------------------------------------------------------------
# Data-collecting env wrapper
# ---------------------------------------------------------------------------

class DataCollectingEnv:
    """
    Wraps MultiAgentEnvWrapper to capture per-step data during algo.evaluate().
    Inherits from the wrapper so gymnasium recognizes it as a valid env.
    Data is saved to disk so the main process can read it after evaluation.
    """

    def __new__(cls, save_dir, **env_kwargs):
        """Dynamically create a subclass of MultiAgentEnvWrapper with data collection."""
        from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper

        class _DataCollectingEnvImpl(MultiAgentEnvWrapper):
            _save_dir = None
            _episode_samples = []
            _episode_count = 0
            _step_count = 0
            _true_cgd = None
            _true_vgm = None

            def __init__(self, save_dir, **kwargs):
                super().__init__(**kwargs)
                self._save_dir = Path(save_dir)
                self._save_dir.mkdir(parents=True, exist_ok=True)
                self._episode_samples = []
                self._episode_count = 0
                self._step_count = 0

            def reset(self, **kwargs):
                if self._episode_samples:
                    self._flush_episode()

                obs, info = super().reset(**kwargs)

                base = self.base_env
                self._true_cgd = base.array.model.Cgd.copy()

                model = base.array.model
                if base.use_barriers:
                    cgd_gates_only = model.cgd_full[:, :model.n_gate]
                    self._true_vgm = -np.linalg.pinv(model.cdd_inv_full @ cgd_gates_only)
                else:
                    self._true_vgm = -np.linalg.pinv(model.cdd_inv_full @ model.cgd_full)
                if model.charge_carrier == 'electrons':
                    self._true_vgm = -self._true_vgm

                self._step_count = 0
                return obs, info

            def step(self, actions):
                base = self.base_env

                # Capture state BEFORE stepping
                predictor = base.capacitance_model["capacitance_predictor"]
                current_cgd_estimate = predictor.get_full_matrix().copy()
                estimated_vgm = base.array.model.gate_voltage_composer.virtual_gate_matrix.copy()

                obs, rewards, terminated, truncated, infos = super().step(actions)

                # Capture state AFTER stepping
                plunger_distance = np.abs(
                    base.device_state["gate_ground_truth"] -
                    base.device_state["current_gate_voltages"]
                )
                barrier_distance = np.abs(
                    base.device_state["barrier_ground_truth"] -
                    base.device_state["current_barrier_voltages"]
                )

                # Re-run model on the normalized obs the env used
                image = base._last_normalized_obs["image"]
                ml_model = base.capacitance_model["ml_model"]
                device = base.capacitance_model["device"]

                batch = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(1).to(device)
                with torch.no_grad():
                    values, log_vars = ml_model(batch)
                values_np = values.cpu().numpy()
                log_vars_np = log_vars.cpu().numpy()

                n_dots = base.num_dots
                nn = base.capacitance_model["nearest_neighbour"]
                get_targets = get_nearest_targets if nn else get_targets_with_nnn

                for scan_idx in range(n_dots - 1):
                    self._episode_samples.append({
                        'episode': self._episode_count,
                        'step': self._step_count,
                        'scan': image[:, :, scan_idx],
                        'capacitance': get_targets(scan_idx, self._true_cgd, n_dots, has_sensor=True),
                        'current_estimate': get_targets(scan_idx, current_cgd_estimate, n_dots, has_sensor=False),
                        'model_values': -values_np[scan_idx],  # negated: qarray sign convention
                        'model_log_vars': log_vars_np[scan_idx],
                        'estimated_vgm': estimated_vgm,
                        'true_vgm': self._true_vgm,
                        'plunger_distance': plunger_distance,
                        'barrier_distance': barrier_distance,
                    })

                self._step_count += 1

                if any(terminated.values()) or any(truncated.values()):
                    self._flush_episode()

                return obs, rewards, terminated, truncated, infos

            def _flush_episode(self):
                if not self._episode_samples:
                    return
                try:
                    path = self._save_dir / f"episode_{self._episode_count}.npy"
                    np.save(path, np.array(self._episode_samples, dtype=object), allow_pickle=True)
                    n_steps = self._episode_samples[-1]['step'] + 1
                    print(f"  Episode {self._episode_count}: {n_steps} steps, "
                          f"saved {len(self._episode_samples)} samples")
                except (OSError, FileNotFoundError):
                    pass  # temp dir may be cleaned up already
                self._episode_samples = []
                self._episode_count += 1

            def close(self):
                if self._episode_samples:
                    self._flush_episode()
                super().close()

        return _DataCollectingEnvImpl(save_dir=save_dir, **env_kwargs)


# ---------------------------------------------------------------------------
# Wandb artifact download
# ---------------------------------------------------------------------------

def download_wandb_artifact(artifact_name):
    """Download a wandb artifact and return the local path."""
    import wandb
    run = wandb.init(entity="rl_agents_for_tuning", project="RLModel")
    artifact = run.use_artifact(artifact_name, type="model_checkpoint")
    artifact_dir = artifact.download()
    wandb.finish()
    return artifact_dir


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(checkpoint_path, config_dir=None):
    """Load training and env configs."""
    checkpoint_path = Path(checkpoint_path)

    if config_dir is not None:
        config_dir = Path(config_dir)
    elif (checkpoint_path / "training_config.yaml").exists():
        config_dir = checkpoint_path
    elif checkpoint_path.name.startswith("iteration"):
        config_dir = checkpoint_path.parent
    else:
        config_dir = DEFAULT_CONFIG_DIR

    with open(config_dir / "training_config.yaml") as f:
        config = yaml.safe_load(f)
    with open(config_dir / "env_config.yaml") as f:
        env_config = yaml.safe_load(f)

    # Patch reward defaults for backward compat with old checkpoints
    reward_defaults = {
        "sparse_reward": False, "plunger_radius": 2, "barrier_radius": 2,
        "outer_plunger_radius": 10, "outer_plunger_reward_max": 0.5,
    }
    for k, v in reward_defaults.items():
        env_config.setdefault("reward", {})[k] = env_config.get("reward", {}).get(k, v)

    # Write patched config to temp file for env construction
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir='/tmp')
    yaml.dump(env_config, tmp, default_flow_style=False)
    patched_config_path = tmp.name
    tmp.close()

    algo_str = config['rl_config']['algorithm'].lower()
    return config, env_config, patched_config_path, algo_str


# ---------------------------------------------------------------------------
# Algo building (follows eval_runs/main.py pattern)
# ---------------------------------------------------------------------------

def build_algo(config, env_config, patched_config_path, algo_str, checkpoint_path,
               data_save_dir, num_episodes=5, explore=False):
    """Build RLlib algo with DataCollectingEnv, restore checkpoint."""
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.sac import SACConfig
    from ray.tune.registry import register_env
    from swarm.training.utils import policy_mapping_fn
    from swarm.training.train_utils import (
        fix_optimizer_betas_after_checkpoint_load,
        create_env_to_module_connector,
    )
    from swarm.voltage_model import create_rl_module_spec

    checkpoint_path = Path(checkpoint_path)

    # Capacitance model weights: from env_config or default
    cap_weights = env_config.get("capacitance_model", {}).get("weights_path", CAP_MODEL_CHECKPOINT)

    # Env factory — creates DataCollectingEnv that records per-step data
    create_env_fn = partial(
        _create_collecting_env,
        save_dir=data_save_dir,
        env_config_path=patched_config_path,
        capacitance_model_checkpoint=cap_weights,
    )
    register_env("qarray_multiagent_env", create_env_fn)

    ray.init(
        include_dashboard=False, log_to_driver=False, logging_level=40,
        runtime_env={
            "working_dir": str(SRC_DIR),
            "excludes": config['ray']['runtime_env']['excludes'],
            "env_vars": {**config['ray']['runtime_env']['env_vars'], "JAX_PLATFORMS": "cuda"},
        },
    )

    rl_module_config = {
        policy: {
            **config['neural_networks'][policy],
            "free_log_std": config['rl_config']['multi_agent']['free_log_std'],
            "log_std_bounds": config['rl_config']['multi_agent']['log_std_bounds'],
        }
        for policy in config['rl_config']['multi_agent']['policies']
    }
    rl_module_spec = create_rl_module_spec(env_config, algo=algo_str, config=rl_module_config)

    # Filter training params by algorithm
    ppo_only = {'lr', 'lambda_', 'clip_param', 'entropy_coeff', 'vf_loss_coeff',
                'kl_target', 'num_epochs', 'minibatch_size', 'train_batch_size'}
    sac_only = {'actor_lr', 'critic_lr', 'alpha_lr', 'twin_q', 'tau', 'initial_alpha',
                'target_entropy', 'n_step', 'clip_actions', 'target_network_update_freq',
                'num_steps_sampled_before_learning_starts', 'replay_buffer_config', 'reward_scale'}

    training_params = config['rl_config']['training'].copy()
    if algo_str == 'sac':
        for p in ppo_only: training_params.pop(p, None)
        AlgoConfig = SACConfig
    else:
        for p in sac_only: training_params.pop(p, None)
        AlgoConfig = PPOConfig

    # Connector setup (matches eval_runs/main.py)
    use_deltas = env_config['simulator']['use_deltas']
    memory_layer = config['neural_networks']['plunger_policy']['backbone'].get('memory_layer')
    has_lstm = memory_layer == 'lstm'
    if use_deltas and has_lstm:
        env_to_module_connector = partial(create_env_to_module_connector, use=True)
    else:
        env_to_module_connector = None

    algo_config = (
        AlgoConfig()
        .environment(env="qarray_multiagent_env")
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
            policies=config['rl_config']['multi_agent']['policies'],
            policies_to_train=config['rl_config']['multi_agent']['policies_to_train'],
            count_steps_by=config['rl_config']['multi_agent']['count_steps_by'],
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=config['rl_config']['env_runners']['rollout_fragment_length'],
            num_gpus_per_env_runner=0.5,
            env_to_module_connector=env_to_module_connector,
            add_default_connectors_to_env_to_module_pipeline=True,
        )
        .learners(num_learners=0)
        .training(**training_params)
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_duration=num_episodes,
            evaluation_duration_unit="episodes",
            evaluation_sample_timeout_s=36000,  # 10 hours
            evaluation_config={"explore": explore},
        )
    )

    algo = algo_config.build()
    algo.restore_from_path(str(checkpoint_path.absolute()))
    fix_optimizer_betas_after_checkpoint_load(algo)
    print(f"Loaded {algo_str.upper()} checkpoint from {checkpoint_path}")
    return algo


def _create_collecting_env(cfg=None, save_dir=None, env_config_path=None,
                           capacitance_model_checkpoint=None):
    """Env factory for Ray — creates DataCollectingEnv."""
    import jax
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    try:
        jax.clear_backends()
    except Exception:
        pass

    return DataCollectingEnv(
        save_dir=save_dir,
        return_voltage=True,
        env_config_path=env_config_path,
        capacitance_model_checkpoint=capacitance_model_checkpoint,
    )


# ---------------------------------------------------------------------------
# Main collection entrypoint
# ---------------------------------------------------------------------------

def collect_episodes(checkpoint_path, num_episodes=5, save_path=None, explore=False,
                     config_dir=None):
    """Run episodes via algo.evaluate(), collecting per-step data."""
    save_path = Path(save_path or DEFAULT_SAVE_PATH)

    # Temp dir for per-episode data files (written by env wrapper in Ray worker)
    data_save_dir = tempfile.mkdtemp(prefix="episode_data_")
    print(f"Temp data dir: {data_save_dir}")

    config, env_config, patched_config_path, algo_str = load_configs(
        checkpoint_path, config_dir
    )

    algo = build_algo(
        config, env_config, patched_config_path, algo_str, checkpoint_path,
        data_save_dir=data_save_dir, num_episodes=num_episodes, explore=explore,
    )

    # Run evaluation — DataCollectingEnv saves per-step data to data_save_dir
    print(f"\nRunning {num_episodes} evaluation episodes (explore={explore})...")
    result = algo.evaluate()

    eval_metrics = result.get('evaluation', {})
    avg_reward = eval_metrics.get('env_runners', {}).get('episode_reward_mean', None)
    if avg_reward is not None:
        print(f"Mean episode reward: {avg_reward:.2f}")

    # Collect all episode files into single .npy
    all_samples = []
    for f in sorted(Path(data_save_dir).glob("episode_*.npy")):
        data = np.load(f, allow_pickle=True)
        all_samples.extend(list(data))
        f.unlink()

    if not all_samples:
        print("WARNING: No data collected! Check if evaluation ran correctly.")
    else:
        n_episodes_actual = max(s['episode'] for s in all_samples) + 1
        solved = sum(
            1 for ep in range(n_episodes_actual)
            if max(s['step'] for s in all_samples if s['episode'] == ep) + 1
            < env_config['simulator']['max_steps']
        )
        print(f"\nCollected {len(all_samples)} samples across {n_episodes_actual} episodes "
              f"({solved}/{n_episodes_actual} solved)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, np.array(all_samples, dtype=object), allow_pickle=True)
    print(f"Saved to {save_path}")

    # Cleanup
    import ray, shutil
    ray.shutdown()
    try:
        os.unlink(patched_config_path)
    except OSError:
        pass
    try:
        shutil.rmtree(data_save_dir, ignore_errors=True)
    except OSError:
        pass

    return all_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect episode data for calibration analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Local checkpoint path. If not provided, downloads from wandb.")
    parser.add_argument("--wandb-artifact", type=str, default=DEFAULT_WANDB_ARTIFACT,
                        help="Wandb artifact name to download if --checkpoint not given")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Config directory (default: checkpoint dir or checkpoints/)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--explore", action="store_true",
                        help="Use stochastic actions (sample from distribution)")
    parser.add_argument("--output", type=str, default=None, help="Output .npy path")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        print(f"Downloading checkpoint from wandb: {args.wandb_artifact}")
        checkpoint = download_wandb_artifact(args.wandb_artifact)

    collect_episodes(
        checkpoint_path=checkpoint,
        num_episodes=args.episodes,
        save_path=args.output,
        explore=args.explore,
        config_dir=args.config_dir,
    )
