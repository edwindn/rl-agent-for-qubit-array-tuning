"""Smoke test for the MAPPO module path. Run as a script.

What it does:
  1. Loads env_config.yaml (the same file train.py reads).
  2. Builds an RLModuleSpec via factory.create_rl_module_spec(algo='mappo')
     — this instantiates CustomMAPPOCatalog and the routing encoder.
  3. Constructs a fake batch with the Dict observations the env wrapper produces
     under return_global_state=True.
  4. Runs forward_train (training path with EMBEDDINGS), _forward (inference),
     and compute_values (value-head path) end-to-end.
  5. Asserts on output shapes and writes a numerical / plot diagnostic dump
     under /tmp/mappo_module_smoke for visual inspection.

Run:
  uv run python src/qadapt/voltage_model/algorithms/test_mappo_smoke.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Resolve src/ on the path so this script works the same way as train.py.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[3]
sys.path.insert(0, str(_SRC))

from ray.rllib.core.columns import Columns

from qadapt.voltage_model.factory import create_rl_module_spec


def _load_env_config():
    path = _SRC / "qadapt" / "environment" / "env_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_neural_networks_config():
    """Minimal MAPPO rl_module_config (per-policy: backbone + heads + centralized_critic)."""
    impala_backbone = {
        "type": "IMPALA",
        "mobilenet_version": "small",
        "feature_size": 256,
        "load_pretrained": False,    # fast smoke test — skip torchvision download
        "freeze_backbone": False,
        "adaptive_pooling": True,
        "num_res_blocks": 2,
        "memory_layer": None,
    }
    base_block = {
        "backbone": impala_backbone,
        "policy_head": {
            "hidden_layers": [32],
            "activation": "relu",
            "use_attention": False,
        },
        "value_head": {  # unused under MAPPO but kept for parity
            "hidden_layers": [32],
            "activation": "relu",
            "use_attention": False,
        },
        "centralized_critic": {
            "backbone": impala_backbone,
            "value_head": {
                "hidden_layers": [32],
                "activation": "relu",
                "use_attention": False,
            },
        },
        "free_log_std": False,
        "log_std_bounds": [-10, 2],
    }
    return {
        "plunger_policy": dict(base_block),
        "barrier_policy": dict(base_block),
    }


def _fake_obs_batch(observation_space, batch_size=4):
    """Sample a fake batch matching the registered Dict observation_space."""
    obs = {}
    for key, sub_space in observation_space.spaces.items():
        # The CNN encoder permutes (B, H, W, C) -> (B, C, H, W) when C <= 8 (see
        # backbones.py); keep the (B, H, W, C) layout the env produces.
        sample = np.stack(
            [sub_space.sample() for _ in range(batch_size)], axis=0
        ).astype(np.float32)
        obs[key] = torch.from_numpy(sample)
    return obs


def _count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main():
    out_dir = Path("/tmp/mappo_module_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== MAPPO module smoke test ===")
    env_config = _load_env_config()
    print(f"Loaded env_config: resolution={env_config['simulator']['resolution']} "
          f"num_dots={env_config['simulator']['num_dots']}")

    nn_config = _build_neural_networks_config()
    spec = create_rl_module_spec(env_config, algo="mappo", config=nn_config)
    print(f"Built MultiRLModuleSpec with policies: {list(spec.rl_module_specs.keys())}")

    # Build the multi-RLModule. RLlib will construct the per-policy modules and
    # wire them with our CustomMAPPOCatalog.
    multi_module = spec.build()
    print(f"Built MultiRLModule: {type(multi_module).__name__}")

    summary_lines = []
    summary_lines.append(f"resolution: {env_config['simulator']['resolution']}")
    summary_lines.append(f"num_dots: {env_config['simulator']['num_dots']}")

    for policy_name in ("plunger_policy", "barrier_policy"):
        print(f"\n--- {policy_name} ---")
        module = multi_module[policy_name]
        obs_space = spec.rl_module_specs[policy_name].observation_space
        print(f"obs space keys: {list(obs_space.spaces.keys())}")
        for key, sub in obs_space.spaces.items():
            print(f"  {key}: shape={sub.shape}")

        # Param accounting
        total, trainable = _count_params(module)
        actor_total, _ = _count_params(module.encoder.actor_encoder)
        if module.encoder.critic_encoder is not None:
            critic_total, _ = _count_params(module.encoder.critic_encoder)
        else:
            critic_total = 0
        pi_total, _ = _count_params(module.pi)
        vf_total, _ = _count_params(module.vf)
        print(f"params: total={total:,} actor_enc={actor_total:,} "
              f"critic_enc={critic_total:,} pi={pi_total:,} vf={vf_total:,}")
        summary_lines.append(
            f"{policy_name}: total={total:,} actor_enc={actor_total:,} "
            f"critic_enc={critic_total:,} pi={pi_total:,} vf={vf_total:,}"
        )

        # Confirm critic_encoder receives global_image's channel count
        first_critic_conv = next(
            m for m in module.encoder.critic_encoder.modules()
            if isinstance(m, torch.nn.Conv2d)
        )
        first_actor_conv = next(
            m for m in module.encoder.actor_encoder.modules()
            if isinstance(m, torch.nn.Conv2d)
        )
        print(f"actor first-conv in_channels={first_actor_conv.in_channels} "
              f"(expected {obs_space['image'].shape[-1]})")
        print(f"critic first-conv in_channels={first_critic_conv.in_channels} "
              f"(expected {obs_space['global_image'].shape[-1]})")
        assert first_actor_conv.in_channels == obs_space["image"].shape[-1]
        assert first_critic_conv.in_channels == obs_space["global_image"].shape[-1]

        # Fake batch
        batch_size = 4
        obs = _fake_obs_batch(obs_space, batch_size=batch_size)
        batch = {Columns.OBS: obs}

        # Inference forward
        module.eval()
        with torch.no_grad():
            inf_out = module._forward(batch)
        assert Columns.ACTION_DIST_INPUTS in inf_out, (
            f"missing ACTION_DIST_INPUTS; got keys={list(inf_out.keys())}"
        )
        action_dim = obs_space.spaces["voltage"].shape[0]
        # PPO's gaussian dist requires 2 * action_dim inputs (mean + log_std)
        expected_logits = 2 * action_dim
        assert inf_out[Columns.ACTION_DIST_INPUTS].shape == (batch_size, expected_logits), (
            f"ACTION_DIST_INPUTS shape {inf_out[Columns.ACTION_DIST_INPUTS].shape} "
            f"!= ({batch_size}, {expected_logits})"
        )
        print(f"inference forward ok: ACTION_DIST_INPUTS"
              f"{tuple(inf_out[Columns.ACTION_DIST_INPUTS].shape)}")

        # Training forward (writes EMBEDDINGS as the centralized critic features)
        module.train()
        train_out = module._forward_train(batch)
        assert Columns.EMBEDDINGS in train_out, (
            f"missing EMBEDDINGS; got keys={list(train_out.keys())}"
        )
        emb = train_out[Columns.EMBEDDINGS]
        # Embeddings are a Dict {image_features, voltage} since the encoders
        # return that dict (see backbones.py). The value head consumes the dict.
        assert isinstance(emb, dict) and "image_features" in emb and "voltage" in emb, (
            f"unexpected EMBEDDINGS structure: {emb if not isinstance(emb, dict) else list(emb.keys())}"
        )
        print(f"training forward ok: EMBEDDINGS keys={list(emb.keys())} "
              f"image_features{tuple(emb['image_features'].shape)} "
              f"voltage{tuple(emb['voltage'].shape)}")
        # Sanity: critic voltage should have num_agents components (the global state),
        # not the per-agent action_dim.
        num_agents = obs_space["global_voltages"].shape[0]
        assert emb["voltage"].shape == (batch_size, num_agents), (
            f"critic voltage shape {tuple(emb['voltage'].shape)} != "
            f"({batch_size}, {num_agents}) -- the critic is not seeing the "
            f"global voltage vector."
        )

        # Compute values from EMBEDDINGS path (cheap, reuses the train output)
        values_with_emb = module.compute_values(batch, embeddings=emb)
        assert values_with_emb.shape == (batch_size,), (
            f"values shape {tuple(values_with_emb.shape)} != ({batch_size},)"
        )
        # Compute values from scratch (re-runs critic encoder; sanity check the
        # fallback path the learner would hit if EMBEDDINGS were dropped)
        values_from_scratch = module.compute_values(batch, embeddings=None)
        assert values_from_scratch.shape == (batch_size,)
        # Both paths should produce identical numbers (same weights, same input).
        max_abs = float((values_with_emb - values_from_scratch).abs().max())
        print(f"compute_values ok: shape{tuple(values_with_emb.shape)} "
              f"max|emb_path - scratch_path|={max_abs:.2e} (must be ~0)")
        assert max_abs < 1e-5, (
            f"values from embeddings vs from scratch differ by {max_abs}; "
            f"the routing encoder is not deterministic w.r.t. its inputs."
        )

        # Diagnostic plot: histogram of value outputs (from a freshly-init'd
        # network these will be tiny and roughly mean-zero — a useful sanity
        # check that the head wasn't built with a degenerate scale).
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(values_with_emb.detach().cpu().numpy(), bins=8)
            ax.set_title(f"{policy_name}: value head outputs (B={batch_size})")
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            fig.tight_layout()
            plot_path = out_dir / f"{policy_name}_values_hist.png"
            fig.savefig(plot_path, dpi=110)
            plt.close(fig)
            print(f"saved value histogram → {plot_path}")
        except ImportError:
            pass

    summary_path = out_dir / "module_smoke_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nWrote summary → {summary_path}")
    print("\n✓ MAPPO module smoke test passed!")


if __name__ == "__main__":
    main()
