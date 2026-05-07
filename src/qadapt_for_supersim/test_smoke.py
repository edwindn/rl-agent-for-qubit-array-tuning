"""
Stage 3 smoke tests for the SuperSims RL plumbing.

Covers both policy_split modes:
  per_qubit — single shared `qubit_policy`, action shape (5,), 10 dist inputs.
  per_param — five shared policies (one per parameter), action shape (1,),
              2 dist inputs each.

Each test:
  1. Build the multi-RLModule spec (catalog + encoder + heads instantiate cleanly).
  2. Forward pass on a fake batch (correct output shapes).
  3. Tiny gradient step (PPO loss math doesn't crash).

Run:
  CUDA_VISIBLE_DEVICES=1 uv run python src/qadapt/voltage_model/test_supersims_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "src"))

from qadapt.voltage_model.create_rl_module import create_rl_module_spec  # noqa: E402
from qadapt.voltage_model.custom_neural_nets import (  # noqa: E402
    MLPEncoder,
    MLPFlatPolicyHead,
    MLPFlatValueHead,
)


_BACKBONE = {
    "backbone": {
        "type": "MLP",
        "hidden_layers": [64, 64],
        "feature_size": 64,
        "activation": "relu",
        "memory_layer": None,
    },
    "policy_head": {
        "hidden_layers": [64, 32],
        "activation": "relu",
        "use_attention": False,
    },
    "value_head": {
        "hidden_layers": [64, 32],
        "activation": "relu",
        "use_attention": False,
    },
    "free_log_std": False,
    "log_std_bounds": [-10, -1.5],
}

_PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]


def _build(env_config: dict, nn_config: dict):
    spec = create_rl_module_spec(env_config, algo="ppo", config=nn_config)
    return spec, spec.build()


def _check_module_layout(module):
    encoder = module.encoder
    if hasattr(encoder, "actor_encoder"):
        inner = encoder.actor_encoder
        layout = "split"
    elif hasattr(encoder, "encoder"):
        inner = encoder.encoder
        layout = "shared"
    else:
        raise AssertionError(f"Unexpected encoder layout: {type(encoder)}")
    assert isinstance(inner, MLPEncoder), f"inner encoder is {type(inner)}"
    assert isinstance(module.pi, MLPFlatPolicyHead), f"pi head is {type(module.pi)}"
    assert isinstance(module.vf, MLPFlatValueHead), f"vf head is {type(module.vf)}"
    return layout, inner


def _fwd_and_grad(module, expected_dist_inputs: int):
    from ray.rllib.core.columns import Columns
    B = 8
    batch = {"obs": {"staircase": torch.rand(B, 21), "params": torch.randn(B, 5)}}
    out = module._forward_train(batch)
    adi = out[Columns.ACTION_DIST_INPUTS]
    assert adi.shape == (B, expected_dist_inputs), adi.shape
    embeddings = out.get(Columns.EMBEDDINGS, None)
    vf = module.compute_values(batch, embeddings=embeddings)
    assert vf.shape == (B,), vf.shape

    optim = torch.optim.Adam(module.parameters(), lr=1e-3)
    loss = (vf - 0.0).pow(2).mean() + adi.pow(2).mean()
    loss.backward()
    grads = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
    assert any(g > 0 for g in grads), "no gradients flowed"
    optim.step()


def _smoke_per_qubit():
    print("\n[per_qubit] build + forward + grad")
    env_config = {"env_type": "supersims", "policy_split": "per_qubit"}
    nn_config = {"qubit_policy": _BACKBONE}
    spec, mm = _build(env_config, nn_config)
    assert list(spec.rl_module_specs.keys()) == ["qubit_policy"]
    layout, inner = _check_module_layout(mm["qubit_policy"])
    print(f"    Encoder layout={layout}, inner.output_dims={inner.output_dims}")
    _fwd_and_grad(mm["qubit_policy"], expected_dist_inputs=10)  # 5 mean + 5 log_std
    print("    PASS")


def _smoke_per_param():
    print("\n[per_param] build + forward + grad for each of 5 policies")
    env_config = {"env_type": "supersims", "policy_split": "per_param"}
    nn_config = {f"{p}_policy": _BACKBONE for p in _PARAM_NAMES}
    spec, mm = _build(env_config, nn_config)
    expected = {f"{p}_policy" for p in _PARAM_NAMES}
    assert set(spec.rl_module_specs.keys()) == expected, list(spec.rl_module_specs.keys())
    for pname in _PARAM_NAMES:
        pol_id = f"{pname}_policy"
        module = mm[pol_id]
        _check_module_layout(module)
        _fwd_and_grad(module, expected_dist_inputs=2)  # 1 mean + 1 log_std per agent
    print(f"    PASS — all 5 per-param policies built, forward/grad OK")


def _smoke_grouped():
    """grouped mode: 2 policies (freq_policy action_dim=3, env_policy action_dim=2)."""
    print("\n[grouped] build + forward + grad for freq_policy and env_policy")
    env_config = {"env_type": "supersims", "policy_split": "grouped"}
    nn_config = {f"{g}_policy": _BACKBONE for g in ["freq", "env"]}
    spec, mm = _build(env_config, nn_config)
    assert set(spec.rl_module_specs.keys()) == {"freq_policy", "env_policy"}, list(spec.rl_module_specs.keys())
    # freq_policy: action_dim=3 → ADI shape (B, 6) (3 mean + 3 log_std)
    _check_module_layout(mm["freq_policy"])
    _fwd_and_grad(mm["freq_policy"], expected_dist_inputs=6)
    # env_policy: action_dim=2 → ADI shape (B, 4) (2 mean + 2 log_std)
    _check_module_layout(mm["env_policy"])
    _fwd_and_grad(mm["env_policy"], expected_dist_inputs=4)
    print(f"    PASS — freq_policy (3-D), env_policy (2-D), forward/grad OK")


def _smoke_free_log_std():
    """free_log_std=True: log_std is a single learnable nn.Parameter, not a network output.
    Build per-param policies in this mode; assert the log_std half is exactly the init
    value across batch rows, the mean half varies, the parameter receives gradient."""
    print("\n[free_log_std] state-independent log_std for per_param policies")
    from ray.rllib.core.columns import Columns
    env_config = {"env_type": "supersims", "policy_split": "per_param"}
    backbone_free = {**_BACKBONE, "free_log_std": True, "log_std_init": -2.3, "log_std_bounds": None}
    nn_config = {f"{p}_policy": backbone_free for p in _PARAM_NAMES}
    spec, mm = _build(env_config, nn_config)
    B = 8
    for pname in _PARAM_NAMES:
        module = mm[f"{pname}_policy"]
        # Confirm policy head has the standalone parameter.
        assert hasattr(module.pi, "log_std_param"), f"{pname}: missing log_std_param"
        assert module.pi.log_std_param.requires_grad, f"{pname}: log_std_param must be trainable"
        assert module.pi.log_std_param.shape == (1,), module.pi.log_std_param.shape
        assert float(module.pi.log_std_param.detach().abs().mean() - 2.3) < 1e-4, (
            f"{pname}: log_std_param not initialised at -2.3"
        )

        batch = {"obs": {"staircase": torch.rand(B, 21), "params": torch.randn(B, 5)}}
        out = module._forward_train(batch)
        adi = out[Columns.ACTION_DIST_INPUTS]
        assert adi.shape == (B, 2), f"{pname}: expected (B,2), got {adi.shape}"
        mean, log_std = adi[:, 0], adi[:, 1]
        # log_std half should be EXACTLY the init value for every row (untrained).
        assert torch.allclose(log_std, torch.full_like(log_std, -2.3), atol=1e-5), (
            f"{pname}: log_std not constant at init: {log_std}"
        )
        # mean half should vary across rows (network is using obs, not constant).
        assert mean.std().item() > 1e-6, f"{pname}: mean is constant across batch ({mean})"

        # Gradient flows into log_std_param.
        loss = adi.pow(2).mean()
        module.pi.log_std_param.grad = None
        loss.backward()
        g = module.pi.log_std_param.grad
        assert g is not None and g.abs().item() > 0, (
            f"{pname}: log_std_param received no gradient ({g})"
        )
    print(f"    PASS — log_std fixed at -2.3, mean varies, gradient flows into log_std_param")


if __name__ == "__main__":
    print("=== SuperSims RLlib plumbing smoke tests ===")
    _smoke_per_qubit()
    _smoke_per_param()
    _smoke_grouped()
    _smoke_free_log_std()
    print("\nAll Stage 3 smoke tests passed.")
