"""
Task 4a+4b diagnostics — static unit tests on the CNN agents and GroupedMAC.
No training required. Run BEFORE kicking off the grouped smoke training.

Usage:
    uv run --extra facmac python benchmarks/MARL/facmac/tests/task_4_diagnostics.py

Produces under benchmarks/MARL/facmac/diagnostics/task_4/:

    1. summary.txt               — all assertion results + per-layer param counts
    2. shape_trace.txt           — detailed shape walk through each CNN agent
    3. sentinel_dispatch.txt     — obs-routing proof inside GroupedMAC
    4. param_isolation.txt       — gradient-update isolation proof (plunger-only loss
                                   must not change barrier params, and vice versa)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn as nn
from gym.spaces import Box

_BENCH_DIR = Path(__file__).resolve().parent.parent
_VENDOR_DIR = _BENCH_DIR / "vendor"
_PROJECT_SRC = _BENCH_DIR.parent.parent / "src"
for p in (_VENDOR_DIR, _PROJECT_SRC, _BENCH_DIR):
    sys.path.insert(0, str(p))

from agents.plunger_cnn import PlungerCNNAgent
from agents.barrier_cnn import BarrierCNNAgent
from grouped_mac import GroupedMAC

OUT_DIR = _BENCH_DIR / "diagnostics" / "task_4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H, W = 48, 48
N_PLUNGER = 4
N_BARRIER = 3
N_AGENTS = N_PLUNGER + N_BARRIER
FLAT_OBS = 2 * H * W
BATCH = 3
HIDDEN_DIM = 64


def _build_args() -> SimpleNamespace:
    return SimpleNamespace(
        n_agents=N_AGENTS,
        n_actions=1,
        rnn_hidden_dim=HIDDEN_DIM,
        agent="grouped",
        agent_output_type=None,
        action_selector=None,
        agent_groups={"plunger": list(range(N_PLUNGER)),
                      "barrier": list(range(N_PLUNGER, N_AGENTS))},
        group_obs_shapes={"plunger": (2, H, W), "barrier": (1, H, W)},
        agent_modules={"plunger": "plunger_cnn", "barrier": "barrier_cnn"},
        action_spaces=[Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)] * N_AGENTS,
        exploration_mode="gaussian",
        start_steps=0,
        act_noise=0.0,
    )


def _build_mac(args) -> GroupedMAC:
    scheme = {
        "obs": {"vshape": FLAT_OBS, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.float},
        "avail_actions": {"vshape": (1,), "group": "agents", "dtype": th.int},
        "state": {"vshape": N_AGENTS * FLAT_OBS},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": N_AGENTS}
    return GroupedMAC(scheme, groups, args)


def shape_trace() -> list[str]:
    args = _build_args()
    plunger = PlungerCNNAgent((2, H, W), args)
    barrier = BarrierCNNAgent((1, H, W), args)

    lines = [f"=== Shape traces (input H x W = {H} x {W}) ==="]
    for name, agent, C in (("plunger", plunger, 2), ("barrier", barrier, 1)):
        dummy = th.zeros(BATCH, C, H, W)
        lines.append(f"\n-- {name} agent --")
        lines.append(f"  input:           {tuple(dummy.shape)}")
        x = dummy
        for i, layer in enumerate(agent.cnn):
            x = layer(x)
            lines.append(f"  cnn[{i}] {type(layer).__name__:<12s} -> {tuple(x.shape)}")
        x = agent.fc1(x)
        lines.append(f"  fc1            -> {tuple(x.shape)}")
        x = agent.fc2(x)
        lines.append(f"  fc2 (actions)  -> {tuple(x.shape)}")

        total = sum(p.numel() for p in agent.parameters())
        lines.append(f"  total params:    {total:,}")
        for pname, p in agent.named_parameters():
            lines.append(f"    {pname:<18s} {tuple(p.shape)}  numel={p.numel():,}")

    out = OUT_DIR / "shape_trace.txt"
    out.write_text("\n".join(lines))
    return lines + [f"\n  [written] {out}"]


def gradient_flow_check() -> list[str]:
    args = _build_args()
    lines = ["=== Gradient flow per CNN agent ==="]
    for name, AgentCls, C in (
        ("plunger", PlungerCNNAgent, 2),
        ("barrier", BarrierCNNAgent, 1),
    ):
        agent = AgentCls((C, H, W), args)
        x = th.randn(BATCH, C, H, W, requires_grad=False)
        hidden = agent.init_hidden().expand(BATCH, -1)
        out = agent(x, hidden)
        loss = out["actions"].mean()
        loss.backward()
        zero_layers = []
        for pname, p in agent.named_parameters():
            g = p.grad
            if g is None:
                zero_layers.append((pname, "NONE"))
            elif float(g.abs().sum()) == 0.0:
                zero_layers.append((pname, "zero-sum"))
        status = "OK (all layers got non-zero grads)" if not zero_layers else f"BAD: {zero_layers}"
        lines.append(f"  {name}: {status}")
    return lines


def sentinel_dispatch() -> list[str]:
    """
    Obs routing test: fill plunger obs with +1.0 and barrier obs with -1.0, then
    monkey-patch each group's agent to output mean(input) instead of a normal
    forward. Correct routing means:
        - plunger output slots (canonical idx 0..3) should be +1.0
        - barrier output slots (canonical idx 4..6) should be -1.0 (pad already stripped)
    A swap or pad-leakage bug changes these values.
    """
    args = _build_args()
    mac = _build_mac(args)

    plunger_sentinel = +1.0
    barrier_sentinel = -1.0

    obs = th.zeros(BATCH, N_AGENTS, FLAT_OBS)
    for i in args.agent_groups["plunger"]:
        obs[:, i] = plunger_sentinel
    for i in args.agent_groups["barrier"]:
        # Barrier's real channel is the first H*W slice; zero-pad is the second.
        obs[:, i, :H * W] = barrier_sentinel
        obs[:, i, H * W:] = 0.0  # pad

    class ProbeAgent(nn.Module):
        native_channels = None
        def __init__(self, expected_C):
            super().__init__()
            self.expected_C = expected_C
            self.native_channels = expected_C
        def init_hidden(self):
            return th.zeros(1, HIDDEN_DIM)
        def forward(self, inputs, hidden_state, actions=None):
            assert inputs.shape[1] == self.expected_C, f"agent saw {inputs.shape[1]} channels, expected {self.expected_C}"
            batch = inputs.shape[0]
            mean_val = inputs.view(batch, -1).mean(dim=1, keepdim=True)
            return {"actions": mean_val, "hidden_state": hidden_state}

    mac.agents = nn.ModuleDict({"plunger": ProbeAgent(2), "barrier": ProbeAgent(1)})
    mac.init_hidden(BATCH)

    class FakeBatch:
        def __init__(self, obs_no_time):
            self.batch_size = obs_no_time.shape[0]
            # Vendor does batch["obs"][:, t], so add a time axis of length 1.
            self._obs = obs_no_time.unsqueeze(1)
        def __getitem__(self, key):
            return self._obs
        @property
        def device(self):
            return th.device("cpu")

    batch = FakeBatch(obs)
    out = mac.forward(batch, t=0, select_actions=True)
    actions = out["actions"].detach().cpu().numpy()

    lines = ["=== Sentinel dispatch test ==="]
    lines.append(f"  plunger sentinel = {plunger_sentinel:+.1f}, barrier sentinel = {barrier_sentinel:+.1f}")
    lines.append(f"  action_out shape = {actions.shape}   (expected ({BATCH}, {N_AGENTS}, 1))")

    plunger_out = actions[:, args.agent_groups["plunger"], 0]
    barrier_out = actions[:, args.agent_groups["barrier"], 0]
    lines.append(f"  mean plunger slots = {plunger_out.mean():+.4f}   (expect +1.0)")
    lines.append(f"  mean barrier slots = {barrier_out.mean():+.4f}   (expect -1.0)")

    plunger_ok = np.allclose(plunger_out, plunger_sentinel, atol=1e-5)
    barrier_ok = np.allclose(barrier_out, barrier_sentinel, atol=1e-5)
    lines.append(f"  plunger routing OK: {plunger_ok}")
    lines.append(f"  barrier routing OK: {barrier_ok}")

    out_path = OUT_DIR / "sentinel_dispatch.txt"
    out_path.write_text("\n".join(lines))
    return lines + [f"  [written] {out_path}"]


def param_isolation_test() -> list[str]:
    """
    One backward pass where the loss touches only the plunger outputs must leave
    barrier agent params untouched, and vice versa.
    """
    args = _build_args()
    mac = _build_mac(args)
    mac.init_hidden(BATCH)

    obs = th.randn(BATCH, N_AGENTS, FLAT_OBS)

    class FakeBatch:
        def __init__(self, obs_no_time):
            self.batch_size = obs_no_time.shape[0]
            # Vendor does batch["obs"][:, t], so add a time axis of length 1.
            self._obs = obs_no_time.unsqueeze(1)
        def __getitem__(self, key):
            return self._obs
        @property
        def device(self):
            return th.device("cpu")

    lines = ["=== Parameter-update isolation ==="]
    for target_group, other_group in (("plunger", "barrier"), ("barrier", "plunger")):
        for p in mac.parameters():
            if p.grad is not None:
                p.grad.zero_()

        out = mac.forward(FakeBatch(obs), t=0, select_actions=True)
        target_idx = args.agent_groups[target_group]
        loss = out["actions"][:, target_idx, :].sum()
        loss.backward()

        target_has_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in mac.agents[target_group].parameters()
        )
        other_has_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in mac.agents[other_group].parameters()
        )

        lines.append(
            f"  loss on {target_group} only:  "
            f"{target_group} grads non-zero = {target_has_grad}, "
            f"{other_group} grads non-zero = {other_has_grad}  "
            f"{'OK' if target_has_grad and not other_has_grad else 'BAD'}"
        )

    out_path = OUT_DIR / "param_isolation.txt"
    out_path.write_text("\n".join(lines))
    return lines + [f"  [written] {out_path}"]


def main() -> None:
    print(f"Writing diagnostics to {OUT_DIR}\n")

    summary: list[str] = []

    print("[1/4] Shape traces")
    summary.extend(shape_trace())
    summary.append("")

    print("[2/4] Gradient flow")
    summary.extend(gradient_flow_check())
    summary.append("")

    print("[3/4] Sentinel dispatch")
    summary.extend(sentinel_dispatch())
    summary.append("")

    print("[4/4] Param isolation")
    summary.extend(param_isolation_test())

    out = OUT_DIR / "summary.txt"
    out.write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\n  [written] {out}")


if __name__ == "__main__":
    main()
