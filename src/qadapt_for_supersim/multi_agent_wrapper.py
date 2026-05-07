"""
Multi-agent wrapper for SuperSimsEnv.

Two modes, selected by `policy_split` in supersims_env_config.yaml:

  policy_split: "per_qubit"
    One agent per qubit (qubit_0, qubit_1, ...). All agents share `qubit_policy`.
    Each agent sees its own staircase + params and emits a 5-d delta.

  policy_split: "per_param"  (default for new runs)
    One agent per (qubit, param) pair: qubit_{i}_omega01, qubit_{i}_omegad,
    qubit_{i}_phi, qubit_{i}_drive, qubit_{i}_beta. Five shared policies
    (omega01_policy, omegad_policy, phi_policy, drive_policy, beta_policy),
    each shared across qubits. Each agent emits a 1-d delta for its parameter.
    Mirrors the barrier/plunger split from the dot env.

In both modes the wrapper assembles the per-agent actions back into the
underlying env's stacked (N_QUBITS, 5) action tensor.
"""
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Add src directory to path for clean imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from qadapt_for_supersim.env import SuperSimsEnv  # noqa: E402


PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]
# Column order matches SuperSimsEnv's params layout: [omega_01, omega_d, phi, Omega, beta].
assert len(PARAM_NAMES) == 5

# Grouped mode: 2 policies covering disjoint param subsets. The freq group acts
# through the qubit/drive frequency response (parameters that interact tightly
# through the All-XY staircase); the env group shapes the pulse envelope.
# Indices match the column order of PARAM_NAMES / SuperSimsEnv params.
GROUP_NAMES = ["freq", "env"]
PARAM_GROUPS = {
    "freq": [0, 1, 2],   # omega_01, omega_d, phi
    "env":  [3, 4],      # Omega ("drive"), beta
}
assert sorted(idx for idxs in PARAM_GROUPS.values() for idx in idxs) == list(range(5))

_QUBIT_POLICY = "qubit_policy"


class SuperSimsMultiAgentWrapper(MultiAgentEnv):
    """Multi-agent wrapper. Mode selected by `policy_split` in env config."""

    def __init__(self, training: bool = True, config_path: str = "supersims_env_config.yaml"):
        super().__init__()
        self.base_env = SuperSimsEnv(training=training, config_path=config_path)

        self.n_qubits = self.base_env.n_qubits
        self.n_allxy = self.base_env.n_allxy
        self.n_params = self.base_env.n_params

        # SuperSimsEnv already loaded and stored the yaml; reuse it.
        self.policy_split = self.base_env.config.get("policy_split", "per_qubit")
        if self.policy_split not in ("per_qubit", "per_param", "grouped"):
            raise ValueError(
                f"policy_split must be 'per_qubit', 'per_param', or 'grouped'; "
                f"got {self.policy_split!r}"
            )

        # Build agent IDs and per-agent action shapes based on mode.
        # `agent_action_shapes` maps agent_id → action shape; used to construct
        # the action_spaces Dict (allows different action dims per agent in
        # grouped mode where freq → (3,) and env → (2,)).
        agent_action_shapes: Dict[str, tuple] = {}
        if self.policy_split == "per_qubit":
            self.all_agent_ids = [f"qubit_{i}" for i in range(self.n_qubits)]
            for aid in self.all_agent_ids:
                agent_action_shapes[aid] = (self.n_params,)
        elif self.policy_split == "per_param":
            self.all_agent_ids = [
                f"qubit_{i}_{pname}"
                for i in range(self.n_qubits)
                for pname in PARAM_NAMES
            ]
            for aid in self.all_agent_ids:
                agent_action_shapes[aid] = (1,)
        else:  # grouped
            self.all_agent_ids = [
                f"qubit_{i}_{gname}"
                for i in range(self.n_qubits)
                for gname in GROUP_NAMES
            ]
            for i in range(self.n_qubits):
                for gname in GROUP_NAMES:
                    agent_action_shapes[f"qubit_{i}_{gname}"] = (len(PARAM_GROUPS[gname]),)

        self._agent_ids = set(self.all_agent_ids)
        self.agents = self._agent_ids.copy()
        self.possible_agents = self._agent_ids.copy()

        # Per-agent obs space matches the underlying env's normalised obs:
        #   staircase: [-1, 1]   (= 2·P(|1⟩) − 1)
        #   params:    [-2, 2]   ((p − midpoint)/half_span; ±2 is the safety-rail headroom).
        per_agent_obs_space = spaces.Dict({
            "staircase": spaces.Box(low=-1.0, high=1.0,
                                    shape=(self.n_allxy,), dtype=np.float32),
            "params":    spaces.Box(low=-2.0, high=2.0,
                                    shape=(self.n_params,), dtype=np.float32),
        })

        self.observation_spaces = spaces.Dict({
            agent_id: per_agent_obs_space for agent_id in self.all_agent_ids
        })
        self.action_spaces = spaces.Dict({
            agent_id: spaces.Box(low=-1.0, high=1.0,
                                 shape=agent_action_shapes[agent_id], dtype=np.float32)
            for agent_id in self.all_agent_ids
        })
        self.observation_space = self.observation_spaces
        self.action_space = self.action_spaces

    def get_agent_ids(self):
        return list(self.all_agent_ids)

    def _qubit_obs(self, global_obs: dict, i: int) -> dict:
        return {
            "staircase": global_obs["staircase"][i].astype(np.float32),
            "params":    global_obs["params"][i].astype(np.float32),
        }

    def _split_obs(self, global_obs: dict) -> Dict[str, dict]:
        """Slice global stacked obs into per-agent dicts."""
        if self.policy_split == "per_qubit":
            return {f"qubit_{i}": self._qubit_obs(global_obs, i) for i in range(self.n_qubits)}
        out = {}
        if self.policy_split == "per_param":
            # Each of the 5 agents on a qubit gets the same per-qubit obs.
            for i in range(self.n_qubits):
                qubit_view = self._qubit_obs(global_obs, i)
                for pname in PARAM_NAMES:
                    out[f"qubit_{i}_{pname}"] = qubit_view
            return out
        # grouped: both group-agents on a qubit share the per-qubit obs.
        for i in range(self.n_qubits):
            qubit_view = self._qubit_obs(global_obs, i)
            for gname in GROUP_NAMES:
                out[f"qubit_{i}_{gname}"] = qubit_view
        return out

    def _stack_actions(self, agent_actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine per-agent actions into a stacked (N_QUBITS, 5) array."""
        if self.policy_split == "per_qubit":
            return np.stack([
                np.asarray(agent_actions[f"qubit_{i}"], dtype=np.float32)
                for i in range(self.n_qubits)
            ], axis=0)
        out = np.zeros((self.n_qubits, self.n_params), dtype=np.float32)
        if self.policy_split == "per_param":
            # Each agent emits a (1,) delta for its (qubit, param).
            for i in range(self.n_qubits):
                for k, pname in enumerate(PARAM_NAMES):
                    a = np.asarray(agent_actions[f"qubit_{i}_{pname}"], dtype=np.float32).reshape(-1)
                    if a.size != 1:
                        raise ValueError(
                            f"Per-param action for qubit_{i}_{pname} must be 1-d, got shape {a.shape}"
                        )
                    out[i, k] = a[0]
            return out
        # grouped: each agent emits an (action_dim,) delta covering its param subset.
        for i in range(self.n_qubits):
            for gname in GROUP_NAMES:
                idxs = PARAM_GROUPS[gname]
                a = np.asarray(agent_actions[f"qubit_{i}_{gname}"], dtype=np.float32).reshape(-1)
                if a.size != len(idxs):
                    raise ValueError(
                        f"Grouped action for qubit_{i}_{gname} must have shape ({len(idxs)},), "
                        f"got shape {a.shape}"
                    )
                for k, idx in enumerate(idxs):
                    out[i, idx] = a[k]
        return out

    def _per_agent_rewards(self, per_qubit_rewards: np.ndarray) -> Dict[str, float]:
        if self.policy_split == "per_qubit":
            return {f"qubit_{i}": float(per_qubit_rewards[i]) for i in range(self.n_qubits)}
        if self.policy_split == "per_param":
            # All 5 param-agents on qubit i share the qubit's reward.
            return {
                f"qubit_{i}_{pname}": float(per_qubit_rewards[i])
                for i in range(self.n_qubits) for pname in PARAM_NAMES
            }
        # grouped: both group-agents on qubit i share the qubit's reward.
        return {
            f"qubit_{i}_{gname}": float(per_qubit_rewards[i])
            for i in range(self.n_qubits) for gname in GROUP_NAMES
        }

    def reset(self, *, seed=None, options=None):
        global_obs, global_info = self.base_env.reset(seed=seed, options=options)
        agent_obs = self._split_obs(global_obs)
        per_qubit_rewards = global_info["per_qubit_rewards"]
        rewards = self._per_agent_rewards(per_qubit_rewards)
        agent_infos = {
            agent_id: {"reward": rewards[agent_id], "step": global_info["step"]}
            for agent_id in self.all_agent_ids
        }
        return agent_obs, agent_infos

    def step(self, agent_actions: Dict[str, np.ndarray]):
        assert set(agent_actions.keys()) == self._agent_ids, (
            f"Agent action keys {set(agent_actions.keys())} != expected {self._agent_ids}"
        )
        global_action = self._stack_actions(agent_actions)
        global_obs, _, terminated, truncated, global_info = self.base_env.step(global_action)

        agent_obs = self._split_obs(global_obs)
        per_qubit_rewards = global_info["per_qubit_rewards"]
        agent_rewards = self._per_agent_rewards(per_qubit_rewards)
        agent_terminated = dict.fromkeys(self.all_agent_ids, terminated)
        agent_terminated["__all__"] = terminated
        agent_truncated = dict.fromkeys(self.all_agent_ids, truncated)
        agent_truncated["__all__"] = truncated
        agent_infos = {
            agent_id: {"reward": agent_rewards[agent_id], "step": global_info["step"]}
            for agent_id in self.all_agent_ids
        }
        return agent_obs, agent_rewards, agent_terminated, agent_truncated, agent_infos


# ----- Smoke tests ----- #

def _smoke_per_qubit():
    print("\n[Test 1] per_qubit mode (legacy): N agents × 5-d action")
    # Force per_qubit by writing a temp config.
    import tempfile, textwrap
    cfg_text = textwrap.dedent("""\
        simulator:
          max_steps: 20
          alone_enabled: false
        env_type: "supersims"
        policy_split: "per_qubit"
    """)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(cfg_text)
        tmp = f.name
    wrapper = SuperSimsMultiAgentWrapper(config_path=tmp)
    agent_ids = wrapper.get_agent_ids()
    assert agent_ids == [f"qubit_{i}" for i in range(wrapper.n_qubits)], agent_ids
    obs, info = wrapper.reset(seed=42)
    assert all(obs[a]["staircase"].shape == (wrapper.n_allxy,) for a in agent_ids)
    actions = {a: np.zeros((wrapper.n_params,), dtype=np.float32) for a in agent_ids}
    obs, rewards, term, trunc, infos = wrapper.step(actions)
    assert set(rewards.keys()) == set(agent_ids)
    assert all(0.0 <= rewards[a] <= 1.0 for a in agent_ids)
    print(f"    PASS — {len(agent_ids)} agents, action shape (5,)")


def _smoke_per_param_shapes():
    print("\n[Test 2] per_param mode: N×5 agents × 1-d action")
    wrapper = SuperSimsMultiAgentWrapper()  # uses default supersims_env_config.yaml
    if wrapper.policy_split != "per_param":
        print("    Skipped — config default is not per_param.")
        return
    agent_ids = wrapper.get_agent_ids()
    expected = [f"qubit_{i}_{p}" for i in range(wrapper.n_qubits) for p in PARAM_NAMES]
    assert agent_ids == expected, agent_ids
    obs, info = wrapper.reset(seed=42)
    for a in agent_ids:
        assert obs[a]["staircase"].shape == (wrapper.n_allxy,), (a, obs[a]["staircase"].shape)
        assert obs[a]["params"].shape == (wrapper.n_params,), (a, obs[a]["params"].shape)
    actions = {a: np.zeros((1,), dtype=np.float32) for a in agent_ids}
    obs, rewards, term, trunc, infos = wrapper.step(actions)
    assert set(rewards.keys()) == set(agent_ids)
    # All 5 param-agents on the same qubit must have IDENTICAL reward.
    for i in range(wrapper.n_qubits):
        rs = [rewards[f"qubit_{i}_{p}"] for p in PARAM_NAMES]
        assert all(r == rs[0] for r in rs), (i, rs)
    # Different qubits should usually have different rewards (sanity).
    qubit0_r = rewards[f"qubit_0_{PARAM_NAMES[0]}"]
    qubit1_r = rewards[f"qubit_1_{PARAM_NAMES[0]}"]
    print(f"    PASS — {len(agent_ids)} agents (4×5 = 20), q0_r={qubit0_r:.3f} q1_r={qubit1_r:.3f}")


def _smoke_per_param_action_assembly():
    print("\n[Test 3] per_param: action assembly == per_qubit equivalent")
    # Build both wrappers with the SAME sampling config so determinism is the only
    # variable being tested. Each env now owns its sampling cfg (no global monkey
    # patching), so we just declare both with the same parameter_config_filename.
    import tempfile, textwrap
    common_sim_cfg = textwrap.dedent("""\
        simulator:
          max_steps: 20
          alone_enabled: false
        env_type: "supersims"
        parameter_config_filename: "parameter_config_medium.json"
    """)
    pq_cfg = common_sim_cfg + 'policy_split: "per_qubit"\n'
    pp_cfg = common_sim_cfg + 'policy_split: "per_param"\n'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pq_cfg)
        pq_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pp_cfg)
        pp_path = f.name
    wrapper_pq = SuperSimsMultiAgentWrapper(config_path=pq_path)
    wrapper_pp = SuperSimsMultiAgentWrapper(config_path=pp_path)

    if wrapper_pp.policy_split != "per_param":
        print("    Skipped — default config not per_param.")
        return

    SEED = 99
    obs_pq, _ = wrapper_pq.reset(seed=SEED)
    obs_pp, _ = wrapper_pp.reset(seed=SEED)

    rng = np.random.default_rng(0)
    for step in range(3):
        flat = rng.uniform(-0.05, 0.05, size=(wrapper_pq.n_qubits, wrapper_pq.n_params)).astype(np.float32)
        actions_pq = {f"qubit_{i}": flat[i] for i in range(wrapper_pq.n_qubits)}
        actions_pp = {
            f"qubit_{i}_{pname}": np.array([flat[i, k]], dtype=np.float32)
            for i in range(wrapper_pp.n_qubits)
            for k, pname in enumerate(PARAM_NAMES)
        }
        _, r_pq, _, _, _ = wrapper_pq.step(actions_pq)
        _, r_pp, _, _, _ = wrapper_pp.step(actions_pp)
        # qubit_i reward in pq mode == any qubit_i_<param> reward in pp mode
        for i in range(wrapper_pq.n_qubits):
            assert abs(r_pq[f"qubit_{i}"] - r_pp[f"qubit_{i}_{PARAM_NAMES[0]}"]) < 1e-6, (
                step, i, r_pq[f"qubit_{i}"], r_pp[f"qubit_{i}_{PARAM_NAMES[0]}"]
            )
    print(f"    PASS — per_param assembled actions reproduce per_qubit rewards exactly across 3 steps.")


def _smoke_grouped_shapes():
    print("\n[Test 4] grouped mode: N×2 agents with action dims (3,) and (2,)")
    import tempfile, textwrap
    cfg_text = textwrap.dedent("""\
        simulator:
          max_steps: 20
          alone_enabled: false
        env_type: "supersims"
        policy_split: "grouped"
    """)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(cfg_text)
        tmp = f.name
    wrapper = SuperSimsMultiAgentWrapper(config_path=tmp)
    agent_ids = wrapper.get_agent_ids()
    expected = [f"qubit_{i}_{g}" for i in range(wrapper.n_qubits) for g in GROUP_NAMES]
    assert agent_ids == expected, agent_ids
    obs, info = wrapper.reset(seed=42)
    for a in agent_ids:
        assert obs[a]["staircase"].shape == (wrapper.n_allxy,), (a, obs[a]["staircase"].shape)
        assert obs[a]["params"].shape == (wrapper.n_params,), (a, obs[a]["params"].shape)
    # Per-agent action shapes match the group sizes.
    for i in range(wrapper.n_qubits):
        assert wrapper.action_spaces[f"qubit_{i}_freq"].shape == (3,)
        assert wrapper.action_spaces[f"qubit_{i}_env"].shape  == (2,)
    # Step with zero actions of the right shape.
    actions = {}
    for i in range(wrapper.n_qubits):
        actions[f"qubit_{i}_freq"] = np.zeros((3,), dtype=np.float32)
        actions[f"qubit_{i}_env"]  = np.zeros((2,), dtype=np.float32)
    obs, rewards, term, trunc, infos = wrapper.step(actions)
    # Both group-agents on the same qubit must have IDENTICAL reward.
    for i in range(wrapper.n_qubits):
        assert rewards[f"qubit_{i}_freq"] == rewards[f"qubit_{i}_env"], i
    print(f"    PASS — {len(agent_ids)} agents (4×2 = 8), shapes (3,) + (2,)")


def _smoke_grouped_action_assembly():
    print("\n[Test 5] grouped: action assembly == per_qubit equivalent")
    # Build both wrappers with the SAME sampling config; verify rewards match.
    import tempfile, textwrap
    common_sim_cfg = textwrap.dedent("""\
        simulator:
          max_steps: 20
          alone_enabled: false
        env_type: "supersims"
        parameter_config_filename: "parameter_config_medium.json"
    """)
    pq_cfg = common_sim_cfg + 'policy_split: "per_qubit"\n'
    gr_cfg = common_sim_cfg + 'policy_split: "grouped"\n'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(pq_cfg)
        pq_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(gr_cfg)
        gr_path = f.name
    wrapper_pq = SuperSimsMultiAgentWrapper(config_path=pq_path)
    wrapper_gr = SuperSimsMultiAgentWrapper(config_path=gr_path)

    SEED = 99
    obs_pq, _ = wrapper_pq.reset(seed=SEED)
    obs_gr, _ = wrapper_gr.reset(seed=SEED)

    rng = np.random.default_rng(0)
    for step in range(3):
        flat = rng.uniform(-0.05, 0.05, size=(wrapper_pq.n_qubits, wrapper_pq.n_params)).astype(np.float32)
        actions_pq = {f"qubit_{i}": flat[i] for i in range(wrapper_pq.n_qubits)}
        actions_gr = {}
        for i in range(wrapper_gr.n_qubits):
            actions_gr[f"qubit_{i}_freq"] = flat[i, PARAM_GROUPS["freq"]]
            actions_gr[f"qubit_{i}_env"]  = flat[i, PARAM_GROUPS["env"]]
        _, r_pq, _, _, _ = wrapper_pq.step(actions_pq)
        _, r_gr, _, _, _ = wrapper_gr.step(actions_gr)
        for i in range(wrapper_pq.n_qubits):
            r_pq_i = r_pq[f"qubit_{i}"]
            r_gr_i = r_gr[f"qubit_{i}_freq"]
            assert abs(r_pq_i - r_gr_i) < 1e-6, (step, i, r_pq_i, r_gr_i)
    print(f"    PASS — grouped actions assembled into the same physical params as per_qubit; "
          f"rewards match exactly across 3 steps.")


if __name__ == "__main__":
    print("=== SuperSimsMultiAgentWrapper smoke tests (per_qubit + per_param + grouped) ===")
    _smoke_per_qubit()
    _smoke_per_param_shapes()
    _smoke_per_param_action_assembly()
    _smoke_grouped_shapes()
    _smoke_grouped_action_assembly()
    print("\nAll smoke tests passed.")
