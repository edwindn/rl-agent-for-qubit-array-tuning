"""
Single-agent gym wrapper around Cornelius's SuperSims All-XY calibration sim.

Mirrors the role of QuantumDeviceEnv (env.py) for the dot tuning task. Per-step:
  1. Apply the agent's normalised delta through the compensation tensor.
  2. Rebuild the compensation tensor (full virtualisation, every step).
  3. Run the All-XY simulation.
  4. Compute per-qubit rewards from the staircase.

Reuses the SuperSims utilities directly — no re-implementation of physics.

The multi-agent wrapper (Stage 2) splits this env's stacked (N_QUBITS, ...) outputs
into per-qubit agents.
"""
import sys
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from gymnasium import spaces

# SuperSims is a sibling top-level directory of src/. Ship it to Ray workers by
# symlinking it under src/swarm/_supersims (so it rides along with working_dir).
# At runtime, the symlink target resolves to either the real SuperSims/ (locally)
# or the unpacked working_dir copy (on workers). Both look the same to Python.
_SUPERSIMS_DIR = Path(__file__).resolve().parent.parent / "_supersims"
if _SUPERSIMS_DIR.exists() and str(_SUPERSIMS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPERSIMS_DIR))

import parameter_generation as pg  # noqa: E402
from parameter_generation import N_QUBITS  # noqa: E402
from all_xy_sequence import N_ALLXY, run_allxy_simulation  # noqa: E402
from compensation_matrix import build_compensation, update_params  # noqa: E402
from reward import allxy_rewards  # noqa: E402
import normalisations as norm  # noqa: E402

jax.config.update("jax_enable_x64", True)


_N_PARAMS = 5  # [omega_01, omega_d, phi, Omega, beta]


class SuperSimsEnv(gym.Env):
    """
    Stacked (single-agent) view of the All-XY calibration env.

    Observation:
        Dict({
            "staircase": Box(0, 1, (N_QUBITS, N_ALLXY)),  # P(|1⟩) per qubit per gate sequence
            "params":    Box(-inf, inf, (N_QUBITS, N_PARAMS)),  # raw physical params
        })

    Action:
        Box(-1, 1, (N_QUBITS, N_PARAMS))  # normalised delta per qubit; ±1 = one half-span

    Reward (single-agent):
        Mean of per-qubit rewards. Per-qubit rewards are returned in info for the multi-agent wrapper.
    """

    metadata = {"render_modes": []}

    def __init__(self, training: bool = True, config_path: str = "supersims_env_config.yaml"):
        super().__init__()
        self.config = self._load_config(config_path)
        self.training = training
        self.max_steps = int(self.config["simulator"]["max_steps"])
        self.alone_enabled = bool(self.config["simulator"]["alone_enabled"])
        self._delta_scale_factor = float(self.config.get("delta_scale_factor", 1.0))

        # Diagnostic knobs: pin selected params at GT and/or zero hardware/crosstalk.
        # `pin_to_gt` is a list of param names from {omega_01, omega_d, phi, Omega, beta}
        # that are overridden to their perfect-pulse setpoint at episode start (after
        # the sampler runs). Useful for fix-N-vary-(5−N) ablations (e.g., let only
        # omega_d vary across the full distribution, others nailed at GT).
        self._pin_to_gt = list(self.config.get("pin_to_gt") or [])
        self._zero_hw = bool(self.config.get("zero_hw", False))
        self._zero_crosstalk = bool(self.config.get("zero_crosstalk", False))
        self._beta_gt = float(self.config.get("beta_gt", 0.5))   # raised-cosine optimum
        valid_pins = {"omega_01", "omega_d", "phi", "Omega", "beta"}
        bad = [p for p in self._pin_to_gt if p not in valid_pins]
        assert not bad, f"pin_to_gt contains unknown param names: {bad} (valid: {sorted(valid_pins)})"

        # Each env owns its sampling config and (re-)applies it before every
        # sample_all call. This avoids the order-of-construction coupling that arises
        # from `pg._cfg` being shared module-global state — multiple envs in the
        # same process can now coexist and each sample from its own distribution.
        # N_QUBITS / N are read at module load time, so all configs must keep the
        # same system size.
        narrow_name = self.config.get("parameter_config_filename")
        if narrow_name:
            import json
            from pathlib import Path as _Path
            narrow_path = _Path(pg.__file__).parent / narrow_name
            self._sampling_cfg = json.loads(narrow_path.read_text())
            assert self._sampling_cfg["system"]["N_QUBITS"] == pg._cfg["system"]["N_QUBITS"], (
                f"{narrow_name}: N_QUBITS must match the canonical config."
            )
        else:
            # Default: snapshot the canonical config at construction time.
            self._sampling_cfg = dict(pg._cfg)

        self.n_qubits = int(N_QUBITS)
        self.n_allxy = int(N_ALLXY)
        self.n_params = int(_N_PARAMS)

        # Observations are normalised to ~[-1, 1] in _make_obs:
        #   staircase: (2 × P(|1⟩) − 1)         → [-1, 1] exactly
        #   params:    (p − midpoint) / half_span → ~[-1, 1] under episode_bounds.
        # The clip_params safety rail allows physical params up to 2× the episode span,
        # so normalised params can reach ±2 transiently — declare bounds accordingly.
        self.observation_space = spaces.Dict({
            "staircase": spaces.Box(low=-1.0, high=1.0,
                                    shape=(self.n_qubits, self.n_allxy), dtype=np.float32),
            "params":    spaces.Box(low=-2.0, high=2.0,
                                    shape=(self.n_qubits, self.n_params), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_qubits, self.n_params), dtype=np.float32)

        # Episode state — populated in reset().
        self._rng_key = None
        self._params = None             # (N_QUBITS, 5), jnp — physical units
        self._C = None                  # (N_QUBITS, 5, N_QUBITS, 5), jnp
        self._hw = None                 # (N_QUBITS, 3)
        self._t_g = None                # scalar
        self._alpha = None              # (N_QUBITS,)
        self._lambda_ = None             # (N_QUBITS, N_QUBITS)
        self._param_mins = None         # (N_QUBITS, 5)
        self._param_maxs = None         # (N_QUBITS, 5)
        self._delta_scales = None       # (5,) — for action mapping (× delta_scale_factor)
        self._param_midpoints = None    # (N_QUBITS, 5) — for obs normalisation
        self._param_half_spans = None   # (5,) — for obs normalisation (no factor)
        self._step_count = 0

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if not path.is_absolute():
            path = Path(__file__).parent / config_path
        with open(path) as f:
            return yaml.safe_load(f)

    def _sample_episode(self, rng_key):
        """Resample episode parameters and rebuild initial compensation.

        Re-applies this env's sampling config to the global pg._cfg before sampling,
        so concurrent envs in the same process don't poison each other's distribution.

        Then optionally pins selected params at GT and/or zeroes hardware imperfections
        and cross-talk for fix-N-vary-M diagnostic experiments.
        """
        pg._cfg = self._sampling_cfg
        omega_01, alpha, lambda_, t_g, omega_d, phi, Omega, beta, hw = pg.sample_all(rng_key)

        # Diagnostic overrides: pin selected params at their GT value.
        Omega_opt = 2 * jnp.pi / t_g
        if "omega_d" in self._pin_to_gt:
            omega_d = omega_01                                  # resonant drive
        if "phi" in self._pin_to_gt:
            phi = jnp.zeros_like(phi)                            # no phase error
        if "Omega" in self._pin_to_gt:
            Omega = jnp.full_like(Omega, Omega_opt)              # perfect π-pulse area
        if "beta" in self._pin_to_gt:
            beta = jnp.full_like(beta, self._beta_gt)            # raised-cosine DRAG opt = 0.5
        # omega_01 has no separate "GT" — it IS the sampled value (intrinsic), so
        # "pinning omega_01 at GT" is a no-op; we still accept the name for symmetry.

        if self._zero_hw:
            hw = jnp.zeros_like(hw).at[:, 2].set(1.0)            # phi_hw=0, t_delay=0, Omega_scale=1
        if self._zero_crosstalk:
            lambda_ = jnp.zeros_like(lambda_)

        params = jnp.column_stack([omega_01, omega_d, phi, Omega, beta])
        param_mins, param_maxs = norm.episode_bounds(omega_01, t_g)
        half_spans = norm.episode_delta_scales(t_g)              # (5,) raw, no factor
        delta_scales = half_spans * self._delta_scale_factor     # (5,) for action mapping

        # True pinning: zero delta_scales[k] for any k in pin_to_gt so the policy's
        # action has zero physical effect on that param throughout the episode.
        # Without this, "pinning" was init-only — agents kept random-walking pinned
        # params via their stochastic policy, costing reward (~0.20 with std=0.05).
        # Note: this is exact only when crosstalk is off (lambda_=0 → C is block-
        # identity). With crosstalk on, the action still routes through C[:,k,:,l]
        # to other params; if you want hard pinning under crosstalk, also re-pin
        # params to GT after update_params each step.
        _PARAM_NAME_TO_IDX = {"omega_01": 0, "omega_d": 1, "phi": 2, "Omega": 3, "beta": 4}
        for name in self._pin_to_gt:
            k = _PARAM_NAME_TO_IDX[name]
            delta_scales = delta_scales.at[k].set(0.0)

        # Per-qubit midpoints for obs normalisation. param_mins/maxs has placeholder
        # zeros for omega_d (its bounds are dynamic in clip_params); midpoint there is
        # the current omega_01 itself (drive resonant with qubit).
        midpoints = (param_mins + param_maxs) / 2
        midpoints = midpoints.at[:, 1].set(omega_01)

        C, _ = build_compensation(params, hw, t_g, alpha, lambda_)
        return (params, C, hw, t_g, alpha, lambda_,
                param_mins, param_maxs, delta_scales, midpoints, half_spans)

    def _run_sim(self):
        """Run the All-XY sim with current episode state."""
        return run_allxy_simulation(self._params, self._hw, self._t_g, self._alpha, self._lambda_)

    def _normalise_params(self, params_jnp):
        """(p − midpoint) / half_span — broadcasts midpoints (N,5) and half_spans (5,)."""
        return (params_jnp - self._param_midpoints) / self._param_half_spans[None, :]

    def _make_obs(self, P1):
        """Returns *normalised* obs in ~[-1, 1] for both keys. Raw params live in
        info["params_raw"] for downstream plotting/eval."""
        staircase_norm = 2.0 * np.asarray(P1, dtype=np.float32) - 1.0
        params_norm = np.asarray(self._normalise_params(self._params), dtype=np.float32)
        return {
            "staircase": staircase_norm,
            "params":    params_norm,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._rng_key = jax.random.PRNGKey(int(seed))

        ep_key, self._rng_key = jax.random.split(self._rng_key)
        (self._params, self._C, self._hw, self._t_g, self._alpha, self._lambda_,
         self._param_mins, self._param_maxs, self._delta_scales,
         self._param_midpoints, self._param_half_spans) = self._sample_episode(ep_key)
        self._step_count = 0

        P1 = self._run_sim()
        rewards, deviations = allxy_rewards(P1)
        obs = self._make_obs(P1)
        info = {
            "per_qubit_rewards": np.asarray(rewards, dtype=np.float32),
            "deviations":        np.asarray(deviations, dtype=np.float32),
            "params_raw":        np.asarray(self._params, dtype=np.float32),
            "step": self._step_count,
        }
        return obs, info

    def step(self, action: np.ndarray):
        action_jnp = jnp.asarray(action, dtype=jnp.float64)
        # Apply virtualised compensated update.
        delta_raw = norm.delta_to_physical(action_jnp, self._delta_scales)
        self._params = update_params(self._params, delta_raw, self._C,
                                     self._param_mins, self._param_maxs)
        # Rebuild compensation at the new params (full virtualisation, every step).
        self._C, _ = build_compensation(self._params, self._hw, self._t_g,
                                        self._alpha, self._lambda_)

        P1 = self._run_sim()
        rewards, deviations = allxy_rewards(P1)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        mean_reward = float(np.mean(rewards_np))

        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        truncated = False

        obs = self._make_obs(P1)
        info = {
            "per_qubit_rewards": rewards_np,
            "deviations":        np.asarray(deviations, dtype=np.float32),
            "params_raw":        np.asarray(self._params, dtype=np.float32),
            "step": self._step_count,
        }
        return obs, mean_reward, terminated, truncated, info


# ----- Smoke tests ----- #

def _smoke_shape_sanity():
    print("\n[Test 1] Shape sanity + obs normalisation")
    env = SuperSimsEnv()
    obs, info = env.reset(seed=42)
    assert obs["staircase"].shape == (env.n_qubits, env.n_allxy), obs["staircase"].shape
    assert obs["params"].shape == (env.n_qubits, env.n_params), obs["params"].shape
    assert obs["staircase"].dtype == np.float32
    assert info["per_qubit_rewards"].shape == (env.n_qubits,)
    assert info["params_raw"].shape == (env.n_qubits, env.n_params)

    # Normalised staircase: physical P(|1⟩) ∈ [0, 1] → 2x − 1 ∈ [-1, 1].
    assert obs["staircase"].min() >= -1.0001 and obs["staircase"].max() <= 1.0001, (
        f"staircase out of [-1, 1]: range=[{obs['staircase'].min():.3f}, {obs['staircase'].max():.3f}]"
    )
    # Normalised params: should land in ~[-1, 1] for sampled init (well inside the
    # ±2 obs_space rail). Fail loudly if a column is hugely off.
    assert obs["params"].min() >= -1.5 and obs["params"].max() <= 1.5, (
        f"params normalised range out of [-1.5, 1.5]: "
        f"per-col min={obs['params'].min(axis=0)}, max={obs['params'].max(axis=0)}"
    )

    zero_action = np.zeros((env.n_qubits, env.n_params), dtype=np.float32)
    obs2, reward, terminated, truncated, info2 = env.step(zero_action)
    assert obs2["staircase"].shape == (env.n_qubits, env.n_allxy)
    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0, f"reward out of [0,1]: {reward}"
    assert terminated is False
    assert truncated is False
    print(f"    n_qubits={env.n_qubits}, n_allxy={env.n_allxy}, n_params={env.n_params}")
    print(f"    reset reward = {float(np.mean(info['per_qubit_rewards'])):.4f}")
    print(f"    step(0) reward = {reward:.4f}")
    print(f"    obs[staircase] range = [{obs['staircase'].min():.3f}, {obs['staircase'].max():.3f}]")
    print(f"    obs[params]    range = [{obs['params'].min():.3f}, {obs['params'].max():.3f}]")
    print("    PASS")


def _smoke_determinism():
    print("\n[Test 2] Episode determinism (fixed seed)")
    env_a = SuperSimsEnv()
    env_b = SuperSimsEnv()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert np.allclose(obs_a["staircase"], obs_b["staircase"]), "reset staircase differs"
    assert np.allclose(obs_a["params"], obs_b["params"]), "reset params differ"

    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.1, 0.1, size=(5, env_a.n_qubits, env_a.n_params)).astype(np.float32)
    rewards_a, rewards_b = [], []
    for a in actions:
        _, r_a, _, _, _ = env_a.step(a)
        _, r_b, _, _, _ = env_b.step(a)
        rewards_a.append(r_a)
        rewards_b.append(r_b)
    assert np.allclose(rewards_a, rewards_b), f"rewards diverged: {rewards_a} vs {rewards_b}"
    print(f"    5 random steps: rewards match across both envs.")
    print("    PASS")


def _smoke_episode_termination():
    print("\n[Test 3] Episode termination after max_steps")
    env = SuperSimsEnv()
    env.reset(seed=7)
    zero_action = np.zeros((env.n_qubits, env.n_params), dtype=np.float32)
    terminated = False
    for i in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(zero_action)
        if i < env.max_steps - 1:
            assert terminated is False, f"terminated early at step {i}"
    assert terminated is True, "did not terminate at max_steps"
    print(f"    Terminated at step {env.max_steps}.")
    print("    PASS")


if __name__ == "__main__":
    print("=== SuperSimsEnv Stage 1 smoke tests ===")
    _smoke_shape_sanity()
    _smoke_determinism()
    _smoke_episode_termination()
    print("\nAll Stage 1 in-file smoke tests passed.")
