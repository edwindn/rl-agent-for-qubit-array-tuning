from functools import partial

from .multiagentenv import MultiAgentEnv
from .matrix_game.cts_matrix_game import Matrixgame as CtsMatrix


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["cts_matrix_game"] = partial(env_fn, env=CtsMatrix)

# Particle / MAMuJoCo / SMAC envs disabled — not needed for this benchmark.
# Additional envs (e.g. pymarl_quantum) are registered by benchmarks/MARL/facmac/train.py.
