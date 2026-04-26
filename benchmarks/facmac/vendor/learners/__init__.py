from .cq_learner import CQLearner
from .facmac_learner import FACMACLearner
from .maddpg_learner import MADDPGLearner

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["maddpg_learner"] = MADDPGLearner

# Discrete learners disabled — upstream bug, missing `multinomial_entropy` symbol.
