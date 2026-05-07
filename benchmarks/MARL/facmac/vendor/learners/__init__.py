from .cq_learner import CQLearner
from .facmac_learner import FACMACLearner
from .maddpg_learner import MADDPGLearner
from .td3_learner import TD3Learner
from .maddpg_antizero_learner import MADDPGAntiZeroLearner

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["td3_learner"] = TD3Learner
REGISTRY["maddpg_antizero_learner"] = MADDPGAntiZeroLearner

# Discrete learners disabled — upstream bug, missing `multinomial_entropy` symbol.
