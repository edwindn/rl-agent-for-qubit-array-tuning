"""Agent module registry for GroupedMAC — populated in task 4a."""

from .plunger_cnn import PlungerCNNAgent
from .barrier_cnn import BarrierCNNAgent
from .plunger_cnn_initbias import PlungerCNNAgentInitBias
from .barrier_cnn_initbias import BarrierCNNAgentInitBias

REGISTRY = {
    "plunger_cnn": PlungerCNNAgent,
    "barrier_cnn": BarrierCNNAgent,
    "plunger_cnn_initbias": PlungerCNNAgentInitBias,
    "barrier_cnn_initbias": BarrierCNNAgentInitBias,
}
