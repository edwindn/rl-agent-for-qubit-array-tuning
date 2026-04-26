"""Agent module registry for GroupedMAC — populated in task 4a."""

from .plunger_cnn import PlungerCNNAgent
from .barrier_cnn import BarrierCNNAgent

REGISTRY = {
    "plunger_cnn": PlungerCNNAgent,
    "barrier_cnn": BarrierCNNAgent,
}
