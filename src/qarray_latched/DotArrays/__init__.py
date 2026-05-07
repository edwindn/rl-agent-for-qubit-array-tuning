# DotArrays package initialization
"""
Quantum dot array simulation package with advanced features:
- Tunnel-coupled charge-sensed dot arrays
- Voltage-dependent capacitances  
- Barrier voltage models
- Advanced latching dynamics
"""

# Main model classes
from .TunnelCoupledChargeSensed import TunnelCoupledChargeSensed
# from .AdvancedLatching import AdvancedLatching

# Core computation modules  
from .ground_state import _ground_state_open
from .charge_states import build_charge_states, create_full_charge_state_space
from .hamiltonian_build import (
    _jit_free_energy, 
    convert_free_energy_into_hamiltionian_form,
    full_physics_informed_tunneling_hamiltonian
)

# Physics models
from .voltage_dependent_capacitance import create_linear_capacitance_model
from .barrier_voltage_model import BarrierVoltageModel



__all__ = [
    # Main classes
    'TunnelCoupledChargeSensed',
    # 'AdvancedLatching',
    
    # Core functions
    '_ground_state_open',
    'build_charge_states', 
    'create_full_charge_state_space',
    '_jit_free_energy',
    'convert_free_energy_into_hamiltionian_form',
    'full_physics_informed_tunneling_hamiltonian',
    
    # Physics models
    'create_linear_capacitance_model',
    'BarrierVoltageModel',
    
    # Alternative implementations
    'full_physics_informed_ground_state',
]