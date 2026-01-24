from ast import Pass
from platform import mac_ver
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from qarray.DotArrays._helper_functions import _validate_vg
from jax.experimental import sparse as jsparse

# Handle both relative and absolute imports

import sys
import os
# Ensure root path is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from qarray_latched.DotArrays.charge_states import build_charge_states
from qarray_latched.DotArrays.hamiltonian_build import convert_free_energy_into_hamiltionian_form
from qarray_latched.DotArrays.fully_sparse_jax_eigensolver import sparse_lanczos_iteration

        


def _ground_state_open(model, vg, vb=None):
    """
    
    """
    #Code path init
    truncate = True if isinstance(model.num_charge_states, int) else False 
    use_sparse = model.use_sparse
    use_barriers = model.barrier_model is not None and vb is not None
    use_variable_capacitances =  model.voltage_capacitance_model is not None

    #Voltage init and reshape
    _validate_vg(vg, model.n_gate)
    vg_shape = vg.shape
    nd_shape = (*vg_shape[:-1], model.n_dot)
    vg = vg.reshape(-1, model.n_gate) #shape (n_points, n_gate)
    
    # Pre-concatenate voltages, this the code paths with and without barriers are almost identical (only difference is tc calculation), all capacitive crosstalk of the barriers should be accounted for    
    if use_barriers:
        vb = vb.reshape(-1, vb.shape[-1])
        v_extended = jnp.concatenate([vg, vb], axis=-1)  # (n_points, n_gate + n_barrier)
    else:
        v_extended = vg  # Just gate voltages

    #miscellaneous init
    kB_T = 8.617333262145e-5 * model.T
    n_dots = model.n_dot
    num_pixels = v_extended.shape[0]

    #setup batched capacitances
    if use_variable_capacitances:
        cdd_batch, cdd_inv_batch, cgd_batch = model.voltage_capacitance_model.compute_all_capacitances(v_extended)
        
        cdd_inv_batch = jnp.array(cdd_inv_batch)
        cgd_batch = jnp.array(cgd_batch)

    else:
        cdd_inv=jnp.array(model.cdd_inv_full)
        cgd=jnp.array(model.cgd_full)

        # Batch the constant capacitances by expanding to match voltage batch dimensions
        cdd_inv_batch = jnp.expand_dims(cdd_inv, 0).repeat(num_pixels, axis=0)
        cgd_batch = jnp.expand_dims(cgd, 0).repeat(num_pixels, axis=0)
        
    #build charge states
    if truncate:
        from qarray_latched.DotArrays.hamiltonian_build import _jit_free_energy, full_physics_informed_tunneling_hamiltonian

        num_charge_states = model.num_charge_states
        charge_state_batch_size = model.charge_state_batch_size

        charge_states = build_charge_states(truncate=True, num_charge_states=num_charge_states, batchsize = charge_state_batch_size, v_extended = v_extended, cdd_inv_batch=cdd_inv_batch, cgd_batch= cgd_batch, n_dot=n_dots)
    
    else:
        from qarray_latched.DotArrays.unbatched_hamiltonian_build import _jit_free_energy_unbatched as _jit_free_energy, full_physics_informed_tunneling_hamiltonian_unbatched as full_physics_informed_tunneling_hamiltonian

        mcc = model.max_charge_carriers
        charge_states = build_charge_states(truncate=False, max_charge_carriers=mcc, n_dot=n_dots)

    #build tc matrix
    if use_barriers:
        vb = jnp.array(vb)
        #TODO: Update this to use v_extended instead, should also make the code there simpler?
        vb_eff_batch = model.barrier_model.compute_effective_barrier_potential(vg, vb, model)
        tc_matrix_batch = model.barrier_model.compute_tc_matrix_batch(vb_eff_batch)

    else:
        tc_matrix_single = jnp.zeros((n_dots, n_dots))
        tc = model.tc
        for i in range(n_dots - 1):
            tc_matrix_single = tc_matrix_single.at[i, i+1].set(tc)
            tc_matrix_single = tc_matrix_single.at[i+1, i].set(tc)  # Symmetric
        
        # Broadcast to match voltage batch dimensions
        batch_shape = v_extended.shape[:-1] if v_extended.ndim > 1 else (1,)
        tc_matrix_batch = jnp.tile(tc_matrix_single[None, :, :], (*batch_shape, 1, 1))

    #build hamiltonian

    #TODO: we already calculate free energy during the truncation of the charge states but discarded the values should use those instead of recalculating. Not a bottleneck so for now doesn't matter.
    F = _jit_free_energy(v_extended, cdd_inv_batch, cgd_batch, charge_states, n_dots)

    H_f = convert_free_energy_into_hamiltionian_form(F, charge_states)

    H_t = full_physics_informed_tunneling_hamiltonian(tc_matrix_batch, charge_states, model.max_charge_carriers, model.tunneling_convention)


    #solve hamiltonian
    if use_sparse:
        # Import the proper sparse matrix creation functions
        from qarray_latched.DotArrays.fully_sparse_jax_eigensolver import create_sparse_tunneling_matrix, fully_sparse_ground_state_lanczos

        # Create sparse tunneling matrix (this needs to be done outside JIT compilation)
        # Convert charge states to numpy array format expected by sparse matrix creation
        if truncate:
            # For truncated case, charge_states is batched, use the first batch element as template
            charge_states_template = np.array(charge_states[0])
        else:
            charge_states_template = np.array(charge_states)

        # Create the sparse tunneling matrix with the right tc value
        tc_value = model.tc if not use_barriers else tc_matrix_batch[0, 0, 1]  # Get representative tc value
        H_sparse = create_sparse_tunneling_matrix(charge_states_template, tc_value, model.max_charge_carriers)

        # Sparse solve function using the properly created sparse matrix
        @jax.jit
        def sparse_solve_single(vg_single, cdd_inv_single, cgd_single):
            return fully_sparse_ground_state_lanczos(
                vg_single, cdd_inv_single, cgd_single, H_sparse,
                jnp.array(charge_states_template), n_iterations=50
            )

        # Apply sparse solver to all voltage points
        sparse_solve_batch = jax.vmap(sparse_solve_single)

        # Extract single voltage points and capacitance matrices for batched computation
        vg_batch = v_extended[:, :model.n_gate] if use_barriers else v_extended
        n = sparse_solve_batch(vg_batch, cdd_inv_batch, cgd_batch[:, :, :model.n_gate])

        # sparse solver returns occupation numbers directly, convert to numpy
        n = np.array(n)

    else:
        H = H_f + H_t
        _, eigen_states = jnp.linalg.eigh(H)

        ground_state = eigen_states[..., :, 0]

        ground_state_probs = jnp.abs(ground_state)**2  # (..., M)
        if truncate:
            # Batched charge states: (..., M, n_dot)
            n = jnp.einsum('...m,...md->...d', ground_state_probs, charge_states)
        else:
            # Non-batched charge states: (M, n_dot) - broadcast for all pixels
            n = jnp.einsum('...m,md->...d', ground_state_probs, charge_states)

        n = np.array(n)

    n = model.latching_model.add_latching(n, measurement_shape=nd_shape)

    return n.reshape(nd_shape)


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    import jax.numpy as jnp
    from qarray import LatchingModel
    from qarray.noise_models import WhiteNoise
    from TunnelCoupledChargeSensed import TunnelCoupledChargeSensed
    from voltage_dependent_capacitance import create_linear_capacitance_model
    from barrier_voltage_model import BarrierVoltageModel
    
    print("Testing _ground_state_open function with 2-dot model...")
    
    # 2-dot capacitance matrices (from tunnelling.py pattern)
    n_dots = 2
    n_gates = n_dots + 1  # 2 plungers + 1 global
    n_barriers = n_dots - 1  # 1 barrier between dots
    
    # Dot-dot capacitance matrix
    Cdd = [[0.0, 0.4], [0.4, 0.0]]
    
    # Gate-dot capacitance matrix (2 dots x 3 gates)
    Cgd = [[1.0, 0.4, 0.0], [0.4, 1.0, 0.0]]
    
    # Dot-sensor capacitance
    Cds = [[0.05, 0.04]]
    
    # Gate-sensor capacitance
    Cgs = [[0.06, 0.05, 1.0]]
    
    # Barrier capacitance matrices
    Cbd = [[0.06], [0.04]]  # Dot-barrier (2 dots x 1 barrier)
    Cbg = [[0.12, 0.08, 0.0]]  # Barrier-gate (1 barrier x 3 gates)
    Cbs = [[0.04]]  # Barrier-sensor (1 sensor x 1 barrier)
    Cbb = [[1.0]]  # Barrier-barrier (1 x 1)
    
    # Create latching model
    latching_model = LatchingModel(
        n_dots=2,
        p_leads=[0.3, 0.3],
        p_inter=[[0.0, 1.0], [1.0, 0.0]]
    )
    
    # Create noise model
    noise_model = WhiteNoise(amplitude=0.01)
    
    # Create barrier model
    barrier_model = BarrierVoltageModel(
        n_barrier=n_barriers,
        n_dot=n_dots,
        tc_base=0.15,
        alpha=[1.0]
    )
    
    # Create model with barriers
    model = TunnelCoupledChargeSensed(
        Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
        Cbd=Cbd, Cbg=Cbg, Cbs=Cbs, Cbb=Cbb,
        barrier_model=barrier_model,
        coulomb_peak_width=0.1, 
        T=0, 
        max_charge_carriers=4,
        tc=0.15,
        noise_model=noise_model, 
        latching_model=latching_model,
        voltage_capacitance_model=None
    )
    
    # Add voltage-dependent capacitances
    voltage_capacitance_model = create_linear_capacitance_model(
        cdd_0=jnp.array(model.cdd_full), 
        cgd_0=jnp.array(model.cgd_full),
        alpha=0.1,  # 10% voltage dependence for Cdd
        beta=0.1    # 10% voltage dependence for Cgd
    )

    model.voltage_capacitance_model = voltage_capacitance_model
    
    # Enable sparse solver
    model.num_charge_states = 32
    model.use_sparse_solver = False
    
    # Create simple test voltage grid
    vg = model.gate_voltage_composer.do2d('P1', -0.5, 0.5, 128, 'P2', -0.5, 0.5, 128)
    optimal_charges = [1, 1, 0.4]  # 2 dots + sensor
    vg += model.optimal_Vg(optimal_charges)
    
    # Create barrier voltage array
    vb = jnp.full((vg.shape[:-1] + (n_barriers,)), np.array([0.0]*n_barriers))  # Fixed barrier voltage = 0.0
    
    print(f"Voltage grid shape: {vg.shape}")
    print(f"Barrier voltage shape: {vb.shape}")
    print("Calling _ground_state_open with voltage-dependent capacitances and barriers...")
    
    try:
        result = _ground_state_open(model, vg, vb)
        print(f"Success! Result shape: {result.shape}")
        print(f"Sample result values: {result[0, 0, :]}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
