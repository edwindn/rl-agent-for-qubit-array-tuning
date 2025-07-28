import numpy as np
import matplotlib.pyplot as plt

import qdarts
from qdarts.plotting import get_polytopes, plot_polytopes
from qdarts.capacitance_model import CapacitanceModel
from qdarts.simulator import CapacitiveDeviceSimulator
from qdarts.plotting import get_CSD_data
from qdarts.tunneling_simulator import NoisySensorDot, ApproximateTunnelingSimulator

import time
start_time = time.time()

N_dots = 3
N_gates = 4 # 2 dot plungers + 1 barrier gate + 1 sensor gate
inner_dots = [0,1]
sensor_dots = [2]

dot_plungers = [0,1] # plunger gates
barrier_plungers = [2] # barrier gate
sensor_plungers = [3] # sensor gate

C_DD = np.array([[10, 10, 4],
                 [10, 10, 1],
                 [4, 1, 10]], dtype=np.float32)
# inner dot 1, inner dot 2, sensor dot

#symmetrize
C_DD = C_DD + C_DD.T

C_DG = np.array([[11, 0, 1.2, 0],
                 [0, 11, 0.8, 0],
                 [1.5, 0.5, 0.5, 11]], dtype=np.float32)
# plunger gate 1, plunger gate 2, barrier gate, sensor gate


#minimum voltages for each plunger gate. 
bounds_limits = -1.0*np.ones(N_gates)

#deviation from the constant interaction model. Set to None to have no deviation
ks = 4*np.ones(N_dots)

print("Creating capacitance model...")
capacitance_model = CapacitanceModel(C_DG, C_DD, bounds_limits, ks=ks)

print("Creating device simulator...")
capacitive_sim = CapacitiveDeviceSimulator(capacitance_model)

target_state = [1,1,5]
print(f"Computing boundaries for target state: {target_state}")
m = capacitive_sim.boundaries(target_state).point_inside

P = np.zeros((N_gates,2))
P[0,0] = 1
P[2,1] = 1

minV = np.array([-0.02,-0.2])
maxV = np.array([ 0.01, 0.05])
# Reduced resolution for faster computation (was 100)
resolution = 250

offset = np.array([0.002, -0.002])  # Adjust values for desired direction
m = m + P @ offset  # Apply the offset using the projection matrix P

print(f"Computing CSD data with {resolution}x{resolution} = {resolution**2} points...")
sliced_csim, CSD_data, states =  get_CSD_data(capacitive_sim, m, P, minV, maxV, resolution, target_state)

xs = np.linspace(minV[0],maxV[0],resolution)
ys = np.linspace(minV[1],maxV[1],resolution)

plt.pcolormesh(xs,ys,CSD_data.T)
# Project the 9D offset into 2D space for plotting
V_offset_2D = P.T @ m
polytopes = get_polytopes(states, sliced_csim, minV, maxV, V_offset_2D)
plt.xlim(minV[0],maxV[0])
plt.ylim(minV[1],maxV[1])
plot_polytopes(plt.gca(),polytopes, skip_dots=[3,4,5], fontsize=16)
plt.savefig('qdarts_plot2.png', dpi=300, bbox_inches='tight')

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
exit()
# -------------- #

tunnel_couplings = np.zeros((N_dots,N_dots))
tunnel_couplings[0,1] = 30*1e-6
tunnel_couplings[0,2] = 30*1e-6
tunnel_couplings[1,2] = 30*1e-6
tunnel_couplings = tunnel_couplings+ tunnel_couplings.T

temperature = 0.1 # 100mK

print(f"Creating sensor model, time elapsed = {time.time() - start_time:.2f}s")

sensor_model = NoisySensorDot(sensor_dots) #a model of the sensor dots that just needs which dot has which index
sensor_model.config_peak(g_max = 1.0, peak_width_multiplier = 20) #make the sensor peak broader
tunneling_sim = ApproximateTunnelingSimulator(capacitive_sim, #the underlying polytope simulation
                                             tunnel_couplings,  #constant tunnel couplings
                                             temperature, #electron temperature, should be <=200mK
                                             sensor_model) #our sensor model simulation

capacitive_sim.set_maximum_polytope_slack(5/tunneling_sim.beta) #adding slack to keep more states that are likely to affect the hamiltonian
tunneling_sim.num_additional_neighbours[sensor_dots] = 2 #adding additional states for the sensor dots

print(f"Creating tunneling simulator, time elapsed = {time.time() - start_time:.2f}s")
state=tunneling_sim.poly_sim.find_state_of_voltage(m, [0,0,0,2,2,2])
# Fixed: removed cache parameter and m is indeed v_offset
sensor_values = tunneling_sim.sensor_scan_2D(m, P, minV, maxV, resolution, state)

print(f"Sensor scan completed, time elapsed = {time.time() - start_time:.2f}s")
plt.pcolormesh(xs,ys,sensor_values[:,:,2].T)
# Fixed: add missing V_offset parameter
V_offset_2D = P.T @ m
polytopes = get_polytopes(states, sliced_csim, minV, maxV, V_offset_2D)
plt.xlim(minV[0],maxV[0])
plt.ylim(minV[1],maxV[1])
print(f"Generating plot, time elapsed = {time.time() - start_time:.2f}s")
plot_polytopes(plt.gca(),polytopes, skip_dots=[3,4,5], fontsize=16)


plt.savefig('qdarts_plot.png', dpi=300, bbox_inches='tight')

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")