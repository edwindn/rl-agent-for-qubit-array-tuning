import numpy as np
import matplotlib.pyplot as plt

from qdarts.plotting import get_polytopes, plot_polytopes
from qdarts.capacitance_model import CapacitanceModel
from qdarts.simulator import CapacitiveDeviceSimulator
from qdarts.tunneling_simulator import NoisySensorDot, ApproximateTunnelingSimulator

import time
start_time = time.time()

N_dots = 3
N_gates = 4 # 2 dot plungers + 1 barrier gate + 1 sensor gate
inner_dots = [0,1]
sensor_dots = [2]

dot_plungers = [0,1]
barrier_plungers = [2]
sensor_plungers = [3]

# Create the capacitance matrices
C_DD = 20 * np.eye(N_dots) / 2
C_DD[inner_dots[0], inner_dots[1]] = 10
C_DD[inner_dots[1], inner_dots[0]] = 10
# Add stronger coupling between sensor dot and inner dots (like 6-dot version)
C_DD[inner_dots[0], sensor_dots[0]] = 4  # sensor dot closest to first inner dot
C_DD[sensor_dots[0], inner_dots[0]] = 4
C_DD[inner_dots[1], sensor_dots[0]] = 1.0  # sensor dot further from second inner dot  
C_DD[sensor_dots[0], inner_dots[1]] = 1.0

# Dot-gate capacitances
C_DG = np.zeros((N_dots, N_gates))
C_DG[inner_dots[0], dot_plungers[0]] = 11
C_DG[inner_dots[1], dot_plungers[1]] = 11
C_DG[inner_dots[0], barrier_plungers[0]] = 1.2
C_DG[inner_dots[1], barrier_plungers[0]] = 0.8
# Sensor dot configuration (like 6-dot version)
C_DG[sensor_dots[0], sensor_plungers[0]] = 11  # sensor plunger to sensor dot
C_DG[sensor_dots[0], dot_plungers[0]] = 1.5   # cross-talk: dot plunger to sensor
C_DG[sensor_dots[0], dot_plungers[1]] = 0.5   # weaker cross-talk to other dot
C_DG[sensor_dots[0], barrier_plungers[0]] = 0.5  # barrier gate to sensor dot

# Minimum voltages for each plunger gate
bounds_limits = -1.0 * np.ones(N_gates)

# Deviation from the constant interaction model
ks = 4 * np.ones(N_dots)

print("Creating capacitance model...")
capacitance_model = CapacitanceModel(C_DG, C_DD, bounds_limits, ks=ks)

print("Creating device simulator...")
capacitive_sim = CapacitiveDeviceSimulator(capacitance_model)

target_state = [1, 1, 5]  # Updated: inner dots [1,1] and sensor dot [5] (like 6-dot version)
print(f"Computing boundaries for target state: {target_state}")
m = capacitive_sim.boundaries(target_state).point_inside

P = np.zeros((N_gates, 2))
P[0, 0] = 1  # First dot plunger along x-axis
P[2, 1] = 1  # Barrier gate along y-axis

from qdarts.plotting import get_CSD_data

minV = np.array([-0.05,-0.05]) # was -0.02
maxV = np.array([ 0.05, 0.05]) # was 0.01

resolution = 100

print(f"Computing CSD data with {resolution}x{resolution} = {resolution**2} points...")
sliced_csim, CSD_data, states =  get_CSD_data(capacitive_sim, m, P, minV, maxV, resolution, target_state)

xs = np.linspace(minV[0],maxV[0],resolution)
ys = np.linspace(minV[1],maxV[1],resolution)

# plt.pcolormesh(xs,ys,CSD_data.T)
# V_offset_2D = P.T @ m
# polytopes = get_polytopes(states, sliced_csim, minV, maxV, V_offset_2D)
# plt.xlim(minV[0],maxV[0])
# plt.ylim(minV[1],maxV[1])
# plot_polytopes(plt.gca(),polytopes, skip_dots=[3,4,5], fontsize=16)

tunnel_couplings = np.zeros((N_dots, N_dots))
tunnel_couplings[0, 1] = 30e-6
tunnel_couplings[1, 0] = 30e-6

tunnel_couplings = tunnel_couplings + tunnel_couplings.T

temperature = 0.1 # 100mK

print(f"Creating sensor model, time elapsed = {time.time() - start_time:.2f}s")

# Update sensor model initialization
sensor_model = NoisySensorDot(sensor_dots) # Define sensor dots for the 2-dot setup
sensor_model.config_peak(g_max = 1.0, peak_width_multiplier = 20) #make the sensor peak broader
tunneling_sim = ApproximateTunnelingSimulator(capacitive_sim, #the underlying polytope simulation
                                             tunnel_couplings,  #constant tunnel couplings
                                             temperature, #electron temperature, should be <=200mK
                                             sensor_model) #our sensor model simulation

capacitive_sim.set_maximum_polytope_slack(5/tunneling_sim.beta) #adding slack to keep more states that are likely to affect the hamiltonian
tunneling_sim.num_additional_neighbours[sensor_dots] = 2 #adding additional states for the sensor dots

print(f"Creating tunneling simulator, time elapsed = {time.time() - start_time:.2f}s")
state=tunneling_sim.poly_sim.find_state_of_voltage(m, [0,0,2])  # Updated: match pattern from 6-dot version
# Fixed: removed cache parameter and m is indeed v_offset
sensor_values = tunneling_sim.sensor_scan_2D(m, P, minV, maxV, resolution, state)

print(f"Sensor scan completed, time elapsed = {time.time() - start_time:.2f}s")

# print(sensor_values)
print(sensor_values.shape)
plt.pcolormesh(xs, ys, sensor_values[:,:,0].T)
# Fixed: add missing V_offset parameter
V_offset_2D = P.T @ m
polytopes = get_polytopes(states, sliced_csim, minV, maxV, V_offset_2D)
plt.xlim(minV[0],maxV[0])
plt.ylim(minV[1],maxV[1])
print(f"Generating plot, time elapsed = {time.time() - start_time:.2f}s")
plot_polytopes(plt.gca(),polytopes, skip_dots=[3,4,5], fontsize=16)


plt.savefig('CSD_plot.png', dpi=300, bbox_inches='tight')

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
