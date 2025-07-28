import numpy as np
import matplotlib.pyplot as plt

import qdarts
from qdarts.plotting import get_polytopes, plot_polytopes
from qdarts.capacitance_model import CapacitanceModel
from qdarts.simulator import CapacitiveDeviceSimulator

import time
start_time = time.time()

N_dots = 6 #number of dots
N_gates = 9 # 6 dot plungers + 3 barrier gates
inner_dots = [0,1,2]
sensor_dots = [3,4,5]

dot_plungers = [0,1,2]
barrier_plungers = [3,4,5]
sensor_plungers = [6,7,8]

#@title create the capaitance matrices

C_DD=20* np.eye((N_dots))/2 #The self-capacitance of each dot, NOTE: factor of 2 due to symmetrization

#capacitances inner dots
C_DD[inner_dots[0],inner_dots[1]] = 10
C_DD[inner_dots[1],inner_dots[2]] = 10
C_DD[inner_dots[2],inner_dots[0]] = 10

#setup the sensor-dot <->inner dot capacitances.
for i in range(3):
    for j in range(3):
        if i == j:
            C_DD[inner_dots[i], sensor_dots[j]] = 4 #sensor dot closest to the inner dot.
        else:
            C_DD[inner_dots[i], sensor_dots[j]] = 1.0 #sensor dot further away

#symmetrize
C_DD = C_DD + C_DD.T

#dot-gate capacitances.
C_DG = np.zeros((N_dots, N_gates))
#dot to plunger-gate capacitances
for i in range(3):
    C_DG[inner_dots[i],dot_plungers[i]] = 11 #ith dot plunger to the ith dot
    C_DG[sensor_dots[i],sensor_plungers[i]] = 11 #ith sensor plunger to the ith sensor
    C_DG[sensor_dots[i],dot_plungers[i]] = 1.5 #ith dot plunger to the ith sensor (cross-talk)
    
C_DG[sensor_dots[0],dot_plungers[1]] = 0.5
C_DG[sensor_dots[0],dot_plungers[2]] = 0.5
C_DG[sensor_dots[1],dot_plungers[0]] = 0.5
C_DG[sensor_dots[1],dot_plungers[2]] = 0.5
C_DG[sensor_dots[2],dot_plungers[0]] = 0.5
C_DG[sensor_dots[2],dot_plungers[1]] = 0.5

#cross-talk inner dots <-> dot plungers for the plungers that are further away
C_DG[inner_dots[0],dot_plungers[1]] = 1.5
C_DG[inner_dots[0],dot_plungers[2]] = 1.5
C_DG[inner_dots[1],dot_plungers[0]] = 1.5
C_DG[inner_dots[1],dot_plungers[2]] = 1.5
C_DG[inner_dots[2],dot_plungers[0]] = 1.5
C_DG[inner_dots[2],dot_plungers[1]] = 1.5

#cross-talk barrier gates <-> inner dots
C_DG[inner_dots[0],barrier_plungers[0]] = 1.2
C_DG[inner_dots[0],barrier_plungers[1]] = 0.8
C_DG[inner_dots[0],barrier_plungers[2]] = 1.2

C_DG[inner_dots[1],barrier_plungers[0]] = 1.2
C_DG[inner_dots[1],barrier_plungers[1]] = 1.2
C_DG[inner_dots[1],barrier_plungers[2]] = 0.8

C_DG[inner_dots[2],barrier_plungers[0]] = 0.8
C_DG[inner_dots[2],barrier_plungers[1]] = 1.2
C_DG[inner_dots[2],barrier_plungers[2]] = 1.2

#cross talk barrier gates -> sensor dots
C_DG[inner_dots[0],barrier_plungers[0]] = 0.5
C_DG[inner_dots[0],barrier_plungers[1]] = 0.1
C_DG[inner_dots[0],barrier_plungers[2]] = 0.5

C_DG[inner_dots[1],barrier_plungers[0]] = 0.5
C_DG[inner_dots[1],barrier_plungers[1]] = 0.5
C_DG[inner_dots[1],barrier_plungers[2]] = 0.1

C_DG[inner_dots[2],barrier_plungers[0]] = 0.1
C_DG[inner_dots[2],barrier_plungers[1]] = 0.5
C_DG[inner_dots[2],barrier_plungers[2]] = 0.5

#minimum voltages for each plunger gate. 
bounds_limits = -1.0*np.ones(N_gates)

#deviation from the constant interaction model. Set to None to have no deviation
ks = 4*np.ones(N_dots)

print("Creating capacitance model...")
capacitance_model = CapacitanceModel(C_DG, C_DD, bounds_limits, ks=ks)

print("Creating device simulator...")
capacitive_sim = CapacitiveDeviceSimulator(capacitance_model)

target_state = [1,1,1,5,5,5]
print(f"Computing boundaries for target state: {target_state}")
m = capacitive_sim.boundaries(target_state).point_inside

P = np.zeros((N_gates,2))
P[0,0] = 1
P[2,1] = 1

from qdarts.plotting import get_CSD_data

minV = np.array([-0.02,-0.02])
maxV = np.array([ 0.01, 0.01])
# Reduced resolution for faster computation (was 100)
resolution = 100  # Use 50 for development, 100 for final plots

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
# -------------- #

tunnel_couplings = np.zeros((N_dots,N_dots))
tunnel_couplings[0,1] = 30*1e-6
tunnel_couplings[0,2] = 30*1e-6
tunnel_couplings[1,2] = 30*1e-6
tunnel_couplings = tunnel_couplings+ tunnel_couplings.T

temperature = 0.1 # 100mK

from qdarts.tunneling_simulator import NoisySensorDot, ApproximateTunnelingSimulator

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


plt.savefig('qdarts_plot_6d.png', dpi=300, bbox_inches='tight')

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")