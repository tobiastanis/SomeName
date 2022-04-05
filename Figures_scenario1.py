"""
Figures of the results of scenario 1
"""
import numpy as np
import matplotlib.pyplot as plt

import Simulation_setup
import scenario1

print("Running [Figures_scenario1.py]")
simulation_time = scenario1.time

### raw
output = scenario1.output
states = scenario1.states
### all in km or km/s
#### Dataset etc
LUMIO_dataset_states = scenario1.LUMIO_Dataset_states*10**-3
LUMIO_states = states[:, 0:6]*10**-3
LUMIO_states_comp = scenario1.LUMIO_for_comparison*10**-3
time_dataset = np.linspace(0, Simulation_setup.simulation_time, len(LUMIO_dataset_states))
Delta_dataset = scenario1.Difference_scenario1*10**-3

#LUMIO_states_wrt_Moon = np.concatenate((output[:, 28:31], output[:, 31:34]), axis=1)
LLOsat_states_wrt_Moon = states[:, 6:12]*10**-3
X_Moon = scenario1.X_Moon*10**-3
LLOsat_states = np.add(LLOsat_states_wrt_Moon, X_Moon)

###### 2D-system view ######
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.set_title('2D-system view in xy-plane')
ax1.set_xlabel('x-direction [km]')
ax1.set_ylabel('y-direction [km]')
ax1.plot(X_Moon[:,0], X_Moon[:,1], color='grey')
ax1.plot(LLOsat_states[:,0], LLOsat_states[:,1], color='green')
ax1.plot(LUMIO_states[:,0], LUMIO_states[:,1], color='orange')
ax1.plot(0, 0, marker='o', markersize=10, color='blue')
ax1.plot(X_Moon[0,0], X_Moon[0,1], marker='o', markersize=5, color='grey')
ax1.plot(LLOsat_states[0,0], LLOsat_states[0,1], marker='o', markersize=3, color='green')
ax1.plot(LUMIO_states[0,0], LUMIO_states[0,1], marker='o', markersize=3, color='orange')
ax2.set_title('2D-system view in xz-plane')
ax2.set_xlabel('x-direction [km]')
ax2.set_ylabel('z-direction [km]')
ax2.plot(X_Moon[:,0], X_Moon[:,2], color='grey')
ax2.plot(LLOsat_states[:,0], LLOsat_states[:,2], color='green')
ax2.plot(LUMIO_states[:,0], LUMIO_states[:,2], color='orange')
ax2.plot(0, 0, marker='o', markersize=10, color='blue')
ax2.plot(X_Moon[0,0], X_Moon[0,2], marker='o', markersize=5, color='grey')
ax2.plot(LLOsat_states[0,0], LLOsat_states[0,2], marker='o', markersize=3, color='green')
ax2.plot(LUMIO_states[0,0], LUMIO_states[0,2], marker='o', markersize=3, color='orange')
ax3.set_title('2D-system view in yz-plane')
ax3.set_xlabel('y-direction [km]')
ax3.set_ylabel('z-direction [km]')
ax3.plot(X_Moon[:,1], X_Moon[:,2], color='grey')
ax3.plot(LLOsat_states[:,1], LLOsat_states[:,2], color='green')
ax3.plot(LUMIO_states[:,1], LUMIO_states[:,2], color='orange')
ax3.plot(0, 0, marker='o', markersize=10, color='blue')
ax3.plot(X_Moon[0,1], X_Moon[0,2], marker='o', markersize=5, color='grey')
ax3.plot(LLOsat_states[0,1], LLOsat_states[0,2], marker='o', markersize=3, color='green')
ax3.plot(LUMIO_states[0,1], LUMIO_states[0,2], marker='o', markersize=3, color='orange')


fig2, axs = plt.subplots(3,2, constrained_layout=True)
fig2.suptitle('State element difference of simulated LUMIO states wrt dataset states')
axs[0,0].set_title('Difference in x-direction')
axs[0,0].set_xlabel('Time [days]')
axs[0,0].set_ylabel('x-direction [km]')
axs[0,0].plot(time_dataset, Delta_dataset[:,0])
axs[0,1].set_title('Delta-v x-direction')
axs[0,1].set_xlabel('Time [days]')
axs[0,1].set_ylabel('x-direction [km/s]')
axs[0,1].plot(time_dataset, Delta_dataset[:,3])
axs[1,0].set_title('Difference in y-direction')
axs[1,0].set_xlabel('Time [days]')
axs[1,0].set_ylabel('y-direction [km]')
axs[1,0].plot(time_dataset, Delta_dataset[:,1])
axs[1,1].set_title('Delta-v y-direction')
axs[1,1].set_xlabel('Time [days]')
axs[1,1].set_ylabel('y-direction [km/s]')
axs[1,1].plot(time_dataset, Delta_dataset[:,4])
axs[2,0].set_title('Difference in z-direction')
axs[2,0].set_xlabel('Time [days]')
axs[2,0].set_ylabel('z-direction [km]')
axs[2,0].plot(time_dataset, Delta_dataset[:,2])
axs[2,1].set_title('Delta-V z-direction')
axs[2,1].set_xlabel('Time [days]')
axs[2,1].set_ylabel('z-direction [km/s]')
axs[2,1].plot(time_dataset, Delta_dataset[:,5])



plt.show()
print("[Figures_scenario1.py] ran successful")