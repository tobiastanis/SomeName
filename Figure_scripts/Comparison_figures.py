""""""
import scenario1
import scenario2
import scenario3
import scenario4
import numpy as np
import matplotlib.pyplot as plt
import Simulation_setup
dataset_LUMIO = scenario1.LUMIO_Dataset_states
X_Moon = scenario1.X_Moon

states1 = scenario1.LUMIO_for_comparison

Delta1 = scenario1.Difference_scenario1
Delta1_norm = scenario1.Difference_scenario1_norm
Delta2 = scenario2.Difference_scenario2
Delta2_norm = scenario2.Difference_scenario2_norm
Delta3 = scenario3.Difference_scenario3
Delta3_norm = scenario3.Difference_scenario3_norm
Delta4 = scenario4.Difference_scenario4
Delta4_norm = scenario4.Difference_scenario4_norm

time = np.linspace(0, Simulation_setup.simulation_time, len(Delta1))

fig1, (ax1, ax2, ax3) =  plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(time, Delta1[:,0])
ax1.plot(time, Delta2[:,0])
ax1.plot(time, Delta3[:,0])
ax1.plot(time, Delta4[:,0])
ax2.plot(time, Delta1[:,1])
ax2.plot(time, Delta2[:,1])
ax2.plot(time, Delta3[:,1])
ax2.plot(time, Delta4[:,1])
ax3.plot(time, Delta1[:,2])
ax3.plot(time, Delta2[:,2])
ax3.plot(time, Delta3[:,2])
ax3.plot(time, Delta4[:,2])


plt.figure(figsize=(17,8))
plt.plot(time, Delta1_norm)
plt.plot(time, Delta2_norm)
plt.plot(time, Delta3_norm)
plt.plot(time, Delta4_norm)
plt.xlabel('Time [days]')
plt.ylabel('Absolute difference [m]')
plt.title('Absolute difference of the scenarios from the dataset states')
plt.legend(['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4'])
print(Delta1_norm)
print(Delta2_norm)
print(Delta3_norm)
print(Delta4_norm)


plt.show()