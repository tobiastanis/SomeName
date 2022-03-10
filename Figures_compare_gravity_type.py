"""
Figures to see if there is a difference in pointmass or spheric harmonic
"""
import numpy as np
import matplotlib.pyplot as plt
import LUMIO_LLO_propagation
import Earth_Moon_PointmassPropagation
print('Running [Figures_compare_gravity_type.py]')
time = LUMIO_LLO_propagation.time

harmonic_states = LUMIO_LLO_propagation.states
pointmass_states = Earth_Moon_PointmassPropagation.states
harmonic_output = LUMIO_LLO_propagation.output
pointmass_output = Earth_Moon_PointmassPropagation.output


harm_LUMIO = harmonic_states[:, 0:6]
harm_LLO = harmonic_states[:, 6:12]
point_LUMIO = pointmass_states[:, 0:6]
point_LLO = pointmass_states[:, 6:12]

LUMIO_state_difference = np.subtract(harm_LUMIO, point_LUMIO)
LLO_state_difference = np.subtract(harm_LLO, point_LLO)

fig1, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, sharey=False)
ax1.plot(time, np.linalg.norm(LUMIO_state_difference[:, 0:3], axis=1))
ax1.set_title('Position difference LUMIO: Earth, Moon harmonic vs Earth pointmass, Moon pointmass')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Difference position  [m]')
ax2.plot(time, np.linalg.norm(LLO_state_difference[:, 0:3], axis=1))
ax2.set_title('Position difference LLO: Earth, Moon harmonic vs Earth pointmass, Moon pointmass')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Difference [m]')

fig2, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, sharey=False)
ax1.plot(time, np.linalg.norm(LUMIO_state_difference[:, 3:6], axis=1))
ax1.set_title('Velocity difference LUMIO: Earth, Moon harmonic vs Earth pointmass, Moon harmonic')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Difference position  [m/s]')
ax2.plot(time, np.linalg.norm(LLO_state_difference[:, 3:6], axis=1))
ax2.set_title('Velocity difference LLO: Earth, Moon harmonic vs Earth pointmass, Moon harmonic')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Difference [m/s]')

fig3, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, sharey=False)
ax1.plot(time, harmonic_output[:, 4])
ax1.plot(time, harmonic_output[:, 5])
ax1.plot(time, pointmass_output[:, 4])
ax1.plot(time, pointmass_output[:, 5])
ax2.plot(time, harmonic_output[:, 18])
ax2.plot(time, harmonic_output[:, 19])
ax2.plot(time, pointmass_output[:, 18])
ax2.plot(time, pointmass_output[:, 19])
ax1.set_title('Acceleration due to Earth and Moon on LUMIO (harmonic gravity and pointmass')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Acceleration [m/$s^{2}$]')
ax1.legend(['Harmonic Earth', 'Harmonic Moon', 'Pointmass Earth', 'Pointmass Moon'])
ax2.set_title('Acceleration due to Earth and Moon on LLO (harmonic gravity and pointmass')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Acceleration [m/$s^{2}$]')

delta_Earth_LUMIO = np.subtract(harmonic_output[:, 4], pointmass_output[:, 4])
delta_Moon_LUMIO = np.subtract(harmonic_output[:, 5], pointmass_output[:, 5])
delta_Earth_LLO = np.subtract(harmonic_output[:, 18], pointmass_output[:, 18])
delta_Moon_LLO = np.subtract(harmonic_output[:, 19], pointmass_output[:, 19])
fig4, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, sharey=False)
ax1.plot(time, delta_Earth_LUMIO)
ax1.plot(time, delta_Moon_LUMIO)
ax1.legend(['delta Earth', 'delta Moon'])
ax1.set_title('Difference in Acceleration for LUMIO')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Acceleration [m/$s^{2}$]')
ax2.plot(time, delta_Earth_LLO)
ax2.plot(time, delta_Moon_LLO)
ax2.set_title('Difference in Acceleration for LLO')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Acceleration [m/$s^{2}$]')



print('[Figures_compare_gravity_type.py] successfully ran')
plt.show()
