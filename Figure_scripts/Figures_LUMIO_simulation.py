"""
Figures obtained from LUMIO_simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import LUMIO_simulation
print("Running [Figure_LUMIO_simulation.py]")

states = LUMIO_simulation.states*10**-3
X_Moon = LUMIO_simulation.X_Moon*10**-3
output = LUMIO_simulation.output
time = LUMIO_simulation.time

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=True)
ax1.plot(0, 0, marker='o', markersize=10, color='blue')
ax1.plot(states[:, 0], states[:, 1], color='orange')
ax1.plot(X_Moon[:, 0], X_Moon[:, 1], color='grey')
ax1.plot(states[0, 0], states[0, 1], marker='o', markersize=4, color='orange')
ax1.plot(X_Moon[0, 0], X_Moon[0, 1],marker='o', markersize=3, color='grey')
ax1.set_title('Trajectory in xy-plane')
ax1.set_xlabel('x-direction [km]')
ax1.set_ylabel('y-direction [km]')
ax2.plot(0, 0, marker='o', markersize=10, color='blue')
ax2.plot(states[:, 0], states[:, 2], color='orange')
ax2.plot(X_Moon[:, 0], X_Moon[:, 2], color='grey')
ax2.plot(states[0, 0], states[0, 2], marker='o', markersize=4, color='orange')
ax2.plot(X_Moon[0, 0], X_Moon[0, 2],marker='o', markersize=3, color='grey')
ax2.set_title('Trajectory in xz-plane')
ax2.set_xlabel('x-direction [km]')
ax2.set_ylabel('z-direction [km]')
ax3.plot(0, 0, marker='o', markersize=10, color='blue')
ax3.plot(states[:, 1], states[:, 2], color='orange')
ax3.plot(X_Moon[:, 1], X_Moon[:, 2], color='grey')
ax3.plot(states[0, 1], states[0, 2], marker='o', markersize=4, color='orange')
ax3.plot(X_Moon[0, 1], X_Moon[0, 2],marker='o', markersize=3, color='grey')
ax3.set_title('Trajectory in yz-plane')
ax3.set_xlabel('y-direction [km]')
ax3.set_ylabel('z-direction [km]')
ax1.legend(['Earth (center)', 'Propagated LUMIO state', 'Moon'], loc='upper left', bbox_to_anchor=(1, 1))

fig2, (bx1, bx2, bx3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
bx1.plot(time, states[:, 0], color='orange')
bx1.set_title('Trajectory LUMIO in x-direction')
bx1.set_xlabel('Time [days]')
bx1.set_ylabel('x-direction [km]')
bx2.plot(time, states[:, 1], color='orange')
bx2.set_title('Trajectory LUMIO in y-direction')
bx2.set_xlabel('Time [days]')
bx2.set_ylabel('y-direction [km]')
bx3.plot(time, states[:, 2], color='orange')
bx3.set_title('Trajectory LUMIO in z-direction')
bx3.set_xlabel('Time [days]')
bx3.set_ylabel('z-direction [km]')
bx1.legend(['Propagated state element'], loc='upper left', bbox_to_anchor=(1, 1))

fig3, (cx1, cx2, cx3) = plt.subplots(3, 1, figsize=(17,7), dpi=100, constrained_layout=True, sharey=False)
cx1.plot(time, states[:, 3], color='orange')
cx1.set_title('Velocity LUMIO in x-direction')
cx1.set_xlabel('Time [days]')
cx1.set_ylabel('Velocity [km/s]')
cx2.plot(time, states[:, 4], color='orange')
cx2.set_title('Velocity LUMIO in y-direction')
cx2.set_xlabel('Time [days]')
cx2.set_ylabel('Velocity [km/s]')
cx3.plot(time, states[:, 5], color='orange')
cx3.set_title('Velocity LUMIO in x-direction')
cx3.set_xlabel('Time [days]')
cx3.set_ylabel('Velocity [km/s]')
cx1.legend(['Propagated state element'], loc='upper right')

plt.figure()
plt.plot(time, np.linalg.norm(output[:, 0:3], axis=1))
plt.plot(time, output[:, 3])
plt.plot(time, output[:, 4])
plt.plot(time, output[:, 5])
plt.plot(time, output[:, 6])
plt.plot(time, output[:, 7])
plt.plot(time, output[:, 8])
plt.plot(time, output[:, 9])
plt.plot(time, output[:, 10])
plt.plot(time, output[:, 11])
plt.plot(time, output[:, 12])
plt.plot(time, output[:, 13])
plt.legend(['Total acceleration', 'Solar Radiation Pressure', 'Spherical harmonic gravity by Earth',
            'Spherical harmonic gravity by Moon', 'Point mass gravity by Sun', 'Point mass gravity by Mercury',
            'Point mass gravity by Venus', 'Point mass gravity by Mars', 'Point mass gravity by Jupiter',
            'Point mass gravity by Saturn', 'Point mass gravity by Uranus', 'Point mass gravity by Neptune']
           , loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Acceleration distribution')
plt.xlabel('Time [days]')
plt.ylabel('Acceleration [m/$s^{2}$]')

plt.figure()
plt.plot(time, output[:, 3])
plt.plot(time, output[:, 6])
plt.plot(time, output[:, 7])
plt.plot(time, output[:, 8])
plt.plot(time, output[:, 9])
plt.plot(time, output[:, 10])
plt.plot(time, output[:, 11])
plt.plot(time, output[:, 12])
plt.plot(time, output[:, 13])
plt.legend(['SRP', 'Point mass gravity by Sun', 'Point mass gravity by Mercury',
            'Point mass gravity by Venus', 'Point mass gravity by Mars', 'Point mass gravity by Jupiter',
            'Point mass gravity by Saturn', 'Point mass gravity by Uranus', 'Point mass gravity by Neptune']
           , loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Detailed acceleration distribution of the point masses and SRP')
plt.xlabel('Time [days]')
plt.ylabel('Acceleration [m/$s^{2}$]')

plt.figure()
plt.plot(time, output[:, 7])
plt.plot(time, output[:, 8])
plt.plot(time, output[:, 9])
plt.plot(time, output[:, 10])
plt.plot(time, output[:, 11])
plt.plot(time, output[:, 12])
plt.plot(time, output[:, 13])
plt.legend(['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus',
            'Neptune'], loc='upper left', bbox_to_anchor=(1,1))
plt.title('Detailed acceleration distribution of the point masses and SRP')
plt.xlabel('Time [days]')
plt.ylabel('Acceleration [m/$s^{2}$]')

plt.figure()
plot = plt.axes(projection='3d')
plot.plot(0, 0, 0, marker='o', markersize=10, color='blue')
plot.plot(states[:, 0], states[:, 1], states[:, 2], color='orange')
plot.plot(X_Moon[:, 0], X_Moon[:, 1], X_Moon[:, 2], color='grey')
plot.plot(states[0, 0], states[0, 1], states[0, 2], marker='o', markersize=4, color='orange')
plot.plot(X_Moon[0, 0], X_Moon[0, 1], X_Moon[0, 2], marker='o', markersize=4, color='grey')
plt.title('Earth-Moon-LUMIO System overview')
plt.legend(['Earth', 'LUMIO', 'Moon'], loc='upper left', bbox_to_anchor=(1, 1))
plot.set_xlabel('x-direction [km]')
plot.set_ylabel('y-direction [km]')
plot.set_zlabel('z-direction [km]')

print("[Figures_LUMIO_simulation.py] successfully ran \n")
plt.show()

