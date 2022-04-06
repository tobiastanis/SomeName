"""
Figures of Dynamic_Model.py
"""
# import external libraries
import numpy as np
import matplotlib.pyplot as plt
# import own libraries
import Dynamic_Model

time = Dynamic_Model.time

x_moon = Dynamic_Model.X_Moon*10**-3
x_lumio = Dynamic_Model.states_LUMIO*10**-3
x_llosat = Dynamic_Model.states_LLOsat*10**-3
x_llosat_moon = Dynamic_Model.states_LLOsat_wrt_Moon*10**-3
relative_velocity_vector = Dynamic_Model.relative_velocity_vector
relative_position_vector = Dynamic_Model.relative_position_vector
lumio_wrt_moon = Dynamic_Model.output[:, 28:31]

test_lumio_wrt_moon = np.subtract(x_lumio,x_moon)
test_relative_position_vector = np.subtract(x_lumio, x_llosat)

#plt.figure()
#plt.plot(time, np.linalg.norm(relative_position_vector, axis=1))
#plt.plot(time, np.linalg.norm(lumio_wrt_moon, axis=1))
#plt.plot(time, np.linalg.norm(test_relative_position_vector, axis=1))
#plt.plot(time, np.linalg.norm(test_lumio_wrt_moon, axis=1))

plt.figure()
plot = plt.axes(projection='3d')
plot.plot(x_lumio[:, 0], x_lumio[:, 1], x_lumio[:, 2], color='orange')
plot.plot(x_llosat[:, 0], x_llosat[:, 1], x_llosat[:, 2], color='green')
plot.plot(x_moon[:, 0], x_moon[:, 1], x_moon[:, 2], color='grey')
plot.plot(0, 0, 0, marker='o', markersize=10, color='blue')
plot.plot(x_lumio[0, 0], x_lumio[0, 1], x_lumio[0, 2], marker='o', markersize=3,  color='orange')
plot.plot(x_llosat[0, 0], x_llosat[0, 1], x_llosat[0, 2], marker='o', markersize=3, color='green')
plot.plot(x_moon[0, 0], x_moon[0, 1], x_moon[0, 2], marker='o', markersize=3, color='grey')
plt.title('Earth-Moon-Satellites System')
plt.legend(['LUMIO', 'LLO orbiter', 'Moon', 'Earth'])
plot.set_xlabel('x-direction [km]')
plot.set_ylabel('y-direction [km]')
plot.set_zlabel('z-direction [km]')

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=True)
ax1.plot(x_lumio[:, 0], x_lumio[:, 1], color='orange')
ax1.plot(x_llosat[:, 0], x_llosat[:, 1], color='green')
ax1.plot(x_moon[:, 0], x_moon[:, 1], color='grey')
ax1.plot(0, 0, marker='o', markersize=10, color='blue')
ax1.plot(x_lumio[0, 0], x_lumio[0, 1], marker='o', markersize=3, color='orange')
ax1.plot(x_moon[0, 0], x_moon[0, 1],marker='o', markersize=3, color='grey')
ax1.plot(x_llosat[0, 0], x_llosat[0, 1], marker='o', markersize=3, color='green')
ax1.set_title('Trajectory in xy-plane')
ax1.set_xlabel('x-direction [km]')
ax1.set_ylabel('y-direction [km]')
ax2.plot(x_lumio[:, 0], x_lumio[:, 2], color='orange')
ax2.plot(x_llosat[:, 0], x_llosat[:, 2], color='green')
ax2.plot(x_moon[:, 0], x_moon[:, 2], color='grey')
ax2.plot(0, 0, marker='o', markersize=10, color='blue')
ax2.plot(x_lumio[0, 0], x_lumio[0, 2], marker='o', markersize=3, color='orange')
ax2.plot(x_moon[0, 0], x_moon[0, 2],marker='o', markersize=3, color='grey')
ax2.plot(x_llosat[0, 0], x_llosat[0, 2], marker='o', markersize=3, color='green')
ax2.set_title('Trajectory in xz-plane')
ax2.set_xlabel('x-direction [km]')
ax2.set_ylabel('z-direction [km]')
ax3.plot(x_lumio[:, 1], x_lumio[:, 2], color='orange')
ax3.plot(x_llosat[:, 1], x_llosat[:, 2], color='green')
ax3.plot(x_moon[:, 1], x_moon[:, 2], color='grey')
ax3.plot(0, 0, marker='o', markersize=10, color='blue')
ax3.plot(x_lumio[0, 1], x_lumio[0, 2], marker='o', markersize=3, color='orange')
ax3.plot(x_moon[0, 1], x_moon[0, 2],marker='o', markersize=3, color='grey')
ax3.plot(x_llosat[0, 1], x_llosat[0, 2], marker='o', markersize=3, color='green')
ax3.set_title('Trajectory in xz-plane')
ax3.set_xlabel('y-direction [km]')
ax3.set_ylabel('z-direction [km]')
ax2.legend(['LUMIO trajectory', 'LLOsat trajectory', 'Moon trajectory', 'Earth'], loc='upper left', bbox_to_anchor=(1, 1))


fig2, (bx1, bx2, bx3) = plt.subplots(3, 1, constrained_layout=True, sharey=True)
bx1.plot(x_llosat_moon[:, 0], x_llosat_moon[:, 1], color='green')
bx1.plot(0, 0, marker='o', markersize=7, color='grey')
bx1.plot(x_llosat_moon[0, 0], x_llosat_moon[0, 1], marker='o', markersize=3, color='darkgreen')
bx1.set_title('Trajectory in xy-plane')
bx1.set_xlabel('x-direction [km]')
bx1.set_ylabel('y-direction [km]')
bx2.plot(x_llosat_moon[:, 0], x_llosat_moon[:, 2], color='green')
bx2.plot(0, 0, marker='o', markersize=7, color='grey')
bx2.plot(x_llosat_moon[0, 0], x_llosat_moon[0, 2], marker='o', markersize=3, color='darkgreen')
bx2.set_title('Trajectory in xz-plane')
bx2.set_xlabel('x-direction [km]')
bx2.set_ylabel('z-direction [km]')
bx2.legend(['LLOsat trajectory', 'Moon'], loc='upper left', bbox_to_anchor=(1, 1))
bx3.plot(x_llosat_moon[:, 1], x_llosat_moon[:, 2], color='green')
bx3.plot(0, 0, marker='o', markersize=7, color='grey')
bx3.plot(x_llosat_moon[0, 1], x_llosat_moon[0, 2], marker='o', markersize=3, color='darkgreen')
bx3.set_title('Trajectory in yz-plane')
bx3.set_xlabel('y-direction [km]')
bx3.set_ylabel('z-direction [km]')


plt.show()
