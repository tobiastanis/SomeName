"""
The LOS-vector between the LLO satellite and LUMIO could be blocked by the Moon. Using vector prjection it is
determined when this happens, so that it is known when to execute intersatellite ranging.

This is done by using the vector h, p and r
r = vector Moon to LUMIO
p = part of vector between LLO and LUMIO so that vector h makes a 90 degree angle with p, which makes h, p, r a
right triangle

All parameters used are in SI units

"""
import Dynamic_Model
import numpy as np
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()

print('Running [Visibility_Analysis.py')
time = Dynamic_Model.time
### Radius Moon
radius_Moon = spice_interface.get_average_radius("Moon")    # [m]

### Getting the states
# LUMIO is wrt Earth and LLO is wrt Moon
states = Dynamic_Model.states
Moon_wrt_Earth = Dynamic_Model.X_Moon
# Both positionvectors are wrt Moon
pos_LUMIO = Dynamic_Model.output[:, 28:31]
pos_LLO = states[:, 6:9]

pos_LUMIO_earth = states[:, 0:3]
pos_LLOsat_earth = Dynamic_Model.output[:, 41:44]

#intersatellite_vector = np.subtract(pos_LUMIO_earth, pos_LLOsat_earth)
intersatellite_vector = np.subtract(pos_LUMIO,pos_LLO)
#intersatellite_vector = Dynamic_Model.relative_position_vector
norm_intersatellite_vector = np.linalg.norm(intersatellite_vector, axis=1)



##### a = vector LUMIO wrt Moon, b = intersatellite distance
a = pos_LUMIO
a_norm = np.linalg.norm(a, axis=1)
b = intersatellite_vector
b_norm = norm_intersatellite_vector
b_unit = []
for i in range(len(b_norm)):
    unit_element = b[i, :]/ b_norm[i]
    b_unit.append(unit_element)

b_unit = np.asarray(b_unit)

a1 = []
for i in range(len(b_norm)):
    a1_scalar_element = np.dot(a[i, :], b_unit[i, :])
    a1.append(a1_scalar_element)
# a1 a scalar array and the distance of the projection of LUMIO to Moon vector on the intersatellite distance vector
# which is as long that a scalar a2 can be obtained so that a2 is perpendicular to a1 and its vector is from the
# center of the Moon to perpendicular on the intersatellite distance vector
a1 = np.asarray(a1)

a2 = []
for i in range(len(a1)):
    a2_element = np.sqrt(a_norm[i]**2 - a1[i]**2)
    a2.append(a2_element)

# If a2 > radius moon, then visibility otherwise blocked by moon
a2 = np.asarray(a2)

import matplotlib.pyplot as plt
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(0, 0, marker='o', markersize=7, color='grey')
ax1.plot(pos_LUMIO[:,0], pos_LUMIO[:, 1], color='orange')
ax1.plot(pos_LLO[:, 0], pos_LLO[:, 1], color='cyan')
ax1.plot(pos_LUMIO[0,0], pos_LUMIO[0,1], marker='o', markersize=4, color='orange')
ax1.plot(pos_LLO[0,0], pos_LLO[0,1], marker='o', markersize=4, color='cyan')
ax1.set_title('LUMIO-LLO-Moon System in xy-plane')
ax1.set_xlabel('Distance in x-direction [m]')
ax1.set_ylabel('Distance in y-direction [m]')
ax2.plot(0, 0, marker='o', markersize=7, color='grey')
ax2.plot(pos_LUMIO[:,0], pos_LUMIO[:, 2], color='orange')
ax2.plot(pos_LLO[:, 0], pos_LLO[:, 2], color='cyan')
ax2.plot(pos_LUMIO[0,0], pos_LUMIO[0,2], marker='o', markersize=4, color='orange')
ax2.plot(pos_LLO[0,0], pos_LLO[0,2], marker='o', markersize=4, color='cyan')
ax2.set_title('LUMIO-LLO-Moon System in xz-plane')
ax2.set_xlabel('Distance in x-direction [m]')
ax2.set_ylabel('Distance in z-direction [m]')
ax3.plot(0, 0, marker='o', markersize=7, color='grey')
ax3.plot(pos_LUMIO[:,1], pos_LUMIO[:, 2], color='orange')
ax3.plot(pos_LLO[:, 1], pos_LLO[:, 2], color='cyan')
ax3.plot(pos_LUMIO[0,1], pos_LUMIO[0,2], marker='o', markersize=4, color='orange')
ax3.plot(pos_LLO[0,1], pos_LLO[0,2], marker='o', markersize=4, color='cyan')
ax3.set_title('LUMIO-LLO-Moon System in yz-plane')
ax3.set_xlabel('Distance in y-direction [m]')
ax3.set_ylabel('Distance in z-direction [m]')
ax1.legend(['Moon', 'LUMIO', 'LLO orbiter'])

plt.figure()
plt.plot(time, a2)
plt.axhline(y=radius_Moon, color='red', linestyle='-')
plt.title('Visibility analysis')
plt.xlabel('Time [days]')
plt.ylabel('Distance center of the Moon perpendicular to LOS-vector')
plt.show()

print('[Visibility_Analysis.py] ran successfully')