"""
Transformation from the CRTBP to the ICRF,

Doesn't work yet
"""
import numpy as np
import CRTBP
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
import Simulation_setup
spice_interface.load_standard_kernels()
print("Running [CRTBP_to_ICRF.py]")
data_norm_states = CRTBP.states

r_norm = data_norm_states[:, 0:3]
v_norm = data_norm_states[:, 3:6]

t0 = Simulation_setup.simulation_start_epoch
tend = Simulation_setup.simulation_end_epoch
period = Simulation_setup.simulation_span*constants.JULIAN_DAY
m_char = CRTBP.m_char
G = constants.GRAVITATIONAL_CONSTANT
mu = CRTBP.mu

x_Moon = []
x_L2orbiter = []
for i in range(len(period)):
    # Time steps
    dt = period[i]
    # Moon state vector
    X_P1P2 = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", t0 + dt)
    x_Moon.append(X_P1P2)
    r_Moon = X_P1P2[0:3]
    v_Moon = X_P1P2[3:6]
    # Characteristic values
    l_char = np.linalg.norm(r_Moon)
    t_char = np.sqrt(l_char ** 3 / (G * m_char))
    v_char = l_char / t_char
    # Primary position in rotating frame (non-dim)
    r_p1 = np.array([-mu, 0, 0])
    v_p1 = np.array([0, 0, 0])
    # Non-dimensional rotational state vector
    r_rot_nd = r_norm[i, :]
    v_rot_nd = v_norm[i, :]
    # Non-dimensional primary centered
    r_pc_nd = r_rot_nd + r_p1
    v_pc_nd = v_rot_nd + v_p1
    # Primary centered dimensional
    r_pc = r_pc_nd * l_char
    v_pc = v_pc_nd * v_char
    sv_pc = np.transpose([np.concatenate((r_pc, v_pc), axis=0)])
    # print(sv_pc)
    # Attitude matrix between inertial and rotating frame
    X_ref = r_Moon / np.linalg.norm(r_Moon)
    Z_ref = np.cross(r_Moon, v_Moon) / np.linalg.norm(np.cross(r_Moon, v_Moon))
    Y_ref = np.cross(Z_ref, X_ref)
    A_ref = np.transpose([X_ref, Y_ref, Z_ref])
    print(X_ref)
    print(A_ref)

    # Instantaneous angular velocity
    omega = np.linalg.norm(np.cross(r_Moon, v_Moon)) / (np.linalg.norm(r_Moon) ** 2)
    # The creation of B
    C11 = A_ref[0, 0]; C12 = A_ref[0, 1]; C13 = A_ref[0, 2]
    C21 = A_ref[1, 0]; C22 = A_ref[1, 1]; C23 = A_ref[1, 2]
    C31 = A_ref[2, 0]; C32 = A_ref[2, 1]; C33 = A_ref[2, 2]
    B_ref = np.array([[omega * C12, -omega * C11, 0],
                      [omega * C22, -omega * C21, 0],
                      [omega * C32, -omega * C31, 0]])
    O_ref = np.zeros((3, 3))
    # Full transformation matrix
    A_top = np.concatenate((A_ref, O_ref), axis=1)
    A_bot = np.concatenate((B_ref, A_ref), axis=1)
    A_full = np.concatenate((A_top, A_bot), axis=0)
    print(A_full)
    quit()
    # State vector in the ephemeris frame
    x_ephem = np.transpose(np.matmul(A_full, sv_pc))[0]
    # print(x_ephem)
    x_L2orbiter.append(x_ephem)

x_Moon = np.array(x_Moon)
x_L2orbiter = np.array(x_L2orbiter)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_L2orbiter[:, 0], x_L2orbiter[:, 1])
plt.plot(x_Moon[:, 0], x_Moon[:, 1])
plt.plot(0, 0, marker='o', markersize=5, color='blue')
plt.figure()
plt.plot(x_L2orbiter[:, 0], x_L2orbiter[:, 2])
plt.plot(x_Moon[:, 0], x_Moon[:, 2])
plt.plot(0, 0, marker='o', markersize=5, color='blue')
plt.figure()
plt.plot(x_L2orbiter[:, 1], x_L2orbiter[:, 2])
plt.plot(x_Moon[:, 1], x_Moon[:, 2])
plt.plot(0, 0, marker='o', markersize=5, color='blue')


plt.show()
print("[CRTBP_to_ICRF.py] ran successfully \n")