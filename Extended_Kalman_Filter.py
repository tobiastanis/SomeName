"""
Extended Kalman Filter
"""
### Imports ###
#General
import numpy as np
import matplotlib.pyplot as plt
#Own libraries
import EKF_formulas_and_input as ekf
import EKF_integrator
import Measurement_Model
import Simulation_setup
from LLO_initial_states import initial_state_pathfinder_earth
from Dynamic_Model_Earth_based import states

#Time
time = Simulation_setup.simulation_span
dt = Simulation_setup.fixed_time_step
ephemeris_time = Simulation_setup.ephemeris_time_span

# Initial States - LUMIO is wrt Earth and Pathfinder is wrt Moon
LUMIO_ini = Simulation_setup.LUMIO_initial_states
Pathfinder_ini = initial_state_pathfinder_earth
X0_nominal = np.concatenate((LUMIO_ini, Pathfinder_ini))
initial_error = np.array([500, 500, 500, 1e-3, 1e-3, 1e-3, 500, 500, 500, 1e-3, 1e-3, 1e-3])
X0 = np.transpose([np.add(X0_nominal,initial_error)])

# Nominal Measurements
Y_nominal = Measurement_Model.observations_array

# Initial covariance matrix
P0 = 10*np.diag((initial_error))

# State compensation matrix Q
#Q = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
Q = np.zeros((12,12))
# Defining weighting observations R
R = 1**2

# Initializing
Pk = P0
Xhat_k = X0
I = np.eye(12, dtype=int)

X_ekf = []
y_ekf = []
P_ekf = []

for i in range(len(time)-1):
    count = i
    print(count)
    ET_k_1 = ephemeris_time[i]
    # Obtaining input for t = k-1
    Xstar_k_1 = Xhat_k
    Yk = Y_nominal[i+1]
    P_k1_k1 = Pk
    # Inegrating Xstar_k_1 to Xstar_k
    Xstar_k = EKF_integrator.state_integrator(ET_k_1, dt, Xstar_k_1)
    # Obtaining Y_ref
    Y_ref = ekf.Y(Xstar_k)
    # Integrating Phi
    Phi_LUMIO = EKF_integrator.Phi_integrator_LUMIO(ET_k_1, dt, Xstar_k_1)
    Phi_LLOsat = EKF_integrator.Phi_integrator_LLOsat(ET_k_1, dt, Xstar_k_1)
    Phi_top = np.concatenate((Phi_LUMIO, np.zeros((6,6))), axis=1)
    Phi_bot = np.concatenate((np.zeros((6,6)), Phi_LLOsat), axis=1)
    Phi = np.concatenate((Phi_top, Phi_bot), axis=0)

    # Updating P_k1_k1 to P_flat_k
    P_flat_k = np.add(np.matmul(np.matmul(Phi, P_k1_k1), np.transpose(Phi)), Q)
    # Computing G(X,tk), called Y_Ref
    y = Yk - Y_ref
    # Defining H
    H = ekf.H_range_2sat_simulation(Xstar_k)
    # Computing the Kalman gain
    K = np.matmul(P_flat_k, np.transpose([H])) * (np.matmul(np.matmul(H, P_flat_k), np.transpose(H)) + R) ** -1
    # Computing covariance matrix
    Pk = np.matmul(np.subtract(I, (K * H)), P_flat_k)
    # State update X_hat_k
    Xhat_k = np.add(Xstar_k,(K*y))
    # Savings
    X_ekf.append(np.transpose(Xhat_k)[0])
    y_ekf.append(y)
    P_ekf.append(Pk)
    """
    print('Xstar_k_1:', Xstar_k_1)
    print('Xstar_k:', Xstar_k)
    print('Yk:', Yk)
    print('Pk1_k1:', P_k1_k1)
    print('Philumio:', Phi_LUMIO)
    print('Phillosat:', Phi_LLOsat)
    print('Pflat_k', P_flat_k)
    print('H', H)
    print('y', y)
    print('K', K)
    print('Pk', Pk)
    print('Xhat_k', Xhat_k)
    quit()
    """


X_ekf = np.vstack((np.transpose(X0)[0], X_ekf))

x_error = []
for i in range(len(time)):
    row = np.subtract(states[i,:], X_ekf[i,:])
    x_error.append(row)
x_error = np.array(x_error)
y_diff = np.array(y_ekf)

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(time, x_error[:, 0])
ax1.set_title('EKF position error LUMIO in x-direction')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Position error [m]')
ax2.plot(time, x_error[:, 1])
ax2.set_title('EKF position error LUMIO in y-direction')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Position error [m]')
ax3.plot(time, x_error[:, 2])
ax3.set_title('EKF position error LUMIO in z-direction')
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('Position error [m]')

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(time, x_error[:, 6])
ax1.set_title('EKF position error LLOsat in x-direction')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Position error [m]')
ax2.plot(time, x_error[:, 7])
ax2.set_title('EKF position error LLOsat in y-direction')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Position error [m]')
ax3.plot(time, x_error[:, 8])
ax3.set_title('EKF position error LLOsat in z-direction')
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('Position error [m]')

fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(time, x_error[:, 3])
ax1.set_title('EKF velocity error LUMIO in x-direction')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('velocity error [m/s]')
ax2.plot(time, x_error[:, 4])
ax2.set_title('EKF velocity error LUMIO in y-direction')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('velocity error [m/s]')
ax3.plot(time, x_error[:, 5])
ax3.set_title('EKF velocity error LUMIO in z-direction')
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('velocity error [m/s]')

fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(time, x_error[:, 9])
ax1.set_title('EKF velocity error LLOsat in x-direction')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('velocity error [m/s]')
ax2.plot(time, x_error[:, 10])
ax2.set_title('EKF velocity error LLOsat in y-direction')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('velocity error [m/s]')
ax3.plot(time, x_error[:, 11])
ax3.set_title('EKF velocity error LLOsat in z-direction')
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('velocity error [m/s]')

plt.figure()
plt.plot(np.linspace(0, 10, len(y_diff)), y_diff)

fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(states[:, 0], states[:, 1], color='orange', label='states lumio')
ax1.plot(X_ekf[:, 0], X_ekf[:, 1], color='blue', label='states x_ekf lumio')
ax1.plot(states[:, 6], states[:, 7], color='red', label='states pathfinder')
ax1.plot(X_ekf[:, 6], states[:, 7], color='yellow', label='states x_ekf path')
ax2.plot(states[:, 0], states[:, 2], color='orange', label='states lumio')
ax2.plot(X_ekf[:, 0], X_ekf[:, 2], color='blue', label='states x_ekf lumio')
ax2.plot(states[:, 6], states[:, 8], color='red', label='states pathfinder')
ax2.plot(X_ekf[:, 6], states[:, 8], color='yellow', label='states x_ekf path')
ax3.plot(states[:, 1], states[:, 2], color='orange', label='states lumio')
ax3.plot(X_ekf[:, 1], X_ekf[:, 2], color='blue', label='states x_ekf lumio')
ax3.plot(states[:, 7], states[:, 8], color='red', label='states pathfinder')
ax3.plot(X_ekf[:, 7], states[:, 8], color='yellow', label='states x_ekf path')
plt.legend()
plt.show()