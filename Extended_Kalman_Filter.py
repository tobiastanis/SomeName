"""
Extended Kalman Filter, at this stage only ranging is taken into account
"""
#own libraries
import Nominal_Simulation
import Measurement_Model
import csv_files
#function_files
import nominal_simulators
import phi_calculator
import functions_ekf
#general libraries
import numpy as np
import csv
#tudatpy libraries
print('Running Extended Kalman Filter')
# Initializing time
dt = Nominal_Simulation.fixed_time_step
estimated_initial_errors = np.array([500, 500, 500, 1e-3, 1e-3, 1e-3, 500, 500, 500, 1e-3, 1e-3, 1e-3])
ephemeris_time = Nominal_Simulation.simulation_span_ephemeris

# States of the satellites
true_states = Nominal_Simulation.states
true_initial_states = Nominal_Simulation.states[0, :]
X0 = np.transpose([np.add(true_initial_states,estimated_initial_errors)])

# Nominal Measurements
Y_nominal = Measurement_Model.observations_array

# Initial covariance matrix
P0 = 10*np.diag((estimated_initial_errors))

# State Compensation Matrix
Qc = np.eye(6)*[0.1, 0.1, 1 , 0.1, 0.1, 0.1]*5e-12
RR1 = np.concatenate((dt**2/2*np.eye(3), np.zeros((3,3))), axis=1)
RR2 = np.concatenate((dt*np.eye(3), np.zeros((3,3))), axis=1)
RR3 = np.concatenate((np.zeros((3,3)), dt**2/2*np.eye(3)), axis=1)
RR4 = np.concatenate((np.zeros((3,3)), dt*np.eye(3)), axis=1)
RR = np.concatenate((np.concatenate((np.concatenate((RR1, RR2), axis=0), RR3), axis=0), RR4), axis=0)

Qdt = np.matmul(np.matmul(RR,Qc), np.transpose(RR))

# Weighting the observations
R = Measurement_Model.sigma**2

# Initializing
Pk = P0
std_Pk = np.sqrt(np.diag(Pk))
Xhat_k = X0
I = np.eye(12, dtype=int)

# Creating the savings
X_ekf = []; y_ekf = []; P_ekf = []; std_Pk_ekf = []
# Adding the initial values
X_ekf.append(np.transpose(Xhat_k)[0]); P_ekf.append(Pk); std_Pk_ekf.append(std_Pk)
for i in range(len(ephemeris_time)-1):
    count = i
    print(count)
    ET_k_1 = ephemeris_time[i]
    # Obtaining input for t = k-1
    Xstar_k_1 = Xhat_k
    Yk = Y_nominal[i+1]
    X_nominal = true_states[i+1, :]
    P_k1_k1 = Pk
    # Inegrating Xstar_k_1 to Xstar_k
    Xstar_k = nominal_simulators.highfidelity_model(ET_k_1, dt, ET_k_1, Xstar_k_1, savings=0)[1, :]
    Xstar_k = np.transpose([Xstar_k])
    # Obtaining Y_ref
    Y_ref = functions_ekf.Y(Xstar_k)
    # Integrating Phi
    Phi_EML2 = phi_calculator.phi_highfidelity_eml2(ET_k_1, dt, Xstar_k_1)
    Phi_ELO = phi_calculator.phi_highfidelity_elo(ET_k_1, dt, Xstar_k_1)

    Phi = functions_ekf.Phi(Phi_EML2, Phi_ELO)
    # Updating P_k1_k1 to P_flat_k
    P_flat_k = np.add(np.matmul(np.matmul(Phi, P_k1_k1), np.transpose(Phi)), Qdt)
    # Computing G(X,tk), called Y_Ref
    y = Yk - Y_ref
    # Computing H
    H = functions_ekf.H_range_2sat_simulation(Xstar_k)
    # Computing the Kalman gain
    K = np.matmul(P_flat_k, np.transpose([H])) * (np.matmul(np.matmul(H, P_flat_k), np.transpose(H)) + R) ** -1
    # Computing covariance matrix
    Pk = np.matmul(np.subtract(I, (K * H)), P_flat_k)
    # State update X_hat_k
    Xhat_k = np.add(Xstar_k, (K * y))
    # Savings
    X_ekf.append(np.transpose(Xhat_k)[0])
    y_ekf.append(y)
    P_ekf.append(Pk)
    std_Pk = np.sqrt(np.diag(Pk))
    std_Pk_ekf.append(std_Pk)


### Results ###
X_ekf = np.array(X_ekf)
std_Pk_ekf = np.array(std_Pk_ekf)
std_Pk_up = 3*std_Pk_ekf
std_Pk_down = -3*std_Pk_ekf
x_error = []
for i in range(len(ephemeris_time)):
    row = np.subtract(true_states[i,:], X_ekf[i,:])
    x_error.append(row)
x_error = np.array(x_error)
y_diff = np.array(y_ekf)


import matplotlib.pyplot as plt
t = Nominal_Simulation.simulation_span

plt.figure()
plt.plot(t, x_error[:, 0], color='red', label='x')
plt.plot(t, x_error[:, 1], color='blue', label='y')
plt.plot(t, x_error[:, 2], color='green', label='z')
plt.plot(t, std_Pk_up[:, 0], color='orange', label='3$\sigma_{x}$')
plt.plot(t, std_Pk_down[:, 0], color='orange')
plt.plot(t, std_Pk_up[:, 1], color='cyan', label='3$\sigma_{y}$')
plt.plot(t, std_Pk_down[:, 1], color='cyan')
plt.plot(t, std_Pk_up[:, 2], color='yellow', label='3$\sigma_{z}$')
plt.plot(t, std_Pk_down[:, 2], color='yellow')
plt.legend()
plt.xlabel('Time since epoch [days]')
plt.ylabel('Estimated position error [m]')
plt.title('EML2 position error')

plt.figure()
plt.plot(t, x_error[:, 3], color='red', label='$\dot{x}$')
plt.plot(t, x_error[:, 4], color='blue', label='$\dot{y}$')
plt.plot(t, x_error[:, 5], color='green', label='$\dot{z}$')
plt.plot(t, std_Pk_up[:, 3], color='orange', label='3$\sigma_{\dot{x}}$')
plt.plot(t, std_Pk_down[:, 3], color='orange')
plt.plot(t, std_Pk_up[:, 4], color='cyan', label='3$\sigma_{\dot{y}}$')
plt.plot(t, std_Pk_down[:, 4], color='cyan')
plt.plot(t, std_Pk_up[:, 5], color='yellow', label='3$\sigma_{\dot{z}}$')
plt.plot(t, std_Pk_down[:, 5], color='yellow')
plt.legend()
plt.xlabel('Time since epoch [days]')
plt.ylabel('Estimated velocity error [m]')
plt.title('EML2 velocity error')

plt.figure()
plt.plot(t, x_error[:, 6], color='red', label='x')
plt.plot(t, x_error[:, 7], color='blue', label='y')
plt.plot(t, x_error[:, 8], color='green', label='z')
plt.plot(t, std_Pk_up[:, 6], color='orange', label='3$\sigma_{x}$')
plt.plot(t, std_Pk_down[:, 6], color='orange')
plt.plot(t, std_Pk_up[:, 7], color='cyan', label='3$\sigma_{y}$')
plt.plot(t, std_Pk_down[:, 7], color='cyan')
plt.plot(t, std_Pk_up[:, 8], color='yellow', label='3$\sigma_{z}$')
plt.plot(t, std_Pk_down[:, 8], color='yellow')
plt.legend()
plt.xlabel('Time since epoch [days]')
plt.ylabel('Estimated position error [m]')
plt.title('ELO position error')

plt.figure()
plt.plot(t, x_error[:, 9], color='red', label='$\dot{x}$')
plt.plot(t, x_error[:, 10], color='blue', label='$\dot{y}$')
plt.plot(t, x_error[:, 11], color='green', label='$\dot{z}$')
plt.plot(t, std_Pk_up[:, 9], color='orange', label='3$\sigma_{\dot{x}}$')
plt.plot(t, std_Pk_down[:, 9], color='orange')
plt.plot(t, std_Pk_up[:, 10], color='cyan', label='3$\sigma_{\dot{y}}$')
plt.plot(t, std_Pk_down[:, 10], color='cyan')
plt.plot(t, std_Pk_up[:, 11], color='yellow', label='3$\sigma_{\dot{z}}$')
plt.plot(t, std_Pk_down[:, 11], color='yellow')
plt.legend()
plt.xlabel('Time since epoch [days]')
plt.ylabel('Estimated velocity error [m]')
plt.title('ELO velocity error')
print('Extended Kalman Filter Finished')
plt.show()

array_to_save = np.concatenate((np.concatenate((np.concatenate((np.transpose([ephemeris_time]), true_states), axis=1), X_ekf), axis=1), std_Pk_ekf), axis=1)
