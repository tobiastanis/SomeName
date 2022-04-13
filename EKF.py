"""
Extended Kalman Filter
"""
### Imports ###
#General
import numpy as np
import matplotlib.pyplot as plt
#Own libraries
import EKF_formulas_and_input as ekf
import Dynamic_Model
import Measurement_Model

# Time stuff
dt = Dynamic_Model.fixed_time_step
time = Dynamic_Model.time
endtime = max(time)
t_ET = Dynamic_Model.ephemeris_time_span

# Reference trajectory states
X_LUMIO_ref = Dynamic_Model.states_LUMIO
X_LLO_ref = Dynamic_Model.states_LLOsat
X_reference = np.concatenate((X_LUMIO_ref, X_LLO_ref), axis=1)

a_LUMIO = Dynamic_Model.output[:, 0:3]
a_LLOsat = Dynamic_Model.output[:, 14:17]

Xdot_reference = np.concatenate((np.concatenate((X_reference[:, 3:6], a_LUMIO), axis=1), \
                 np.concatenate((X_reference[:, 9:12], a_LLOsat), axis=1)), axis=1)

# Reference measurements
relative_position_vector = Dynamic_Model.relative_position_vector
relative_velocity_vector = Dynamic_Model.relative_velocity_vector
norm_position_vector = Measurement_Model.norm_position_vector
observation_array = Measurement_Model.observations_array

# Initial errors and P0
x_error_ini = np.array([10, 10, 10, 0.1, 0.1, 0.1, 4, 4, 4, 0.05, 0.05, 0.05])
a = np.random.normal(0,1); b = np.random.normal(0, 0.5)
P0 = np.diagflat([[np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1),
                   np.random.normal(0, 0.5), np.random.normal(0, 0.5), np.random.normal(0, 0.5)],
                  [np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1),
                   np.random.normal(0, 0.5), np.random.normal(0, 0.5), np.random.normal(0, 0.5)]])

# Adding errors on top of the initial true states
X_initial_estimated = X_reference[0, :] + x_error_ini

# Defining Q, the state noise compensation matrix
Q = np.diagflat([[np.random.normal(0,0.2), np.random.normal(0,0.2), np.random.normal(0,0.2),
                  np.random.normal(0,0.2), np.random.normal(0,0.2), np.random.normal(0,0.2)],
                 [np.random.normal(0,0.2), np.random.normal(0,0.2), np.random.normal(0,0.2),
                  np.random.normal(0,0.2), np.random.normal(0,0.2), np.random.normal(0,0.2)]])

# Initializing
#x_k1_k1 = np.transpose([X_initial_estimated])
P_k1_k1 = P0
x_hat_k1 = np.transpose([x_error_ini])

# Savings for later
x_error = []
X_estimated = []
for i in range(len(time)):
    count = i
    print(count)
    X_star = np.transpose([X_reference[i]])
    # Defining Phi, the state transition matrix
    Phi = ekf.Phi(i)
    # Calculating error forward using Phi
    x_flat_k = np.matmul(Phi, x_hat_k1)
    # Defining Xdot from reference solution
    Xdot = np.transpose([Xdot_reference[i]])
    # Covariance matrix propagated
    P_hat = np.add(np.matmul(np.matmul(Phi, P_k1_k1), np.transpose(Phi)), Q)
    # Relating the observation to the state with H
    H = ekf.H_range_2sat_simulation(X_star)
    H_trans = np.transpose([H])
    # Kalman gain K
    Y = observation_array[i]
    Y_ref = norm_position_vector[i]
    y = Y - Y_ref
    K = np.matmul(P_hat,H_trans)*(Y-Y_ref)
    # Updating Covariance matrix P
    P = (np.eye(12) - K*H)*P_hat
    # Update estimated state error
    x_hat = x_flat_k + K*(y-np.matmul(H, x_flat_k)[0])
    x_error.append(x_hat)
    # Update estimated state X_est
    X_est = X_star + x_hat
    X_estimated.append(X_est)
    # Updating state error and covaiance matrix
    x_hat_k1 = x_hat
    P_k1_k1 = P

x_error = np.array(x_error)

plt.figure()
plt.plot(time, x_error[:, 0])



plt.show()


