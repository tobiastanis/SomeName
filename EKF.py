"""
Input for EKF
"""
import numpy as np
import Measurement_Model
import Dynamic_Model
import State_Transition_Matrix_LUMIO
from EKF_formulas_and_input import H_range_LUMIO_simulation
from tudatpy.kernel import constants
Phi_LUMIO = State_Transition_Matrix_LUMIO.state_transition_matrices_lumio
#1.######################################################
# Observations array
y = Measurement_Model.observations_array
y_hat = Measurement_Model.norm_position_vector
rho = Dynamic_Model.relative_position_vector
#2.######################################################
# Trajectory Dynamic_Model.py
X_true_initial = Dynamic_Model.states_LUMIO
a_LUMIO = Dynamic_Model.output[:, 0:3]
Xdot_true_initial = np.concatenate((X_true_initial[:, 3:6], a_LUMIO), axis=1)

#Initial state error [m] n [m/s]
x_err = np.array([10, 10, 10, 0.1, 0.1, 0.1])
# Initial covariance matrix
P0 = np.diagflat([[1, 1, 1],[0.1, 0.1, 0.1]])
# Timestep
dt = Dynamic_Model.fixed_time_step
# Timespan
time = Dynamic_Model.time
t_ET = Dynamic_Model.ephemeris_time_span

# Endtime [days]
simulation_end_epoch = 10.0 #days

#eyematrix diagonal 6x6
I = np.diagflat([[1,1,1], [1,1,1]])

#3.###################################################
# Initial estimated state
X_initial_estimated = X_true_initial[0,:] + x_err

#4.################################################
#Dummy Q (state noise compensation)
Q = np.diagflat([[0.0001, 0.0001, 0.0001], [0.0001, 0.0001, 0.0001]])

#5.###############################################
# Initial conditions
x_k1_k1 = X_initial_estimated
P_k1_k1 = P0

STORE_x_err = []
STORE_x = []
for i in range(len(time)):
    xdot = Xdot_true_initial[i]
    # Predicting the state (Forward Euler)
    x_k1_k = x_k1_k1 + xdot*dt
    # Covariance matrix propagated forward in time
    Phi = Phi_LUMIO[t_ET[i]]
    P_hat_k = np.add(np.matmul(np.matmul(Phi, P_k1_k1), np.transpose(Phi)), Q)
    # Computing H
    H = H_range_LUMIO_simulation(rho[i,0], rho[i,1], rho[i,2])
    # White Gaussian noise matrix R (prop not necessary
    R = np.array([[np.random.normal(0,1), 0, 0, 0, 0, 0],
                  [0, np.random.normal(0,1), 0, 0, 0, 0],
                  [0, 0, np.random.normal(0,1), 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    # Calculating the Kalman gain matrix K
    K = np.matmul(H,P_hat_k)*(y[i]-y_hat[i])
    # Updating Covariance matrix P
    P = np.matmul((np.subtract(I,np.matmul(K,H))),P_hat_k)
    # State error
    x_error = y[i]*K
    # State
    X_new = x_k1_k + x_error
    ### Savings
    STORE_x.append(X_new)
    STORE_x_err.append(x_error)
    # Updating
    x_k1_k1 = X_new
    P_k1_k1 = P

x_error = np.array(STORE_x_err)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(time, x_error[:,0])



plt.show()