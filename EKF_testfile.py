"""
Extended Kalman Filter
"""
### Imports ###
#General
import numpy as np
import os
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

#### X'es ####
X_LUMIO_ref = Dynamic_Model.states_LUMIO
X_LLO_ref = Dynamic_Model.states_LLOsat
# True initial states (reference trajectory)
X0 = np.concatenate((X_LUMIO_ref, X_LLO_ref), axis=1)
# Error over all nominal states
x_error = np.array([100, 100, 100, 0.001, 0.001, 0.001, 100, 100, 100, 0.0005, 0.0005, 0.0005])
# Nominal states
X_est = []
for i in range(len(time)):
    row = np.add(X0[i, :],x_error)
    X_est.append(row)
X_est = np.array(X_est)

a_LUMIO = Dynamic_Model.output[:, 0:3]
a_LLOsat = Dynamic_Model.output[:, 14:17]
# Xdot where the velocity elements are from X_star and acceleration elements are obtained from dynamic model
Xdot_star = np.concatenate((np.concatenate((X_est[:, 3:6], a_LUMIO), axis=1),
                                 np.concatenate((X_est[:, 9:12], a_LLOsat), axis=1)), axis=1)

### Y ###
Y_star = Measurement_Model.observations_array

# Initial errors and P0
P0 = np.diagflat([[np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01),
                   np.random.normal(0, 0.001), np.random.normal(0, 0.001), np.random.normal(0, 0.01)],
                  [np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01),
                   np.random.normal(0, 0.001), np.random.normal(0, 0.001), np.random.normal(0, 0.001)]])

# Defining Q, the state noise compensation matrix
Q = np.diagflat([[np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02),
                  np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02)],
                 [np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02),
                  np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02)]])
# Defining R
"""
R = np.diagflat([[np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2,
                  np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2],
                 [np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2,
                  np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2, np.random.normal(0,0.1)**2]])
"""
R = np.random.normal(0,0.01)
# Initializing
#x_k1_k1 = np.transpose([X_initial_estimated])
P_k1_k1 = P0
#x_k1_k1 = np.transpose([X_est[0,:]])
x_k1_k1 = np.transpose([x_error])
Xk = np.transpose([X_est[0,:]])
I = np.eye(12, dtype=int)

xk_hat = []
X_ekf = []
for i in range(len(time)):
    count = i
    print(count)
    Xk_1 = Xk
    Y_ref = Y_star[i]
    Phi = ekf.Phi(i)
    # Updating X
    x_k1_k = np.matmul(Phi,x_k1_k1)
    # Upating P_k1_k1 to P_flat_k
    P_hat = np.add(np.matmul(np.matmul(Phi, P_k1_k1), np.transpose(Phi)), Q)
    # Obtaining Y from x_k1_k and defining y
    Y = ekf.Y(Xk)[0]
    y = Y - Y_ref
    # Defining H
    H = ekf.H_range_2sat_simulation(Xk)
    # Computing the Kalman gain
    K = np.matmul(P_hat,np.transpose([H]))*(np.matmul(np.matmul(H,P_hat), np.transpose(H)) - R)**-1
    # Computing covariance matrix
    Pk = np.matmul(np.subtract(I,(K*H)),P_hat)
    # Computing xk and Xstar_new
    xk = np.add(x_k1_k,(K*(y - np.matmul(H,x_k1_k)[0])))
    Xk = np.add(Xk_1, xk)
    # Savings
    xk_hat.append(xk)
    X_ekf.append(Xk)
    # Measurement update
    P_k1_k1 = Pk
    x_k1_k1 = xk


xk_hat = np.array(xk_hat)
X_ekf = np.array(X_ekf)

plt.figure()
plt.plot(time[0:100], xk_hat[0:100, 0])



plt.show()