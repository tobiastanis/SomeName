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

#### X'es ####
X_LUMIO_ref = Dynamic_Model.states_LUMIO
X_LLO_ref = Dynamic_Model.states_LLOsat
# True initial states (reference trajectory)
X0 = np.concatenate((X_LUMIO_ref, X_LLO_ref), axis=1)
# Error over all nominal states
x_error = np.array([100, 100, 100, 6e-4, 6e-4, 6e-4, 10, 10, 10, 6e-5, 6e-5, 6e-5])
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
                   np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)],
                  [np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01),
                   np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)]])

# Defining Q, the state noise compensation matrix
Q = np.diagflat([[np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02),
                  np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02)],
                 [np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02),
                  np.random.normal(0,0.02), np.random.normal(0,0.02), np.random.normal(0,0.02)]])
# Defining R
R = np.random.normal(0,0.01)

# Initializing
#x_k1_k1 = np.transpose([X_initial_estimated])
P_k1_k1 = P0
X_hat_k = np.transpose([X_est[0,:]])
#x_k1_k1 = np.transpose([x_error])
I = np.eye(12, dtype=int)

X_ekf = []
y_diff = []
for i in range(len(time)):
    count = i
    #print(count)
    #Xstar_k_1 = np.transpose([X_est[i, :]])
    Xstar_k_1 = X_hat_k
    Y_ref = Y_star[i]
    Phi = ekf.Phi(i)
    # Updating X
    #x_k1_k = np.matmul(Phi,x_k1_k1)
    Xstar_k = np.matmul(Phi,Xstar_k_1)
    #Xstar_k = Xstar_k_1
    # Updating P_k1_k1 to P_flat_k
    P_hat = np.add(np.matmul(np.matmul(Phi, P_k1_k1), np.transpose(Phi)), Q)
    # Obtaining Y from x_k1_k and defining y
    Y = ekf.Y(Xstar_k)[0]
    y = Y_ref - Y
    # Defining H
    H = ekf.H_range_2sat_simulation(Xstar_k)
    # Computing the Kalman gain
    K = np.matmul(P_hat,np.transpose([H]))*(np.matmul(np.matmul(H,P_hat), np.transpose(H)) + R)**-1
    # Computing covariance matrix
    Pk = np.matmul(np.subtract(I,(K*H)),P_hat)
    # Computing xk and Xstar_new
    #xk = np.add(x_k1_k,(K*(y - np.matmul(H,x_k1_k)[0])))
    X_hat_k = np.add(Xstar_k,(K*y))
    #xk = K*y
    #Xstar_k = np.add(Xstar_k_1, xk)
    # Savings
    X_ekf.append(X_hat_k)
    #X_ekf.append(Xstar_k)
    y_diff.append(y)
    # Measurement update
    P_k1_k1 = Pk
    #x_k1_k1 = xk

X_ekf = np.array(X_ekf)
x_error = []
for i in range(len(time)):
    row = np.subtract(X_est[i,:], np.transpose(X_ekf[i,:])[0])
    x_error.append(row)
x_error = np.array(x_error)
y_diff = np.array(y_diff)

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
plt.plot(time, y_diff)
plt.title('Measurement difference in y')
plt.xlabel('Time [days]')
plt.ylabel('Measurement difference [m]')








plt.show()