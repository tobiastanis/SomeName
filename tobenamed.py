"""
Snippets and trash
"""
# General libraries
import numpy as np
import sympy as sp
# Own libraries
import Dynamic_Model
import Simulation_setup
# tudatpy libraries
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice

# Load spice kernels
spice.load_standard_kernels()
print('Running [tobenamed.py]')
#importing data from the Dynamic_Model
time = Dynamic_Model.time
simulation_start_epoch = Simulation_setup.simulation_start_epoch
simulation_end_epoch = Simulation_setup.simulation_end_epoch
fixed_time_step = Simulation_setup.fixed_time_step
states = Dynamic_Model.states
output = Dynamic_Model.output
#Defining data
X_Moon = Dynamic_Model.X_Moon
states_LUMIO = states[:, 0:6]
states_LLOsat_wrt_Moon = states[:, 6:12]
states_LLOsat = np.add(X_Moon, states_LLOsat_wrt_Moon)
#Nominal state vector
X_star = np.concatenate((states_LUMIO, states_LLOsat), axis=1)
#Accelerations
a_LUMIO = output[:, 0:3]
a_LLOsat = output[:, 14:17]
X_dot_LUMIO = np.concatenate((states_LUMIO[:, 3:6], a_LUMIO), axis=1)
X_dot_LLOsat = np.concatenate((states_LLOsat[:, 3:6], a_LLOsat), axis=1)
#Nominal derivative of the nominal state vector
X_dot_star = np.concatenate((X_dot_LUMIO, X_dot_LLOsat), axis=1)

#Intersatellite stuff
relative_position_vector = Dynamic_Model.relative_position_vector
relative_velocity_vector = Dynamic_Model.relative_velocity_vector
#Intersatellite vector
relative_state_vector = np.concatenate((relative_position_vector, relative_velocity_vector), axis=1)
# Ideal range and range rate
rho_vector = relative_position_vector
rho_dot_vector = relative_velocity_vector
rho_abs = np.linalg.norm(rho_vector, axis=1)
rho_dot_abs = []
for i in range(len(rho_vector)):
    a = rho_vector[i, :]; b = rho_dot_vector[i, :]
    element = np.dot(a,b)/rho_abs[i]
    rho_dot_abs.append(element)
rho_dot_abs = np.array(rho_dot_abs)

## Computing matrix H, no errors (noise and bias) taken into account, so this is more like R in [50_turanranging]
H_wave_top = []
for i in range(len(rho_abs)):
    H_top_element = np.array([(X_star[i,0]-X_star[i,6])/rho_abs[i], (X_star[i,1]-X_star[i,7])/rho_abs[i],
    (X_star[i, 2] - X_star[i, 8]) / rho_abs[i], 0, 0, 0, -(X_star[i,0]-X_star[i,6])/rho_abs[i], -(X_star[i,1]-X_star[i,7])/rho_abs[i],
    -(X_star[i, 2] - X_star[i, 8]) / rho_abs[i], 0, 0, 0])
    H_wave_top.append(H_top_element)
H_wave_top = np.array(H_wave_top)

H_wave_bot = []
for i in range(len(rho_abs)):
    rho = rho_abs[i]; X = X_star[i, :]
    H_bot_element = np.array([1/rho*(X[3]-X[9]), 1/rho*(X[4]-X[10]), 1/rho*(X[5]-X[11]),
                              1/rho*(X[0]-X[6]), 1/rho*(X[1]-X[7]), 1/rho*(X[2]-X[8]),
                              1/rho*(-X[3]+X[9]), 1/rho*(-X[4]+X[10]), 1/rho*(-X[5]+X[11]),
                              1/rho*(-X[0]+X[6]), 1/rho*(-X[1]+X[7]), 1/rho*(-X[2]+X[8])])
    H_wave_bot.append(H_bot_element)
H_wave_bot = np.array(H_wave_bot)

########################### State Transition Matrix and matrix A ######################
# phi initial is I
# A is the partial derivative matrix of X_dot_star (12x12)
#phi_dot = A*phi

"""
Input for EKF
"""
import numpy as np

import EKF_formulas_and_input
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
    i = i
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