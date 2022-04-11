"""
This is the second part of the Orbit determination model, which is the Measurement Model
"""
# General libraries
import numpy as np
import sympy as sp
# Own libraries
import Dynamic_Model
import Simulation_setup
# tudatpy libraries
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

########################### State Transition Matrix #######################