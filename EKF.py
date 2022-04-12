"""
Input for EKF
"""
import numpy as np
from Measurement_Model import observations_array
import Dynamic_Model
import State_Transition_Matrix_LUMIO
import State_Transition_Matrix_LLOsat
from tudatpy.kernel import constants
Phi_LUMIO = State_Transition_Matrix_LUMIO.state_transition_matrices_lumio
#1.######################################################
# Observations array
observations_array = observations_array
#2.######################################################
# Trajectory Dynamic_Model.py
X_true_initial = Dynamic_Model.states_LUMIO
#Initial state error [m] n [m/s]
x_err = np.array([10, 10, 10, 0.1, 0.1, 0.1])
# Initial covariance matrix
P0 = np.array([[10, 0, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0], [0, 0, 10, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
# Timestep
dt = Dynamic_Model.fixed_time_step
# Timespan
time = Dynamic_Model.time
# Endtime [days]
simulation_end_epoch = 10.0 #days
#3.###################################################
# Initial estimated state
X_initial_estimated = X_true_initial[0,:] + x_err
#4.################################################
#Dummy Q (state noise compensation)
Q = np.array([[0.1, 0, 0, 0, 0, 0],
              [0, 0.1, 0, 0, 0, 0],
              [0, 0, 0.1, 0, 0, 0],
              [0, 0, 0, 0.1, 0, 0],
              [0, 0, 0, 0, 0.1, 0],
              [0, 0, 0, 0, 0, 0.1]])
#5.###############################################
for i in range(len(time)):
