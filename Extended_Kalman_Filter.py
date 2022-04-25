"""
Extended Kalman Filter, at this stage only ranging is taken into account
"""
#own libraries
import Nominal_Simulation
import Measurement_Model
#function_files
import nominal_simulators
import phi_calculator
import functions_ekf
#general libraries
import numpy as np
#tudatpy libraries

# Initializing time
dt = Nominal_Simulation.fixed_time_step
estimated_initial_errors = np.array([500, 500, 500, 1e-3, 1e-3, 1e-3, 500, 500, 500, 1e-3, 1e-3, 1e-3])
ephemeris_time = Nominal_Simulation.simulation_span_ephemeris

# States of the satellites
true_initial_states = Nominal_Simulation.states[0, :]
X0 = np.transpose([np.add(true_initial_states,estimated_initial_errors)])

# Nominal Measurements
Y_nominal = Measurement_Model.observations_array

# Initial covariance matrix
P0 = 10*np.diag((estimated_initial_errors))

# State Compensation Matrix
Q = np.diag((1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1))*5e-19

