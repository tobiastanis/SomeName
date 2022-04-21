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
import LLO_initial_states
#Time
time = Simulation_setup.simulation_span
dt = Simulation_setup.fixed_time_step
ephemeris_time = Simulation_setup.ephemeris_time_span

# Initial States - LUMIO is wrt Earth and Pathfinder is wrt Moon
LUMIO_ini = Simulation_setup.LUMIO_initial_states
Pathfinder_ini = LLO_initial_states.initial_state_pathfinder
X0_nominal = np.concatenate((LUMIO_ini, Pathfinder_ini))
initial_error = np.array([100, 100, 100, 6e-4, 6e-4, 6e-4, 10, 10, 10, 6e-5, 6e-5, 6e-5])
X0 = np.transpose([np.add(X0_nominal,initial_error)])

# Nominal Measurements
Y_nominal = Measurement_Model.observations_array

# Initial covariance matrix
P0 = np.diag([np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01),
           np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01),
           np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01), np.random.normal(0,0.01)])

# State compensation matrix Q
Q = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# Defining weighting observations R
R = np.random.normal(0, 0.01)**2

# Initializing
P_k1_k1 = P0
Xhat_k = X0
I = np.eye(12, dtype=int)
Phi_k1_k1 = I


X_ekf = []
y_ekf = []
P_ekf = []

for i in range(len(time)):
    ET_k_1 = ephemeris_time[i]
    # obtaining t = k-1
    Xstar_k_1 = Xhat_k
    Y = Y_nominal[i]
    P_k1_k1 = P0

    # Inegrating Xstar_k_1 to Xstar_k
    Xstar_k = EKF_integrator.state_integrator(ET_k_1, dt, Xstar_k_1)
    print(Xstar_k_1)
    print(Xstar_k)
    quit()
