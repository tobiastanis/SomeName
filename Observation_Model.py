"""
Observation setup and measurement etc
"""

# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
# Load own modules
import LUMIO_LLO_propagation
import Simulation_setup
# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observations
from tudatpy.kernel.astro import element_conversion
spice.load_standard_kernels()
print("Running [Observation_Model.py]")

# Simulation Settings
simulation_start_epoch = Simulation_setup.simulation_start_epoch
simulation_end_epoch = Simulation_setup.simulation_end_epoch
fixed_step_size = Simulation_setup.fixed_time_step
# Satellites states wrt Earth
states = LUMIO_LLO_propagation.states
states_LUMIO = states[:, 0:6]
pos_LUMIO = states_LUMIO[:, 0:3]
velo_LUMIO = states_LUMIO[:, 3:6]
states_LLOorbiter = states[:, 6:12]
pos_LLOorbiter = states_LLOorbiter[:, 0:3]
velo_LLOorbiter = states_LLOorbiter[:, 3:6]

# Integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    simulation_start_epoch, fixed_step_size, propagation_setup.integrator.rkf_78, fixed_step_size, fixed_step_size, 5.0,
    5.0)

##### Observation Setup #####
transmitters = ['LUMIO', 'LLOorbiter']
receivers = ['LUMIO', 'LLOorbiter']

transmitter_position = list()




print("[Observation_Model.py] ran succesfully \n")