"""
This file contains the nominal simulation of the trajectory for both ELO and EML2. Firs the simulation set-up is defined
and then the satellites are propagated based on the simulation set-up. Note that by changing the simulation set-up, the
sates automatically change using the states_obtainer
"""
#import own libraries
from Satellites import EML2
from Satellites import ELO
import states_obtainer
from Simulation_Models import Nominal_Simulation_Models
#import general libraries
import math
import numpy as np
from datetime import datetime
#import tudatpy libraries
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup

## Loading SPICE kernels
spice_interface.load_standard_kernels()
# Measuring dynamic model runtime
starttime = datetime.now()
print('Running [Nominal_Simulation.py]')

#### Simulation set_up ####
# CONTROL PANEL (changeable)
simulation_start_epoch_mjd = 60390.00           # Modified Julian Time
simulation_time_days = 9                        # Simulation Time [days]
fixed_time_step = 0.001*constants.JULIAN_DAY    # Fixed Time Step [s]

# simulation_start_epoch (Ephemeris Time)
simulation_start_epoch = states_obtainer.simulation_start_epoch(simulation_start_epoch_mjd)
# Simulation_start_epoch (Ephemeris Time)
simulation_end_epoch = simulation_start_epoch + simulation_time_days*constants.JULIAN_DAY

# number of steps
n_steps = math.floor((simulation_end_epoch-simulation_start_epoch)/fixed_time_step)+1
# Simulation span from 0 [days]
simulation_span = np.linspace(0, simulation_time_days, n_steps)
# Ephemeris time span [s]
simulation_span_ephemeris = np.linspace(simulation_start_epoch, simulation_end_epoch, n_steps)

#### Initializing ####
EML2_initial = states_obtainer.initial_states_eml2(simulation_start_epoch_mjd)
ELO_initial = states_obtainer.initial_states_elo(simulation_start_epoch_mjd)

initial_states = np.vstack([EML2_initial.reshape(-1,1), ELO_initial.reshape(-1,1)])
# Saving trajectory Moon from ephemeris
X_Moon = states_obtainer.moon_ephemeris(simulation_start_epoch, simulation_end_epoch, n_steps)

[states, output] = Nominal_Simulation_Models.higherfidelity_model(
    simulation_start_epoch,
    fixed_time_step,
    simulation_end_epoch,
    initial_states,
    savings=1
)


