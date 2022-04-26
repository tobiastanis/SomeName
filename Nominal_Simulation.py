"""
This file contains the nominal simulation of the trajectory for both ELO and EML2. Firs the simulation set-up is defined
and then the satellites are propagated based on the simulation set-up. Note that by changing the simulation set-up, the
sates automatically change using the states_obtainer
"""
#import own libraries
import states_obtainer
import nominal_simulators
#import general libraries
import math
import numpy as np
#import tudatpy libraries
from tudatpy.kernel import constants

# Measuring dynamic model runtime
print('Running [Nominal_Simulation.py]')

#### Simulation set_up ####
# CONTROL PANEL (changeable)
simulation_start_epoch_mjd = 60390.00           # Modified Julian Time
simulation_time_days = 6                        # Simulation Time [days]
fixed_time_step = 60    # Fixed Time Step [s]

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

# Simulating nominal simulation
[states, output] = nominal_simulators.higherfidelity_model(
    simulation_start_epoch,
    fixed_time_step,
    simulation_end_epoch,
    initial_states,
    savings=1
)

print('Finished running Nominal Simulations')
