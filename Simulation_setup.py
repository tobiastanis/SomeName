"""
In this file the simulation set-up is presented, so that all simulations or propagations use the same time interval
Adjust the simulation_time and fixed_time_steps only.
Simulation_time for the duration and fixed_time_step for the interval between two epochs within the simulation_time.
"""
import math
import numpy as np
from tudatpy.kernel import constants
import Dataset_reader
#from Dataset_reader import simulation_start_epoch
print("Running [Simulation_setup.py]")
### MJD times for datareading ###
t0_mjd = 60390.00           # Start time 21-03-2024 (next few days no stationkeeping
t1_mjd = 60418.00           # 18-04-2024 Next few days no stationkeeping
tend_mjd = 60755.00         # End of life time 21-03-2025
t2_mjd = 59914.00           # comparison time for paper NaviMoon

simulation_time = 10           ####### Simulation time in days
# simulation start epoch gives the time in seconds from 01-01-2000 00:00, which is used to define celestial positions
simulation_start_epoch = Dataset_reader.simulation_start_epoch(t0_mjd)
simulation_end_epoch = simulation_start_epoch+simulation_time*constants.JULIAN_DAY

fixed_time_step = 0.005*constants.JULIAN_DAY

n_steps = math.floor((simulation_end_epoch-simulation_start_epoch)/fixed_time_step)+1
simulation_span = np.linspace(0, simulation_time, n_steps)

# Needed to obtain states of celestial bodies over an interval
ephemeris_time_span = np.linspace(simulation_start_epoch, simulation_end_epoch, n_steps)

##### Initial states of LUMIO
LUMIO_initial_states = Dataset_reader.initial_state(t0_mjd)
LUMIO_mass = 22.8           # kg
reference_area_radiation_LUMIO = 0.410644     # Total radiating area (002_LUMIO...)
radiation_pressure_coefficient_LUMIO = 1.08   # From thesis stefano send
occulting_bodies_LUMIO = ["Moon", "Earth"]


print("[Simulation_setup.py] ran successfully \n")

