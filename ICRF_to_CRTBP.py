"""
Converts the dataset states to CRTBP
"""
# Load standard modules
import numpy as np

# Load own modules
import Simulation_setup
import Dataset_reader

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()

t0_mjd = Simulation_setup.t0_mjd
tend_mjd = Simulation_setup.tend_mjd

simulation_start_epoch = Simulation_setup.simulation_start_epoch
simulation_end_epoch = Simulation_setup.simulation_end_epoch
period = Simulation_setup.simulation_span*constants.JULIAN_DAY

states_dataset = Dataset_reader.state_lumio(t0_mjd, tend_mjd)
pos_dataset = states_dataset[:, 0:3]
velo_dataset = states_dataset[:, 3:6]

