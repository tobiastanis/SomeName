"""
Function to obtain Moon cartesian state over a time interval. No light time correction is taken into account. However,
that can be adjusted
"""
import numpy as np
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()

def Moon_ephemeris(t0, tend, n_steps):
    period = np.linspace(t0, tend, n_steps)
    X_Moon = []
    for i in range(len(period)):
        t_n = period[i]
        state = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", t_n)
        X_Moon.append(state)
    return np.array(X_Moon)