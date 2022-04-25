"""
This file imports the data of LUMIO and the Moon over an epoch of Modified Julian Time 59091.50000 to 61325.00000,
meaning from date 2020-08-30 12:00:00.000 to 2026-10-12 00:00:00.000.

Two important times at which the initial state of LUMIO are considered are at 21-03-2024 and 18-04-2024, since no
stationkeeping is performed from then to a certain amount of days

See below for function description (from line 46)
"""
import numpy as np
import pandas as pd
import csv
#tudatpy
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()
print("Running [states_obtainer.py]")
# Opening csv datasets
LUMIO_datacsv = open('LUMIO_states.csv')
Moon_datacsv = open('Moon_states.csv')
# Reading
csvreader_LUMIO = csv.reader(LUMIO_datacsv)
csvreader_Moon = csv.reader(Moon_datacsv)
# Extruding
header_LUMIO = next(csvreader_LUMIO)
header_Moon = next(csvreader_Moon)
rows_LUMIO = []
rows_Moon = []
for row in csvreader_LUMIO:
    rows_LUMIO.append(row)
for row in csvreader_Moon:
    rows_Moon.append(row)
data_LUMIO = np.array(rows_LUMIO).astype(float)
data_Moon = np.array(rows_Moon).astype(float)

t0_data_mjd = data_Moon[0, 0]
tend_data_mjd = data_Moon[(len(data_Moon) - 1), 0]

LUMIO_dataframe = pd.DataFrame(
    {'MJD': data_LUMIO[:, 0], 'ET': data_LUMIO[:, 1], 'x': data_LUMIO[:, 2], 'y': data_LUMIO[:, 3],
     'z': data_LUMIO[:, 4], 'vx': data_LUMIO[:, 5], 'vy': data_LUMIO[:, 6], 'vz': data_LUMIO[:, 7]})
Moon_dataframe = pd.DataFrame(
    {'MJD': data_Moon[:, 0], 'ET': data_Moon[:, 1], 'x': data_Moon[:, 2], 'y': data_Moon[:, 3],
     'z': data_Moon[:, 4], 'vx': data_Moon[:, 5], 'vy': data_Moon[:, 6], 'vz': data_Moon[:, 7]})

"""
Function file which can be used to obtain data over a time interval defined in Modified Julian Time from 59091.50000 to
61325.00000. The data set has time epochs of 0.25 MJD. However, numbers in between can be used as well, but notice that 
for t0, data will always be rounded up and for tend the data will always be rounded down.

Five functions. Data functions output are in dataframe format and state parameters in [km]. state functions are in 
np.array format and state parameters are in [m]. Last is obtaining the ephemeris time used to define states of celestial
bodies
"""

def data_moon(t0, tend):
    return Moon_dataframe.loc[(Moon_dataframe['MJD'] >= t0) & (Moon_dataframe['MJD'] <= tend)]

def data_lumio(t0, tend):
    return LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] >= t0) & (LUMIO_dataframe['MJD'] <= tend)]

def states_moon(t0, tend):
    return np.asarray(Moon_dataframe.loc[(Moon_dataframe['MJD'] >= t0) & (Moon_dataframe['MJD'] <= tend)])[:, 2: 8]*10**3

def states_eml2(t0,tend):
    return np.asarray(LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] >= t0) & (LUMIO_dataframe['MJD'] <= tend)])[:, 2: 8]*10**3

def initial_states_eml2(t0):
    data = np.asarray(LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] == t0)])[0]
    return data[2: 8]*10**3

def simulation_start_epoch(t0):
    data = np.asarray(LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] == t0)])[0]
    return np.asscalar(data[1])

def initial_states_elo(t0):
    # ESA's pathfinder like
    initial_states_ELO_Moon = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=spice_interface.get_body_gravitational_parameter("Moon"),
        semi_major_axis=5737.4E3,
        eccentricity=0.61,
        inclination=np.deg2rad(57.82),
        argument_of_periapsis=np.deg2rad(90),
        longitude_of_ascending_node=np.rad2deg(61.552),
        true_anomaly=np.deg2rad(0)
    )
    ephemeris_start_epoch = (np.asarray(LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] == t0)])[0])[1]
    moon_initial_states = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", ephemeris_start_epoch)
    return np.add(initial_states_ELO_Moon, moon_initial_states)

def moon_ephemeris(t0, tend, n_steps):
    period = np.linspace(t0, tend, n_steps)
    x_moon = []
    for i in range(len(period)):
        t_n = period[i]
        state = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", t_n)
        x_moon.append(state)
    return np.array(x_moon)

print("[states_obtainer.py] ran successfully \n")
