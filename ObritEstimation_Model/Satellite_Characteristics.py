"""
This file contains the initial states of the satellites. The Earth-Moon L2 Orbiter (EML2) is based on the
characteristics of LUMIO and based on its trajectory. The initial states of EML2 are obtained from the
dataset_reader.py file, which reads the states of LUMIO provided by Milano and ESA.

There is a chance that for the EML2 satellite, CRTBP states will be used as well. More over wil follow...

As second satellite, an Elliptic Moon Orbiter (ELO) is propagated. The ELO satellite is based on the exisiting ESA's
Lunar Pathfinder. Ephemeris of the Pathfinder couldn't be obtained for the right time span, so the trajectory of ELO
is based on the Keplerian elements of the Lunar Pathfinder.

Lastly, maybe a Lunar polar orbiter (LuPO) will be propagated, but more over that later...
"""
from ObritEstimation_Model import Dataset_reader
import numpy as np
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()


class Satellite:
    def __init__(self, name, initial_states, mass, reference_area, radiation_pressure_coefficient, occulting_bodies):
        self.name = name
        self.initial_states = initial_states
        self.mass = mass
        self.reference_area = reference_area
        self.radiation_pressure_coefficient = radiation_pressure_coefficient
        self.occulting_bodies = occulting_bodies


t0_mjd = 60390.00           # Start time at 00:00:0000 21-03-2024
t1_mjd = 60418.00           # Start time at 00:00:0000 18-04-2024

# EML2
# Initial states from Dataset reader
initial_states_EML2_20240321 = Dataset_reader.initial_states_(t0_mjd)
initial_states_EML2_20240418 = Dataset_reader.initial_state(t1_mjd)
# EML2's characteristics regarding mass and SRP
mass_EML2 = 22.8           # kg
reference_area_radiation_EML2 = 0.410644     # Total radiating area (002_LUMIO...)
radiation_pressure_coefficient_EML2 = 1.08   # From thesis stefano send
occulting_bodies_EML2 = ["Moon", "Earth"]


#ELO
# Initial states based on Pathfinder
moon_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Moon")

# ESA's pathfinder like
initial_states_ELO_Moon = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=5737.4E3,
    eccentricity=0.61,
    inclination=np.deg2rad(57.82),
    argument_of_periapsis=np.deg2rad(90),
    longitude_of_ascending_node=np.rad2deg(61.552),
    true_anomaly=np.deg2rad(0)
)
Moon_20240321 = Dataset_reader.state_moon(t0_mjd, t0_mjd)[0]
Moon_20240418 = Dataset_reader.state_moon(t1_mjd, t1_mjd)[0]
initial_states_ELO_20240321 = np.add(initial_states_ELO_Moon, Moon_20240321)
initial_states_ELO_20240418 = np.add(initial_states_ELO_Moon, Moon_20240418)


# ELO's characteristics regarding mass and SRP
mass_LLOsat = 280       # kg
reference_area_radiation_LLOsat = 3
radiation_pressure_coefficient_LLOsat = 1.8
occulting_bodies_LLOsat = ["Moon", "Earth"]


