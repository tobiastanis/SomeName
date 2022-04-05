import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface

spice_interface.load_standard_kernels()
print("Running [LLO_initial_states.py]")
moon_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Moon")

# Circular in xy-plane
initial_state_circ1 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=60E3,
    eccentricity=0.0,
    inclination=np.deg2rad(0.0),
    argument_of_periapsis=np.deg2rad(0.0),
    longitude_of_ascending_node=np.deg2rad(0.0),
    true_anomaly=np.deg2rad(0.0)
)
# Circular in yz-plane
initial_state_circ2 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=60E3,
    eccentricity=0.0,
    inclination=np.deg2rad(90.0),
    argument_of_periapsis=np.deg2rad(0.0),
    longitude_of_ascending_node=np.deg2rad(90.0),
    true_anomaly=np.deg2rad(0.0)
)
# ESA's pathfinder like
initial_state_pathfinder = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=moon_gravitational_parameter,
    semi_major_axis=5737.4E3,
    eccentricity=0.61,
    inclination=np.deg2rad(57.82),
    argument_of_periapsis=np.deg2rad(90),
    longitude_of_ascending_node=np.rad2deg(61.552),
    true_anomaly=np.deg2rad(30)
)

mass_LLOsat = 280       # kg
reference_area_radiation_LLOsat = 1.0
radiation_pressure_coefficient_LLOsat = 1.0
occulting_bodies_LLOsat = ["Moon", "Earth"]

