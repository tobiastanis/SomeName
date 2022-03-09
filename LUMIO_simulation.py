"""
Propagation of LUMIO using tudatpy libraries. Propagation starts at the same initial condition as the dataset, but over
time it is expected that the propagated by tudat trajectory will deviate due to no stationkeeping and or dynamic errors
not taken into account.
Errors that are taken into account: Spherical harmonic gravity Earth and Moon, pointmass Sun, Mercury, Venus, Mars,
Jupiter, Saturn, Uranus and Neptune and also Solar Radiation Pressure is taken into account.

The results of the propagation are compared with the state elements of the Dataset.

Note that the numbers used for SRP, are an estimate and not factual yet...
"""
import Dataset_reader
import Simulation_setup
import numpy as np
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from Ephemeris_obtainer import Moon_ephemeris
spice_interface.load_standard_kernels()


print("Running [LUMIO_simulation.py]")
# Adjust simulation setting in [Simulation_setup.py]
t0 = Simulation_setup.t0_mjd
tend = t0+Simulation_setup.simulation_time
fixed_time_step = Simulation_setup.fixed_time_step
n_steps = Simulation_setup.n_steps
simulation_start_epoch = Simulation_setup.simulation_start_epoch
simulation_end_epoch = Simulation_setup.simulation_end_epoch
ephemeris_time_span = Simulation_setup.ephemeris_time_span
time = Simulation_setup.simulation_span

#######
LUMIO_initial_state = Dataset_reader.initial_state(t0)
X_Moon = Moon_ephemeris(simulation_start_epoch, simulation_end_epoch, n_steps)

### Environment Setup ###
# The creation of bodies
bodies_to_create = [
    "Earth", "Moon", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"
]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
initial_time = simulation_start_epoch
final_time = simulation_end_epoch
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create, initial_time, final_time, global_frame_origin, global_frame_orientation, fixed_time_step
)

body_system = environment_setup.create_system_of_bodies(body_settings)

# Adding LUMIO to the fray
body_system.create_empty_body("LUMIO")
body_system.get("LUMIO").mass = 22.3

bodies_to_propagate = ["LUMIO"]
central_bodies = ["Earth"]

### Acceleration Setup ###
# SRP
reference_area_radiation = 1.0
radiation_pressure_coefficient = 1.0
occulting_bodies = ["Moon"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)
environment_setup.add_radiation_pressure_interface(body_system,"LUMIO", radiation_pressure_settings)

acceleration_settings_LUMIO = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(40,40)],
    Sun=[propagation_setup.acceleration.point_mass_gravity(),
         propagation_setup.acceleration.cannonball_radiation_pressure()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings = {
    "LUMIO": acceleration_settings_LUMIO
}

acceleration_models = propagation_setup.create_acceleration_models(
    body_system, acceleration_settings, bodies_to_propagate, central_bodies)

### Initial States ###
initial_states = np.transpose([LUMIO_initial_state])

### Savings ###
# Is adjustable
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("LUMIO"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "LUMIO", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "LUMIO", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "LUMIO", "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Mercury"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Saturn"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Uranus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Neptune"
    )
]

### Propagating ###
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagation_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    termination_condition,
    output_variables=dependent_variables_to_save
)
### Integrating ###
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_time_step
)

### Dynamic Simulator ###
dynamic_simulator = numerical_simulation.SingleArcSimulator(
    body_system, integrator_settings, propagation_settings
)

########################################################################################################################
####### RESULTS #################
output_dict = dynamic_simulator.dependent_variable_history
states_dict = dynamic_simulator.state_history
output = np.vstack(list(output_dict.values()))
states = np.vstack(list(states_dict.values()))

print("[LUMIO_simulation.py] successfully ran \n")