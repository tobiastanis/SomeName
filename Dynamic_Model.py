"""
This is the first part of the Orbit Determination model, which consists of a Dynamic model, Measurement model and a
Estimation model.

This is the dynamic model of the propagation of LUMIO and the LLOsat. The simulation time is 10 days with a time step of
0.01 JULIAN DAY. The simulation provides the nominal states of both LUMIO and LLOsat. Also, the model provides the
acceleration inputs of all perturbations and the total acceleration acting on the satellites.
"""
from datetime import datetime
import Dataset_reader
import Simulation_setup
import numpy as np
import LLO_initial_states
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from Ephemeris_obtainer import Moon_ephemeris

## Loading SPICE kernels
spice_interface.load_standard_kernels()
# Measuring dynamic model runtime
starttime = datetime.now()

print("Running [Dynamic_Model.py]")
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
LUMIO_Dataset_states = Dataset_reader.state_lumio(t0, tend)
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
body_system.get("LUMIO").mass = Simulation_setup.LUMIO_mass
# Adding LLO Satellite to system
body_system.create_empty_body("LLOsat")
body_system.get("LLOsat").mass = LLO_initial_states.mass_LLOsat

bodies_to_propagate = ["LUMIO", "LLOsat"]
central_bodies = ["Earth", "Moon"]
### Acceleration Setup ###
# SRP
reference_area_radiation_LUMIO = Simulation_setup.reference_area_radiation_LUMIO
radiation_pressure_coefficient_LUMIO = Simulation_setup.radiation_pressure_coefficient_LUMIO
occulting_bodies_LUMIO = Simulation_setup.occulting_bodies_LUMIO
radiation_pressure_settings_LUMIO = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation_LUMIO, radiation_pressure_coefficient_LUMIO, occulting_bodies_LUMIO
)
reference_area_radiation_LLOsat = LLO_initial_states.reference_area_radiation_LLOsat
radiation_pressure_coefficient_LLOsat = LLO_initial_states.radiation_pressure_coefficient_LLOsat
occulting_bodies_LLOsat = LLO_initial_states.occulting_bodies_LLOsat
radiation_pressure_settings_LLOsat = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation_LLOsat, radiation_pressure_coefficient_LLOsat, occulting_bodies_LLOsat
)

environment_setup.add_radiation_pressure_interface(body_system,"LUMIO", radiation_pressure_settings_LUMIO)
environment_setup.add_radiation_pressure_interface(body_system, "LLOsat",radiation_pressure_settings_LLOsat)
acceleration_settings_LUMIO = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
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

acceleration_settings_LLOsat = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(12,12)],
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
    "LUMIO": acceleration_settings_LUMIO,
    "LLOsat": acceleration_settings_LLOsat
}

acceleration_models = propagation_setup.create_acceleration_models(
    body_system, acceleration_settings, bodies_to_propagate, central_bodies)

### Initial States ###
LLO_initial_state = LLO_initial_states.initial_state_pathfinder
initial_states = np.vstack([
    LUMIO_initial_state.reshape(-1,1),
    LLO_initial_state.reshape(-1,1)
])

### Savings ###
# Is adjustable
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("LUMIO"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "LUMIO", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Moon"
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
    ),
    propagation_setup.dependent_variable.total_acceleration("LLOsat"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "LLOsat", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "LLOsat", "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Mercury"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Saturn"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Uranus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOsat", "Neptune"
    ),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("LUMIO", "Moon"),
    propagation_setup.dependent_variable.relative_velocity("LUMIO", "Moon"),
    propagation_setup.dependent_variable.relative_velocity("LUMIO", "LLOsat"),
    propagation_setup.dependent_variable.relative_position("LUMIO", "LLOsat"),
    propagation_setup.dependent_variable.relative_distance("LUMIO", "LLOsat"),
    propagation_setup.dependent_variable.relative_position("LLOsat", "Earth")
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

###### RESULTS ######
output_dict = dynamic_simulator.dependent_variable_history
states_dict = dynamic_simulator.state_history
output = np.vstack(list(output_dict.values()))
states = np.vstack(list(states_dict.values()))

states_LUMIO = states[:, 0:6]
states_LLOsat_wrt_Moon = states[:, 6:12]
states_LLOsat = np.add(states_LLOsat_wrt_Moon, X_Moon)
relative_position_vector = output[:, 37:40]
relative_velocity_vector = output[:, 34:37]

endtime = datetime.now()
print('Maximum inter-satellite distance:', max(np.linalg.norm(relative_position_vector, axis=1))*10**-3, 'km \n')
print('Minimum inter-satellite distance:', min(np.linalg.norm(relative_position_vector, axis=1))*10**-3, 'km \n')

print('Dynamic model runtime: {}'.format(endtime - starttime))
print("[Dynamic_Model.py] successfully ran \n")


