"""
This is the first part of the Orbit Determination model, which consists of a Dynamic model, Measurement model and a
Estimation model.

This is the dynamic model of the propagation of LUMIO and the LLOsat. The simulation time is 10 days with a time step of
0.01 JULIAN DAY. The simulation provides the nominal states of both LUMIO and LLOsat. Also, the model provides the
acceleration inputs of all perturbations and the total acceleration acting on the satellites.
"""
import Simulation_setup
import LLO_initial_states
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup

## Loading SPICE kernels
spice_interface.load_standard_kernels()

print("Running [State_Transition_Matrix_LLOsat.py]")
# Adjust simulation setting in [Simulation_setup.py]
t0 = Simulation_setup.t0_mjd
tend = t0+Simulation_setup.simulation_time
fixed_time_step = Simulation_setup.fixed_time_step
n_steps = Simulation_setup.n_steps
simulation_start_epoch = Simulation_setup.simulation_start_epoch
simulation_end_epoch = Simulation_setup.simulation_end_epoch
ephemeris_time_span = Simulation_setup.ephemeris_time_span
time = Simulation_setup.simulation_span

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

# Adding LLO Satellite to system
body_system.create_empty_body("LLOsat")
body_system.get("LLOsat").mass = LLO_initial_states.mass_LLOsat

bodies_to_propagate = ["LLOsat"]
central_bodies = ["Moon"]
### Acceleration Setup ###
# SRP

reference_area_radiation_LLOsat = LLO_initial_states.reference_area_radiation_LLOsat
radiation_pressure_coefficient_LLOsat = LLO_initial_states.radiation_pressure_coefficient_LLOsat
occulting_bodies_LLOsat = LLO_initial_states.occulting_bodies_LLOsat
radiation_pressure_settings_LLOsat = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation_LLOsat, radiation_pressure_coefficient_LLOsat, occulting_bodies_LLOsat
)

environment_setup.add_radiation_pressure_interface(body_system, "LLOsat",radiation_pressure_settings_LLOsat)

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
    "LLOsat": acceleration_settings_LLOsat
}

acceleration_models = propagation_setup.create_acceleration_models(
    body_system, acceleration_settings, bodies_to_propagate, central_bodies)

### Initial States ###
initial_llosat = LLO_initial_states.initial_state_pathfinder

### Propagating ###
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagation_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_llosat,
    termination_condition,
)

### Integrating ###
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_time_step
)

### Dynamic Simulator ###
dynamic_simulator = numerical_simulation.SingleArcSimulator(
    body_system, integrator_settings, propagation_settings
)

parameter_settings = estimation_setup.parameter.initial_states(propagation_settings, body_system)
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Earth"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Moon"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Sun"))
parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient("LLOsat"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Mercury"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Venus"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Mars"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Jupiter"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Saturn"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Uranus"))
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Neptune"))

variational_equations_solver = numerical_simulation.SingleArcVariationalSimulator(
    body_system, integrator_settings, propagation_settings,
    estimation_setup.create_parameters_to_estimate(parameter_settings, body_system), integrate_on_creation=1
)

state_transition_matrices_llosat = variational_equations_solver.state_transition_matrix_history
print(state_transition_matrices_llosat)
print("[State_Transition_Matrix_LLOsat.py] successfully ran \n")


