"""
State transition matrices are obtained by using parts of the dynamic model.
"""
import Dataset_reader
import Simulation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from Ephemeris_obtainer import Moon_ephemeris

## Loading SPICE kernels
spice_interface.load_standard_kernels()

print("Running [State_Transition_Matrix_LUMIO.py]")
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

bodies_to_propagate = ["LUMIO"]
central_bodies = ["Earth"]
### Acceleration Setup ###
# SRP
reference_area_radiation_LUMIO = Simulation_setup.reference_area_radiation_LUMIO
radiation_pressure_coefficient_LUMIO = Simulation_setup.radiation_pressure_coefficient_LUMIO
occulting_bodies_LUMIO = Simulation_setup.occulting_bodies_LUMIO
radiation_pressure_settings_LUMIO = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation_LUMIO, radiation_pressure_coefficient_LUMIO, occulting_bodies_LUMIO
)

environment_setup.add_radiation_pressure_interface(body_system,"LUMIO", radiation_pressure_settings_LUMIO)

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
acceleration_settings = {
    "LUMIO": acceleration_settings_LUMIO
}

acceleration_models = propagation_setup.create_acceleration_models(
    body_system, acceleration_settings, bodies_to_propagate, central_bodies)

### Initial States ###
initial_lumio = LUMIO_initial_state

### Propagating ###
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagation_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_lumio,
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
parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient("LUMIO"))
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

state_transition_matrices_lumio = variational_equations_solver.state_transition_matrix_history
print("[State_Transition_Matrix_LUMIO.py] successfully ran \n")



