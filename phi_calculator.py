# General libraries
import numpy as np
from Satellites import EML2
from Satellites import ELO
# tudatpy libraries
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup

spice_interface.load_standard_kernels()

def phi_higherfidelity_eml2(t, dt, X):
    simulation_start_epoch = t
    simulation_end_epoch = t + 3*dt
    fixed_time_step = dt
    initial_states = np.transpose(X[0:6])[0]

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
    # Defining to be propagated satellites from Satellites.py
    body_system.create_empty_body("EML2")
    body_system.get("EML2").mass = EML2.mass

    bodies_to_propagate = ["EML2"]
    central_bodies = ["Earth"]
    ### Acceleration Setup ###
    # SRP
    reference_area_radiation_eml2 = EML2.reference_area
    radiation_pressure_coefficient_eml2 = EML2.radiation_pressure_coefficient
    occulting_bodies_eml2 = EML2.occulting_bodies
    radiation_pressure_settings_eml2 = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation_eml2, radiation_pressure_coefficient_eml2, occulting_bodies_eml2
    )

    environment_setup.add_radiation_pressure_interface(body_system, "EML2", radiation_pressure_settings_eml2)
    acceleration_settings_eml2 = dict(
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
        "EML2": acceleration_settings_eml2
    }

    acceleration_models = propagation_setup.create_acceleration_models(
        body_system, acceleration_settings, bodies_to_propagate, central_bodies)
    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
    propagation_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_states,
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

    parameter_settings = estimation_setup.parameter.initial_states(
        propagation_settings, body_system)
    parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Earth"))
    parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Moon"))
    parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Sun"))
    parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient("EML2"))
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

    state_transition_matrices_eml2 = variational_equations_solver.state_transition_matrix_history[t+dt]

    return state_transition_matrices_eml2


def phi_higherfidelity_elo(t, dt, X):
    simulation_start_epoch = t
    simulation_end_epoch = t + 3*dt
    fixed_time_step = dt
    initial_states = np.transpose(X[6:12])[0]

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
    # Defining to be propagated satellites from Satellites.py
    body_system.create_empty_body("ELO")
    body_system.get("ELO").mass = ELO.mass

    bodies_to_propagate = ["ELO"]
    central_bodies = ["Earth"]
    ### Acceleration Setup ###
    # SRP
    reference_area_radiation_elo = ELO.reference_area
    radiation_pressure_coefficient_elo = ELO.radiation_pressure_coefficient
    occulting_bodies_elo = ELO.occulting_bodies
    radiation_pressure_settings_elo = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation_elo, radiation_pressure_coefficient_elo, occulting_bodies_elo
    )
    environment_setup.add_radiation_pressure_interface(body_system, "ELO", radiation_pressure_settings_elo)

    acceleration_settings_elo = dict(
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(12, 12)],
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
        "ELO": acceleration_settings_elo
    }

    acceleration_models = propagation_setup.create_acceleration_models(
        body_system, acceleration_settings, bodies_to_propagate, central_bodies)

    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
    propagation_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_states,
        termination_condition,
    )

    ### Integrating ###
    integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
        simulation_start_epoch, fixed_time_step
    )

    parameter_settings = estimation_setup.parameter.initial_states(
        propagation_settings, body_system)
    parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Earth"))
    parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Moon"))
    parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Sun"))
    parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient("ELO"))
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

    state_transition_matrices_elo = variational_equations_solver.state_transition_matrix_history[t+dt]

    return state_transition_matrices_elo



