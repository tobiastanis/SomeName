"""
EKF integrator Xhat_k_1 to Xhat_k and obtaining state transition matrix Phi
"""

# General libraries
import numpy as np
# Own libraries
import LLO_initial_states
import Simulation_setup
# tudatpy libraries
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup

## Loading SPICE kernels
spice_interface.load_standard_kernels()

def state_integrator(t, dt, X):
    simulation_start_epoch = t
    simulation_end_epoch = t+dt
    fixed_time_step = dt
    initial_states = X

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

    environment_setup.add_radiation_pressure_interface(body_system, "LUMIO", radiation_pressure_settings_LUMIO)
    environment_setup.add_radiation_pressure_interface(body_system, "LLOsat", radiation_pressure_settings_LLOsat)
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
        "LUMIO": acceleration_settings_LUMIO,
        "LLOsat": acceleration_settings_LLOsat
    }

    #obtaining relative position vector
    dependent_variables_to_save = [
    propagation_setup.dependent_variable.relative_position("LUMIO", "LLOsat")
    ]

    acceleration_models = propagation_setup.create_acceleration_models(
        body_system, acceleration_settings, bodies_to_propagate, central_bodies)
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
    output_dict = dynamic_simulator.dependent_variable_history
    relative_position_vector = output_dict[t+dt]
    norm_relative_position_vector = np.linalg.norm(relative_position_vector)
    states = np.transpose([dynamic_simulator.state_history[simulation_end_epoch]])

    return [states, norm_relative_position_vector]

def Phi_integrator_LUMIO(t, dt, X):
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

    environment_setup.add_radiation_pressure_interface(body_system, "LUMIO", radiation_pressure_settings_LUMIO)

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

    ### Propagating ###
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

    state_transition_matrices_lumio = variational_equations_solver.state_transition_matrix_history[t+dt]

    return state_transition_matrices_lumio

def Phi_integrator_LLOsat(t, dt, X):
    simulation_start_epoch = t
    simulation_end_epoch = t + dt
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

    environment_setup.add_radiation_pressure_interface(body_system, "LLOsat", radiation_pressure_settings_LLOsat)

    acceleration_settings_LLOsat = dict(
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
        "LLOsat": acceleration_settings_LLOsat
    }

    acceleration_models = propagation_setup.create_acceleration_models(
        body_system, acceleration_settings, bodies_to_propagate, central_bodies)

    ### Initial States ###
    initial_llosat = initial_states

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
    return state_transition_matrices_llosat[t+dt]