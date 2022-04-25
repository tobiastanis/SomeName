"""
High fidelity simulation model
"""
#general
import numpy as np
#own
from Satellites import EML2
from Satellites import ELO
#tudatpy
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup

## Loading SPICE kernels
spice_interface.load_standard_kernels()

"""
higherfidelity_model: 
    includes all solar system planets and the Sun. Only Moon is 12,12 order,degree spherical harmonic 
highfidelity_model:
    Only includes accelerations above 12 m/s^2, which are
    EML2: SRP, pointmasses Sun, Moon, Earth, Mercury, Venus, Mars, Jupiter, Saturn
    ELO: SRP, spherical harmonic gravity Moon 12,12 and pointmasses Sun, Earth, Jupiter
"""
def higherfidelity_model(t0, dt, tend, X, savings):
    simulation_start_epoch = t0
    simulation_end_epoch = tend
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
    # Defining to be propagated satellites from Satellites.py
    body_system.create_empty_body("EML2")
    body_system.get("EML2").mass = EML2.mass
    body_system.create_empty_body("ELO")
    body_system.get("ELO").mass = ELO.mass

    bodies_to_propagate = ["EML2", "ELO"]
    central_bodies = ["Earth", "Earth"]
    ### Acceleration Setup ###
    # SRP
    reference_area_radiation_eml2 = EML2.reference_area
    radiation_pressure_coefficient_eml2 = EML2.radiation_pressure_coefficient
    occulting_bodies_eml2 = EML2.occulting_bodies
    radiation_pressure_settings_eml2 = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation_eml2, radiation_pressure_coefficient_eml2, occulting_bodies_eml2
    )
    reference_area_radiation_elo = ELO.reference_area
    radiation_pressure_coefficient_elo = ELO.radiation_pressure_coefficient
    occulting_bodies_elo = ELO.occulting_bodies
    radiation_pressure_settings_elo = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation_elo, radiation_pressure_coefficient_elo, occulting_bodies_elo
    )

    environment_setup.add_radiation_pressure_interface(body_system, "EML2", radiation_pressure_settings_eml2)
    environment_setup.add_radiation_pressure_interface(body_system, "ELO", radiation_pressure_settings_elo)
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
        "EML2": acceleration_settings_eml2,
        "ELO": acceleration_settings_elo
    }

    acceleration_models = propagation_setup.create_acceleration_models(
        body_system, acceleration_settings, bodies_to_propagate, central_bodies)

    if savings == 1:
        ### Savings ###
        # Is adjustable
        dependent_variables_to_save = [
            propagation_setup.dependent_variable.total_acceleration("EML2"),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.cannonball_radiation_pressure_type, "EML2", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Earth"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Moon"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Mercury"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Venus"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Mars"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Jupiter"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Saturn"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Uranus"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Neptune"
            ),
            propagation_setup.dependent_variable.total_acceleration("ELO"),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.cannonball_radiation_pressure_type, "ELO", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Earth"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.spherical_harmonic_gravity_type, "ELO", "Moon"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Mercury"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Venus"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Mars"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Jupiter"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Saturn"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Uranus"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Neptune"
            ),
            propagation_setup.dependent_variable.central_body_fixed_cartesian_position("EML2", "Moon"),
            propagation_setup.dependent_variable.relative_velocity("EML2", "ELO"),
            propagation_setup.dependent_variable.relative_position("EML2", "ELO"),
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
    else:
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

    ###### RESULTS ######

    states_dict = dynamic_simulator.state_history
    states = np.vstack(list(states_dict.values()))

    if savings == 1:
        output_dict = dynamic_simulator.dependent_variable_history
        output = np.vstack(list(output_dict.values()))
        return [states, output]
    else:
        return states


def highfidelity_model(t0, dt, tend, X, savings):
    simulation_start_epoch = t0
    simulation_end_epoch = tend
    fixed_time_step = dt

    initial_states = X

    ### Environment Setup ###
    # The creation of bodies
    bodies_to_create = [
        "Earth", "Moon", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"
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
    body_system.create_empty_body("ELO")
    body_system.get("ELO").mass = ELO.mass

    bodies_to_propagate = ["EML2", "ELO"]
    central_bodies = ["Earth", "Earth"]
    ### Acceleration Setup ###
    # SRP
    reference_area_radiation_eml2 = EML2.reference_area
    radiation_pressure_coefficient_eml2 = EML2.radiation_pressure_coefficient
    occulting_bodies_eml2 = EML2.occulting_bodies
    radiation_pressure_settings_eml2 = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation_eml2, radiation_pressure_coefficient_eml2, occulting_bodies_eml2
    )
    reference_area_radiation_elo = ELO.reference_area
    radiation_pressure_coefficient_elo = ELO.radiation_pressure_coefficient
    occulting_bodies_elo = ELO.occulting_bodies
    radiation_pressure_settings_elo = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation_elo, radiation_pressure_coefficient_elo, occulting_bodies_elo
    )

    environment_setup.add_radiation_pressure_interface(body_system, "EML2", radiation_pressure_settings_eml2)
    environment_setup.add_radiation_pressure_interface(body_system, "ELO", radiation_pressure_settings_elo)
    acceleration_settings_eml2 = dict(
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Moon=[propagation_setup.acceleration.point_mass_gravity()],
        Sun=[propagation_setup.acceleration.point_mass_gravity(),
             propagation_setup.acceleration.cannonball_radiation_pressure()],
        Mercury=[propagation_setup.acceleration.point_mass_gravity()],
        Venus=[propagation_setup.acceleration.point_mass_gravity()],
        Mars=[propagation_setup.acceleration.point_mass_gravity()],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        Saturn=[propagation_setup.acceleration.point_mass_gravity()]
    )

    acceleration_settings_elo = dict(
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(12, 12)],
        Sun=[propagation_setup.acceleration.point_mass_gravity(),
             propagation_setup.acceleration.cannonball_radiation_pressure()],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()]
    )

    acceleration_settings = {
        "EML2": acceleration_settings_eml2,
        "ELO": acceleration_settings_elo
    }

    acceleration_models = propagation_setup.create_acceleration_models(
        body_system, acceleration_settings, bodies_to_propagate, central_bodies)

    if savings == 1:
        ### Savings ###
        # Is adjustable
        dependent_variables_to_save = [
            propagation_setup.dependent_variable.total_acceleration("EML2"),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.cannonball_radiation_pressure_type, "EML2", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Earth"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Moon"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Mercury"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Venus"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Mars"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Jupiter"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "EML2", "Saturn"
            ),
            propagation_setup.dependent_variable.total_acceleration("ELO"),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.cannonball_radiation_pressure_type, "ELO", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Earth"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.spherical_harmonic_gravity_type, "ELO", "Moon"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Sun"
            ),
            propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, "ELO", "Jupiter"
            ),
            propagation_setup.dependent_variable.central_body_fixed_cartesian_position("EML2", "Moon"),
            propagation_setup.dependent_variable.relative_velocity("EML2", "ELO"),
            propagation_setup.dependent_variable.relative_position("EML2", "ELO"),
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
    else:
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

    ###### RESULTS ######

    states_dict = dynamic_simulator.state_history
    states = np.vstack(list(states_dict.values()))

    if savings == 1:
        output_dict = dynamic_simulator.dependent_variable_history
        output = np.vstack(list(output_dict.values()))
        return [states, output]
    else:
        return states






