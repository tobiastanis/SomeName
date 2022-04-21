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
    n_steps = 1
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

    states = np.transpose([dynamic_simulator.state_history[simulation_end_epoch]])

    return states