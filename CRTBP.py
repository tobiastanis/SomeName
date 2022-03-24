"""
Some text about the CRTBP
"""
import numpy as np
from tudatpy.kernel.interface import spice_interface
from scipy.integrate import odeint
from tudatpy.kernel import constants
import Simulation_setup
spice_interface.load_standard_kernels()
print("Running [CRTBP.py]")

def crtbp(x, t, mu):
    r1 = np.sqrt((mu + x[0]) ** 2 + x[1] ** 2 + x[2] ** 2)              #Distance to Primary (Earth)
    r2 = np.sqrt((1 - mu - x[0]) ** 2 + x[1] ** 2 + x[2] ** 2)          #Distance to Secondary (Moon)
    # Normalized masses of the primaries
    mu = spice_interface.get_body_gravitational_parameter("Moon") / \
         (spice_interface.get_body_gravitational_parameter("Moon") +
          spice_interface.get_body_gravitational_parameter("Earth"))
    xdot = [x[3],
            x[4],
            x[5],
            x[0] + 2 * x[4] - (1 - mu) * (x[0] + mu) / r1 ** 3 - mu * (x[0] + mu - 1) / r2 ** 3,
            -2 * x[3] + (1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3) * x[1],
            ((mu - 1) / r1 ** 3 - mu / r2 ** 3) * x[2]
            ]
    return xdot

####### Normalizing Units #######
G = constants.GRAVITATIONAL_CONSTANT
M_Earth = spice_interface.get_body_gravitational_parameter("Earth")/G
M_Moon = spice_interface.get_body_gravitational_parameter("Moon")/G

L_char = 384400e3
m_char = M_Earth+M_Moon
t_char = np.sqrt(L_char**3/(constants.GRAVITATIONAL_CONSTANT*m_char))
v_char = L_char/t_char
mu = spice_interface.get_body_gravitational_parameter("Moon")/(spice_interface.get_body_gravitational_parameter("Earth")+spice_interface.get_body_gravitational_parameter("Moon"))


t_span = np.linspace(0, Simulation_setup.simulation_time*constants.JULIAN_DAY/t_char, Simulation_setup.n_steps)
x_ini = np.array([1.1435, 0, -0.1579, 0, -0.2220, 0])

states = odeint(crtbp, x_ini, t_span, args=(mu,), rtol=1e-12, atol=1e-12)

print("[CRTBP.py] ran successfully \n")