"""
Some text about the CRTBP
"""
import numpy as np
from tudatpy.kernel.interface import spice_interface

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

