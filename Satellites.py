"""
Class file for the satellites. May be later include ranging parameters as well..
"""

class satellite:
    def __init__(self, name, mass, reference_area, radiation_pressure_coefficient, occulting_bodies):
        self.name = name
        self.mass = mass
        self.reference_area = reference_area
        self.radiation_pressure_coefficient = radiation_pressure_coefficient
        self.occulting_bodies = occulting_bodies

# EML2 orbiter based on LUMIO
EML2 = satellite("EML2", 22.8, 0.410644, 1.08, ["Moon", "Earth"])

# ELO based on Lunar Pathfinder
ELO = satellite("ELO", 280, 3.0, 1.8, ["Moon", "Earth"])