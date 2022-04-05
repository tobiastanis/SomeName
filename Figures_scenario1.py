"""
Figures of the results of scenario 1
"""
import numpy as np
import matplotlib.pyplot as plt
import scenario1

print("Running [Figures_scenario1.py]")
simulation_time = scenario1.time

### raw
output = scenario1.output
states = scenario1.states
###
LUMIO_dataset_states = scenario1.LUMIO_Dataset_states
LUMIO_states = states[:, 0:6]
LLOsat_states = states[:, 6:12]






print("[Figures_scenario1.py] ran successful")