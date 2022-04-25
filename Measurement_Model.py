"""
Measurement Model
"""
import numpy as np
import Nominal_Simulation
import functions_ekf

states = Nominal_Simulation.states
intersatellite_distance = []
for i in range(len(states)):
    Y = functions_ekf.Y(states[i])
    intersatellite_distance.append(Y)
intersatellite_distance = np.array(intersatellite_distance)

sigma = 10
sigma_noise = np.random.normal(0, sigma, len(intersatellite_distance))
#Defining the observation array
observations_array = np.add(intersatellite_distance, sigma_noise)

#### From 38_lumiopowerdistr.dpf
# inter-satellite downlink budget assumptions
# Minimum operation distance [m]
#LUMIO_OD_min = 31772E3
# Maximum operation distance [m]
#LUMIO_OD_max = 89870E3
# Frequency [Hz]
#LUMIO_freq = 2200E6
# Transmission power [dBW]
#LUMIO_Tx_power = 3
# Transmission losses [dB]
#Tx_pathlosses = 1
# Transmission antenna gain [dBi]
#LUMIO_Tx_gain_min = 4.5
#LUMIO_Tx_gain_max = 6.5
# Polarisation losses [dB]
#LUMIO_Pol_loss = 0.5
# Datarate [bps]
#LUMIO_datarate_min = 500
#LUMIO_datarate_max = 4000
# Req Eb/No [dB]
#LUMIO_ebno = 2.5
# Link margin [dB]
#LUMIO_linkmargin = 3



