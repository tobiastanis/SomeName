"""
Measurement Model
"""
import numpy as np
from Dynamic_Model import relative_position_vector
from Dynamic_Model import relative_velocity_vector
from Dynamic_Model import states_LUMIO
from Dynamic_Model import states_LLOsat

norm_position_vector = np.linalg.norm(relative_position_vector, axis=1)
sigma_noise = np.random.normal(0, 10, len(norm_position_vector))
#Defining the observation array
observations_array = np.add(norm_position_vector, sigma_noise)

### From 38_lumiopowerdistr.dpf
# inter-satellite downlink budget assumptions
# Minimum operation distance [m]
LUMIO_OD_min = 31772E3
# Maximum operation distance [m]
LUMIO_OD_max = 89870E3
# Frequency [Hz]
LUMIO_freq = 2200E6
# Transmission power [dBW]
LUMIO_Tx_power = 3
# Transmission losses [dB]
Tx_pathlosses = 1
# Transmission antenna gain [dBi]
LUMIO_Tx_gain_min = 4.5
LUMIO_Tx_gain_max = 6.5
# Polarisation losses [dB]
LUMIO_Pol_loss = 0.5
# Datarate [bps]
LUMIO_datarate_min = 500
LUMIO_datarate_max = 4000
# Req Eb/No [dB]
LUMIO_ebno = 2.5
# Link margin [dB]
LUMIO_linkmargin = 3



