"""
Simple measurement model, to be writen
"""
import numpy as np
import LUMIO_LLO_propagation
import matplotlib.pyplot as plt

print("Running [Simple_Measurement_Model.py]")
time = LUMIO_LLO_propagation.time

x_Moon = LUMIO_LLO_propagation.X_Moon
states_LUMIO = LUMIO_LLO_propagation.states[:, 0:6]
states_LLO_wrt_Moon = LUMIO_LLO_propagation.states[:, 6:12]
states_LLO = np.add(states_LLO_wrt_Moon, x_Moon)

LOS_vector = np.subtract(states_LUMIO[:, 0:3], states_LLO[:, 3:6])
intersatelltie_distance = np.linalg.norm(LOS_vector, axis=1)

measurement_noise = np.random.normal(0, 10e5, len(time))

measured_distance = np.add(intersatelltie_distance, measurement_noise)

plt.figure()
plt.plot(time, intersatelltie_distance, color='red')
plt.plot(time, measured_distance, color='blue')
plt.title('((Measured) Intersatellite distance')
plt.xlabel('Time [days]')
plt.ylabel('Distance [m]')
plt.legend(['Nominal intersatellite distance', 'Measured intersatellite distance'])

print('[Simple_Measurement_Model.py] successfully ran')
plt.show()

