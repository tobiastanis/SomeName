"""
Simple measurement model, to be writen
"""
import numpy as np
import LUMIO_LLO_propagation
import matplotlib.pyplot as plt

print("Running [Simple_Measurement_Model.py]")
time = LUMIO_LLO_propagation.time
output = LUMIO_LLO_propagation.output
position_LUMIO_wrt_Moon = output[:, 28:31]
x_Moon = LUMIO_LLO_propagation.X_Moon
states_LLO_wrt_Moon = LUMIO_LLO_propagation.states[:, 6:12]
position_LLO_wrt_Moon = states_LLO_wrt_Moon[:, 3:6]

LOS_vector = np.subtract(position_LUMIO_wrt_Moon, position_LLO_wrt_Moon)
intersatelltie_distance = np.linalg.norm(LOS_vector, axis=1)

measurement_noise = np.random.normal(0, 10e4, len(time))

measured_distance = np.add(intersatelltie_distance, measurement_noise)

plt.figure()
plt.plot(time, intersatelltie_distance, color='red')
plt.plot(time, measured_distance, color='blue')
plt.title('(Measured) Intersatellite distance')
plt.xlabel('Time [days]')
plt.ylabel('Distance [m]')
plt.legend(['Nominal intersatellite distance', 'Measured intersatellite distance'])

print('Maximum intersatellite distance \n', max(intersatelltie_distance)*10**-3, '[km]')
print('Minimum intersatellite distance \n', min(intersatelltie_distance)*10**-3, '[km]')


print('[Simple_Measurement_Model.py] successfully ran')
plt.show()

