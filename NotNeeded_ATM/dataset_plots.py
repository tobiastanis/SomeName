import numpy as np
import Dataset_reader
import matplotlib.pyplot as plt
t0 = 60390.00000
sim_time = 30
tend = t0+sim_time

X_LUMIO = Dataset_reader.state_lumio(t0,tend)
X_Moon = Dataset_reader.state_moon(t0, tend)
time = np.linspace(0, sim_time, len(X_LUMIO))
velo_LUMIO = X_LUMIO[:, 3:6]
vel_norm = np.linalg.norm(velo_LUMIO, axis=1)
print(vel_norm)
plt.figure()
plt.plot(time, vel_norm)



plt.show()