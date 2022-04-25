import numpy as np

# Possible H functions
def H_range_2sat_simulation(X):
    X = np.transpose(X)[0]
    x1 = X[0]; y1 = X[1]; z1 = X[2]; x2 = X[6]; y2 = X[7]; z2 = X[8]
    rho_abs = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    return np.array([(x1-x2)/rho_abs, (y1-y2)/rho_abs, (z1-z2)/rho_abs, 0, 0, 0, (-x1+x2)/rho_abs, (-y1+y2)/rho_abs,
                     (-z1+z2)/rho_abs, 0, 0, 0])

# Intersatellite distance
def Y(X):
    x1 =X[0]; y1 =X[1]; z1=X[2]; x2=X[6]; y2=X[7]; z2=X[8]
    return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

# Phi
def Phi(Phi_EML2, Phi_ELO):
    Phi_top = np.concatenate((Phi_EML2, np.zeros((6, 6))), axis=1)
    Phi_bot = np.concatenate((np.zeros((6, 6)), Phi_ELO), axis=1)
    Phi = np.concatenate((Phi_top, Phi_bot), axis=0)
    return Phi
