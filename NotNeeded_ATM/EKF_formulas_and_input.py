import numpy as np


def H_range_LUMIO_simulation(rho1, rho2, rho3):
    rho_abs = (rho1**2 + rho2**2 + rho3**2)**0.5
    return np.array([rho1/rho_abs, rho2/rho_abs, rho3/rho_abs, 0, 0, 0])

def H_range_2sat_simulation(X):
    X = np.transpose(X)[0]
    x1 = X[0]; y1 = X[1]; z1 = X[2]; x2 = X[6]; y2 = X[7]; z2 = X[8]
    rho_abs = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    return np.array([(x1-x2)/rho_abs, (y1-y2)/rho_abs, (z1-z2)/rho_abs, 0, 0, 0, (-x1+x2)/rho_abs, (-y1+y2)/rho_abs,
                     (-z1+z2)/rho_abs, 0, 0, 0])

def Y(X):
    x1 =X[0]; y1 =X[1]; z1=X[2]; x2=X[6]; y2=X[7]; z2=X[8]
    return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5


