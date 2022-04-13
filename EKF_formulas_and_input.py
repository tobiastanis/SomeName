import numpy as np
import State_Transition_Matrix_LUMIO
import State_Transition_Matrix_LLOsat


def H_range_LUMIO_simulation(rho1, rho2, rho3):
    rho_abs = (rho1**2 + rho2**2 + rho3**2)**0.5
    return np.array([rho1/rho_abs, rho2/rho_abs, rho3/rho_abs, 0, 0, 0])

def H_range_2sat_simulation(X_star):
    X_star = np.transpose(X_star)[0]
    x1 = X_star[0]; y1 = X_star[1]; z1 = X_star[2]; x2 = X_star[6]; y2 = X_star[7]; z2 = X_star[8]
    rho_abs = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    return np.array([(x1-x2)/rho_abs, (y1-y2)/rho_abs, (z1-z2)/rho_abs, 0, 0, 0, (-x1+x2)/rho_abs, (-y1+y2)/rho_abs,
                     (-z1+z2)/rho_abs, 0, 0, 0])



def Phi(i):
    t_ET = State_Transition_Matrix_LUMIO.ephemeris_time_span
    zero_room = np.zeros((6,6))
    Phi_LUMIO = State_Transition_Matrix_LUMIO.state_transition_matrices_lumio
    Phi_LLOsat = State_Transition_Matrix_LLOsat.state_transition_matrices_llosat
    top = np.concatenate((Phi_LUMIO[t_ET[i]], zero_room), axis=1)
    bot = np.concatenate((zero_room, Phi_LLOsat[t_ET[i]]), axis=1)
    return np.concatenate((top, bot), axis=0)


