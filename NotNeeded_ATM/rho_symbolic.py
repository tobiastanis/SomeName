import numpy as np
import sympy as sp
"""
Ideal range and range rate
"""
#rho = sp.Symbol('rho')
x1 = sp.Symbol('x1'); y1 = sp.Symbol('y1'); z1 = sp.Symbol('z1');
vx1 = sp.Symbol('vx1'); vy1 = sp.Symbol('vy1') ; vz1 = sp.Symbol('vz1')
x2 = sp.Symbol('x2'); y2 = sp.Symbol('y2'); z2 = sp.Symbol('z2');
vx2 = sp.Symbol('vx2'); vy2 = sp.Symbol('vy2') ; vz2 = sp.Symbol('vz2')

rho = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

rho_par_x1 = rho.diff('x1'); rho_par_y1 = rho.diff('y1'); rho_par_z1 = rho.diff('z1')
rho_par_vx1 = rho.diff('vx1'); rho_par_vy1 = rho.diff('vy1'); rho_par_vz1 = rho.diff('vz1')
rho_par_x2 = rho.diff('x2'); rho_par_y2 = rho.diff('y2'); rho_par_z2 = rho.diff('z2')
rho_par_vx2 = rho.diff('vx2'); rho_par_vy2 = rho.diff('vy1'); rho_par_vz2 = rho.diff('vz2')

rho_abs = sp.Symbol('rho_abs')

rho_dot = 1/rho_abs*((x1-x2)*(vx1-vx2) + (y1-y2)*(vy1-vy2) + (z1-z2)*(vz1-vz2))

rho_dot_par_x1 = rho_dot.diff('x1'); rho_dot_par_y1 = rho_dot.diff('y1'); rho_dot_par_z1 = rho_dot.diff('z1')
rho_dot_par_vx1 = rho_dot.diff('vx1'); rho_dot_par_vy1 = rho_dot.diff('vy1'); rho_dot_par_vz1 = rho_dot.diff('vz1')
rho_dot_par_x2 = rho_dot.diff('x2'); rho_dot_par_y2 = rho_dot.diff('y2'); rho_dot_par_z2 = rho_dot.diff('z2')
rho_dot_par_vx2 = rho_dot.diff('vx2'); rho_dot_par_vy2 = rho_dot.diff('vy2'); rho_dot_par_vz2 = rho_dot.diff('vz2')

#print(rho_dot_par_x1, rho_dot_par_y1, rho_dot_par_z1, rho_dot_par_vx1, rho_dot_par_vy1, rho_dot_par_vz1)
#print(rho_dot_par_x2, rho_dot_par_y2, rho_dot_par_z2, rho_dot_par_vx2, rho_dot_par_vy2, rho_dot_par_vz2)

H_wave_symbolic = [rho_par_x1, rho_par_y1, rho_par_z1, rho_par_vx1, rho_par_vy1, rho_par_vz1, rho_par_x2,
                  rho_par_y2, rho_par_z2, rho_par_vx2, rho_par_vy2, rho_par_vz2], [rho_dot_par_x1, rho_dot_par_y1,
                  rho_dot_par_z1, rho_dot_par_vx1, rho_dot_par_vy1, rho_dot_par_vz1, rho_dot_par_x2, rho_dot_par_y2,
                  rho_dot_par_z2, rho_dot_par_vx2, rho_dot_par_vy2, rho_dot_par_vz2]

print(H_wave_symbolic)

