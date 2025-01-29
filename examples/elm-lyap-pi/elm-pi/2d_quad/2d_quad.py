import sympy as sp
import numpy as np
import lyznet
import torch


lyznet.utils.set_random_seed()

x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6')
symbolic_vars = (x1, x2, x3, x4, x5, x6)

gc, m, I_m = 9.81, 1.0, 1.0

K = sp.Matrix([
    [0.0000, 1.0000, 0.0000, 0.0000, 1.7321, 0.0000],
    [1.0000, 0.0000, 13.5165, 1.9380, 0.0000, 5.2946]
])

initial_u = -K * sp.Matrix([x1, x2, x3, x4, x5, x6])

# u_e = sp.Matrix([gc, 0])
# target = sp.Matrix([0, 0, 0, 0, 0, 0])

# initial_u = -K * (sp.Matrix([x1, x2, x3, x4, x5, x6]) - target) + u_e

f = sp.Matrix([
    x4,
    x5,
    x6,
    gc/m*sp.sin(x3),
    -gc/m+gc/m*sp.cos(x3),
    0
])

g = sp.Matrix([
    [0, 0],  
    [0, 0], 
    [0, 0],
    [sp.sin(x3)/m, 0],
    [sp.cos(x3)/m, 0],
    [0, 1/I_m]
    ])


def f_numpy(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # x1 = x[:, 0]   # x position
    # x2 = x[:, 1]   # y position
    x3 = x[:, 2]   # theta angle
    x4 = x[:, 3]   # velocity in x
    x5 = x[:, 4]   # velocity in y
    x6 = x[:, 5]   # angular velocity
        
    # Equations of motion
    dx1 = x4
    dx2 = x5
    dx3 = x6
    dx4 = (gc/m) * np.sin(x3)
    dx5 = -gc/m + (gc/m) * np.cos(x3)
    dx6 = np.zeros_like(x[:, 0])    
    return np.stack([dx1, dx2, dx3, dx4, dx5, dx6], axis=1)


def g_numpy(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    N = x.shape[0]
    x3 = x[:, 2]
    m = 1
    I_m = 1

    zero_col = np.zeros((N, 1))
    sin_x3_m = np.sin(x3) / m
    sin_x3_m = np.expand_dims(sin_x3_m, axis=1)
    cos_x3_m = np.cos(x3) / m
    cos_x3_m = np.expand_dims(cos_x3_m, axis=1)
    one_I_m = np.tile(np.array([[1 / I_m]]), (N, 1))

    col1 = np.concatenate(
        (zero_col, zero_col, zero_col, sin_x3_m, cos_x3_m, zero_col), axis=1)
    col2 = np.concatenate(
        (zero_col, zero_col, zero_col, zero_col, zero_col, one_I_m), axis=1)
    result = np.stack((col1, col2), axis=-1)
    return result


Q = np.diag([10, 10, 10, 1, 1, 1])
# R = np.array([[0.1, 0.05], [0.05, 0.1]])
# R = np.eye(2)

domain = [[-0.4, 0.4]] * 6
sys_name = f"2d_quad_{domain[0]}"

system = lyznet.ControlAffineSystem(f, g, domain, sys_name, Q=Q)
print("Eigenvalues of linearization: ", np.linalg.eigvals(system.A))

lyznet.elm_pi(system, initial_u=initial_u, num_of_iters=10, width=6400, 
              num_colloc_pts=12000, g_numpy=g_numpy, f_numpy=f_numpy, 
              final_plot=True, final_test=True
              )
