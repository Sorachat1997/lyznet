import sympy as sp
import numpy as np
import lyznet
import torch

lyznet.utils.set_random_seed()

x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
symbolic_vars = (x1, x2, x3, x4)

m = 0.1
M = 1
L = 0.8
gc = 9.8

theta = x1
omega = x2
p = x3
v = x4

# Define dynamics and cost
f = sp.Matrix([
    omega, 
    (-m * L * omega**2 * sp.sin(theta) * sp.cos(theta) 
     + (M + m) * gc * sp.sin(theta)) / (L * (M + m * (sp.sin(theta)**2))),
    v, 
    (m * sp.sin(theta) * (L * omega**2 - gc * sp.cos(theta))) / 
    (M + m * sp.sin(theta)**2)
    ])

g = sp.Matrix([
    0, 
    -sp.cos(theta) / (L * (M + m * (sp.sin(theta)**2))), 
    0, 
    1 / (M + m * sp.sin(theta)**2)
    ])


def g_numpy(x):
    # x is a numpy array of shape (N, d)
    # Return a numpy array of shape (N, d, k), where k is the number of inputs
    if x.ndim == 1:
        x = x.reshape(1, -1)
    m = 0.1
    M = 1
    L = 0.8
    theta = x[:, 0]
    omega = x[:, 1]

    g1 = np.zeros_like(theta)
    g2 = -np.cos(theta) / (L * (M + m * (np.sin(theta)**2)))
    g3 = np.zeros_like(theta)
    g4 = 1 / (M + m * np.sin(theta)**2)

    return np.stack([g1, g2, g3, g4], axis=1)[:, :, np.newaxis]


Q = np.diag([60, 1.5, 180, 45])

domain = [[-0.2, 0.2]] * 4
sys_name = f"cartpole_{domain[0]}"


K = sp.Matrix([[-32.8482,   -9.1512,   -1.0000,   -2.3460]])

initial_u = -K * sp.Matrix([x1, x2, x3, x4])


system = lyznet.ControlAffineSystem(f, g, domain, sys_name, Q=Q)
print("Eigenvalues of linearization: ", np.linalg.eigvals(system.A))

lyznet.elm_pi(system, initial_u=initial_u, num_of_iters=10, width=400, 
              num_colloc_pts=3000, g_numpy=g_numpy, 
              final_plot=True, final_test=True
              )
