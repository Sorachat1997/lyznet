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


# def f_torch(x):
#     # x is a tensor of shape (N, d)
#     # Return a tensor of shape (N, d, 1) - evaluation of vector field at x
#     m = 0.1
#     M = 1
#     L = 0.8
#     g = 9.8
#     theta = x[:, 0]
#     omega = x[:, 1]
#     x1 = x[:, 2]
#     x2 = x[:, 3]

#     theta_d = omega
#     omega_d = ((-m * L * omega**2 * torch.sin(theta) * torch.cos(theta) 
#                + (M + m) * g * torch.sin(theta)) / 
#                (L * (M + m * (torch.sin(theta)**2))))
#     x1_d = x2
#     x2_d = (m * torch.sin(theta) * (L * omega**2 - g * torch.cos(theta))) / \
#            (M + m * torch.sin(theta)**2)

#     return torch.stack([theta_d, omega_d, x1_d, x2_d], dim=1)


# define a torch function for g and pass as argument (optional)
# if not provided, g will be matrix-lamdified to for g_torch (less efficient)

def g_torch(x):
    # x is a tensor of shape (N, d)
    # Return a tensor of shape (N, d, k), where k is the number of inputs 
    m = 0.1
    M = 1
    L = 0.8
    theta = x[:, 0]
    omega = x[:, 1]

    return torch.stack(
        [torch.zeros_like(theta), 
         -torch.cos(theta) / (L * (M + m * (torch.sin(theta)**2))), 
         torch.zeros_like(theta), 
         1 / (M + m * torch.sin(theta)**2)], dim=1).unsqueeze(2)


Q = np.diag([60, 1.5, 180, 45])

domain = [[-0.2, 0.2]] * 4
sys_name = f"cartpole_{domain[0]}"


K = sp.Matrix([[-32.8482,   -9.1512,   -1.0000,   -2.3460]])

initial_u = -K * sp.Matrix([x1, x2, x3, x4])


system = lyznet.ControlAffineSystem(f, g, domain, sys_name, Q=Q)
print("Eigenvalues of linearization: ", np.linalg.eigvals(system.A))


lyznet.neural_pi(system, 
                 initial_u=initial_u, 
                 # f_torch=f_torch,
                 g_torch=g_torch, 
                 num_of_iters=10, lr=0.001, layer=2, width=10, 
                 num_colloc_pts=300000, max_epoch=10)
