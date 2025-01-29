import sympy as sp
import numpy as np
import lyznet
import torch


lyznet.utils.set_random_seed()

x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9')
symbolic_vars = (x1, x2, x3, x4, x5, x6, x7, x8, x9)

gc = 9.81

# New matrix K
K = sp.Matrix([
    [-0.0000, 0.0000, 1.0000, -0.0000, 0.0000, 1.7321, -0.0000, 0.0000, 0.0000],
    [0.0000, 1.0000, -0.0000, 0.0000, 1.4515, -0.0000, 5.4294, -0.0000, -0.0000],
    [-1.0000, 0.0000, -0.0000, -1.4515, -0.0000, -0.0000, -0.0000, 5.4294, 0.0000],
    [0.0000, 0.0000, -0.0000, 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, 1.0000]
])

# Assuming x1 to x9 are your state variables
initial_u = -K * sp.Matrix([x1, x2, x3, x4, x5, x6, x7, x8, x9])

f = sp.Matrix([
    x4,
    x5,
    x6,
    -gc * sp.sin(x8),
    gc * sp.cos(x8) * sp.sin(x7), 
    gc * sp.cos(x8) * sp.cos(x7) - gc, 
    0,
    0,
    0
])


def f_torch(x):    
    # State variables
    px = x[:, 0]   # x position
    py = x[:, 1]   # y position
    pz = x[:, 2]   # z position
    pdot_x = x[:, 3]   # velocity in x
    pdot_y = x[:, 4]   # velocity in y
    pdot_z = x[:, 5]   # velocity in z
    # f = x[:, 6]   # net normalized thrust
    phi = x[:, 6]  # roll angle
    theta = x[:, 7] # pitch angle
    psi = x[:, 8]   # yaw angle
    
    # Equations of motion
    dx1 = pdot_x
    dx2 = pdot_y
    dx3 = pdot_z
    # dx4 = torch.zeros_like(x[:, 0])
    # dx5 = torch.zeros_like(x[:, 0])
    dx4 = - gc * torch.sin(theta)
    dx5 = gc * torch.cos(theta) * torch.sin(phi)
    dx6 = gc * torch.cos(theta) * torch.cos(phi) - gc
    dx7 = torch.zeros_like(x[:, 0])
    dx8 = torch.zeros_like(x[:, 0])
    dx9 = torch.zeros_like(x[:, 0])
    
    return torch.stack([dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9], dim=1)


g = sp.Matrix([
    [0, 0, 0, 0],  
    [0, 0, 0, 0], 
    [0, 0, 0, 0],
    [-sp.sin(x8), 0, 0, 0],
    [sp.cos(x8) * sp.sin(x7), 0, 0, 0],
    [sp.cos(x8) * sp.cos(x7), 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])


def g_torch(x):
    N = x.shape[0]
    x7 = x[:, 6]
    x8 = x[:, 7]

    zero_col = torch.zeros(N, 1)
    sin_x8 = - torch.sin(x8).unsqueeze(1)
    cos_x8_sin_x7 = (torch.cos(x8) * torch.sin(x7)).unsqueeze(1)
    cos_x8_cos_x7 = (torch.cos(x8) * torch.cos(x7)).unsqueeze(1)
    one_col = torch.tensor(1).expand(N, 1)

    col1 = torch.cat(
        (zero_col, zero_col, zero_col, sin_x8, cos_x8_sin_x7, cos_x8_cos_x7, 
         zero_col, zero_col, zero_col), dim=1)
    col2 = torch.cat(
        (zero_col, zero_col, zero_col, zero_col, zero_col, zero_col,
         one_col, zero_col, zero_col), dim=1)
    col3 = torch.cat(
        (zero_col, zero_col, zero_col, zero_col, zero_col, zero_col,
         zero_col, one_col, zero_col), dim=1)
    col4 = torch.cat(
        (zero_col, zero_col, zero_col, zero_col, zero_col, zero_col,
         zero_col, zero_col, one_col), dim=1)
    result = torch.stack((col1, col2, col3, col4), dim=-1)
    # print("g: ", result)
    return result


Q = np.diag(np.concatenate(([10] * 6, [1] * 3)))
R = np.eye(4)

domain = [[-0.2, 0.2]] * 9
sys_name = f"3d_quad_{domain[0]}"

system = lyznet.ControlAffineSystem(f, g, domain, sys_name, Q=Q, R=R)
print("Eigenvalues of linearization: ", np.linalg.eigvals(system.A))

lyznet.neural_pi(system, 
                 initial_u=initial_u, 
                 f_torch=f_torch,
                 g_torch=g_torch, 
                 num_of_iters=10, lr=0.001, layer=2, width=10, 
                 num_colloc_pts=300000, max_epoch=10,
                 # data=True
                 )
