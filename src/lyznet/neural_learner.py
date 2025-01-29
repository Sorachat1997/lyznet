import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import lyznet

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


class Net(nn.Module):
    def __init__(self, num_inputs, num_layers, width):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.torch.tanh(x)
        x = self.final_layer(x)
        return x


class ReLUNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width):
        super(ReLUNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.torch.tanh(x)
        x = self.final_layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width, num_outputs):
        super(UNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        # Use num_outputs in the final layer
        self.final_layer = nn.Linear(width, num_outputs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.tanh(x)   
        x = self.final_layer(x)
        return x


class CMNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width, num_outputs):
        super(CMNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        # Use num_outputs in the final layer
        self.final_layer = nn.Linear(width, num_outputs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.tanh(x)   
        x = self.final_layer(x)
        return x


class PolyNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width, deg=2, zero_bias=False):
        super(PolyNet, self).__init__()
        self.deg = deg
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1) 

        if zero_bias: 
            self._set_biases_to_zero()

    def _set_biases_to_zero(self):
        for layer in self.layers:
            nn.init.constant_(layer.bias, 0.0)
            layer.bias.requires_grad = False
        nn.init.constant_(self.final_layer.bias, 0.0)
        self.final_layer.bias.requires_grad = False

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = x.pow(self.deg) 
        x = self.final_layer(x)
        return x


class PosNet(torch.nn.Module):
    def __init__(self, num_inputs, num_layers, width):
        super(PosNet, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(width, width))

        # Note: No need for final linear layer now.
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        # Compute dot product with itself to ensure positive definite output
        x = (x**2).sum(dim=1, keepdim=True)
        return x


class HomoNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width, deg=1):
        super(HomoNet, self).__init__()
        self.deg = deg
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1)

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / norm
        for layer in self.layers:
            x_normalized = layer(x_normalized)
            x_normalized = torch.torch.tanh(x_normalized)
        output = self.final_layer(x_normalized)
        return output * (norm ** self.deg)


class HomoUNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width, num_outputs=1, deg=1):
        super(HomoUNet, self).__init__()
        self.deg = deg
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        # Allowing the final layer to have more than one output
        self.final_layer = nn.Linear(width, num_outputs)

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / norm
        for layer in self.layers:
            x_normalized = layer(x_normalized)
            x_normalized = torch.torch.tanh(x_normalized)
        output = self.final_layer(x_normalized)
        return output * (norm ** self.deg)


class HomoPolyNet(nn.Module):
    def __init__(self, num_inputs, num_layers, width, deg=1):
        super(HomoPolyNet, self).__init__()
        self.deg = deg
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1)

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / norm
        for layer in self.layers:
            x_normalized = layer(x_normalized)
            x_normalized = x_normalized.pow(2)
        output = self.final_layer(x_normalized)
        return output * (norm ** self.deg)


class SimpleNet(nn.Module):
    def __init__(self, d, out_dim=1):
        super(SimpleNet, self).__init__()
        
        # One-node layers for each dimension
        self.initial_layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(d)])
        
    def forward(self, x):
        outputs = []
        
        for i in range(x.shape[1]):
            xi = x[:, i].view(-1, 1)  
            out = self.initial_layers[i](xi) 
            out = torch.torch.tanh(out)
            outputs.append(out)
        
        concatenated = torch.cat(outputs, dim=1)
        final_output = (concatenated ** 2).sum(dim=1, keepdim=True)
        
        return final_output


def evaluate_dynamics(f, x):
    x_split = torch.split(x, 1, dim=1)
    result = []
    for fi in f:
        args = [x_s.squeeze() for x_s in x_split]
        result.append(fi(*args))
    return result


def Zubov_loss(x, net, system, mu=0.1, beta=1.0, c1=0.01, c2=1, 
               transform="tanh", v_max=None):
    # Learning Lyapunov function that characertizes maximal ROA
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero_tensor.requires_grad = True
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    f_values = evaluate_dynamics(system.f_torch, x)
    f_tensor = torch.stack(f_values, dim=1)

    V_dot = (V_grad * f_tensor).sum(dim=1)
    norm_sq = (x**2).sum(dim=1)
    
    # mask = (beta**2 - norm_sq) > 0
    # lower_bound = torch.where(mask, torch.clamp(torch.torch.tanh(c1 * norm_sq) - V, 
    #                           min=0)**2, torch.zeros_like(norm_sq))
    # upper_bound = torch.where(mask, torch.clamp(torch.torch.tanh(c2 * norm_sq) - V, 
    #                           max=0)**2, torch.zeros_like(norm_sq))

    if v_max is not None: 
        mu = 20/v_max

    if transform == "exp":
        pde_loss = (V_dot + mu * norm_sq * (1-V))**2
    else:
        pde_loss = (V_dot + mu * norm_sq * (1-V) * (1+V))**2 

    loss = ( 
            pde_loss  
            # + lower_bound 
            # + upper_bound 
            # + hessian_loss
            + V_zero**2 
           ).mean()
    
    return loss


def DTS_Zubov_loss(x, net, system, mu=0.2, dt=3e-3):
    # Learning Lyapunov function that characertizes maximal ROA
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero_tensor.requires_grad = True
    V_zero = net(zero_tensor)
    V = net(x).squeeze()

    f_values = evaluate_dynamics(system.f_torch, x)
    f_tensor = torch.stack(f_values, dim=1)

    V_next = net(f_tensor).squeeze()
    V_diff = V_next - V
    norm_sq = (x**2).sum(dim=1)
    
    difference_loss = (V_diff + (torch.exp(mu*dt*norm_sq)-1) * (1-V))**2

    loss = ( 
            difference_loss  
            + V_zero**2 
           ).mean()
    
    return loss


def Zubov_control_loss(x, net, system, mu=0.1, f_torch=None, g_torch=None):
    # Learning maximal control Lyapunov function using Zubov's equation
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(x.device)
    zero_tensor.requires_grad = True
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    # Compute dynamics
    if f_torch is None: 
        f_values = evaluate_dynamics(system.f_torch, x)
        f_tensor = torch.stack(f_values, dim=1)
    else: 
        f_tensor = f_torch(x)

    if g_torch is None:
        g_torch = system.g_torch

    g_values = g_torch(x)

    N = x.shape[0]
    R_inv_torch = torch.tensor(
        np.linalg.inv(system.R), dtype=torch.float32
    ).repeat(N, 1, 1)

    g_torch_value = g_torch(x).transpose(1, 2)

    # Compute u using the original formula
    u_values = -0.5 * torch.bmm(
        torch.bmm(R_inv_torch, g_torch_value), V_grad.unsqueeze(2)
    )

    # Compute f(x) * mu * (1 - V^2)
    f_term = f_tensor * mu * (1 - V**2).unsqueeze(1)

    # Compute g(x) * u
    gu_term = torch.bmm(g_values, u_values).squeeze()

    # Combine both terms
    f_u_values = f_term + gu_term

    # Cost terms
    Q_torch = torch.tensor(system.Q, dtype=torch.float32).repeat(N, 1, 1)
    R_torch = torch.tensor(system.R, dtype=torch.float32).repeat(N, 1, 1)

    # u^T * R * u
    u_cost = torch.bmm(
        torch.bmm(u_values.transpose(1, 2), R_torch), u_values
    ).squeeze()

    # mu^2 * (1 - V^2)^2 * x^T * Q * x
    x_unsqueezed = x.unsqueeze(2)
    x_cost = mu**2 * (1 - V**2)**2 * torch.bmm(
        torch.bmm(x_unsqueezed.transpose(1, 2), Q_torch), x_unsqueezed
    ).squeeze()

    # Total cost
    omega = u_cost + x_cost

    # V_dot = DV * (f_u_values)
    V_dot = (V_grad * f_u_values).sum(dim=1)

    # Loss term
    pde_loss = V_dot + omega

    # Final loss
    loss = (
        pde_loss**2 + V_zero**2
    ).mean()
    
    return loss


def Lyapunov_loss(x, net, system, mu=0.1):
    # Learning a Lyapunov function on a region of interest (inside maximal ROA)
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    f_values = evaluate_dynamics(system.f_torch, x)
    f_tensor = torch.stack(f_values, dim=1)

    V_dot = (V_grad * f_tensor).sum(dim=1)
    norm_sq = (x**2).sum(dim=1)

    loss = (
            torch.relu(V_dot + mu * norm_sq)**2 
            + V_zero**2 
           ).mean()

    return loss


# def Lyapunov_control_loss(x, net, system, mu=0.1, f_torch=None, g_torch=None):
#     # Learning control Lyapunov function using Lyapunov PDI
#     x.requires_grad = True
#     zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(x.device)
#     zero_tensor.requires_grad = True
#     V_zero = net(zero_tensor)
#     V = net(x).squeeze()
#     V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

#     # Compute dynamics
#     if f_torch is None: 
#         f_values = evaluate_dynamics(system.f_torch, x)
#         f_tensor = torch.stack(f_values, dim=1)
#     else: 
#         f_tensor = f_torch(x)

#     if g_torch is None:
#         g_torch = system.g_torch

#     g_values = g_torch(x)

#     N = x.shape[0]
#     R_inv_torch = torch.tensor(
#         np.linalg.inv(system.R), dtype=torch.float32
#     ).repeat(N, 1, 1)

#     g_torch_value = g_torch(x).transpose(1, 2)

#     u_values = -0.5 * torch.bmm(
#         torch.bmm(R_inv_torch, g_torch_value), V_grad.unsqueeze(2)
#     )

#     f_term = f_tensor
#     gu_term = torch.bmm(g_values, u_values).squeeze()
#     f_u_values = f_term + gu_term

#     Q_torch = torch.tensor(system.Q, dtype=torch.float32).repeat(N, 1, 1)
#     R_torch = torch.tensor(system.R, dtype=torch.float32).repeat(N, 1, 1)

#     u_cost = torch.bmm(
#         torch.bmm(u_values.transpose(1, 2), R_torch), u_values
#     ).squeeze()

#     x_unsqueezed = x.unsqueeze(2)
#     x_cost = torch.bmm(
#         torch.bmm(x_unsqueezed.transpose(1, 2), Q_torch), x_unsqueezed
#     ).squeeze()

#     omega = u_cost + x_cost
#     V_dot = (V_grad * f_u_values).sum(dim=1)

#     # print("V_dot: ", V_dot.shape)
#     # print("omega: ", omega.shape)

#     # pdi_loss = torch.relu(V_dot + 0.01*omega)

#     norm_sq = (x**2).sum(dim=1)
#     pdi_loss = torch.relu(V_dot + 0.01*norm_sq)

#     positivity_loss = torch.relu(-V + 0.01*norm_sq)

#     loss = (
#         pdi_loss**2 
#         + positivity_loss**2
#         + V_zero**2
#     ).mean()
    
#     return loss


# def Lyapunov_control_loss(x, net, system, f_torch=None, g_torch=None): 
#     # training a CLF using Sontag's universal formula for stabilizing
#     x.requires_grad = True
#     if f_torch is None: 
#         f_values = evaluate_dynamics(system.f_torch, x)
#         f_tensor = torch.stack(f_values, dim=1)
#     else: 
#         f_tensor = f_torch(x)
#     # print("f_tensor: ", f_tensor.shape)

#     if g_torch is None:
#         g_torch = system.g_torch
#     g_values = g_torch(x)
#     # print("g_values: ", g_values.shape)
#     g_transposed = g_values.transpose(1, 2)

#     zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
#     V_zero = net(zero_tensor)
#     V = net(x).squeeze()
#     V_grad = torch.autograd.grad(V.sum(), x, create_graph=True, 
#                                  retain_graph=True)[0]
#     V_grad_unsqueezed = V_grad.unsqueeze(-1)
#     # print("V_grad: ", V_grad.shape)    

#     a_x = torch.sum(V_grad * f_tensor, dim=1)
#     # print("a_x: ", a_x.shape)
    
#     b_x = torch.bmm(g_transposed, V_grad_unsqueezed).squeeze(-1)
#     # print("b_x: ", b_x.shape)

#     b_x_norm_squared = torch.sum(b_x ** 2, dim=1, keepdim=True)
#     b_x_norm_fourth = b_x_norm_squared ** 2

#     threshold = 1e-8
#     mask = b_x_norm_squared > threshold

#     u_sontag = torch.zeros_like(b_x)

#     # Apply the mask directly with operations
#     masked_a_x = a_x[mask.squeeze(-1)]
#     masked_b_x = b_x[mask.squeeze(-1)]
#     masked_b_x_norm_squared = b_x_norm_squared[mask].squeeze(-1)
#     masked_b_x_norm_fourth = b_x_norm_fourth[mask].squeeze(-1)

#     u_sontag_temp = - (masked_a_x + torch.sqrt(masked_a_x**2 
#                        + masked_b_x_norm_fourth)) / masked_b_x_norm_squared

#     u_sontag[mask.squeeze(-1)] = u_sontag_temp[:, None] * masked_b_x
#     # print("u_sontag:", u_sontag.shape)

#     f_u_values = f_tensor + torch.bmm(
#         g_values, u_sontag.unsqueeze(2)
#         ).squeeze()
#     # print("f_u: ", f_u_values.shape)

#     V_dot = (V_grad * f_u_values).sum(dim=1)
#     # print("V_dot", V_dot.shape)

#     norm_sq = (x**2).sum(dim=1)
    
#     loss = (
#             torch.relu(V_dot + 0.01*norm_sq)**2 
#             + torch.relu(-V + 0.01*norm_sq)**2 
#             + V_zero**2 
#            ).mean()

#     return loss


def Lyapunov_control_loss(x, net, system, f_torch=None, g_torch=None):
    # Training a CLF using Sontag's universal formula for stabilizing
    x.requires_grad = True
    if f_torch is None:
        f_values = evaluate_dynamics(system.f_torch, x)
        f_tensor = torch.stack(f_values, dim=1)
    else:
        f_tensor = f_torch(x)
    # print("f_tensor: ", f_tensor.shape)

    if g_torch is None:
        g_torch = system.g_torch
    g_values = g_torch(x)
    # print("g_values: ", g_values.shape)
    g_transposed = g_values.transpose(1, 2)

    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(
        V.sum(), x, create_graph=True, retain_graph=True
    )[0]
    V_grad_unsqueezed = V_grad.unsqueeze(-1)
    # print("V_grad: ", V_grad.shape)

    a_x = torch.sum(V_grad * f_tensor, dim=1)
    # print("a_x: ", a_x.shape)

    b_x = torch.bmm(g_transposed, V_grad_unsqueezed).squeeze(-1)
    # print("b_x: ", b_x.shape)

    b_x_norm_squared = torch.sum(b_x ** 2, dim=1)  # Shape: (N,)
    b_x_norm_fourth = b_x_norm_squared ** 2  # Shape: (N,)

    epsilon = 1e-8
    b_x_norm_squared_safe = b_x_norm_squared + epsilon  # Shape: (N,)

    u_sontag_temp = -(
        a_x + torch.sqrt(a_x ** 2 + b_x_norm_fourth)
    ) / b_x_norm_squared_safe  # Shape: (N,)

    u_sontag = u_sontag_temp.unsqueeze(-1) * b_x  # Shape: (N, m)
    # print("u_sontag:", u_sontag.shape)

    f_u_values = f_tensor + torch.bmm(
        g_values, u_sontag.unsqueeze(2)
    ).squeeze()
    # print("f_u: ", f_u_values.shape)

    V_dot = (V_grad * f_u_values).sum(dim=1)
    # print("V_dot", V_dot.shape)

    norm_sq = (x ** 2).sum(dim=1)

    loss = (
        torch.relu(V_dot + 0.01 * norm_sq) ** 2
        + torch.relu(-V + 0.01 * norm_sq) ** 2
        + V_zero ** 2
    ).mean()

    return loss


def Universal_Lyapunov_loss(x, net, system, mu=0.1):
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero_tensor.requires_grad = True
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    V_grad_zero = torch.autograd.grad(V.sum(), zero_tensor, create_graph=True)[0]

    f_values = evaluate_dynamics(system.f_torch, x)
    f_tensor = torch.stack(f_values, dim=1)

    V_dot = (V_grad * f_tensor).sum(dim=1)
    norm_sq = (x**2).sum(dim=1)

    loss = (
            torch.relu(V_dot + mu * norm_sq)**2 
            + V_zero**2 
           ).mean()

    return loss


def default_omega(x):
    # Default omega function: norm square of x
    return (x**2).sum(dim=1)


def Lyapunov_PDE_loss(x, net, system, omega=default_omega):
    # Learning a Lyapunov function on a region of interest (inside maximal ROA)
    # by solving the Lyapunov PDE DV*f = - omega
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    f_values = evaluate_dynamics(system.f_torch, x)
    f_tensor = torch.stack(f_values, dim=1)

    V_dot = (V_grad * f_tensor).sum(dim=1)
    omega_tensor = omega(x)

    loss = (
            (V_dot + omega_tensor)**2 
            + V_zero**2 
           ).mean()

    return loss


def compute_u_gradients_at_zero(net, g_torch, R_inv, zero_tensor):
    # Ensure zero_tensor requires gradient
    zero_tensor.requires_grad = True

    # Compute necessary values at zero_tensor
    V_zero = net(zero_tensor).squeeze()
    V_grad_zero = torch.autograd.grad(V_zero, zero_tensor, create_graph=True)[0]
    g_values_zero = g_torch(zero_tensor)

    R_inv_torch_zero = torch.tensor(R_inv, dtype=torch.float32, requires_grad=True).repeat(zero_tensor.shape[0], 1, 1).to(zero_tensor.device)

    # Compute u_values at zero_tensor
    u_values_zero = -0.5 * torch.bmm(
        torch.bmm(R_inv_torch_zero, g_values_zero.transpose(1, 2)), V_grad_zero.unsqueeze(2)
    ).squeeze()

    # Initialize a matrix to hold gradients
    grad_matrix = torch.zeros_like(u_values_zero, device=zero_tensor.device)

    # Compute gradient for each element in u_values_zero
    for i in range(u_values_zero.shape[0]):
        grad_matrix[i] = torch.autograd.grad(u_values_zero[i], zero_tensor, retain_graph=True)[0].squeeze()

    return grad_matrix


def Lyapunov_GHJB_loss(x, net, system, omega=None, u_func=None, 
                       f_torch=None, g_torch=None, R_inv=None, K=None):
    # Learning a Lyapunov (value) function that solves the "generalized" HJB
    # DV*(f+g*u) = - omega (= - [x^T*Q*x + u^T*R*u])
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero_tensor.requires_grad = True
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    DV_zero = torch.autograd.grad(V_zero, zero_tensor, create_graph=True)[0]
    # print("DV_zero: ", DV_zero.shape)

    if f_torch is None: 
        f_values = evaluate_dynamics(system.f_torch, x)
        f_tensor = torch.stack(f_values, dim=1)
    else: 
        f_tensor = f_torch(x)

    if g_torch is None:
        g_torch = system.g_torch
        # print("-"*50)
    g_values = g_torch(x)
    # print("g:", g_values.shape)

    u_values = u_func(x)
    # print("u:", u_values.shape)

    f_u_values = f_tensor + torch.bmm(
        g_values, u_values
        ).squeeze()

    V_dot = (V_grad * f_u_values).sum(dim=1)
    omega_tensor = omega(x)

    # match linear approximation of kappa to K_{i+1}
    g_values_zero = g_torch(zero_tensor)
    # print("g: ", g_values_zero.shape)
    g_V = torch.bmm(g_values_zero.transpose(1, 2), 
                    DV_zero.unsqueeze(0).transpose(1, 2)).squeeze()

    R_inv_tensor = torch.tensor(R_inv, dtype=torch.float32).unsqueeze(0)
    # print("R_inv: ", R_inv_tensor.shape)
    u_zero = -0.5 * torch.bmm(
        torch.bmm(R_inv_tensor, g_values_zero.transpose(1, 2)), 
        DV_zero.unsqueeze(0).transpose(1, 2)
    ).squeeze(0)

    # print("u_zero: ", u_zero)

    # Only works for one input
    # u_grad_zero = torch.autograd.grad(
    #     u_zero.sum(), zero_tensor, create_graph=True)[0].squeeze()

    # Compute the Jacobian
    jacobian_list = []
    for i in range(u_zero.shape[0]):  # Loop over the output dimensions
        # Compute gradient for each output component with respect to the inputs
        grad = torch.autograd.grad(
            u_zero[i], zero_tensor, create_graph=True, retain_graph=True)[0]
        jacobian_list.append(grad)

    # Stack the gradients to form the Jacobian matrix
    u_grad_zero = torch.stack(jacobian_list, dim=1)

    # print("u_grad: ", u_grad_zero.shape)
    # print("K: ", K.shape)

    # print("omega: ", omega_tensor.shape)
    # print("V_dot: ", V_dot.shape)

    loss = (
            (V_dot + omega_tensor)**2 
            + V_zero**2 
            + torch.norm(g_V)**2
            # + torch.norm(u_zero)**2  
            + torch.norm(u_grad_zero - K)**2 
           ).mean()

    return loss


def Homo_Lyapunov_loss(x, net, system, mu=1e-2):
    # Normalize x to lie on the unit sphere
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    # print(norm)
    x_unit = x / norm
    # print(x_unit)

    x_unit.requires_grad = True
    # zero_tensor = torch.zeros_like(x_unit[0]).unsqueeze(0).to(device)
    # V_zero = net(zero_tensor)
    V = net(x_unit).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x_unit, create_graph=True)[0]

    f_values = evaluate_dynamics(system.f_torch, x_unit)
    f_tensor = torch.stack(f_values, dim=1)

    # print(f_tensor)
    
    V_dot = (V_grad * f_tensor).sum(dim=1)

    # Loss components: penalty for negative V and positive V_dot
    loss = (
        torch.relu(-V + mu)**2   # Penalize negative V
        + torch.relu(V_dot + mu)**2   # Penalize positive V_dot
        # V_zero**2  
    ).mean()

    return loss


def Homo_Lyapunov_Control_loss(x, net, u_net, system, f_torch, g_torch, 
                               mu=1e-2):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x_unit = x / norm
    x_unit.requires_grad = True

    V = net(x_unit).squeeze()
    u = u_net(x_unit).squeeze()

    if f_torch is None: 
        f_values = evaluate_dynamics(system.f_torch, x)
        f_tensor = torch.stack(f_values, dim=1)
    else: 
        f_tensor = f_torch(x)

    if g_torch is not None:
        g_values = g_torch(x_unit)
    else:
        g_values = system.g_torch(x_unit)

    # print("f: ", f_tensor.shape)
    # print("g: ", g_values.shape)
    # print("u (before reshape): ", u.shape)

    # Reshape `u` to ensure it has the correct shape for bmm
    if u.dim() == 1:
        u = u.view(-1, 1, 1)  # For a single output, make it (N, 1, 1)
    elif u.dim() == 2:
        u = u.unsqueeze(-1)  # For multiple outputs (N, k), make it (N, k, 1)
    
    # print("u (after reshape): ", u.shape)

    f_u_values = f_tensor + torch.bmm(
        g_values, u
        ).squeeze()

    V_grad = torch.autograd.grad(V.sum(), x_unit, create_graph=True)[0]

    V_dot = (V_grad * f_u_values).sum(dim=1)

    loss = (
        torch.relu(-V + mu)**2   # Penalize negative V
        + torch.relu(V_dot + mu)**2   # Penalize positive V_dot
        # V_zero**2  
    ).mean()

    return loss


def compute_jacobian(f, x):
    # x is a tensor of shape (N, d)
    # Jacobian will be of shape (N, d, d)
    x = x.requires_grad_(True)
    y = f(x)

    jacobian = []
    for i in range(y.shape[1]):
        grads = torch.autograd.grad(outputs=y[:, i], inputs=x, 
                                    grad_outputs=torch.ones_like(y[:, i]),
                                    create_graph=True, retain_graph=True, 
                                    only_inputs=True)[0]
        jacobian.append(grads)

    jacobian = torch.stack(jacobian, dim=2).transpose(1, 2)  # (N, d, d)
    return jacobian


def Contraction_EP_loss(x, net, system, f_torch, V_net, c2_V, mu=1.0):
    # Learning a neural contraction metric for an EP on a region of attraction
    # by solving the matrix PDI Df(x).T*M(x) + M(x)*Df(x) + DM(x)*f(x) <= - Q(x)
    # Only data points within a verified ROA {x: V_net(x)<=c2_V} are used 
    x.requires_grad = True

    # Mask the data if an invariant level set {x: V_net(x)<=c2_V}  is given
    if V_net is not None and c2_V is not None:
        mask = V_net(x).squeeze() <= c2_V    
        x = x[mask]

    batch_size = x.size(0)
    d = x.size(1)

    if x.size(0) == 0:  # If no data points satisfy the condition, return zero loss
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    M_net = net(x)

    M = torch.zeros(batch_size, d, d, device=x.device)
    indices = torch.triu_indices(d, d)
    M[:, indices[0], indices[1]] = M_net
    M = M + M.transpose(1, 2)
    M = M - torch.diag_embed(M.diagonal(dim1=1, dim2=2) / 2)

    f_tensor = f_torch(x)
    Df_x = compute_jacobian(f_torch, x)

    M_dot = torch.zeros_like(M)
    for i in range(d):
        for j in range(d):
            V = M[:, i, j]
            Mij_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
            Mij_dot = (Mij_grad * f_tensor).sum(dim=1)
            M_dot[:, i, j] = Mij_dot

    # Compute the loss term
    Df_x_T_M = torch.einsum('bji,bjk->bik', Df_x, M)
    M_Df_x = torch.einsum('bij,bjk->bik', M, Df_x)

    # Calculate the residual
    Q_x = mu * torch.eye(d, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    residual = Df_x_T_M + M_Df_x + M_dot + Q_x

    # Ensure M(x) - Q(x) is positive definite
    M_minus_Q = M - Q_x
    M_minus_Q = M 
    eigenvalues_M_minus_Q = torch.linalg.eigvalsh(M_minus_Q)
    positive_definite_penalty = torch.relu(-eigenvalues_M_minus_Q).pow(2).mean()

    # Ensure residual is negative definite
    eigenvalues_residual = torch.linalg.eigvalsh(residual)
    negative_definite_penalty = torch.relu(eigenvalues_residual).pow(2).mean()

    # Combine loss components
    loss = positive_definite_penalty + negative_definite_penalty
    return loss
    

def Contraction_EP_PDE_loss(x, net, system, f_torch, V_net, c2_V, mu=1.0):
    # Learning a neural contraction metric for an EP on a region of attraction
    # by solving the matrix PDE Df(x).T*M(x) + M(x)*Df(x) + DM(x)*f(x) = - Q(x)
    # Only data points within a verified ROA {x: V_net(x)<=c2_V} are used 
    x.requires_grad = True

    # Mask the data if an invariant level set {x: V_net(x)<=c2_V}  is given
    if V_net is not None and c2_V is not None:
        mask = V_net(x).squeeze() <= c2_V    
        x = x[mask]

    batch_size = x.size(0)
    d = x.size(1)

    if x.size(0) == 0:  # If no data points satisfy the condition, return zero loss
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    M_net = net(x)

    M = torch.zeros(batch_size, d, d, device=x.device)
    indices = torch.triu_indices(d, d)
    M[:, indices[0], indices[1]] = M_net
    M = M + M.transpose(1, 2)
    M = M - torch.diag_embed(M.diagonal(dim1=1, dim2=2) / 2)

    f_tensor = f_torch(x)
    Df_x = compute_jacobian(f_torch, x)

    M_dot = torch.zeros_like(M)
    for i in range(d):
        for j in range(d):
            V = M[:, i, j]
            Mij_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
            Mij_dot = (Mij_grad * f_tensor).sum(dim=1)
            M_dot[:, i, j] = Mij_dot

    # Compute the loss term
    Df_x_T_M = torch.einsum('bji,bjk->bik', Df_x, M)
    M_Df_x = torch.einsum('bij,bjk->bik', M, Df_x)

    # Calculate the residual
    Q_x = mu * torch.eye(d, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    residual = Df_x_T_M + M_Df_x + M_dot + Q_x

    # Compute the mean squared loss
    loss = (residual ** 2).mean()

    return loss


def sample_boundary_points(batch_size, domain, device):
    dim = len(domain)
    boundary_x = []
    for d in domain:
        # Randomly choose an edge for each dimension
        edge_vals = torch.tensor(d, device=device)
        boundary_points = (torch.rand(batch_size, 1, device=device) > 0.5).float() * edge_vals[1] + \
                          (torch.rand(batch_size, 1, device=device) <= 0.5).float() * edge_vals[0]
        boundary_x.append(boundary_points)
    
    # Randomly fix one dimension to be on the boundary for each point
    for i in range(batch_size):
        fixed_dim = torch.randint(0, dim, (1,))
        boundary_x[fixed_dim][i] = torch.tensor(
            domain[fixed_dim], 
            device=device).view(2, 1)[torch.randint(0, 2, (1,))]

    boundary_x = torch.cat(boundary_x, dim=1)
    return boundary_x


def Sontag_CLF_loss(x, net, system, f_torch=None, g_torch=None): 
    # training a CLF using Sontag's universal formula for stabilizing
    x.requires_grad = True
    if f_torch is None: 
        f_values = evaluate_dynamics(system.f_torch, x)
        f_tensor = torch.stack(f_values, dim=1)
    else: 
        f_tensor = f_torch(x)
    # print("f_tensor: ", f_tensor.shape)

    if g_torch is None:
        g_torch = system.g_torch
    g_values = g_torch(x)
    # print("g_values: ", g_values.shape)
    g_transposed = g_values.transpose(1, 2)

    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True, 
                                 retain_graph=True)[0]
    V_grad_unsqueezed = V_grad.unsqueeze(-1)
    # print("V_grad: ", V_grad.shape)    

    a_x = torch.sum(V_grad * f_tensor, dim=1)
    # print("a_x: ", a_x.shape)
    
    b_x = torch.bmm(g_transposed, V_grad_unsqueezed).squeeze(-1)
    # print("b_x: ", b_x.shape)

    # epsilon = 1e-8
    # b_x_denom = b_x.pow(2).clamp(min=epsilon)  

    b_x_norm_squared = torch.sum(b_x ** 2, dim=1, keepdim=True)
    b_x_norm_fourth = b_x_norm_squared ** 2

    threshold = 1e-8
    mask = b_x_norm_squared > threshold

    u_sontag = torch.zeros_like(b_x)
    # Apply the mask directly with operations
    masked_a_x = a_x[mask.squeeze(-1)]
    masked_b_x = b_x[mask.squeeze(-1)]
    masked_b_x_norm_squared = b_x_norm_squared[mask].squeeze(-1)
    masked_b_x_norm_fourth = b_x_norm_fourth[mask].squeeze(-1)

    u_sontag_temp = - (masked_a_x + torch.sqrt(masked_a_x**2 
                       + masked_b_x_norm_fourth)) / masked_b_x_norm_squared

    u_sontag[mask.squeeze(-1)] = u_sontag_temp[:, None] * masked_b_x
    # print("u_sontag:", u_sontag.shape)

    f_u_values = f_tensor + torch.bmm(
        g_values, u_sontag.unsqueeze(2)
        ).squeeze()
    # print("f_u: ", f_u_values.shape)

    V_dot = (V_grad * f_u_values).sum(dim=1)
    # print("V_dot", V_dot.shape)

    # Sampling boundary points
    boundary_x = sample_boundary_points(x.size(0), system.domain, device)
    
    # Compute V for boundary points
    V_boundary = net(boundary_x).squeeze()

    norm_sq = (x**2).sum(dim=1)
    
    loss = (
            torch.relu(V_dot + 0.1*norm_sq)**2 
            + torch.relu(-V + 0.1*norm_sq)**2 
            + V_zero**2 
            + torch.relu(V - 1)**2 
            + torch.relu(1 - V_boundary)**2
           ).mean()

    return loss


def Sontag_controller_loss(x, net, system, f_torch=None, g_torch=None, 
                           V_clf=None): 
    # training a neural controller using a control Lyapunov function V_clf
    # and Sontag's universal formula 
    # V_clf is defaulted to the quadratic Lyapunov function x^T*P*x, where P
    # solves the ARE for LQR, i.e. P = system.P

    if V_clf is None and system.P is not None:
        P = torch.tensor(system.P, dtype=torch.float32)
        
        def V_clf(x):
            return torch.sum(x @ P * x, dim=1)

    if V_clf is None:
        V_clf = lambda x: torch.sum(x**2, dim=1)
        # V_clf = lambda x: x[:, 0]**2 + x[:, 0]*x[:, 1] + x[:, 1]**2

    x.requires_grad = True
    if f_torch is None: 
        f_values = evaluate_dynamics(system.f_torch, x)
        f_tensor = torch.stack(f_values, dim=1)
    else: 
        f_tensor = f_torch(x)
    # print("f_tensor: ", f_tensor.shape)

    if g_torch is None:
        g_torch = system.g_torch
    g_values = g_torch(x)
    # print("g_values: ", g_values.shape)
    g_transposed = g_values.transpose(1, 2)

    V = V_clf(x)
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True, 
                                 retain_graph=True)[0]
    V_grad_unsqueezed = V_grad.unsqueeze(-1)

    # print("V_grad: ", V_grad.shape)    
    # a(x) = dV * f_torch
    a_x = torch.sum(V_grad * f_tensor, dim=1)
    # print("a_x: ", a_x.shape)
    
    # b(x) = dV * g_torch
    b_x = torch.bmm(g_transposed, V_grad_unsqueezed).squeeze(-1)
    # print("b_x: ", b_x.shape)

    # epsilon = 1e-8
    # b_x_denom = b_x.pow(2).clamp(min=epsilon)  

    b_x_norm_squared = torch.sum(b_x ** 2, dim=1, keepdim=True)
    b_x_norm_fourth = b_x_norm_squared ** 2

    threshold = 1e-8
    mask = b_x_norm_squared > threshold

    u_sontag = torch.zeros_like(b_x)
    # Apply the mask directly with operations
    masked_a_x = a_x[mask.squeeze(-1)]
    masked_b_x = b_x[mask.squeeze(-1)]
    masked_b_x_norm_squared = b_x_norm_squared[mask].squeeze(-1)
    masked_b_x_norm_fourth = b_x_norm_fourth[mask].squeeze(-1)

    u_sontag_temp = - (masked_a_x + torch.sqrt(masked_a_x**2 
                       + masked_b_x_norm_fourth)) / masked_b_x_norm_squared

    u_sontag[mask.squeeze(-1)] = u_sontag_temp[:, None] * masked_b_x
    u = net(x).squeeze()

    # print(u_sontag.shape)
    # print("u: ", u.shape)

    loss = torch.mean(u_sontag - u)**2

    return loss


def HJB_loss(x, net, system, omega=None, u_func=None, 
             f_torch=None, g_torch=None, R_inv=None):
    # Learning optimal value function and control by solving HJB
    # DV*(f+g*u) = - [x^T*Q*x + u^T*R*u]), where u = -0.5*inv(R)*g^T*DV^T
    x.requires_grad = True
    # print("x: ", x)
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero_tensor.requires_grad = True
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    # print("V_grad: ", V_grad.shape)

    if f_torch is None: 
        f_values = evaluate_dynamics(system.f_torch, x)
        f_tensor = torch.stack(f_values, dim=1)
    else: 
        f_tensor = f_torch(x)

    # print("f: ", f_tensor.shape)

    if g_torch is None:
        g_torch = system.g_torch
    g_values = g_torch(x)
    # print("g:", g_values.shape)

    R_inv_numpy = np.linalg.inv(system.R)
    R_inv_tensor = torch.tensor(R_inv_numpy, dtype=torch.float32).unsqueeze(0)

    # print("R_inv: ", R_inv_tensor.shape)

    u_values = -0.5 * torch.matmul(
        torch.matmul(
            R_inv_tensor, g_values.transpose(1, 2)), V_grad.unsqueeze(2)
        )
    # print("u:", u_values)

    f_u_values = f_tensor + torch.bmm(
        g_values, u_values
        ).squeeze()

    # print("f_u: ", f_u_values.shape)

    V_dot = (V_grad * f_u_values).sum(dim=1)
    # print(V_dot)

    Q_tensor = torch.tensor(system.Q, dtype=torch.float32)
    # print("Q: ", Q_tensor)
    xQ = torch.matmul(x, Q_tensor)
    x_cost = (x * xQ).sum(dim=1)    
    # print("x_cost: ", x_cost)

    R_tensor = torch.tensor(system.R, dtype=torch.float32)
    R_u = torch.matmul(R_tensor, u_values.unsqueeze(2))
    u_cost = torch.sum(u_values * R_u.squeeze(2), dim=1)
    # print("u_cost: ", u_cost)

    omega_tensor = x_cost + u_cost

    # # matching D^2V(0)=P
    DV_zero = torch.autograd.grad(V_zero, zero_tensor, create_graph=True)[0]
    # print("DV_zero: ", DV_zero.shape)
    hessian_list = []
    for i in range(DV_zero.size(1)):  # Iterate over the components
        grad = torch.autograd.grad(
            DV_zero[0][i], zero_tensor, create_graph=True, retain_graph=True)[0]
        hessian_list.append(grad.squeeze())

    Hessian_V_zero = torch.stack(hessian_list)
    # print(Hessian_V_zero)
    P_tensor = torch.tensor(system.P, dtype=torch.float32).to(device)

    # match controller gain at x=0
    g_values_zero = g_torch(zero_tensor)
    u_zero = -0.5 * torch.bmm(
        torch.bmm(R_inv_tensor, g_values_zero.transpose(1, 2)), 
        DV_zero.unsqueeze(0).transpose(1, 2)
    ).squeeze(0)

    # print("u_zero: ", u_zero)

    # Compute the Jacobian
    u_jacobian_list = []
    for i in range(u_zero.shape[0]):  # Loop over the output dimensions
        # Compute gradient for each output component with respect to the inputs
        grad = torch.autograd.grad(
            u_zero[i], zero_tensor, create_graph=True, retain_graph=True)[0]
        u_jacobian_list.append(grad)

    # Stack the gradients to form the Jacobian matrix
    u_grad_zero = torch.stack(u_jacobian_list, dim=1)
    K_tensor = torch.tensor(system.K, dtype=torch.float32).to(device)

    loss = (
            (V_dot + omega_tensor)**2 
            + V_zero**2 
            + torch.norm(Hessian_V_zero - P_tensor)**2 
            + torch.norm(u_grad_zero - K_tensor)**2 
           ).mean()

    return loss


def loss_function_selector(loss_mode, net, x, system, data_tensor, 
                           v_max, transform, omega, u_func, f_torch, g_torch, 
                           R_inv, K, V_clf, V_net, c2_V, u_net):
    if loss_mode == 'Zubov':
        loss = Zubov_loss(x, net, system, v_max=v_max, transform=transform)
        if data_tensor is not None:
            data_loss = torch.mean(
                (net(data_tensor[0]) - data_tensor[1])**2
                )
            loss += data_loss

    elif loss_mode == 'DTS_Zubov':
        loss = DTS_Zubov_loss(x, net, system) 
        if data_tensor is not None:
            data_loss = torch.mean(
                (net(data_tensor[0]) - data_tensor[1])**2
                )
            loss += data_loss

    elif loss_mode == 'Zubov_Control':
        loss = Zubov_control_loss(x, net, system, g_torch=g_torch)
        if data_tensor is not None:
            data_loss = torch.mean(
                (net(data_tensor[0]) - data_tensor[1])**2
                )
            loss += data_loss
            
    elif loss_mode == 'Lyapunov':
        loss = Lyapunov_loss(x, net, system)

    elif loss_mode == 'Lyapunov_Control':
        loss = Lyapunov_control_loss(x, net, system, f_torch=f_torch,
                                     g_torch=g_torch)        

    elif loss_mode == 'Lyapunov_PDE':
        loss = Lyapunov_PDE_loss(x, net, system, omega)

    elif loss_mode == 'Lyapunov_GHJB':
        loss = Lyapunov_GHJB_loss(x, net, system, omega, u_func, f_torch, 
                                  g_torch, R_inv, K)

    elif loss_mode == 'Homo_Lyapunov':
        loss = Homo_Lyapunov_loss(x, net, system)

    elif loss_mode == 'Universal_Lyapunov': 
        loss = Universal_Lyapunov_loss(x, net, system)

    elif loss_mode == 'HJB':
        loss = HJB_loss(x, net, system, f_torch, g_torch)

    elif loss_mode == 'Sontag_CLF':
        loss = Sontag_CLF_loss(x, net, system, f_torch, g_torch)
        
    elif loss_mode == 'Sontag_Controller':
        loss = Sontag_controller_loss(x, net, system, f_torch, g_torch, V_clf)

    elif loss_mode == 'Contraction_EP': 
        loss = Contraction_EP_loss(x, net, system, f_torch, V_net, c2_V)

    elif loss_mode == 'Contraction_EP_PDE': 
        loss = Contraction_EP_PDE_loss(x, net, system, f_torch, V_net, c2_V)
        if data_tensor is not None:
            data_loss = torch.mean(
                (net(data_tensor[0]) - data_tensor[1])**2
                )
            loss = loss + data_loss

    elif loss_mode == 'Homo_Lyapunov_Control': 
        loss = Homo_Lyapunov_Control_loss(x, net, u_net, system, f_torch, 
                                          g_torch)

    elif loss_mode == 'Data':
        if data_tensor is not None:
            loss = torch.mean(
                (net(data_tensor[0]) - data_tensor[1])**2
                )
        else:
            raise ValueError("No data provided for 'Data' loss mode.")
            
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    return loss


def generate_model_path(system, data, N, max_epoch, 
                        layer, width, lr, loss_mode, net_type):
    if not os.path.exists('results'):
        os.makedirs('results')
    base_path = (
        f"results/{system.name}"
        f"_loss={loss_mode}_N={N}_epoch={max_epoch}_layer={layer}"
        f"_width={width}_lr={lr}"
    )
    if data is not None:
        num_data_points = data.shape[0]
        base_path = f"{base_path}_data={num_data_points}"
    if net_type is not None:
        base_path = f"{base_path}_net={net_type}"
    
    return f"{base_path}.pt"


def training_loop(system, net, x_train, optimizer, data_tensor, max_epoch, 
                  batch_size, loss_mode, v_max, transform, omega, u_func,
                  f_torch, g_torch, R_inv, K, V_clf, V_net, c2_V, u_net,
                  optimizer_u):
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size

    max_epoch_loss = float('inf')
    average_epoch_loss = float('inf')
    epoch = 0

    start_time = time.time()

    while ((average_epoch_loss > 1e-5 or max_epoch_loss > 1e-5) 
           and epoch < max_epoch):
        total_loss = 0.0
        losses = []

        indices = torch.randperm(num_samples)
        x_train_shuffled = x_train[indices]

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}", 
                            unit="batch", leave=False)
        for i in progress_bar:
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            x_batch = x_train_shuffled[batch_start:batch_end]

            loss = loss_function_selector(loss_mode, net, x_batch, 
                                          system, data_tensor, 
                                          v_max, transform, omega, u_func,
                                          f_torch, g_torch, R_inv, K, V_clf,
                                          V_net, c2_V, u_net)

            optimizer.zero_grad()
            if optimizer_u is not None:
                optimizer_u.zero_grad()

            loss.backward()
            optimizer.step()
            if optimizer_u is not None:
                optimizer_u.step()

            total_loss += loss.item()
            losses.append(loss.item())

            progress_bar.set_postfix(loss=loss.item())

        average_epoch_loss = total_loss / num_batches
        max_epoch_loss = max(losses)
        print(f"Epoch {epoch + 1} completed. " 
              f"Average epoch loss: {average_epoch_loss:.5g}. " 
              f"Max epoch loss: {max_epoch_loss:.5g}")
        epoch += 1

    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds.")


def neural_learner(system, data=None, loss_mode='Zubov', layer=3, width=10, 
                   num_colloc_pts=100000, batch_size=32, max_epoch=5, 
                   lr=0.01, overwrite=False, transform=None, v_max=200, 
                   net_type=None, omega=default_omega, u_func=None,
                   f_torch=None, g_torch=None, R_inv=None, K=None,
                   initial_net=None, homo_deg=1, V_clf=None, 
                   V_net=None, c2_V=None, u_net=None, homo_deg_u=1, 
                   poly_deg=2): 
    
    domain = system.domain
    d = len(system.symbolic_vars)

    if initial_net is not None:  
        net = initial_net.to(device)

    elif net_type == "Simple":
        net = SimpleNet(d).to(device) 

    elif net_type == "Poly":            
        net = PolyNet(d, layer, width, deg=poly_deg, zero_bias=True).to(device)       

    elif net_type == "Homo":
        net = HomoNet(d, layer, width, deg=homo_deg).to(device)

    elif net_type == "HomoPoly":
        net = HomoPolyNet(d, layer, width, deg=homo_deg).to(device)   
        
    elif net_type == "Positive":
        net = PosNet(d, layer, width).to(device)

    elif loss_mode == "Sontag_Controller":
        k = system.symbolic_g.shape[1]
        net = UNet(d, layer, width, k).to(device)

    elif loss_mode == "Contraction_EP" or loss_mode == "Contraction_EP_PDE":
        n = len(system.symbolic_f)
        net = CMNet(d, layer, width, int(n*(n+1)/2)).to(device)    

    elif loss_mode == "Homo_Lyapunov_Control":
        net = HomoNet(d, layer, width, deg=homo_deg).to(device)
        num_outputs = system.symbolic_g.shape[1]
        u_net = HomoUNet(d, layer, width, num_outputs=num_outputs, 
                         deg=homo_deg_u).to(device)  

    elif net_type == "ReLU":
        net = ReLUNet(d, layer, width).to(device)        

    else: 
        net = Net(d, layer, width).to(device)

    model_path = generate_model_path(system, data, num_colloc_pts, max_epoch, 
                                     layer, width, lr, loss_mode, net_type)

    if u_net is not None:
        net_type = "Homo_u_net"
        model_path_u = generate_model_path(system, data, num_colloc_pts, max_epoch, 
                                           layer, width, lr, loss_mode, net_type)

    print('_' * 50)

    if loss_mode == "Contraction_EP" or loss_mode == "Contraction_EP_PDE":
        if V_net is not None and c2_V is not None: 
            print(f"Learning neural contraction metric on V<={c2_V}:")        
        else:
            print(f"Learning neural contraction metric on domain {system.domain}:")                    
    
    elif loss_mode == "Homo_Lyapunov_Control":
        print("Learning homogeneous neural Lyapunov function and controller:")
    else:
        print("Learning neural Lyapunov function:")

    lyznet.utils.check_cuda()

    if not overwrite:
        if os.path.isfile(model_path):
            print("Model exists. Loading model...")
            net = torch.load(model_path, map_location=device)
            print(f"Model loaded: {model_path}")
            if u_net is not None and os.path.isfile(model_path_u):
                print(f"Loading existing u_net model from {model_path_u}")
                u_net = torch.load(model_path_u, map_location=device)
                print(f"u_net model loaded: {model_path_u}")
                return net, u_net, model_path, model_path_u
            else:
                return net, model_path

    print(f"Model is on device: {next(net.parameters()).device}")

    # print model info
    print("Network depth: ", layer)
    print("Network width: ", width)
    print("Loss mode: ", loss_mode)
    print("Epochs to train: ", max_epoch)
    print("Learning rate: ", lr)
    print("Number of collocation points: ", num_colloc_pts)
    print("Batch size: ", batch_size)
    print("Data used:", len(data) if data is not None else "None")

    print("Training model...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    optimizer_u = None
    if u_net is not None:
        optimizer_u = torch.optim.Adam(u_net.parameters(), lr=lr)

    ranges = torch.tensor(domain).to(device)
    x_train = torch.rand((num_colloc_pts, d)).to(device)
    x_train = x_train * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    if data is not None:
        x_data = data[:, :d]
        y_data = data[:, d:]        
        x_data_tensor = torch.FloatTensor(x_data).to(device)
        y_data_tensor = torch.FloatTensor(y_data).to(device)
        data_tensor = (x_data_tensor, y_data_tensor)
    else:
        data_tensor = None

    training_loop(system, net, x_train, optimizer, data_tensor, max_epoch, 
                  batch_size, loss_mode, v_max, transform, omega, u_func,
                  f_torch, g_torch, R_inv, K, V_clf, V_net, c2_V, u_net, 
                  optimizer_u)

    print(f"Model trained: {model_path}")
    torch.save(net, model_path)

    if u_net is not None:
        torch.save(u_net, model_path_u)
        return net, u_net, model_path, model_path_u

    return net, model_path
