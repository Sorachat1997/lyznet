import numpy as np
import time
import torch
import torch.nn as nn
import lyznet
import scipy


class ElmNet(nn.Module):

    def activation(x):
        return torch.tanh(x)

    def __init__(self, input_dim, hidden_dim, weights_np, bias_np):
        super(ElmNet, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        # nn.init.normal_(self.hidden.weight, mean=0, std=1)
        # nn.init.normal_(self.hidden.bias, mean=0, std=1)

        # Convert numpy arrays to tensors
        weights = torch.tensor(weights_np, dtype=torch.float64)
        bias = torch.tensor(bias_np, dtype=torch.float64)

        # Replace the weights and biases of the layer
        self.hidden.weight = nn.Parameter(weights)
        self.hidden.bias = nn.Parameter(bias)

    def forward(self, x):
        return ElmNet.activation(self.hidden(x))


def torch_elm_learner(system, num_hidden_units=200, num_colloc_pts=3000):

    def evaluate_dynamics(f, x):
        x_split = torch.split(x, 1, dim=1)
        result = []
        for fi in f:
            args = [x_s.squeeze() for x_s in x_split]
            result.append(fi(*args))
        return result

    def compute_residuals(net, samples):
        x = samples.clone().requires_grad_(True)
        H = net(x)
        residuals = []    
        f_values = evaluate_dynamics(system.f_torch, x)
        f_x = torch.stack(f_values, dim=1)
        Q_x = (x**2).sum(dim=1)
        for i in range(H.shape[1]):
            H_i = H[:, i]
            grad_H_i, = torch.autograd.grad(
                H_i, x, grad_outputs=torch.ones_like(H_i), create_graph=True
            )
            lie_derivative_H_i = (grad_H_i * f_x).sum(dim=1, keepdim=True)
            pde_residual_per_unit = lie_derivative_H_i 
            residuals.append(pde_residual_per_unit.squeeze())
        residuals = torch.stack(residuals).t()
        b = - Q_x.unsqueeze(1)
        return residuals, b

    def compute_boundary(net, samples):
        x = samples.clone().requires_grad_(True)
        H = net(x)
        b = torch.zeros(x.shape[0], 1)
        return H, b

    d = len(system.symbolic_vars)
    m = num_hidden_units
    N = num_colloc_pts
    weights = np.random.randn(m, d)
    bias = np.random.randn(m, 1)
    samples = np.array([np.random.uniform(dim[0], dim[1], N) 
                        for dim in system.domain]).T

    net = ElmNet(d, m, weights, bias.squeeze())  
    net = net.double()

    samples_tensor = torch.tensor(samples, dtype=torch.float64)
    boundary_point = torch.zeros(1, 2, dtype=torch.float64)
    BOUNDARY_WEIGHT = 100
    print("Computing residuals to set up ELM optimization...")
    lyznet.tik()
    A1, b1 = compute_residuals(net, samples_tensor)
    lyznet.tok()
    A2, b2 = compute_boundary(net, boundary_point)
    A = torch.cat([A1, BOUNDARY_WEIGHT * A2], dim=0)
    b = torch.cat([b1, BOUNDARY_WEIGHT * b2], dim=0)

    print("Trainning ELM with linear least squares...")
    # beta = torch.linalg.lstsq(A, b).solution
    lyznet.tik()
    beta = torch.linalg.lstsq(A, b, driver='gelsd').solution
    lyznet.tok()

    model_path = (f"results/{system.name}_torch_elm_m={num_hidden_units}"
                  f"_N={num_colloc_pts}")

    beta_numpy = beta.squeeze().detach().cpu().numpy()
    return weights, bias, beta_numpy, model_path


def numpy_elm_learner(system, num_hidden_units=100, num_colloc_pts=3000, 
                      loss_mode="Lyapunov", data=None, lambda_reg=0.1, mu=0.1, 
                      omega=None, u_func=None, f_numpy=None, g_numpy=None,
                      test=None, next_K=None, pre_K=None, 
                      weights=None, bias=None, samples=None, Dg_zero=None,
                      return_test_loss=False, one_boundary=True, c2_P=None): 
    def activation(x):
        return np.tanh(x)

    def activation_prime(x):
        return 1 - np.square(np.tanh(x))

    def Q(x):
        return sum(xi**2 for xi in x)

    def default_omega(samples):
        return np.sum(samples**2, axis=1)

    def compute_residuals_vectorized(samples, weights, bias, omega=omega, 
                                     g_numpy=g_numpy, f_numpy=f_numpy):
        if omega is None:
            omega = default_omega
        H = np.dot(samples, weights.T) + bias.T
        sigma_H = activation(H)
        sigma_prime_H = activation_prime(H)
        if f_numpy is None:
            f_numpy = system.f_numpy_vectorized
        f_x = f_numpy(samples)
        # print("f_x: ", f_x.shape)
        # print("sigma_H: ", sigma_H.shape)
        # print("sigma_prime_H: ", sigma_prime_H.shape)
        # print("weights: ", weights.shape)

        batched_gradients = np.einsum('nm,md->mnd', sigma_prime_H, weights)
        omega_values = omega(samples)  
        # print("omega: ", omega_values.shape)
        if loss_mode == "Zubov": 
            A = (np.einsum('mnd,nd->mn', batched_gradients, f_x) 
                 - mu * np.einsum('nm,n->mn', sigma_H, omega_values))
            b = - mu * omega_values  

        elif loss_mode == "Lyapunov_GHJB":
            if g_numpy is None:
                g_numpy = system.g_numpy_vectorized
            g_x = g_numpy(samples)
            # print("g_x: ", g_x.shape)
            u = u_func(samples)
            # print("u: ", u.shape)
            f_u = f_x + np.einsum('ndk,nk->nd', g_x, u)
            # print("f_u: ", f_u.shape)

            A = np.einsum('mnd,nd->mn', batched_gradients, f_u)
            b = - omega_values
        
        elif loss_mode == "Zubov_GHJB":
            if g_numpy is None:
                g_numpy = system.g_numpy_vectorized
            g_x = g_numpy(samples)
            u = u_func(samples)
            f_u = f_x + np.einsum('ndk,nk->nd', g_x, u)
            A = (np.einsum('mnd,nd->mn', batched_gradients, f_u)
                 - mu * np.einsum('nm,n->mn', sigma_H, omega_values))
            b = - mu * omega_values

        else:
            A = np.einsum('mnd,nd->mn', batched_gradients, f_x)
            b = - omega_values
        return A, b

    def compute_residuals(samples, weights, bias):
        N = samples.shape[0]
        m = weights.shape[0]
        A = np.zeros((m, N))
        b = np.zeros(N)

        for i in range(N):
            x = samples[i].reshape(-1, 1)
            H = np.matmul(weights, x) + bias
            sigma_H = activation(H)
            sigma_prime_H = activation_prime(H)
            diag_sigma_prime_H = np.diagflat(sigma_prime_H)
            if f_numpy is None: 
                f_x = [func(*x) for func in system.f_numpy]
            else:
                f_x = f_numpy(x)
            if loss_mode == "Zubov": 
                mu = 0.1
                A[:, i] = (
                    np.matmul(diag_sigma_prime_H, weights) @ f_x
                    - mu * sum(xi**2 for xi in x) * sigma_H
                    ).squeeze()
                b[i] = - mu * Q(x)

            elif loss_mode == "Lyapunov_GHJB":
                if g_numpy is not None: 
                    g_x = g_numpy(x)
                else:
                    g_x = system.g_numpy(*x)

                u = u_func(x)
                # print("f_x: ", f_x.shape)
                # print("g_x: ", g_x.shape)
                # print("u: ", u.shape)
                f_u = f_x + np.dot(g_x, u)
                # print("f_u: ", f_u.shape)

                A[:, i] = (
                    np.matmul(diag_sigma_prime_H, weights) @ f_u
                    ).squeeze() 
                b[i] = - omega(x)  
            else: 
                A[:, i] = (
                    np.matmul(diag_sigma_prime_H, weights) @ f_x
                    ).squeeze() 
                b[i] = - Q(x)
        return A, b

    def compute_zero_boundary(samples, weights, bias):
        N = samples.shape[0]
        m = weights.shape[0]
        A = np.zeros((m, N))
        b = np.zeros(N)    
        for i in range(N):
            x = samples[i].reshape(-1, 1)
            H = np.matmul(weights, x) + bias
            sigma_H = activation(H)
            A[:, i] = sigma_H.squeeze() 
        return A, b    

    def compute_one_boundary(samples, weights, bias):
        H = np.dot(weights, samples.T) + bias
        A = activation(H)
        N = samples.shape[0]
        b = np.ones(N) 
        return A, b    

    def generate_boundary_points(domain, num_points_per_edge):
        if len(domain) == 1:
            x_min, x_max = domain[0]
            return np.array([[x_min], [x_max]])

        if len(domain) >= 2:
            # Generate points for the first two dimensions
            x_min, x_max = domain[0]
            y_min, y_max = domain[1]
            x_edge_points = np.linspace(x_min, x_max, num_points_per_edge)
            y_edge_points = np.linspace(y_min, y_max, num_points_per_edge)

            # Generate edge points for the first two dimensions
            edge_points = np.array([[x, y] for x in [x_min, x_max] 
                                    for y in y_edge_points] +
                                   [[x, y] for x in x_edge_points 
                                    for y in [y_min, y_max]])

            if len(domain) == 2:
                return edge_points
            # For other dimensions, fix points at their min and max values
            other_dims_fixed_points = []
            for dim_values in domain[2:]:
                min_val, max_val = dim_values
                for fixed_val in [min_val, max_val]:
                    fixed_points = np.full((len(edge_points), len(domain)), 
                                           fixed_val)
                    fixed_points[:, :2] = edge_points  
                    other_dims_fixed_points.append(fixed_points)

            return np.vstack(other_dims_fixed_points)

    def compute_data_boundary(data, weights, bias):
        x_data, y_data = data[:, :-1], data[:, -1]
        N = x_data.shape[0]
        m = weights.shape[0]
        A = np.zeros((m, N))
        for i in range(N):
            x = x_data[i].reshape(-1, 1)
            H = np.matmul(weights, x) + bias
            sigma_H = activation(H)
            A[:, i] = sigma_H.squeeze()
        b = y_data
        return A, b

    def compute_max_residual(weights, bias, system, beta, N):
        test_samples = np.array([np.random.uniform(dim[0], dim[1], 2 * N) 
                                 for dim in system.domain]).T
        # A_test, b_test = compute_residuals(test_samples, weights, bias)
        A_test, b_test = compute_residuals_vectorized(test_samples, weights, bias)
        residual_errors = np.dot(A_test.T, beta) - b_test
        return np.max(np.abs(residual_errors))

    def compute_controller_jacobian_loss_at_zero(weights, bias, K):
        result = lyznet.utils.compute_controller_gain_ELM_loss_dreal(
            weights, bias, system
            )
        A = result.reshape(-1, m).T        
        b = K.flatten()
        return A, b

    def compute_gain_loss(weights, bias, next_K): 
        result = lyznet.utils.compute_controller_gain_ELM_loss_numpy(
            weights, bias, np.linalg.inv(system.R), (system.B).T, Dg_zero
            )
        A = result.reshape(-1, m).T        
        b = next_K.flatten()
        return A, b

    d = len(system.symbolic_vars)
    m = num_hidden_units
    N = num_colloc_pts

    if weights is None:
        weights = np.random.randn(m, d)
    if bias is None:
        bias = np.random.randn(m, 1)
    if samples is None:
        samples = np.array([np.random.uniform(dim[0], dim[1], N) 
                            for dim in system.domain]).T

    x0 = np.zeros((1, d))

    print('_' * 50)
    print("Learning ELM Lyapunov function:")
    print("Computing residuals to set up ELM optimization...")
    lyznet.tik()
    A1, b1 = compute_residuals_vectorized(samples, weights, bias)
    # A1, b1 = compute_residuals(samples, weights, bias)
    lyznet.tok()

    A2, b2 = compute_zero_boundary(x0, weights, bias)

    boundary_weight = 100

    if (((loss_mode == "Zubov" or loss_mode == "Zubov_GHJB") and data is None)
            and one_boundary):
        # Add extra 1-boundary condition for solving Zubov equation
        boundary_points = generate_boundary_points(system.domain, 100)
        A3, b3 = compute_one_boundary(boundary_points, weights, bias)
        A = np.hstack((A1, boundary_weight * A2, boundary_weight * A3))
        b = np.hstack((b1, boundary_weight * b2, boundary_weight * b3))
    else:
        A = np.hstack((A1, boundary_weight * A2))
        b = np.hstack((b1, boundary_weight * b2))

    if data is not None:
        A_data, b_data = compute_data_boundary(data, weights, bias)
        A = np.hstack((A, A_data))
        b = np.hstack((b, b_data))

    if next_K is not None:
        A4, b4 = compute_gain_loss(weights, bias, next_K)
        A = np.hstack((A, A4))
        b = np.hstack((b, b4))

    if c2_P is not None and (loss_mode == "Zubov" or loss_mode == "Zubov_GHJB"):
        # Sampling points that satisfy xT*system.P*x <= c2_P
        valid_samples = []
        while len(valid_samples) < 1000:
            sample = np.random.uniform([dim[0] for dim in system.domain], 
                                       [dim[1] for dim in system.domain], (1, d))
            if np.dot(np.dot(sample, system.P), sample.T) <= c2_P:
                valid_samples.append(sample)
        valid_samples = np.vstack(valid_samples)
        
        # Computing A_extra, b_extra with 100 weight
        A_extra, b_extra = compute_residuals_vectorized(valid_samples, weights, bias)
        A_extra *= 100  # Applying weight
        b_extra *= 100  # Applying weight
        
        # Augmenting A and b with A_extra and b_extra
        A = np.hstack((A, A_extra))
        b = np.hstack((b, b_extra))

    reg_matrix = np.eye(A.shape[0]) * lambda_reg
    A_reg = np.vstack((A.T, np.sqrt(lambda_reg) * reg_matrix))
    b_reg = np.hstack((b, np.zeros(A.shape[0])))

    print("Trainning ELM with linear least squares...")
    lyznet.tik()
    # beta, residuals, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
    beta, residuals, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    # beta, residuals, _, _ = scipy.linalg.lstsq(A_reg, b_reg, 
    #                                            lapack_driver='gelsy')
    lyznet.tok()

    print("Square Residual Sum:", residuals)  
    if test is True:
        print("Testing residual errors...")
        lyznet.tik() 
        max_residual_error = compute_max_residual(
            weights, bias, system, beta, N
            )
        print("Maximum Test Residual Error:", max_residual_error)
        lyznet.tok()   
    
    model_path = (f"results/{system.name}_numpy_elm_m={num_hidden_units}"
                  f"_loss_mode={loss_mode}"
                  f"_N={num_colloc_pts}")

    if data is not None:
        num_data_points = data.shape[0]
        model_path = f"{model_path}_data={num_data_points}"

    if return_test_loss:
        return weights, bias, beta, model_path, max_residual_error
    else:
        return weights, bias, beta, model_path
