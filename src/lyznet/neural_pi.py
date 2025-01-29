import sympy as sp
from sympy.utilities.lambdify import lambdify
import numpy as np
import torch
import lyznet
import time
import scipy


def get_closed_loop_f_expr(f_expr, g_expr, u_expr, symbolic_vars):
    # print("Computing closed-loop dynamics with a given controller...")
    # lyznet.tik()
    f_u_matrix = (sp.Matrix(f_expr) 
                  + lyznet.utils.sympy_matrix_multiply_dreal(
                    sp.Matrix(g_expr), sp.Matrix(u_expr), symbolic_vars
                    )
                  )
    f_u = [element[0] for element in f_u_matrix.tolist()]
    # lyznet.tok()    
    return f_u


def get_closed_loop_f_numpy(system, f_numpy, g_numpy, u_func):
    def closed_loop_f_numpy(samples):
        f_x = f_numpy(samples)
        g_x = g_numpy(samples)
        u = u_func(samples)
        f_u = f_x + np.einsum('ndk,nk->nd', g_x, u)
        return f_u    
    return closed_loop_f_numpy


def stage_cost_numpy(x, u_func, Q_numpy, R_numpy):
    u = u_func(x)  
    x_cost = np.einsum('ni,ij,nj->n', x, Q_numpy, x)
    u_cost = np.einsum('ni,ij,nj->n', u, R_numpy, u)
    omega = x_cost + u_cost
    return omega


def stage_cost_torch(x, u_func, Q_numpy, R_numpy):
    u = u_func(x) 
    N, k, n = u.shape
    if k > n:
        u = u.transpose(1, 2)

    Q_torch = torch.tensor(
        Q_numpy, dtype=torch.float32
        ).repeat(N, 1, 1)

    R_torch = torch.tensor(
        R_numpy, dtype=torch.float32
        ).repeat(N, 1, 1)

    u_cost = torch.bmm(
        torch.bmm(u, R_torch), u.transpose(1, 2)
        ).squeeze().squeeze()    
    x_unsqueezed = x.unsqueeze(2)  
    x_cost = torch.bmm(
        torch.bmm(x_unsqueezed.transpose(1, 2), Q_torch), x_unsqueezed
        ).squeeze().squeeze()
    return x_cost + u_cost


def get_controller_expr_from_Net(net, R_sym, g_expr, symbolic_vars):
    R_inv_sym = sp.Matrix(R_sym).inv()
    g_matrix = sp.Matrix(g_expr).T
    start_time = time.time()
    print("Computing controller from neural value function...")
    u_matrix = lyznet.utils.get_controller_from_Net_dreal(
        net, R_inv_sym, g_matrix, symbolic_vars
        )
    u_expr = u_matrix.tolist()
    # print("u_expr:", u_expr)
    elapsed_time = time.time() - start_time
    print(f"Total time for computing controller: {elapsed_time:.2f} seconds.")
    return sp.Matrix(u_expr)


def get_controller_expr_from_ELM(W, b, beta, R_sym, g_expr, symbolic_vars):
    R_inv_sym = sp.Matrix(R_sym).inv()
    g_matrix = sp.Matrix(g_expr).T
    start_time = time.time()
    print("Computing controller from neural value function...")
    u_matrix = lyznet.utils.get_controller_from_ELM_dreal(
        W, b, beta, R_inv_sym, g_matrix, symbolic_vars
        )
    u_expr = u_matrix.tolist()
    # print("u_expr:", u_expr)
    elapsed_time = time.time() - start_time
    print(f"Total time for computing controller: {elapsed_time:.2f} seconds.")
    return sp.Matrix(u_expr)


def get_controller_torch_from_Net(net, g_torch, R_inv_numpy):
    def u_func(x, net):
        x.requires_grad_(True)
        V = net(x)
        V_x = torch.autograd.grad(
            outputs=V, inputs=x, grad_outputs=torch.ones_like(V), 
            create_graph=True, only_inputs=True)[0]
        N = x.shape[0]
        R_inv_torch = torch.tensor(
            R_inv_numpy, dtype=torch.float32
            ).repeat(N, 1, 1)

        g_torch_value = g_torch(x).transpose(1, 2)

        u_values = -0.5 * torch.bmm(
            torch.bmm(R_inv_torch, g_torch_value), V_x.unsqueeze(2)
            )
        return u_values
    return lambda x: u_func(x, net)


def get_controller_numpy_from_ELM(W, b, beta, g_numpy, R_inv_numpy, loss_mode):
    def activation(x):
        return np.tanh(x)

    def activation_prime(x):
        return 1 - np.square(np.tanh(x))

    def u_func_vectorized(x):
        x = np.atleast_2d(x)
        H = (np.dot(W, x.T) + b).T
        sigma_prime_H = activation_prime(H)
        g_x = g_numpy(x)
        g_x_transposed = np.transpose(g_x, (0, 2, 1))
        beta_reshaped = beta.reshape(-1, 1)
        batched_gradients = np.einsum('nm,md->nmd', sigma_prime_H, W)
        dv_x = np.einsum('mi,nmd->ndi', beta_reshaped, batched_gradients)
        k_x_temp = np.einsum('nkd,ndi->nki', g_x_transposed, dv_x)
        k_x = -0.5 * np.einsum('ij,njk->nik', R_inv_numpy, k_x_temp)
        k_x = np.squeeze(k_x, axis=-1)
        # for Zubov_GHJB
        if loss_mode == "Zubov_GHJB":
            sigma_H = activation(H)
            v_x = np.einsum('mi,nm->ni', beta_reshaped, sigma_H)
            divisor = 0.1 * (1 - v_x)
            divisor_reshaped = divisor.reshape(-1, 1)
            # k_x = k_x / divisor_reshaped
            mask = v_x > 0.99
            k_x[mask] = 0
            k_x[~mask] = k_x[~mask] / divisor_reshaped[~mask]        
        return k_x

    return u_func_vectorized   


def elm_pi(system, initial_u=None, num_of_iters=10, num_colloc_pts=3000,
           width=10, data=None, n_samples=3000, f_numpy=None, g_numpy=None,
           plot_each_iteration=None, verify=None, test=None, final_plot=None,
           gain_loss=None, final_test=True, mode=None, one_boundary=True,
           return_model=None):
    start_time = time.time()
    print("-" * 50)
    print("Extreme learning machine policy iteration (ELM-PI):")
    print(f"Control system model ({system.name}): ")
    print("Domain: \n", system.domain)
    print("Uncontrolled vector field f: \n", system.symbolic_f)
    print("Control matrix g: \n", system.symbolic_g)
    print("Linearized vector field A: \n", system.A)
    print("Linearized control matrix B: \n", system.B)
    print("State cost matrix Q: \n", system.Q)
    print("Control cost matrix R: \n", system.R)
    print("LQR gain K: ", system.K)

    if initial_u is None:
        initial_u = sp.Matrix(system.K) * sp.Matrix(system.symbolic_vars)

    print("Initial controller: ", initial_u)

    u_expr = initial_u
    init_u_func = sp.lambdify(system.symbolic_vars, u_expr, modules=['numpy'])

    def u_func(x):
        x = np.atleast_2d(x)
        u_value_transposed = np.transpose(init_u_func(*x.T))
        # print("u_value_transposed: ", u_value_transposed.shape)
        u_value = np.transpose(u_value_transposed, (0, 2, 1))
        # print("u_value: ", u_value.shape)
        output = np.squeeze(u_value, axis=-1)
        # print("u_output: ", output)
        return output

    Q_numpy = system.Q
    R_numpy = system.R
    R_inv_numpy = np.linalg.inv(R_numpy)

    if f_numpy is None:
        f_numpy = system.f_numpy_vectorized

    if g_numpy is None:
        g_numpy = system.g_numpy_vectorized

    sys_name = system.name

    # calculating initial linear gain and prediction
    pre_K = None  
    Dg_zero = None
    if gain_loss is not None:
        ep = [0] * len(system.symbolic_vars)
        K = np.array(
                lyznet.utils.compute_jacobian_np_dreal(
                    u_expr, system.symbolic_vars, ep
                    )
                )
        Dg_zero = lyznet.utils.compute_Dg_zero(system)
        # print("Dg_zero", Dg_zero)

    for iter_num in range(1, num_of_iters + 1):
        print("-" * 50)
        print(f"Iteration {iter_num}...")
        iter_name = f"{sys_name}_iter_{iter_num}"

        # Evaluate u_func at the origin
        origin = np.zeros((1, len(system.symbolic_vars)))   
        u_at_origin = u_func(origin)
        print("u_func value at origin:", u_at_origin)

        omega = lambda x: stage_cost_numpy(x, u_func, Q_numpy, R_numpy)

        # if data is not None and iter_num == 1: 
        #     data = lyznet.generate_data(
        #         closed_loop_sys, n_samples=n_samples, 
        #         omega=omega
        #         )

        if gain_loss is not None:
            P = scipy.linalg.solve_continuous_lyapunov(
                (system.A + system.B @ K).T, -Q_numpy - K.T @ R_numpy @ K
                )

            eigenvalues = np.linalg.eigvals(system.A + system.B @ K)
            # Print eigenvalues
            print("Eigenvalues of linearization:", eigenvalues)

            next_K = - R_inv_numpy @ system.B.T @ P
            print(f"Linear gain at iteration {iter_num}:", K)
            print(f"Predicted linear gain at iteration {iter_num+1}:", next_K)
        else:
            next_K = None 

        system.name = iter_name

        if mode == "Zubov":
            loss_mode = "Zubov_GHJB"
        else:
            loss_mode = "Lyapunov_GHJB"

        W, b, beta, model_path = lyznet.numpy_elm_learner(
            system, num_hidden_units=width, 
            num_colloc_pts=num_colloc_pts,
            lambda_reg=0.0, loss_mode=loss_mode, omega=omega,
            u_func=u_func, f_numpy=f_numpy, g_numpy=g_numpy, test=test,
            next_K=next_K, Dg_zero=Dg_zero, c2_P=0.3, one_boundary=one_boundary
            )

        if plot_each_iteration is True: 
            f_u_numpy = get_closed_loop_f_numpy(system, f_numpy, 
                                                g_numpy, u_func)

            lyznet.plot_V(system, elm_model=[W, b, beta], 
                          model_path=model_path, 
                          phase_portrait=True, 
                          plot_trajectories=True, 
                          plot_cost=True, u_func=u_func, Q=system.Q, 
                          R=system.R, closed_loop_f_numpy=f_u_numpy)

        if iter_num <= num_of_iters:
            u_func = get_controller_numpy_from_ELM(
                W, b, beta, g_numpy, R_inv_numpy, loss_mode
                )
            if gain_loss is not None:
                pre_K = lyznet.utils.compute_controller_gain_ELM_loss_numpy(
                    W, b, R_inv_numpy, (system.B).T, Dg_zero
                    )
                K = np.einsum('kdm,m->kd', pre_K, beta.squeeze())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("-" * 50)
    print(f"Total time for EML training: {elapsed_time:.2f} seconds.")
            
    # verify and generate final plot 
    if final_plot is True: 
        f_u_numpy = get_closed_loop_f_numpy(system, f_numpy, g_numpy, u_func)
        lyznet.plot_V(system, elm_model=[W, b, beta], 
                      model_path=model_path, 
                      phase_portrait=True,
                      plot_trajectories=True, 
                      plot_cost=True, u_func=u_func, Q=system.Q, 
                      R=system.R, closed_loop_f_numpy=f_u_numpy)

    if final_test is True: 
        W, b, beta, model_path = lyznet.numpy_elm_learner(
            system, num_hidden_units=width, 
            num_colloc_pts=num_colloc_pts,
            lambda_reg=0.0, loss_mode=loss_mode, omega=omega,
            u_func=u_func, f_numpy=f_numpy, g_numpy=g_numpy, test=True
            )
    if return_model:
        return W, b, beta
        
    if verify is True:
        sys_name = f"{sys_name}_final_verified"
        W, b, beta, model_path = lyznet.numpy_elm_learner(
            system, num_hidden_units=width, 
            num_colloc_pts=num_colloc_pts,
            lambda_reg=0.1, loss_mode=loss_mode, omega=omega,
            u_func=u_func, f_numpy=f_numpy, g_numpy=g_numpy
            )

        u_expr = get_controller_expr_from_ELM(
            W, b, beta, system.R, system.symbolic_g, system.symbolic_vars
            )
        f_u = get_closed_loop_f_expr(system.symbolic_f, system.symbolic_g, 
                                     u_expr, system.symbolic_vars)
        closed_loop_sys = lyznet.DynamicalSystem(f_u, system.domain, sys_name)   

        c1_P = lyznet.local_stability_verifier(closed_loop_sys)
        c2_P = lyznet.quadratic_reach_verifier(closed_loop_sys, c1_P)
        c1_V, c2_V = lyznet.numpy_elm_verifier(closed_loop_sys, W, b, beta, 
                                               c2_P)

        lyznet.plot_V(closed_loop_sys, elm_model=[W, b, beta], 
                      model_path=model_path, 
                      c2_V=c2_V, c2_P=c2_P)


def neural_pi(system, f_torch=None, g_torch=None, initial_u=None, 
              num_of_iters=10, lr=0.001, 
              layer=2, width=10, num_colloc_pts=300000, max_epoch=5,
              data=None, n_samples=3000, plot_each_iteration=None,
              final_plot=None, verify=None):

    sys_name = system.name
    domain = system.domain
    symbolic_vars = system.symbolic_vars
    f_expr = system.symbolic_f
    g_expr = system.symbolic_g

    print("-" * 50)
    print("Physics-informed neural network policy iteration (PINN-PI):")

    print(f"Control system model ({sys_name}): ")
    print("Domain: ", domain)
    print("f: ", system.symbolic_f)
    print("g: ", system.symbolic_g)
    print("A: ", system.A)
    print("B: ", system.B)
    print("Q: ", system.Q)
    print("R: ", system.R)
    print("K: ", system.K)

    if initial_u is None:
        initial_u = sp.Matrix(system.K) * sp.Matrix(system.symbolic_vars)

    print("Initial controller: ", initial_u)

    u_expr = initial_u
    u_numpy = sp.lambdify(system.symbolic_vars, u_expr, modules=['numpy'])
    u_func = lambda x: lyznet.utils.numpy_to_torch(x, u_numpy)

    Q_numpy = system.Q
    R_numpy = system.R
    R_inv_numpy = np.linalg.inv(R_numpy)

    if g_torch is None:
        g_torch = system.g_torch

    initial_net = None

    for iter_num in range(1, num_of_iters + 1):
        print("-" * 50)
        print(f"Iteration {iter_num}...")
        iter_sys_name = f"{sys_name}_iter_{iter_num}"

        # Evaluate u_func at the origin
        origin = torch.zeros(1, len(symbolic_vars))
        u_at_origin = u_func(origin)
        print("u_func value at origin:", u_at_origin)

        # Evaluate u_expr at the origin
        u_expr_at_origin = lyznet.utils.evaluate_at_origin_dreal(
            u_expr, symbolic_vars)
        print("u_expr evaluated at origin:", u_expr_at_origin)

        f_u = get_closed_loop_f_expr(f_expr, g_expr, u_expr, symbolic_vars)

        closed_loop_sys = lyznet.DynamicalSystem(f_u, domain, iter_sys_name)

        # print("Closed-loop system dynamics: x' = ", closed_loop_sys.symbolic_f)
        print("Eigenvalues of linearization: ", 
              np.linalg.eigvals(closed_loop_sys.A))

        # calculating linear gain and prediction
        ep = [0] * len(symbolic_vars)
        K = np.array(
                lyznet.utils.compute_jacobian_np_dreal(
                    u_expr, symbolic_vars, ep
                    )
                )
        print(f"Linear gain at iteration {iter_num}:", K)
        P = scipy.linalg.solve_continuous_lyapunov(
            closed_loop_sys.A.T, -Q_numpy - K.T @ R_numpy @ K
            )

        next_K = - R_inv_numpy @ system.B.T @ P
        print(f"Predicted linear gain at iteration {iter_num+1}:", next_K)
        next_K_tensor = torch.tensor(next_K, dtype=torch.float32)

        omega = lambda x: stage_cost_torch(x, u_func, Q_numpy, R_numpy)

        if data is not None and iter_num == 1: 
            data = lyznet.generate_data(
                closed_loop_sys, n_samples=n_samples, 
                omega=omega
                )

        system.name = iter_sys_name
        loss_mode = "Lyapunov_GHJB"

        net, model_path = lyznet.neural_learner(
            system, data=data, lr=lr, layer=layer, width=width, 
            num_colloc_pts=num_colloc_pts, 
            max_epoch=max_epoch, loss_mode=loss_mode, 
            omega=omega, u_func=u_func,
            f_torch=f_torch, g_torch=g_torch, R_inv=R_inv_numpy, 
            K=next_K_tensor, initial_net=initial_net
            )

        if plot_each_iteration: 
            lyznet.plot_V(closed_loop_sys, net, model_path, 
                          phase_portrait=True, 
                          plot_trajectories=True,
                          plot_cost=True, u_func=u_func, Q=system.Q, 
                          R=system.R)

        if iter_num <= num_of_iters:
            u_expr = get_controller_expr_from_Net(
                net, system.R, g_expr, symbolic_vars
                )

            u_func = get_controller_torch_from_Net(
                net, g_torch, R_inv_numpy
                )
            initial_net = net

    if final_plot:     
        f_u = get_closed_loop_f_expr(f_expr, g_expr, u_expr, symbolic_vars)
        closed_loop_sys = lyznet.DynamicalSystem(f_u, domain, sys_name) 
        lyznet.plot_V(closed_loop_sys, net, model_path, 
                      phase_portrait=True, 
                      plot_trajectories=True, 
                      plot_cost=True, u_func=u_func, Q=system.Q, R=system.R)

    if verify: 
        if final_plot is None: 
            f_u = get_closed_loop_f_expr(f_expr, g_expr, u_expr, symbolic_vars)
        c1_P = lyznet.local_stability_verifier(closed_loop_sys)
        c2_P = lyznet.quadratic_reach_verifier(closed_loop_sys, c1_P)
        c1_V, c2_V = lyznet.neural_verifier(closed_loop_sys, net, 
                                            c2_P, c2_V=10)
        lyznet.plot_V(closed_loop_sys, net, model_path, 
                      phase_portrait=True, 
                      plot_trajectories=True, 
                      c1_P=c1_P,
                      c2_P=c2_P,
                      c1_V=c1_V,
                      c2_V=c2_V, 
                      )
