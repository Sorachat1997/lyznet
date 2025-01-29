from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import numpy as np
import os
from scipy.integrate import solve_ivp
import random
import pandas as pd
import lyznet


plt.rcParams['pdf.fonttype'] = 42
# rc('font', **{'family': 'Linux Libertine O'})
# plt.rcParams['font.size'] = 16
# plt.rcParams['mathtext.fontset'] = 'stix'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_accumulated_cost(system, model_path, elm_model, u_func, Q, R, T=10, 
                          step_size=0.02, save_csv=True, 
                          closed_loop_f_numpy=None):
    domain = system.domain

    if system.system_type == "DynamicalSystem":
        f_np = lambda t, x: system.f_numpy_vectorized(x) 
    else: 
        # For non-autonomous systems, ensure closed_loop_f_numpy is provided
        if closed_loop_f_numpy is None:
            # raise ValueError("The system is not autonomous, "
            #                  "but a closed-loop vector field is not given.")
            closed_loop_f_numpy = system.closed_loop_f_numpy
        f_np = lambda t, x: closed_loop_f_numpy(x)

    initial_conditions = [d[0]/3 for d in domain]

    fig, ax = plt.subplots()

    # Solve the differential equation from the given initial conditions
    sol = solve_ivp(f_np, [0, T], initial_conditions, method='RK45', 
                    t_eval=np.linspace(0, T, int(T/step_size)))

    # Calculate cost at each time step
    cost_values = []
    for x in sol.y.T:
        x_np = np.array(x)
        # print("x: ", x_np.shape)
        if hasattr(u_func, 'is_numpy_func') and u_func.is_numpy_func:
            u_np = u_func(x_np).T
            Q = system.Q
            R = system.R
        elif elm_model is None:
            x_tensor = torch.tensor(x_np, dtype=torch.float32).view(1, -1)
            u_np = u_func(x_tensor).squeeze(0).detach().numpy()  
        else:
            u_np = u_func(x_np).T 

        # print("Q: ", Q.shape)
        # print("x: ", x_np.shape)
        # print("R: ", R.shape)
        # print("u: ", u_np.shape)        
        cost = x_np.T @ Q @ x_np + u_np.T @ R @ u_np
        cost_values.append(cost)

    accumulated_cost = np.cumsum(np.array(cost_values) * step_size)

    ax.plot(sol.t, accumulated_cost, label='Accumulated Cost')

    ax.set_xlabel('Time')
    ax.set_ylabel('Accumulated Cost')
    ax.set_title('Accumulated Cost Over Time')

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'{model_path}_cost.pdf', format='pdf', dpi=300)
    plt.close(fig)

    if save_csv:
        cost_data = pd.DataFrame({'Time': sol.t, 'Cost': accumulated_cost})
        csv_file_path = f'{model_path}_cost.csv'
        cost_data.to_csv(csv_file_path, index=False)


def simulate_trajectories(system, model_path, n=50, T=20, 
                          closed_loop_f_numpy=None, plot_control=False,
                          u_func=None):
    print(f"Simulating {n} trajectories from random intitial conditions...")
    fig, ax = plt.subplots() 
    domain = system.domain  
    if system.system_type == "DynamicalSystem":
        f = lambda t, x: system.f_numpy_vectorized(x) 
    else: 
        # For non-autonomous systems, ensure closed_loop_f_numpy is provided
        if closed_loop_f_numpy is None:
            # raise ValueError("The system is not autonomous, "
            #                  "but a closed-loop vector field is not given.")
            closed_loop_f_numpy = system.closed_loop_f_numpy
        f = lambda t, x: closed_loop_f_numpy(x)

    all_u_values = []
    all_t_values = []

    lyznet.tik()
    for _ in range(n):
        # Random initial conditions within the full domain
        # initial_conditions = [random.uniform(*d) for d in domain]

        # Try initial conditions within a smaller domain
        half_domain = [[d[0] / 3, d[1] / 3] for d in domain]
        initial_conditions = [random.uniform(*d) for d in half_domain]

        # Solve differential equation using RK45
        sol = solve_ivp(f, [0, T], initial_conditions, method='RK45', 
                        t_eval=np.linspace(0, T, 500))
        # Plot each dimension of the solution against time
        for i in range(sol.y.shape[0]):
            ax.plot(sol.t, sol.y[i], linewidth=1, 
                    label=f'Dimension {i+1}' if _ == 0 else "")

        # Calculate and store u_func values if plot_control is True
        if plot_control and u_func is not None:
            u_values = []
            for x in sol.y.T:
                u_np = u_func(x)
                u_values.append(u_np)
            all_u_values.append(u_values)
            all_t_values.append(sol.t)

    ax.set_xlabel('Time')
    ax.set_ylabel('States')
    ax.set_title('States Over Time')

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'{model_path}_{n}_trajectories_[0,{T}].pdf', 
                format='pdf', dpi=300)
    plt.close(fig)

    # If plot_control is True, create a separate figure for control inputs
    if plot_control and u_func is not None:
        fig_u, ax_u = plt.subplots(figsize=(10, 6))
        for u_vals, t_vals in zip(all_u_values, all_t_values):
            u_vals = np.array(u_vals)
            ax_u.plot(t_vals, u_vals[:, 0], label="Control Input u(t)")
        
        ax_u.set_xlabel('Time')
        ax_u.set_ylabel('Control Inputs')
        ax_u.set_title('Control Inputs Over Time')

        plt.savefig(f'{model_path}_{n}_control_inputs_[0,{T}].pdf', 
                    format='pdf', dpi=300)
        plt.close(fig_u)

    lyznet.tok()


def evaluate_vector_field(x1, x2, system, closed_loop_f_numpy):
    d = len(system.symbolic_vars)
    zero_dims = d - 2
    x_dims = [np.zeros_like(x1) for _ in range(zero_dims)]
    input_array = np.vstack(
        [x1.ravel(), x2.ravel()] + [x_dim.ravel() for x_dim in x_dims]
    ).T
    # print("input_array: ", input_array.shape)
    # f_values = [f_np(*input_array.T) for f_np in system.f_numpy]
    # f_values = system.f_numpy(*input_array.T)
    if system.system_type == "DynamicalSystem":
        f_values = system.f_numpy_vectorized(input_array).T 
        # print(f_values.shape)
    else: 
        # # For non-autonomous systems, ensure closed_loop_f_numpy is provided
        if closed_loop_f_numpy is None:
            # raise ValueError("The system is not autonomous, "
            #                  "but a closed-loop vector field is not given.")
            closed_loop_f_numpy = system.closed_loop_f_numpy
        f_values = closed_loop_f_numpy(input_array).T
    f_values = np.array(f_values)[:2]  # Only taking first two dimensions
    # print("f_values: ", f_values.shape)
    return f_values[0].reshape(x1.shape), f_values[1].reshape(x2.shape)


def plot_phase_portrait(Xd, Yd, system, closed_loop_f_numpy=None):
    DX, DY = evaluate_vector_field(Xd, Yd, system, closed_loop_f_numpy)
    # print("DX: ", DX)
    # print("DY: ", DY)
    plt.streamplot(Xd, Yd, DX, DY, color='gray', linewidth=0.5, density=0.8, 
                   arrowstyle='-|>', arrowsize=1)


def evaluate_dynamics(f, x):
    x_split = torch.split(x, 1, dim=1)
    result = []
    for fi in f:
        args = [x_s.squeeze() for x_s in x_split]
        result.append(fi(*args))
    return result


def plot_lie_derivative(system, net, x_tensor, x1, x2, ax):
    # Calculate V and its gradient
    x_tensor.requires_grad = True
    V = net(x_tensor).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x_tensor, create_graph=True)[0]

    # Calculate dynamics and V_dot
    f_values = evaluate_dynamics(system.f_torch, x_tensor)
    f_tensor = torch.stack(f_values, dim=1)
    V_dot = (V_grad * f_tensor).sum(dim=1)

    # Reshape V_dot for plotting
    V_dot_reshaped = V_dot.detach().cpu().numpy().reshape(x1.shape)

    # Overlay the lie derivative plot on the same axes
    ax.plot_surface(x1, x2, V_dot_reshaped, color='red', alpha=0.5, 
                    label="Lie Derivative")


def plot_obstacles(ax, obstacles):
    for obstacle in obstacles:
        if obstacle['type'] == 'rectangle':
            center = obstacle['center']
            width = obstacle['width']
            height = obstacle['height']
            lower_left_corner = (center[0] - width / 2, center[1] - height / 2)
            rect = Rectangle(lower_left_corner, width, height, color='grey', 
                             alpha=0.7)
            ax.add_patch(rect)
        elif obstacle['type'] == 'circle':
            center = obstacle['center']
            radius = obstacle['radius']
            circle = Circle(center, radius, color='grey', alpha=0.7)
            ax.add_patch(circle)


def plot_V(system, net=None, model_path=None, V_list=None, c_lists=None, 
           c1_V=None, c2_V=None, c1_P=None, c2_P=None, 
           phase_portrait=None, elm_model=None, lie_derivative=None, 
           plot_trajectories=None, n_trajectories=50, plot_cost=None,
           u_func=None, Q=None, R=None, closed_loop_f_numpy=None,
           obstacles=None, plot_control=True):

    domain = system.domain
    d = len(system.symbolic_vars)

    print("Plotting learned Lyapunov function and level sets...")
    lyznet.tik()
    if d == 1:
        x1 = np.linspace(*domain[0], 400)
        x1 = x1.reshape(-1, 1)  # Reshape to 2D array for consistency
        input_array = x1
        P = system.P

        if P is not None:
            def quad_V(x):
                return x.T @ P @ x

            quad_V_test = np.array(
                [quad_V(x) for x in input_array]
                ).reshape(x1.shape)

        if elm_model is not None: 
            weights, bias, beta = elm_model

            def elm_V(x):
                H = np.matmul(x, weights.T) + bias.T
                return np.tanh(H) @ beta
            V_test = np.array(
                [elm_V(x) for x in input_array]
                ).reshape(x1.shape)
        elif net is not None: 
            x_test = torch.tensor(input_array, dtype=torch.float32).to(device)
            V_net = net(x_test)
            V_test = V_net.detach().cpu().numpy().reshape(x1.shape)
        else: 
            if P is not None:
                V_test = quad_V_test
            else:
                V_test = None

        fig = plt.figure(figsize=(12, 6))  # Set figure size
        ax1 = fig.add_subplot(121)
        if V_test is not None:
            ax1.plot(x1, V_test, label="Learned Lyapunov Function")

        # Plotting level sets for 1D

        if c1_V is not None or c2_V is not None: 
            level_values = [c1_V, c2_V]  # Add other level values if needed
            for level in level_values:
                if level is not None:
                    level_points = x1[(V_test < level) 
                                      & (np.abs(V_test - level) < 1e-2)]
                    ax1.plot(level_points, [level] * len(level_points), 'ro')  

            for x_val in level_points:
                ax1.axvline(x=x_val, color='k', linestyle='--', alpha=0.7)

        ax1.set_xlabel(r"$x_1$", fontsize=24)
        ax1.set_ylabel("V(x)", fontsize=24)
        ax1.set_title("Learned Lyapunov Function")
        ax1.legend()

    else: 
        # Generate samples for contour plot
        x1 = np.linspace(*domain[0], 200)
        x2 = np.linspace(*domain[1], 200)
        x1, x2 = np.meshgrid(x1, x2)
        # Plots are projected to (x1,x2) plane. Set other dimensions to zero.
        zero_dims = d - 2
        x_dims = [np.zeros_like(x1) for _ in range(zero_dims)]
        # Stack x1, x2, and zero dimensions to form the input tensor
        input_array = np.vstack(
            [x1.ravel(), x2.ravel()] + [x_dim.ravel() for x_dim in x_dims]
        ).T

        if system.P is not None:
            P = system.P

            def quad_V(x):
                return x.T @ P @ x

            quad_V_test = np.array([quad_V(x) for x in input_array]).reshape(x1.shape)

        if elm_model is not None: 
            weights, bias, beta = elm_model

            def elm_V(x):
                H = np.matmul(x, weights.T) + bias.T
                return np.tanh(H) @ beta
            V_test = np.array([elm_V(x) for x in input_array]).reshape(x1.shape)
        elif net is not None: 
            x_test = torch.tensor(input_array, dtype=torch.float32).to(device)
            V_net = net(x_test)
            V_test = V_net.detach().cpu().numpy().reshape(x1.shape)
        else: 
            if system.P is not None:
                V_test = quad_V_test
            else:
                V_test = None

        fig = plt.figure(figsize=(12, 6))  # Set figure size

        # Subplot 1: 3D surface plot of the learned function
        ax1 = fig.add_subplot(121, projection="3d")
        if V_test is not None:
            ax1.plot_surface(x1, x2, V_test)  # Plot the learned function
        ax1.set_xlabel(r"$x_1$", fontsize=24)
        ax1.set_ylabel(r"$x_2$", fontsize=24)
        # ax1.set_zlabel("V(x)")
        ax1.set_title("Learned Lyapunov Function")

        if net is not None and lie_derivative is not None:
            plot_lie_derivative(system, net, x_test, x1, x2, ax1)

        # Subplot 2: Contour plot of target set and level sets
        ax2 = fig.add_subplot(122)

        if V_list is not None and c_lists is not None:
            color_list = ['g', 'b', 'c', 'm', 'y', 'k']  # Add more colors if needed
            for func, levels, color in zip(V_list, c_lists, color_list):
                func_input_split = np.split(input_array, input_array.shape[1], axis=1)
                func_eval = func(*func_input_split).reshape(x1.shape)
                ax2.contour(x1, x2, func_eval, levels=levels,
                            colors=color, linewidths=2, linestyles='-.')

        # if V_list is not None and c_lists is not None:
        #     for func, levels in zip(V_list, c_lists):
        #         func_input_split = np.split(input_array, input_array.shape[1], 
        #                                     axis=1)
        #         func_eval = func(*func_input_split).reshape(x1.shape)
        #         ax2.contour(x1, x2, func_eval, levels=levels,
        #                     colors='g', linewidths=2, linestyles='-.')    

        if c1_P is not None: 
            ax2.contour(x1, x2, quad_V_test, levels=[c1_P], 
                        colors='r', linewidths=2, linestyles='--')
        
        if c2_P is not None: 
            ax2.contour(x1, x2, quad_V_test, levels=[c2_P], 
                        colors='r', linewidths=2, linestyles='--')
        
        if c1_V is not None:
            ax2.contour(x1, x2, V_test, colors='b', levels=[c1_V])
        
        if c2_V is not None:
            cs = ax2.contour(x1, x2, V_test, colors='b', levels=[c2_V], 
                             linewidths=3)    

        if phase_portrait is not None:
            plot_phase_portrait(x1, x2, system, 
                                closed_loop_f_numpy=closed_loop_f_numpy)

        if c2_V is not None:
            ax2.clabel(cs, inline=1, fontsize=10)
        ax2.set_xlabel(r'$x_1$', fontsize=24)
        ax2.set_ylabel(r'$x_2$', fontsize=24)
        ax2.set_title('Level sets')
    lyznet.tok()    

    if obstacles is not None:
        plot_obstacles(ax2, obstacles)

    if plot_trajectories is not None:
        simulate_trajectories(system, model_path, 
                              closed_loop_f_numpy=closed_loop_f_numpy,
                              plot_control=plot_control,
                              u_func=u_func
                              )

    if plot_cost is not None:
        plot_accumulated_cost(system, model_path, elm_model, 
                              u_func, Q, R, T=10, 
                              step_size=0.02, save_csv=True,
                              closed_loop_f_numpy=closed_loop_f_numpy)
    plt.tight_layout()
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'{model_path}.pdf', format='pdf', dpi=300)
    plt.close(fig)
    # plt.show()
