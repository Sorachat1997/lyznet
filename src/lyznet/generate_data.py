import os
import time 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

import sympy as sp
from scipy.integrate import solve_bvp

import warnings


def generate_data(system, n_samples=None, v_max=200, 
                  overwrite=False, n_nodes=32, plot=True, 
                  transform="tanh", omega=None):
    domain = system.domain
    d = len(domain)  

    X_MAX = 1e+6
    eps = 1e-7
    T = 1

    def augmented_dynamics(t, z):
        # dz_ = [func(*z[:-1]) for func in system.f_numpy]
        dz_ = list(system.f_numpy(*z[:-1]))
        if omega is not None: 
            # data generation for Lyapunov equation Dv*f = -omega(x)
            # Makes it 2D with shape (1, len(z) - 1)
            z_tensor = torch.tensor([z[:-1]], dtype=torch.float32)  
            dz = omega(z_tensor).detach().numpy()
            # print("dz: ", dz)
        else: 
            dz = sum([s**2 for s in z[:-1]])
        return dz_ + [dz]

    def get_train_output(x, z, depth=0):
        # print("x: ", x)
        if np.linalg.norm(x) <= eps:
            if omega is not None:
                # data generation for Lyapunov equation
                y = z  # no transform needed
                z_T = z
            elif transform == "exp":
                # y = 1 - np.exp(-40/v_max*z)  # 1-exp(-40) is practically 1
                y = 1 - np.exp(-20/v_max*z) 
                z_T = z                
            else:
                y = np.tanh(20/v_max*z)  # tanh(20) is practically 1
                z_T = z
        elif z > v_max or np.linalg.norm(x) > X_MAX:
            y = 1.0
            z_T = v_max  # the largest recorded value for unstable initial cdts 
        else:
            sol = solve_ivp(lambda t, z: augmented_dynamics(t, z), [0, T], 
                            list(x) + [z], rtol=1e-6, atol=1e-9)
            current_x = np.array([sol.y[i][-1] for i in range(len(x))])
            current_z = sol.y[len(x)][-1]
            y, z_T = get_train_output(current_x, current_z, depth=depth+1)
        return [y, z_T]

    def generate_train_data(x):
        y, z_T = get_train_output(x, 0)
        return [x, y, z_T]

    if not os.path.exists('results'):
        os.makedirs('results')
    t_filename = (f'results/{system.name}_data_{n_samples}_samples'
                  f'_v_max_{v_max}.npy')
    z_values = None
    print('_' * 50)
    print("Generating training data from numerical integration:")
    if os.path.exists(t_filename) and not overwrite:
        print("Data exists. Loading training data...")
        t_data = np.load(t_filename)
        x_train, y_train = t_data[:, :-1], t_data[:, -1]
    else:
        print("Generating new training data...")
        start_time = time.time()    
        x_train = np.array([np.random.uniform(dim[0], dim[1], n_samples) 
                            for dim in domain]).T
        results = Parallel(n_jobs=n_nodes)(
            delayed(generate_train_data)(x) for x in tqdm(x_train)
            )
        x_train = np.array([res[0] for res in results])
        y_train = np.array([res[1] for res in results])
        z_values = np.array([res[2] for res in results])
        print("Saving training data...")
        t_data = np.column_stack((x_train, y_train))
        np.save(t_filename, t_data)
        end_time = time.time()
        print(f"Time taken for generating data: " 
              f"{end_time - start_time} seconds.\n")

    if plot:  
        plt.figure()    

        if d == 1:
            plt.scatter(x_train, np.zeros_like(x_train), c=y_train, 
                        cmap='coolwarm', s=1)  
        else:
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, 
                        cmap='coolwarm', s=1)  

        plt.savefig(
            f"results/{system.name}_data_{n_samples}_samples_v_max_{v_max}.pdf"
            )
        plt.close()

        if z_values is not None: 
            plt.figure()
            plt.scatter(z_values, np.zeros_like(z_values) + 0.5)
            plt.yticks([])
            plt.xlabel('Value')
            plt.xlim([0, v_max])
            plt.title('Z Values Clustering')
            plt.savefig(f'results/{system.name}_data_{n_samples}_samples'
                        f'_v_max_{v_max}_z_values.pdf')
            plt.close()

    return np.column_stack((x_train, y_train))


def generate_CM_data(system, n_samples=None, overwrite=False, n_nodes=32, plot=True, Q_func=None):
    domain = system.domain
    d = len(domain)

    X_MAX = 1e+6
    eps = 1e-7
    T = 100

    def augmented_dynamics(t, z):
        x = z[:d]
        Phi_flat = z[d:d + d * d].reshape((d, d))
        integral = z[d + d * d:].reshape((d, d))  

        # print("x: ", x)
        # print("Phi_flat: ", Phi_flat)
        # print("integral: ", integral)

        f_val = system.f_numpy(*x)
        A = system.compute_linearization(point=x)
        Phi = Phi_flat

        # print("f_val: ", f_val.shape)
        # print("A: ", A)
        # print("Phi: ", Phi)

        Phi_dot = A @ Phi
        if Q_func is not None:  
            integral_dot = Phi.T @ Q_func(x) @ Phi
        else:
            integral_dot = Phi.T @ Phi  

        return np.concatenate([f_val.flatten(), Phi_dot.flatten(), 
                               integral_dot.flatten()])  

    def solve_ode(x0):
        Phi0 = np.eye(d).flatten()
        integral0 = np.zeros((d, d)).flatten()
        z0 = np.concatenate([x0, Phi0, integral0])

        # terminate if the fundamental matrix solution converged
        def termination_criterion(t, z):
            Phi_flat = z[d:d + d * d]
            norm = np.linalg.norm(Phi_flat)
            # print(f"Termination check at time {t}: norm = {norm}")
            return norm - eps

        termination_criterion.terminal = True
        termination_criterion.direction = -1

        sol = solve_ivp(augmented_dynamics, [0, T], z0, 
                        events=termination_criterion, rtol=1e-6, atol=1e-9)
        return sol.y[:, -1]  

    def generate_train_data(x):
        result = solve_ode(x)
        x_data = x
        integral_matrix = result[-d * d:].reshape((d, d))  # Only the last dxd integral value
        y_data = integral_matrix[np.triu_indices(d)]  # Extract upper triangular elements
        return x_data, y_data

    if not os.path.exists('results'):
        os.makedirs('results')
    t_filename = f'results/{system.name}_contraction_metric_data_{n_samples}_samples.npy'

    print('_' * 50)
    print("Generating training data for neural contraction metric from numerical integration:")

    if os.path.exists(t_filename) and not overwrite:
        print("Data exists. Loading data...")
        contraction_data = np.load(t_filename)
    else:
        print("Generating new data...")
        x_train = np.array([np.random.uniform(dim[0], dim[1], n_samples) 
                            for dim in domain]).T
        results = Parallel(n_jobs=n_nodes)(delayed(generate_train_data)(x) 
                                           for x in tqdm(x_train))
        x_data = np.array([res[0] for res in results])
        # print("x_data: ", x_data.shape)
        y_data = np.array([res[1] for res in results])
        # print("y_data: ", y_data.shape)
        contraction_data = np.hstack((x_data, y_data))
        np.save(t_filename, contraction_data)

    return contraction_data


def generate_dts_data(system, n_samples=None, v_max=200, 
                      overwrite=False, n_nodes=32, plot=True, dt=0.003, mu=0.2):
    domain = system.domain
    d = len(domain)  

    X_MAX = 1e+6
    eps = 1e-7
    warning_steps = 1e+6

    def augmented_dynamics_discrete(x, z):
        dz = sum([s**2 for s in x])  # stage cost
        x_next = system.f_numpy(*x)

        # Ensure x_next is a numpy array with consistent dtype
        x_next = np.array(x_next, dtype=np.float64).flatten()
        z_next = z + dt*dz
        z_next = np.array(z_next, dtype=np.float64).flatten()
        return x_next, z_next

    def generate_train_data(x):
        current_x = np.array(x, dtype=np.float64).flatten()
        current_z = np.float64(0.0)
        steps_taken = 0

        while steps_taken < warning_steps:  # Limit to warning_steps iterations
            current_x, current_z = augmented_dynamics_discrete(current_x, current_z)
            steps_taken += 1

            # Check stopping criteria
            if np.linalg.norm(current_x) <= eps:
                y = 1 - np.exp(-mu*current_z)
                z_T = current_z
                break
            elif current_z > v_max or np.linalg.norm(current_x) > X_MAX:
                y = 1.0
                z_T = v_max  
                break
        else:
            # If the loop completes without breaking, raise a warning/exception
            warnings.warn(
                f"Warning: Maximum steps reached ({steps_taken}) without meeting stopping criteria. "
                f"x: {current_x}, z: {current_z}, norm(x): {np.linalg.norm(current_x)}",
                RuntimeWarning
            )
            y = np.nan  # Assign NaN to indicate failure
            z_T = np.nan

        # Ensure y and z_T are consistent types
        y = np.array(y, dtype=np.float64).flatten()
        z_T = np.array(z_T, dtype=np.float64).flatten()

        return [x, y, z_T]

    if not os.path.exists('results'):
        os.makedirs('results')
    t_filename = (f'results/{system.name}_data_{n_samples}_samples'
                  f'_v_max_{v_max}.npy')
    z_values = None
    print('_' * 50)
    print("Generating training data from numerical integration:")
    if os.path.exists(t_filename) and not overwrite:
        print("Data exists. Loading training data...")
        t_data = np.load(t_filename)
        x_train, y_train = t_data[:, :-1], t_data[:, -1]
    else:
        print("Generating new training data...")
        start_time = time.time()    
        x_train = np.array([np.random.uniform(dim[0], dim[1], n_samples) 
                            for dim in domain], dtype=np.float64).T
        results = Parallel(n_jobs=n_nodes)(
            delayed(generate_train_data)(x) for x in tqdm(x_train)
            )
        x_train = np.array([np.squeeze(res[0]) for res in results])
        y_train = np.array([np.squeeze(res[1]) for res in results])
        z_values = np.array([np.squeeze(res[2]) for res in results])

        # Final check to ensure no object dtype is used
        if y_train.dtype == np.object or z_values.dtype == np.object:
            raise ValueError("Detected object dtype, please check the data processing steps.")

        print("Saving training data...")
        t_data = np.column_stack((x_train, y_train))
        np.save(t_filename, t_data)
        end_time = time.time()
        print(f"Time taken for generating data: " 
              f"{end_time - start_time} seconds.\n")

    if plot:  
        plt.figure()    
        if d == 1:
            plt.scatter(x_train, np.zeros_like(x_train), c=y_train, 
                        cmap='coolwarm', s=1)  
        else:
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, 
                        cmap='coolwarm', s=1)  

        plt.savefig(
            f"results/{system.name}_data_{n_samples}_samples_v_max_{v_max}.pdf"
            )
        plt.close()

        if z_values is not None: 
            plt.figure()
            plt.scatter(z_values, np.zeros_like(z_values) + 0.5)
            plt.yticks([])
            plt.xlabel('Value')
            plt.xlim([0, v_max])
            plt.title('Z Values Clustering')
            plt.savefig(f'results/{system.name}_data_{n_samples}_samples'
                        f'_v_max_{v_max}_z_values.pdf')
            plt.close()

    return np.column_stack((x_train, y_train))


# Generate Jacobians using symbolic functions and precompile them
def generate_jacobians(f, g, variables):
    df_x = f.jacobian(variables)
    
    # Compute Jacobian of each column of g
    dg_x_cols = [g[:, j].jacobian(variables) for j in range(g.shape[1])]

    # Lambdify for efficient numeric evaluation
    df_x_func = sp.lambdify(variables, df_x, modules='numpy')
    dg_x_func_list = [sp.lambdify(variables, dg_x_col, modules='numpy') 
                      for dg_x_col in dg_x_cols]

    return df_x_func, dg_x_func_list


def generate_pmp_data(system, n_samples=None, T=200, N=1000, tol=1e-5, mu=0.1, 
                      max_nodes=20000, plot=True, save=True, df_x_numpy=None, 
                      dg_x_numpy=None, overwrite=False):
    domain = system.domain
    d = len(domain)
    m = system.symbolic_g.shape[1]  # Number of control inputs (m)
    variables = system.symbolic_vars
    Q = system.Q
    R = system.R

    t_filename = f"results/{system.name}_pmp_data_{n_samples}_samples.npy"

    # If no numerical Jacobians are provided, use symbolic ones
    if df_x_numpy is None or dg_x_numpy is None:
        df_x_func, dg_x_func_list = generate_jacobians(
            system.symbolic_f, system.symbolic_g, variables)

    t = np.linspace(0, T, N)

    # Check if data already exists and load it
    if os.path.exists(t_filename) and not overwrite:
        print(f"Data exists. Loading data from {t_filename}...")
        data = np.load(t_filename)
        return data

    # PMP ODEs for f, lambda, and V
    def pmp_odes(t, y):
        x = y[:d, :]
        lambda_ = y[d:2*d, :]
        V = y[2*d:, :]

        # Compute system dynamics using vectorized methods
        f_x = np.transpose(system.f_numpy_vectorized(x.T), (1, 0))  
        g_x = np.transpose(system.g_numpy_vectorized(x.T), (1, 2, 0))  

        # Compute the optimal control law u*
        u_star = -0.5 * np.linalg.inv(R) @ np.einsum('nmN,nN->mN', g_x, lambda_)

        # Compute the state derivatives dx/dt
        dxdt = f_x + np.einsum('nmN,mN->nN', g_x, u_star)

        # Use the provided or precompiled Jacobians
        if df_x_numpy is not None and dg_x_numpy is not None:
            df_dx = df_x_numpy(x)  # Direct use of provided numpy function
            dg_dx_reshaped = dg_x_numpy(x)
        else:
            # Use the precompiled symbolic Jacobians
            df_dx = np.array([df_x_func(*x[:, i]) for i in range(x.shape[1])])  # Shape: (N, d, d)
            dg_x_numeric = np.array([np.hstack([dg_func(*x[:, i]) 
                                     for dg_func in dg_x_func_list]) 
                                     for i in range(x.shape[1])])  # Shape: (N, d * m)

            # Reshape df_dx to (d, d, N)
            df_dx = np.moveaxis(df_dx, 0, -1)

            # Reshape dg_x to (N, d, m, d) and move axes to (d, m, d, N)
            dg_dx_reshaped = dg_x_numeric.reshape(x.shape[1], d, m, d).transpose(1, 2, 3, 0)

        # Compute the co-state derivatives d(lambda)/dt
        dlambdadt = - Q @ x - np.einsum('jik,jk->ik', df_dx, lambda_)

        # Compute dg contraction contribution to co-state derivatives
        dg_contraction = np.einsum('iN,jN,ijnN->nN', lambda_, u_star, 
                                    dg_dx_reshaped)
        dlambdadt -= dg_contraction

        # Compute the derivative of the value function dV/dt
        dVdt = -(np.einsum('ij,ji->i', x.T @ Q, x) 
                 + np.einsum('ij,ji->i', u_star.T @ R, u_star))

        return np.vstack((dxdt, dlambdadt, dVdt))


    # Boundary conditions for the PMP
    def boundary_conditions(ya, yb, x0):
        x0_cond = ya[:d] - x0
        lambda_T_cond = yb[d:2*d]  # lambda(T) = 0
        V_T_cond = yb[2*d:]  # V(T) = 0
        return np.hstack((x0_cond, lambda_T_cond, V_T_cond))

    # Function to run a single simulation
    def run_simulation(seed):
        np.random.seed(seed)
        x0 = np.random.uniform([dim[0] for dim in domain], [dim[1] for dim in domain])
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Initial guess for the solution
        y_guess = np.zeros((2 * d + 1, t.size))
        for i in range(d):
            y_guess[i] = np.linspace(x0[i], 0, t.size)

        # Solve the TPBVP using solve_bvp
        solution = solve_bvp(lambda t, y: pmp_odes(t, y), 
                             lambda ya, yb: boundary_conditions(ya, yb, x0), t, 
                             y_guess, tol=tol, max_nodes=max_nodes)

        if solution.success:
            y_initial = solution.sol(0)
            V_initial = y_initial[2*d:]  # Extract V at t=0
            w_fine = np.tanh(mu * V_initial)
            return np.array(x0), w_fine
        else:
            return None, None

    # Run simulations in parallel
    print("Generating new training data...")
    results = Parallel(n_jobs=-1)(
        delayed(run_simulation)(i) for i in tqdm(range(n_samples), 
                desc="Generating PMP data"))

    # Collect and return valid results
    collected_data_x = []
    collected_data_w = []

    for result in results:
        if result is not None:
            x_data, w_data = result
            if x_data is not None and w_data is not None:
                collected_data_x.append(x_data)  # state x
                collected_data_w.append(w_data)  # value function w

    collected_data_x = np.array(collected_data_x)
    collected_data_w = np.array(collected_data_w)

    # Save data if requested
    if save and collected_data_x.size > 0 and collected_data_w.size > 0:
        data = np.column_stack((collected_data_x, collected_data_w))
        np.save(t_filename, data)
        print(f"Data saved to {t_filename}")

    # Plot results if requested
    if plot and collected_data_x.size > 0 and collected_data_w.size > 0:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(collected_data_x[:, 0], collected_data_x[:, 1], 
                              c=collected_data_w, cmap='coolwarm', s=50)
        plt.colorbar(scatter, label="Transformed Value Function")
        plt.xlabel("State $x_1$")
        plt.ylabel("State $x_2$")
        plt.title("Transformed Value Function at Initial Conditions")

        # Fix the axes to match the domain
        plt.xlim([domain[0][0], domain[0][1]])
        plt.ylim([domain[1][0], domain[1][1]])

        # Save the plot as a PDF
        plot_filename = f"results/{system.name}_pmp_data_{n_samples}_samples.pdf"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved to {plot_filename}")

    return np.column_stack((collected_data_x, collected_data_w))
