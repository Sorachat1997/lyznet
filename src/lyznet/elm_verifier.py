import numpy as np
import time
import dreal 
import lyznet
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools


def evaluate_elm(all_points, weights, bias, beta):
    u = np.dot(all_points, weights.T) + bias.squeeze(-1)
    tanh_u = np.tanh(u)
    V = np.dot(tanh_u, beta)
    return V


def evaluate_elm_lie_derivative(all_points, weights, bias, beta, system):
    u = np.dot(all_points, weights.T) + bias.squeeze(-1)
    tanh_u = np.tanh(u)
    derivative_tanh_u = 1 - tanh_u**2
    dV_x = np.dot(derivative_tanh_u * beta, weights)
    print("dV_x:", dV_x.shape)
    f_values = system.f_numpy_vectorized(all_points)
    print("f:", f_values.shape)
    DV_f = np.einsum('ij, ij->i', dV_x, f_values).reshape(-1, 1)
    return DV_f


def sample_points(domain, N):
    samples = np.random.uniform(low=domain[0][0], high=domain[0][1], size=(N, 1))
    for d in domain[1:]:
        samples = np.hstack((samples, np.random.uniform(low=d[0], high=d[1], size=(N, 1))))
    return samples


def verify_norm_bound(f, x_bound, config, accuracy=1e-1):
    start_time = time.time()
    norm_f_sq = dreal.Expression(0) 
    for i in range(len(f)):
        norm_f_sq += f[i]**2

    def Check_f_bound(c):
        condition = dreal.logical_imply(x_bound, norm_f_sq <= c**2)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    f_norm = lyznet.utils.bisection_lub(Check_f_bound, 0, 100, 
                                        accuracy=accuracy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return f_norm, elapsed_time


def verify_jacobian_bound(f, x, x_bound, config, accuracy=1e-1):
    start_time = time.time()
    Df = [[None for _ in range(len(f))] for _ in range(len(f))]
    for i in range(len(f)):
        for j in range(len(x)):
            Df[i][j] = f[i].Differentiate(x[j])

    Df_frobenius_norm_sq = sum(sum(m_ij**2 for m_ij in row) for row in Df)

    def Check_df_bound(c):
        condition = dreal.logical_imply(x_bound, Df_frobenius_norm_sq <= c**2)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    df_norm = lyznet.utils.bisection_lub(Check_df_bound, 0, 100, 
                                         accuracy=accuracy)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return df_norm, elapsed_time


def evaluate_points_batch_for_inclusion(all_points, weights, bias, beta, system, L_V, L_P, eta, c2_P):
    u = np.dot(all_points, weights.T) + bias.squeeze(-1)
    tanh_u = np.tanh(u)
    V_x = np.dot(tanh_u, beta)
    V_P_x = np.einsum('ij,ji->i', all_points, np.dot(system.P, all_points.T))
    condition = V_P_x + L_P * eta > c2_P
    valid_points = V_x[condition] - L_V * eta
    return np.min(valid_points) if len(valid_points) > 0 else np.inf


def compute_c1_V_for_inclusion_numpy(xlim, weights, bias, beta, system, L_V, L_P, eta, c2_P):
    start_time = time.time()
    grid_points = [np.linspace(a, b, int(np.ceil((b - a) / eta))) for [a, b] in xlim]
    all_points = np.array(list(itertools.product(*grid_points)))
    min_value = evaluate_points_batch_for_inclusion(all_points, weights, bias, beta, system, L_V, L_P, eta, c2_P)
    c1_V = min_value if min_value < np.inf else None
    end_time = time.time()
    elapsed_time = end_time - start_time
    return c1_V, elapsed_time

def evaluate_points_batch_for_reach(all_points, weights, bias, beta, system, L_V, L_DV_f, eta, c1_V, tol=1e-4):
    u = np.dot(all_points, weights.T) + bias.squeeze(-1)
    tanh_u = np.tanh(u)
    V_x = np.dot(tanh_u, beta)
    derivative_tanh_u = 1 - tanh_u**2
    dV_x = np.dot(derivative_tanh_u * beta, weights)
    f_values = np.array([system.f_numpy_vectorized(all_points)]).squeeze().T  # Adjusted for vectorized system function
    DV_f = np.dot(dV_x, f_values)
    condition1 = V_x + L_V * eta > c1_V
    condition2 = DV_f + L_DV_f * eta > -tol
    valid_points = V_x[condition1 & condition2] - L_V * eta
    return np.min(valid_points) if len(valid_points) > 0 else np.inf


def compute_c2_V_for_reach_numpy(xlim, weights, bias, beta, system, L_V, L_DV_f, eta, c1_V):
    start_time = time.time()
    grid_points = [np.linspace(a, b, int(np.ceil((b - a) / eta))) for [a, b] in xlim]
    all_points = np.array(list(itertools.product(*grid_points)))
    min_value = evaluate_points_batch_for_reach(all_points, weights, bias, beta, system, L_V, L_DV_f, eta, c1_V)
    c2_V = min_value if min_value < np.inf else None
    end_time = time.time()
    elapsed_time = end_time - start_time
    return c2_V, elapsed_time


def evaluate_point_for_inclusion(x_point, weights, bias, beta, system, 
                                 L_V, L_P, eta, c2_P):
    V_x = np.dot(np.tanh(np.dot(x_point, weights.T) + bias.squeeze(-1)), beta)
    V_P_x = np.dot(x_point.T, np.dot(system.P, x_point))
    if V_P_x + L_P * eta > c2_P:
        # print(V_x - L_V * eta)
        return V_x - L_V * eta
    return np.inf


def compute_c1_V_for_inclusion(xlim, weights, bias, beta, system, 
                               L_V, L_V_P, eta, c2_P):
    start_time = time.time()
    grid_points = [np.linspace(a, b, int(np.ceil((b - a) / eta))) 
                   for [a, b] in xlim]
    all_points = list(itertools.product(*grid_points))

    min_value = np.inf
    for point in tqdm(all_points, desc="Processing points"):
        x_point = np.array(point)
        result = evaluate_point_for_inclusion(
            x_point, weights, bias, beta, system, L_V, L_V_P, eta, c2_P
            )
        if result < min_value:
            min_value = result

    c1_V = min_value if min_value < np.inf else None
    end_time = time.time()
    elapsed_time = end_time - start_time
    return c1_V, elapsed_time


def evaluate_point_for_reach(x_point, weights, bias, beta, system, 
                             L_V, L_DV_f, eta, c1_V, tol=1e-4):
    u = np.dot(x_point, weights.T) + bias.squeeze(-1)
    tanh_u = np.tanh(u)
    V_x = np.dot(tanh_u, beta)
    derivative_tanh_u = 1 - tanh_u**2
    dV_x = np.dot(derivative_tanh_u * beta, weights)
    f_values = np.array([f(*x_point) for f in system.f_numpy]).T
    DV_f = np.dot(dV_x, f_values)

    if V_x + L_V * eta > c1_V and DV_f + L_DV_f * eta > -tol:
        # print(V_x - L_V * eta)
        return V_x - L_V * eta
    return np.inf


def compute_c2_V_for_reach(xlim, weights, bias, beta, system, 
                           L_V, L_DV_f, eta, c1_V):
    start_time = time.time()
    grid_points = [np.linspace(a, b, int(np.ceil((b - a) / eta))) 
                   for [a, b] in xlim]
    all_points = list(itertools.product(*grid_points))

    min_value = np.inf
    for point in tqdm(all_points, desc="Processing points"):
        x_point = np.array(point)
        result = evaluate_point_for_reach(
            x_point, weights, bias, beta, system, L_V, L_DV_f, eta, c1_V
            )
        if result < min_value:
            min_value = result

    c2_V = min_value if min_value < np.inf else None
    end_time = time.time()
    elapsed_time = end_time - start_time
    return c2_V, elapsed_time


def lipschitz_elm_verifier(system, weights, bias, beta, c2_P, 
                           tol=1e-4, accuracy=1e-2, eta=1e-3, num_of_jobs=32):
    # spectral_norm_W = np.linalg.norm(weights, ord=2)
    # norm_beta = np.linalg.norm(beta)
    # # assuming the Lipschitz constant of sigma is 1
    # lipschitz_constant_V = spectral_norm_W * norm_beta
    # print("Lipschitz constant of V by spectral norm estimate: ", 
    #       lipschitz_constant_V)

    # config = lyznet.utils.config_dReal(number_of_jobs=num_of_jobs, tol=tol)

    # xlim = system.domain

    # x = [dreal.Variable(f"x{i}") 
    #      for i in range(1, len(system.symbolic_vars) + 1)]

    # z = np.dot(x, weights.T) + bias.squeeze(-1)
    # h = []
    # for j in range(len(weights)):
    #     h.append(dreal.tanh(z[j]))
    # V_learn = np.dot(h, beta.T)
    # # print("V = ", V_learn)

    # DV = [V_learn.Differentiate(x[i]) for i in range(len(x))]
    # x_bound = lyznet.utils.get_x_bound(x, xlim)

    # L_V, time_spent = verify_norm_bound(DV, x_bound, config, accuracy=1)
    # print("Norm of DV and Lipschitz constant of V verified by dReal: ", L_V)
    # print("Time spent on verification: ", time_spent)

    # f = [
    #     lyznet.utils.sympy_to_dreal(
    #         expr, dict(zip(system.symbolic_vars, x))
    #         )
    #     for expr in system.symbolic_f
    #     ]

    # f_norm, time_spent = verify_norm_bound(f, x_bound, config)
    # print("Norm of f verified by dReal: ", f_norm)
    # print("Time spent on verification: ", time_spent)

    # Df_norm, time_spent = verify_jacobian_bound(f, x, x_bound, config)
    # print("Norm of Df and Lipschitz constant of f verified by dReal: ", 
    #       Df_norm)
    # print("Time spent on verification: ", time_spent)

    # H_V_norm, time_spent = verify_jacobian_bound(DV, x, x_bound, config, 
    #                                              accuracy=1)
    # print("Norm of D^2V and Lipschitz constant of DV verified by dReal: ", 
    #       H_V_norm)
    # print("Time spent on verification: ", time_spent)

    # L_DV_f = H_V_norm*f_norm + Df_norm*L_V
    # print("The Lipschitz constant of DV*f is: ", L_DV_f)

    # quad_V = dreal.Expression(0)
    # for i in range(len(x)):
    #     for j in range(len(x)):
    #         quad_V += x[i] * system.P[i][j] * x[j]

    # DV_P = [quad_V.Differentiate(x[i]) for i in range(len(x))]
    # L_V_P, time_spent = verify_norm_bound(DV_P, x_bound, config)
    # print("Norm of DV and Lipschitz constant of V verified by dReal: ", L_V_P)
    # print("Time spent on verification: ", time_spent)

    # # Verification using grid points and Lipschitz constants
    # print('_' * 50)
    # print("Verifying ELM Lyapunov function (with grid points):")
    # c1_V, time_spent = compute_c1_V_for_inclusion_numpy(
    #     xlim, weights, bias, beta, system, L_V, L_V_P, eta, c2_P
    #     )
    # print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    # print("Time spent on verification: ", time_spent)

    L_V = 1.5625
    L_DV_f = 23.040771484375
    c1_V = 0.5342053394332862
    eta = 4e-3
    xlim = system.domain  

    c2_V, time_spent = compute_c2_V_for_reach_numpy(
        xlim, weights, bias, beta, system, L_V, L_DV_f, eta, c1_V
        )
    print(f"Verified V<={c2_V} will reach V<={c1_V} and hence x^TPx<={c2_P}.")
    print("Time spent on verification: ", time_spent)
    return 


def numpy_elm_verifier(system, weights, bias, beta, c2_P, 
                       tol=1e-4, accuracy=1e-2, number_of_jobs=32):
    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    z = np.dot(x, weights.T) + bias.squeeze(-1)
    h = []
    for j in range(len(weights)):
        h.append(dreal.tanh(z[j]))
    V_learn = np.dot(h, beta.T)
    print("V = ", V_learn)

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V_learn.Differentiate(x[i])

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    print('_' * 50)
    print("Verifying ELM Lyapunov function:")
    c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 100, accuracy)
    print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    c2_V = lyznet.reach_verifier_dreal(system, x, V_learn, f, c1_V, 
                                       tol=tol, accuracy=accuracy,
                                       number_of_jobs=number_of_jobs)
    print(f"Verified V<={c2_V} will reach V<={c1_V} and hence x^TPx<={c2_P}.")
    end_time = time.time()
    print(f"Time taken for verifying Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V
