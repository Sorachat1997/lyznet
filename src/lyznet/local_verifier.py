import time 

import numpy as np
import dreal
import lyznet
import sympy as sp

def compositional_local_stability_verifier(system, tol=1e-4, accuracy=1e-5):
    def x_bound_c(vars_list, c):
        if not vars_list:
            return None
        
        subformulas = []
        for var in vars_list:
            subformula = dreal.logical_and(var <= c, var >= -c)
            subformulas.append(subformula)
        
        final_formula = dreal.logical_and(*subformulas)
        return final_formula

    config = lyznet.utils.config_dReal(number_of_jobs=32, tol=tol)
    epsilon = 1e-5

    # Create dReal variables based on the number of symbolic variables
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    print('_' * 50)
    print("Verifying local stability using quadratic Lyapunov function and "
          "compositional analysis: ")
    print("x^TPx = ", V)

    # Create dReal expressions for f based on the symbolic expressions
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    Q = np.array(system.Q).astype(np.float64)
    r = np.min(np.linalg.eigvalsh(Q)) - epsilon

    g = [
        f[i] - sum(system.A[i][j] * x[j] for j in range(len(x))) 
        for i in range(len(x))
        ]

    Dg = [[None for _ in range(len(x))] for _ in range(len(g))]
    for i in range(len(g)):
        for j in range(len(x)):
            Dg[i][j] = g[i].Differentiate(x[j])

    P_Dg = np.dot(system.P, Dg)
    rows, cols = np.shape(P_Dg)
    variables = [f"x{i}" for i in range(1, len(system.symbolic_vars) + 1)]

    eig_P = np.linalg.eigvalsh(system.P)
    min_eig_P = eig_P[0] - epsilon

    dreal_vars_in_P_Dg = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            expr = P_Dg[i][j]
            # print(f"P_Dg[{i}][{j}]: ", expr)
            expr_str = expr.to_string()
            found_vars = [var for var in variables if var in expr_str]
            dreal_vars_in_P_Dg[i][j] = [x[variables.index(var)] 
                                        for var in found_vars]
                                        
    def Check_local_stability(c):
        frobenius_norm_sq_bound = 0.0
        x_norm_bound = np.sqrt(c/min_eig_P)
        bound_not_verified = False
        for i in range(rows):
            for j in range(cols):
                expr = P_Dg[i][j]
                x_bounds = x_bound_c(dreal_vars_in_P_Dg[i][j], x_norm_bound)

                def Check_P_Dg_ij_bound(c): 
                    condition = dreal.logical_imply(x_bounds, expr**2 <= c)
                    return dreal.CheckSatisfiability(
                        dreal.logical_not(condition), config
                        )
                if x_bounds is not None: 
                    c_best = lyznet.utils.bisection_lub(
                        Check_P_Dg_ij_bound, 0, 1000, accuracy
                        )
                else:
                    c_best = 0
                if c_best is not None:
                    frobenius_norm_sq_bound += c_best
                else:
                    bound_not_verified = True 

        if (2**2 * frobenius_norm_sq_bound <= r**2 and
                bound_not_verified is False):
            return None 
        else:
            return "Bound not verified!"

    start_time = time.time()
    c_best = lyznet.utils.bisection_glb(
        Check_local_stability, 0, 100, accuracy
        )
    end_time = time.time()
    print(f"Largest level set x^T*P*x <= {c_best} verified by linearization " 
          f"and compositional analysis.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")    
    return c_best


def local_stability_verifier(system, tol=1e-4, accuracy=1e-5):
    eigenvalues = np.linalg.eigvals(system.A)
    if system.system_type == "DiscreteDynamicalSystem":
        if any(np.abs(ev) > 1 + 1e-7 for ev in eigenvalues):
            print("The system is unstable.")
            return
        elif any(1 - 1e-7 < np.abs(ev) <= 1 + 1e-7 for ev in eigenvalues):
            print("The linearization is inconclusive for stability analysis.")
            return
    else:
        if any(np.real(ev) > 1e-7 for ev in eigenvalues):
            print("The system is unstable (continuous-time).")
            return
        elif any(-1e-7 < np.real(ev) < 1e-7 for ev in eigenvalues):
            print("The linearization is inconclusive for stability analysis.")
            return

    config = lyznet.utils.config_dReal(number_of_jobs=32, tol=tol)
    xlim = system.domain
    epsilon = 1e-4

    # Create dReal variables based on the number of symbolic variables
    x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * system.P[i][j] * x[j]
    print('_' * 50)
    print("Verifying local stability using quadratic Lyapunov function:")
    print("x^TPx = ", V)

    # Create dReal expressions for f based on the symbolic expressions
    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    Q = np.array(system.Q).astype(np.float64)
    r = np.min(np.linalg.eigvalsh(Q)) - epsilon

    g = [
        f[i] - sum(system.A[i][j] * x[j] for j in range(len(x))) 
        for i in range(len(x))
        ]

    Dg = [[None for _ in range(len(x))] for _ in range(len(g))]
    for i in range(len(g)):
        for j in range(len(x)):
            Dg[i][j] = g[i].Differentiate(x[j])

    P_Dg = np.dot(system.P, Dg)
    if system.system_type == "DiscreteDynamicalSystem": 
        A_P_Dg = np.dot(system.A.T, P_Dg)
        frobenius_norm_sq_A_P_Dg = sum(sum(m_ij**2 for m_ij in row) 
                                       for row in A_P_Dg)
        frobenius_norm_sq_Dg = sum(sum(m_ij**2 for m_ij in row) for row in Dg)
        P_norm = np.linalg.norm(system.P, ord=2)
        norm_sq_bound = 8*frobenius_norm_sq_A_P_Dg + 2*P_norm**2*frobenius_norm_sq_Dg**2
        h = norm_sq_bound <= r**2
    else:
        # Frobenios norm as an overa-approximation of 2-norm
        # frobenius_norm_sq_P_Dg = sum(sum(m_ij**2 for m_ij in row) for row in P_Dg)
        # h = (2**2 * frobenius_norm_sq_P_Dg) <= r**2
        P_Dg_symmetric = P_Dg + P_Dg.T
        frobenius_norm_sq_P_Dg = sum(sum(m_ij**2 for m_ij in row) for row in P_Dg_symmetric)
        h = frobenius_norm_sq_P_Dg <= r**2

    start_time = time.time()
    if dreal.CheckSatisfiability(dreal.logical_not(h), config) is None:
        end_time = time.time()
        print("Global asymptotic stability verified by linearization!")
        print(f"Time taken for verification: {end_time - start_time}"
              " seconds.\n")
    else:
        print("Quadratic Lyapunov function can't verify global stability.\n"
              "Verifying local stability...")

    def Check_local_stability(c):    
        x_bound = lyznet.utils.get_bound(x, xlim, V, c2_V=c)
        omega = V <= c
        stability = dreal.logical_imply(
            dreal.logical_and(omega, x_bound), h
            )
        x_boundary = dreal.logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = dreal.logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = dreal.logical_or(x[i] == xlim[i][1], x_boundary)
        set_inclusion = dreal.logical_imply(
            x_bound, dreal.logical_not(x_boundary)
            )
        condition = dreal.logical_and(stability, set_inclusion)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

    start_time = time.time()
    c_best = lyznet.utils.bisection_glb(
        Check_local_stability, 0, 1000, accuracy
        )
    end_time = time.time()
    print(f"Largest level set x^T*P*x <= {c_best} verified by linearization.")
    print(f"Time taken for verification: {end_time - start_time} seconds.\n")
    return c_best


def local_stabilization_verifier(system, tol=1e-4, accuracy=1e-5, c_max=100):
    if system.P is None:
        return
    print("Verifying stabilization by linear controller...")
    u = sp.Matrix(system.K) * sp.Matrix(system.symbolic_vars)
    f_u = lyznet.get_closed_loop_f_expr(system.f, system.g, u, 
                                        system.symbolic_vars)
    closed_loop_sys = lyznet.DynamicalSystem(f_u, system.domain, system.name)
    closed_loop_sys.P = system.P
    c1_P = lyznet.local_stability_verifier(closed_loop_sys)
    c2_P = lyznet.quadratic_reach_verifier(closed_loop_sys, c1_P, c_max=c_max)
    return c2_P
